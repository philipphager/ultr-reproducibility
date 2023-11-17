import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, Callable

import flax.linen as nn
import jax
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import jit
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.log import print_metric_table
from src.util import EarlyStopping, collect_metrics, aggregate_metrics, save_state

logger = logging.getLogger("rich")


class TrainState(train_state.TrainState):
    dropout_key: jax.Array


class Trainer:
    def __init__(
        self,
        random_state: int,
        optimizer,
        criterion,
        metric_fns: Dict[str, Callable],
        epochs: int,
        early_stopping: EarlyStopping,
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.global_step = 0

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainState:
        state = self._init_train_state(model, train_loader)
        best_model_state = None

        for epoch in range(self.epochs):
            state, epoch_loss, epoch_loss_per_rank = self._train_epoch(
                model, state, train_loader, f"Epoch: {epoch} - Training"
            )
            epoch_loss_per_rank = {f"loss@{i}": val for i,val in zip([1,2,5,10], jax.device_get(epoch_loss_per_rank))}
            val_df = self._eval_epoch(model, state, val_loader, f"Epoch: {epoch} - Val")
            val_metrics = aggregate_metrics(val_df)

            wandb.log({"AggMetrics/val" : val_metrics, 
                       "AggMetrics/train.loss" : epoch_loss, 
                       "RankMetrics/train": epoch_loss_per_rank}, step = epoch)

            has_improved, should_stop = self.early_stopping.update(val_metrics)
            logger.info(f"Epoch {epoch}: {val_metrics}, has_improved: {has_improved}")

            if has_improved:
                best_model_state = state
                save_state(state, Path(os.getcwd()), "best_state")

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        return best_model_state

    def test(
        self,
        model: nn.Module,
        state: TrainState,
        test_loader: DataLoader,
        description: str = "Testing",
    ) -> DataFrame:
        test_df = self._eval_epoch(model, state, test_loader, description)
        test_metrics = aggregate_metrics(test_df)
        wandb.log({"AggMetrics/test" : test_metrics})
        print_metric_table(test_metrics, description)

        return test_df

    def _init_train_state(self, model, train_loader):
        key = jax.random.PRNGKey(self.random_state)
        key, param_key, dropout_key = jax.random.split(key, num=3)

        init_rngs = {"params": param_key, "dropout": dropout_key}
        init_data = next(iter(train_loader))
        params = model.init(init_rngs, init_data, training=True)

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self.optimizer,
            dropout_key=dropout_key,
        )

    def _train_epoch(self, model, state, loader, description):
        epoch_loss = 0
        epoch_loss_per_rank = jax.numpy.zeros(4)
        for batch in tqdm(loader, desc=description):
            state, loss, loss_per_rank = self._train_step(model, state, batch)
            epoch_loss += loss
            epoch_loss_per_rank += loss_per_rank[jax.numpy.array([0, 1, 4, 9])]
            self.global_step += loader.batch_size
        epoch_loss /= len(loader)
        epoch_loss_per_rank /= len(loader)
        return state, epoch_loss, epoch_loss_per_rank

    def _eval_epoch(self, model, state, loader, description):
        metrics = []

        for batch in tqdm(loader, desc=description):
            metrics.append(self._eval_step(model, state, batch))

        return collect_metrics(metrics)

    @partial(jit, static_argnums=(0, 1))
    def _train_step(self, model, state, batch):
        dropout_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

        def loss_fn(params):
            y_predict = model.apply(
                params,
                batch,
                training=True,
                rngs={"dropout": dropout_key},
            )
            loss_per_rank = self.criterion(y_predict, batch["click"], where=batch["mask"], reduce_fn = None)
            return loss_per_rank.mean(), loss_per_rank.mean(axis = 0)

        (loss, loss_per_rank), grads = jax.value_and_grad(loss_fn, has_aux = True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss, loss_per_rank

    @partial(jit, static_argnums=(0, 1))
    def _eval_step(self, model, state, batch):
        query_ids, label, mask = batch["query_id"], batch["label"], batch["mask"]

        y_predict = model.apply(
            state.params,
            batch,
            training=False,
            rngs={"dropout": state.dropout_key},
            method=model.predict_relevance,
        )

        results = {"query_id": query_ids}

        for name, metric_fn in self.metric_fns.items():
            results[name] = metric_fn(y_predict, label, where=mask, reduce_fn=None)

        return results
