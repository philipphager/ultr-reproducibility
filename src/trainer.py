import enum
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, Callable

import flax.linen as nn
import jax
import wandb
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import jit
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.log import print_metric_table
from src.util import EarlyStopping, collect_metrics, aggregate_metrics, save_state

logger = logging.getLogger("rich")


class TrainState(train_state.TrainState):
    dropout_key: jax.Array


class Stage(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Trainer:
    def __init__(
        self,
        random_state: int,
        optimizer,
        criterion,
        metric_fns: Dict[str, Callable],
        epochs: int,
        early_stopping: EarlyStopping,
        checkpoint: bool,
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
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
            state = self._train_epoch(
                model, state, train_loader, f"Epoch: {epoch} - Training"
            )
            val_df = self._eval_epoch(model, state, val_loader, f"Epoch: {epoch} - Val")
            val_metrics = aggregate_metrics(val_df)

            wandb.log({Stage.VAL: val_metrics}, epoch)

            has_improved, should_stop = self.early_stopping.update(val_metrics)
            logger.info(f"Epoch {epoch}: {val_metrics}, has_improved: {has_improved}")

            if has_improved:
                best_model_state = state

                if self.checkpoint:
                    save_state(state, Path(os.getcwd()), "best_state")

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        return best_model_state

    def eval(
        self,
        model: nn.Module,
        state: TrainState,
        loader: DataLoader,
        stage: Stage = Stage.TEST,
    ) -> DataFrame:
        eval_df = self._eval_epoch(model, state, loader, stage)
        metrics = aggregate_metrics(eval_df)
        wandb.log({stage: metrics})
        print_metric_table(metrics, stage)

        return eval_df

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
        for batch in tqdm(loader, desc=description):
            state, loss = self._train_step(model, state, batch)
            if self.global_step % (100 * loader.batch_size) == 0:
                wandb.log({Stage.TRAIN: {"loss": loss}}, self.global_step)
            self.global_step += loader.batch_size

        return state

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
            return self.criterion(y_predict, batch["click"], where=batch["mask"])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

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
