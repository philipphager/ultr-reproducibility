import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, Callable, Tuple, Optional

import flax.linen as nn
import jax
from jax import jit
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import time
from flax.training import train_state
class TrainState(train_state.TrainState):
    dropout_key: jax.Array

from src.log import print_metric_table
from src.util import EarlyStopping, collect_metrics, aggregate_metrics, save_state

logger = logging.getLogger("rich")


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
        val_click_loader: Optional[DataLoader],
        val_rel_loader: Optional[DataLoader],
        log_metrics: bool = True,
    ) -> TrainState:
        state = self._init_train_state(model, train_loader)
        best_model_state = state

        start_time = time.time()
        for epoch in range(self.epochs):
            state, epoch_loss = self._train_epoch(
                model, state, train_loader, f"Epoch: {epoch} - Training"
            )
            val_click_df, val_rel_df = self._eval_epoch(
                model, state, val_click_loader, val_rel_loader, f"Epoch: {epoch} - Val"
            )
            val_metrics = aggregate_metrics(val_click_df, val_rel_df)

            has_improved, should_stop = self.early_stopping.update(val_metrics)
            logger.info(f"Epoch {epoch}: {val_metrics}, has_improved: {has_improved}")

            if has_improved:
                best_model_state = state
                save_state(state, Path(os.getcwd()), "best_state")

            if log_metrics:
                wandb.log(
                    {
                        "Metrics/val": val_metrics,
                        "Metrics/train.loss": epoch_loss,
                        "Misc/TimePerEpoch": (time.time() - start_time) / (epoch + 1),
                    },
                    step=epoch,
                )

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        return best_model_state

    def test(
        self,
        model: nn.Module,
        state: TrainState,
        test_click_loader: Optional[DataLoader],
        test_rel_loader: Optional[DataLoader],
        description: str = "Testing",
        log_metrics: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        test_click_df, test_rel_df = self._eval_epoch(model, state, test_click_loader, test_rel_loader, description)
        test_metrics = aggregate_metrics(test_click_df, test_rel_df)
        if log_metrics and description == "Testing":
            wandb.log({"Metrics/test": test_metrics})
        print_metric_table(test_metrics, description)

        return test_click_df, test_rel_df

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
        for batch in tqdm(loader, desc=description):
            state, loss = self._train_step(model, state, batch)
            epoch_loss += loss
            self.global_step += loader.batch_size
        epoch_loss /= len(loader)
        return state, epoch_loss

    def _eval_epoch(self, model, state, click_loader, rel_loader, description):
        click_metrics, rel_metrics = [], []

        for batch in tqdm(click_loader, desc=description):
            click_metrics.append(self._eval_click_step(model, state, batch))
        for batch in tqdm(rel_loader, desc=description):
            rel_metrics.append(self._eval_rel_step(model, state, batch))

        return collect_metrics(click_metrics), collect_metrics(rel_metrics)

    @partial(jit, static_argnums=(0, 1))
    def _train_step(self, model, state, batch):
        dropout_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

        def loss_fn(params):
            y_predict, _, _ = model.apply(
                params,
                batch,
                training=True,
                rngs={"dropout": dropout_key},
            )
            loss = self.criterion(y_predict, batch["click"], where=batch["mask"])
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    @partial(jit, static_argnums=(0, 1))
    def _eval_click_step(self, model, state, batch):
        y_predict, rel_predict, _ = model.apply(
            state.params,
            batch,
            training=False,
            rngs={"dropout": state.dropout_key},
        )
        reduce_fn = lambda a, where: a.reshape(len(a), -1).mean(axis=1, where=where)
        # This is an issue with rax's API: reduce_fn behaves differently for pointwise and listwise losses
        results = {
            "click_loss": self.criterion(
                y_predict, batch["click"], where=batch["mask"], reduce_fn=reduce_fn
            )
        }

        for name, metric_fn in self.metric_fns.items():
            results[f"click_{name}"] = metric_fn(
                rel_predict, 1 / batch["position"], where=batch["mask"], reduce_fn=None
            )
        return results

    @partial(jit, static_argnums=(0, 1))
    def _eval_rel_step(self, model, state, batch):
        query_ids, label, mask, frequency_buckets = batch["query_id"], batch["label"], batch["mask"], batch["frequency_bucket"]

        y_predict = model.apply(
            state.params,
            batch,
            training=False,
            rngs={"dropout": state.dropout_key},
            method=model.predict_relevance,
        )

        results = {"query_id": query_ids, "frequency_bucket": frequency_buckets}

        for name, metric_fn in self.metric_fns.items():
            results[name] = metric_fn(y_predict, label, where=mask, reduce_fn=None)

        return results
