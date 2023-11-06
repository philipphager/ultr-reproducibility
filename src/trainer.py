import logging
from functools import partial
from typing import Dict, Callable

import flax.linen as nn
import jax
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import jit
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.log import print_metric_table
from src.util import EarlyStopping, collect_metrics, aggregate_metrics

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

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainState:
        state = self._init_train_state(model, train_loader)
        best_model_state = None

        for epoch in range(self.epochs):
            state = self._train_epoch(state, train_loader, f"Epoch: {epoch} - Training")
            val_df = self._eval_epoch(state, val_loader, f"Epoch: {epoch} - Val")
            val_metrics = aggregate_metrics(val_df)

            has_improved, should_stop = self.early_stopping.update(val_metrics)
            logger.info(f"Epoch {epoch}: {val_metrics}, has_improved: {has_improved}")

            if has_improved:
                best_model_state = state

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        return best_model_state

    def test(
        self,
        state: TrainState,
        test_loader: DataLoader,
    ) -> DataFrame:
        test_df = self._eval_epoch(state, test_loader, "Testing")
        test_metrics = aggregate_metrics(test_df)
        print_metric_table(test_metrics)

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

    def _train_epoch(self, state, loader, description):
        for batch in tqdm(loader, desc=description):
            state, loss = self._train_step(state, batch)

        return state

    def _eval_epoch(self, state, loader, description):
        metrics = []

        for batch in tqdm(loader, desc=description):
            metrics.append(self._eval_step(state, batch))

        return collect_metrics(metrics)

    @partial(jit, static_argnums=(0,))
    def _train_step(self, state, batch):
        dropout_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

        def loss_fn(params):
            y_predict = state.apply_fn(
                params,
                batch,
                training=True,
                rngs={"dropout": dropout_key},
            )
            return self.criterion(y_predict, batch["click"], where=batch["mask"])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    @partial(jit, static_argnums=(0,))
    def _eval_step(self, state, batch):
        label, mask = batch["label"], batch["mask"]
        y_predict = state.apply_fn(
            state.params,
            batch,
            training=False,
            rngs={"dropout": state.dropout_key},
        )

        results = {"query_id": batch["query_id"]}

        for name, metric_fn in self.metric_fns.items():
            results[name] = metric_fn(y_predict, label, where=mask, reduce_fn=None)

        return results
