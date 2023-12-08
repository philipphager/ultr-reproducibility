import pandas as pd
import enum
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Callable, Tuple, Optional

import flax.linen as nn
import jax
import wandb
from flax.training import train_state
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
        save_checkpoints: bool = True,
        log_metrics: bool = True,
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.save_checkpoints = save_checkpoints
        self.log_metrics = log_metrics

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_click_loader: Optional[DataLoader],
        val_rel_loader: Optional[DataLoader],
    ) -> Tuple[TrainState, pd.DataFrame]:
        start_time = time.time()
        state = self._init_train_state(model, train_loader)
        best_model_state = state
        history = []

        for epoch in range(self.epochs):
            state, train_loss = self._train_epoch(
                model, state, train_loader, f"Epoch: {epoch} - Training"
            )

            val_click_df, val_rel_df = self._eval_epoch(
                model, state, val_click_loader, val_rel_loader, f"Epoch: {epoch} - Val"
            )

            val_click_metrics = aggregate_metrics(val_click_df)
            val_rel_metrics = aggregate_metrics(val_rel_df)
            val_metrics = {**val_click_metrics, **val_rel_metrics}

            has_improved, should_stop = self.early_stopping.update(val_metrics)

            if has_improved:
                best_model_state = state

                if self.save_checkpoints:
                    save_state(state, Path(os.getcwd()), "best_state")

            epoch_metrics = {
                "Val/": val_metrics,
                "Train/loss": train_loss,
                "Misc/TimePerEpoch": (time.time() - start_time) / (epoch + 1),
                "Misc/Epoch": epoch,
            }

            # Flatten nested dictionary to a single level:
            flat_metrics = pd.json_normalize(epoch_metrics, sep="")
            logger.info(f"Epoch {epoch}: {flat_metrics}, has_improved: {has_improved}")
            history.append(flat_metrics)

            if self.log_metrics:
                wandb.log(epoch_metrics, step=epoch)

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        history_df = pd.DataFrame(history)
        return best_model_state, history_df

    def eval(
        self,
        model: nn.Module,
        state: TrainState,
        test_click_loader: Optional[DataLoader],
        test_rel_loader: Optional[DataLoader],
        stage: Stage = Stage.TEST,
    ) -> Tuple[DataFrame, DataFrame]:
        test_click_df, test_rel_df = self._eval_epoch(
            model,
            state,
            test_click_loader,
            test_rel_loader,
            stage,
        )
        test_click_metrics = aggregate_metrics(test_click_df)
        test_rel_metrics = aggregate_metrics(test_rel_df)
        test_metrics = {**test_click_metrics, **test_rel_metrics}
        print_metric_table(test_metrics, stage)

        if self.log_metrics and stage == Stage.TEST:
            wandb.log({"Test/": test_metrics})

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

            return self.criterion(y_predict, batch["click"], where=batch["mask"])

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

        # This is an issue with rax's API: reduce_fn behaves differently for pointwise and listwise losses
        reduce_fn = lambda a, where: a.reshape(len(a), -1).mean(axis=1, where=where.reshape(len(where), -1))
        loss = self.criterion(
            y_predict,
            batch["click"],
            where=batch["mask"],
            reduce_fn=reduce_fn,
        )

        results = {"click_loss": loss}

        for name, metric_fn in self.metric_fns.items():
            results[f"BC_{name}"] = metric_fn(
                rel_predict,
                1 / batch["position"],
                where=batch["mask"],
                reduce_fn=None,
            )

        return results

    @partial(jit, static_argnums=(0, 1))
    def _eval_rel_step(self, model, state, batch):
        y_predict = model.apply(
            state.params,
            batch,
            training=False,
            rngs={"dropout": state.dropout_key},
            method=model.predict_relevance,
        )

        results = {
            "query_id": batch["query_id"],
            "frequency_bucket": batch["frequency_bucket"],
        }

        for name, metric_fn in self.metric_fns.items():
            results[name] = metric_fn(
                y_predict,
                batch["label"],
                where=batch["mask"],
                reduce_fn=None,
            )

        return results
