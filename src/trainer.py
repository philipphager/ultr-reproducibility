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
    random_model_key: jax.Array


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
        progress_bar: bool = True,
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.save_checkpoints = save_checkpoints
        self.log_metrics = log_metrics
        self.progress_bar = progress_bar

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
            logger.info(f"Epoch {epoch}: {val_metrics}, has_improved: {has_improved}")

            if has_improved:
                best_model_state = state

                if self.save_checkpoints:
                    save_state(state, Path(os.getcwd()), "best_state")

            epoch_metrics = {
                "Val/": val_metrics,
                "Train/loss": float(train_loss),
                "Misc/TimePerEpoch": (time.time() - start_time) / (epoch + 1),
                "Misc/Epoch": epoch,
            }
            history.append(pd.json_normalize(epoch_metrics, sep=""))

            if self.log_metrics:
                wandb.log(epoch_metrics, step=epoch)

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        history_df = pd.concat(history) if len(history) > 0 else pd.DataFrame([])
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
        key, param_key, dropout_key, random_model_key = jax.random.split(key, num=4)
        init_rngs = {
            "params": param_key,
            "dropout": dropout_key,
            "random_model": random_model_key,
        }
        init_data = next(iter(train_loader))
        params = model.init(init_rngs, init_data, training=True)

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self.optimizer,
            dropout_key=dropout_key,
            random_model_key=random_model_key,
        )

    def _train_epoch(self, model, state, loader, description):
        epoch_loss = 0

        for batch in tqdm(loader, desc=description, disable=not self.progress_bar):
            state, loss = self._train_step(model, state, batch, state.step)
            epoch_loss += loss

        epoch_loss /= len(loader)
        return state, epoch_loss

    def _eval_epoch(self, model, state, click_loader, rel_loader, description):
        click_metrics, rel_metrics = [], []

        for step, batch in tqdm(
            enumerate(click_loader), desc=description, disable=not self.progress_bar
        ):
            click_metrics.append(self._eval_click_step(model, state, batch, step))

        for step, batch in tqdm(
            enumerate(rel_loader), desc=description, disable=not self.progress_bar
        ):
            rel_metrics.append(self._eval_rel_step(model, state, batch, step))

        return collect_metrics(click_metrics), collect_metrics(rel_metrics)

    @partial(jit, static_argnums=(0, 1))
    def _train_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        def loss_fn(params):
            y_predict, _, _ = model.apply(
                params,
                batch,
                training=True,
                rngs=rngs,
            )

            return self.criterion(y_predict, batch["click"], where=batch["mask"])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    @partial(jit, static_argnums=(0, 1))
    def _eval_click_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        y_predict, rel_predict, _ = model.apply(
            state.params,
            batch,
            training=False,
            rngs=rngs,
        )

        # This is an issue with rax's API: reduce_fn behaves differently for pointwise and listwise losses
        reduce_fn = lambda a, where: a.reshape(len(a), -1).mean(axis=1,where=where.reshape(len(where), -1))
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
    def _eval_rel_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        y_predict = model.apply(
            state.params,
            batch,
            training=False,
            rngs=rngs,
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

    @staticmethod
    def generate_rngs(state: TrainState, step: int) -> Dict[str, jax.Array]:
        # Folding in the step number to generate a new random key.
        dropout = jax.random.fold_in(key=state.dropout_key, data=step)
        random_model = jax.random.fold_in(key=state.random_model_key, data=step)

        return {"dropout": dropout, "random_model": random_model}
