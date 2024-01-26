import enum
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional

import flax.linen as nn
import jax
import pandas as pd
import wandb
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
from jax import jit
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.util import save_state, collect_metrics, aggregate_metrics, reciprocal_rank


class TrainState(train_state.TrainState):
    dropout_key: jax.Array
    random_model_key: jax.Array


class Stage(str, enum.Enum):
    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


class Trainer:
    def __init__(
        self,
        random_state: int,
        optimizer,
        metric_fns: Dict[str, Callable],
        click_metric_fns: Dict[str, Callable],
        epochs: int,
        early_stopping: EarlyStopping,
        save_checkpoints: bool = True,
        log_metrics: bool = True,
        progress_bar: bool = True,
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.metric_fns = metric_fns
        self.click_metric_fns = click_metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.save_checkpoints = save_checkpoints
        self.log_metrics = log_metrics
        self.progress_bar = progress_bar

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[TrainState, pd.DataFrame]:
        start_time = time.time()
        state = self._init_train_state(model, train_loader)
        best_model_state = state
        history = []

        for epoch in range(self.epochs):
            state, train_loss = self._train_epoch(
                model, state, train_loader, f"Epoch: {epoch} - Training"
            )

            val_loss = self._val_epoch(
                model, state, val_loader, f"Epoch: {epoch} - Val"
            )

            has_improved, self.early_stopping = self.early_stopping.update(val_loss)

            if has_improved:
                best_model_state = state

                if self.save_checkpoints:
                    save_state(state, Path(os.getcwd()), "best_state")

            epoch_metrics = {
                "Train/loss": float(train_loss),
                "Val/loss": float(val_loss),
                "Misc/TimePerEpoch": (time.time() - start_time) / (epoch + 1),
                "Misc/Epoch": epoch,
            }
            logging.info(f"Epoch {epoch}: {epoch_metrics}, improved: {has_improved}")
            history.append(pd.json_normalize(epoch_metrics, sep=""))

            if self.log_metrics:
                wandb.log(epoch_metrics, step=epoch)

            if self.early_stopping.should_stop:
                logging.info(f"Epoch: {epoch}: Stopping early")
                break

        history_df = pd.concat(history) if len(history) > 0 else pd.DataFrame([])
        return best_model_state, history_df

    def test_clicks(
        self,
        model: nn.Module,
        state: TrainState,
        loader: DataLoader,
        eval_behavior_cloning: bool = False,
        log_stage: Optional[Stage] = None,
    ) -> DataFrame:
        metrics = []

        for batch in tqdm(loader, disable=not self.progress_bar):
            metric = self._test_click_step(
                model, state, batch, state.step, eval_behavior_cloning
            )
            metrics.append(metric)

        metric_df = collect_metrics(metrics)

        if self.log_metrics and log_stage is not None:
            agg_metrics = aggregate_metrics(metric_df)
            wandb.log({f"{log_stage}/": agg_metrics})
            logging.info(f"{log_stage}: {agg_metrics}")

        return metric_df

    def test_relevance(
        self,
        model: nn.Module,
        state: TrainState,
        loader: Optional[DataLoader] = None,
        log_stage: Optional[Stage] = None,
    ) -> DataFrame:
        metrics = []

        for batch in tqdm(loader, disable=not self.progress_bar):
            metric = self._test_relevance_step(model, state, batch, state.step)
            metrics.append(metric)

        metric_df = collect_metrics(metrics)

        if self.log_metrics and log_stage is not None:
            agg_metrics = aggregate_metrics(metric_df)
            wandb.log({f"{log_stage}/": agg_metrics})
            print(f"{log_stage}: {agg_metrics}")

        return metric_df

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

    def _val_epoch(self, model, state, loader, description):
        epoch_loss = 0

        for batch in tqdm(loader, desc=description, disable=not self.progress_bar):
            epoch_loss += self._val_step(model, state, batch, state.step)

        epoch_loss /= len(loader)
        return epoch_loss

    @partial(jit, static_argnums=(0, 1))
    def _train_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        def loss_fn(params):
            output = model.apply(
                params,
                batch,
                training=True,
                rngs=rngs,
            )
            return model.get_loss(output, batch).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    @partial(jit, static_argnums=(0, 1))
    def _val_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        output = model.apply(
            state.params,
            batch,
            training=False,
            rngs=rngs,
        )

        return model.get_loss(output, batch).mean()

    @partial(jit, static_argnums=(0, 1, 5))
    def _test_click_step(self, model, state, batch, step, behavior_cloning):
        rngs = self.generate_rngs(state, step)

        output = model.apply(
            state.params,
            batch,
            training=False,
            rngs=rngs,
        )

        loss = model.get_loss(output, batch)

        metrics = {
            "query_id": batch["query_id"],
            "loss": loss,
        }

        if behavior_cloning:
            # Evaluate how predictions relate to the original Baidu logging policy:
            labels = reciprocal_rank(batch)

            for name, metric_fn in self.metric_fns.items():
                metrics[f"BC_{name}"] = metric_fn(
                    scores=output.relevance,
                    labels=labels,
                    where=batch["mask"],
                    reduce_fn=None,
                )

        if not hasattr(output, "click"):
            # Skip click metrics for models not directly predicting clicks:
            return metrics

        for name, metric_fn in self.click_metric_fns.items():
            metrics[name] = metric_fn(
                scores=output.click,
                labels=batch["click"],
                where=batch["mask"],
            )

        return metrics

    @partial(jit, static_argnums=(0, 1))
    def _test_relevance_step(self, model, state, batch, step):
        rngs = self.generate_rngs(state, step)

        relevance = model.apply(
            state.params,
            batch,
            training=False,
            rngs=rngs,
            method=model.predict_relevance,
        )

        metrics = {
            "query_id": batch["query_id"],
            "frequency_bucket": batch["frequency_bucket"],
        }

        for name, metric_fn in self.metric_fns.items():
            metrics[name] = metric_fn(
                relevance,
                batch["label"],
                where=batch["mask"],
                reduce_fn=None,
            )

        return metrics

    @staticmethod
    def generate_rngs(state: TrainState, step: int) -> Dict[str, jax.Array]:
        # Folding in the step number to generate a new random key:
        dropout = jax.random.fold_in(key=state.dropout_key, data=step)
        random_model = jax.random.fold_in(key=state.random_model_key, data=step)
        return {"dropout": dropout, "random_model": random_model}
