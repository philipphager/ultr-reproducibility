import logging
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import rax
from flax.training.train_state import TrainState
from jax import jit
from rich.progress import track
from torch.utils.data import DataLoader

from src.log import print_metric_table
from src.util import EarlyStopping, aggregate_metrics

logger = logging.getLogger("rich")


class Trainer:
    def __init__(
        self,
        random_state: int = 0,
        optimizer=optax.adam(learning_rate=0.0001),
        criterion=rax.pointwise_sigmoid_loss,
        metric_fns={
            "ndcg@10": partial(rax.ndcg_metric, topn=10),
            "mrr@10": partial(rax.mrr_metric, topn=10),
            "dcg@01": partial(rax.dcg_metric, topn=1),
            "dcg@03": partial(rax.dcg_metric, topn=3),
            "dcg@05": partial(rax.dcg_metric, topn=5),
            "dcg@10": partial(rax.dcg_metric, topn=10),
        },
        epochs: int = 25,
        early_stopping: EarlyStopping = EarlyStopping(metric="ndcg@10", patience=0),
    ):
        self.random_state = random_state
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fns = metric_fns
        self.epochs = epochs
        self.early_stopping = early_stopping

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        state = self._init_train_state(model, train_loader)
        best_model_state = None

        for epoch in range(self.epochs):
            state = self.train_epoch(state, train_loader, f"Epoch: {epoch} - Training")
            val_metrics = self.eval_epoch(state, val_loader, f"Epoch: {epoch} - Val")
            has_improved, should_stop = self.early_stopping.update(val_metrics)
            logger.info(
                f"Epoch: {epoch} - {val_metrics} - has_improved: {has_improved}"
            )

            if has_improved:
                best_model_state = state

            if should_stop:
                logger.info(f"Epoch: {epoch}: Stopping early")
                break

        return best_model_state

    def test(self, state: TrainState, test_loader: DataLoader):
        metrics = self.eval_epoch(state, test_loader, "Testing")
        print_metric_table(metrics)
        return metrics

    def train_epoch(self, state: TrainState, loader: DataLoader, description: str):
        for batch in track(loader, description=description):
            state, loss = self._train_step(state, batch)

        return state

    def eval_epoch(self, state: TrainState, data_loader: DataLoader, description: str):
        metrics = []

        for batch in track(data_loader, description=description):
            metrics.append(self._eval_step(state, batch))

        return aggregate_metrics(metrics)

    def _init_train_state(self, model, train_loader):
        batch = next(iter(train_loader))
        key = jax.random.PRNGKey(self.random_state)
        params = model.init(key, batch, training=True)

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=self.optimizer,
        )

    @partial(jit, static_argnums=(0,))
    def _train_step(self, state, batch):
        def loss_fn(params):
            y_predict = state.apply_fn(params, batch, training=True)
            return self.criterion(y_predict, batch["click"], where=batch["mask"])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    @partial(jit, static_argnums=(0,))
    def _eval_step(self, state, batch):
        label, mask = batch["label"], batch["mask"]
        y_predict = state.apply_fn(state.params, batch, training=False)

        return {
            name: metric_fn(y_predict, label, where=mask, reduce_fn=jnp.mean)
            for name, metric_fn in self.metric_fns.items()
        }
