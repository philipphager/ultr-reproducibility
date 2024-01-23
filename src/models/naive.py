from typing import Dict, Callable

from flax import linen as nn
from flax.struct import dataclass
from jax import Array

from src.models.base import RelevanceModel


@dataclass
class NaiveConfig:
    loss_fn: Callable
    dims: int
    layers: int
    dropout: float


@dataclass
class NaiveOutput:
    loss: Array
    click: Array
    relevance: Array


class NaiveModel(nn.Module):
    config: NaiveConfig

    def setup(self):
        self.model = RelevanceModel(self.config)

    def __call__(self, batch: Dict, training: bool) -> NaiveOutput:
        relevance = self.predict_relevance(batch, training=training)
        loss = self.config.loss_fn(relevance, batch["click"], where=batch["mask"])

        return NaiveOutput(
            loss=loss,
            click=relevance,
            relevance=relevance,
        )

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.model(batch, training=training)
