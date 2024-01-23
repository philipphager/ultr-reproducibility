from typing import Dict, Callable

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.models.base import RelevanceModel
from src.util import reduce_per_query


@dataclass
class NaiveConfig:
    dims: int
    layers: int
    dropout: float
    loss_fn: Callable = rax.pointwise_sigmoid_loss
    reduce_fn: ReduceFn = reduce_per_query


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

        loss = self.config.loss_fn(
            relevance,
            batch["click"],
            where=batch["mask"],
            reduce_fn=self.config.reduce_fn,
        )

        return NaiveOutput(
            loss=loss,
            click=relevance,
            relevance=relevance,
        )

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.model(batch, training=training)
