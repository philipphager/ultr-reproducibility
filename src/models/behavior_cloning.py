from typing import Dict, Callable

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.models.base import RelevanceModel
from src.util import reduce_per_query


@dataclass
class BehaviorCloningConfig:
    dims: int
    layers: int
    dropout: float
    features: FeatureType
    loss_fn: Callable = rax.pairwise_logistic_loss
    lambdaweight_fn: Callable = rax.dcg_lambdaweight
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class BehaviorCloningOutput:
    loss: Array
    relevance: Array


class BehaviorCloningModel(nn.Module):
    config: BehaviorCloningConfig

    def setup(self):
        self.model = RelevanceModel(self.config)

    def __call__(self, batch: Dict, training: bool) -> BehaviorCloningOutput:
        relevance = self.predict_relevance(batch, training=training)
        labels = 1 / batch["position"]

        loss = self.config.loss_fn(
            scores=relevance,
            labels=labels,
            where=batch["mask"],
            lambdaweight_fn=self.config.lambdaweight_fn,
            reduce_fn=self.config.reduce_fn,
        )

        return BehaviorCloningOutput(
            loss=loss,
            relevance=relevance,
        )

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.model(batch, training=training)
