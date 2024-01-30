from typing import Callable, Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.loss import dual_learning_algorithm
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)
from src.util import reduce_per_query


@dataclass
class DLAConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float
    positions: int
    clip: float
    loss_fn: Callable = rax.softmax_loss
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class DLAOutput:
    examination: Array
    relevance: Array


class DualLearningAlgorithm(nn.Module):
    config: DLAConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.examination_model = ExaminationModel(positions=config.positions)

    def __call__(self, batch: Dict, training: bool) -> DLAOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)

        return DLAOutput(
            examination=examination,
            relevance=relevance,
        )

    def get_loss(self, output: DLAOutput, batch: Dict) -> Array:
        max_weight = 1 / self.config.clip

        return dual_learning_algorithm(
            examination=output.examination,
            relevance=output.relevance,
            labels=batch["click"],
            where=batch["mask"],
            loss_fn=self.config.loss_fn,
            reduce_fn=self.config.reduce_fn,
            max_weight=max_weight,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
