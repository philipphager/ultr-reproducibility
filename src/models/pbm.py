from typing import Callable, Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)
from src.util import reduce_per_query


@dataclass
class PBMConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float
    positions: int
    loss_fn: Callable = rax.pointwise_sigmoid_loss
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class PBMOutput:
    click: Array
    examination: Array
    relevance: Array


class PositionBasedModel(nn.Module):
    config: PBMConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.examination_model = ExaminationModel(positions=config.positions)

    def __call__(self, batch: Dict, training: bool) -> PBMOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)
        click = examination + relevance

        return PBMOutput(
            click=click,
            examination=examination,
            relevance=relevance,
            reduce_fn=self.config.reduce_fn,
        )

    def get_loss(self, output: PBMOutput, batch: Dict) -> Array:
        return self.config.loss_fn(
            scores=output.click,
            labels=batch["click"],
            where=batch["mask"],
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
