from typing import Callable, Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.loss import regression_em
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)
from src.util import reduce_per_query


@dataclass
class RegressionEMConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float
    positions: int
    loss_fn: Callable = rax.pointwise_sigmoid_loss
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class RegressionEMOutput:
    click: Array
    examination: Array
    relevance: Array


class RegressionEM(nn.Module):
    config: RegressionEMConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.examination_model = ExaminationModel(positions=config.positions)

    def __call__(self, batch: Dict, training: bool) -> RegressionEMOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)
        click = examination + relevance

        return RegressionEMOutput(
            click=click,
            examination=examination,
            relevance=relevance,
        )

    def get_loss(self, output: RegressionEMOutput, batch: Dict) -> Array:
        # Note that this model does not actually learn.
        # Returning NLL to comply with the trainer setup.
        return regression_em(
            examination=output.examination,
            relevance=output.relevance,
            labels=batch["click"],
            where=batch["mask"],
            loss_fn=self.config.loss_fn,
            reduce_fn=self.config.reduce_fn,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
