from typing import Optional, Callable, Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array

from src.loss import regression_em
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)


@dataclass
class RegressionEMConfig:
    dims: int
    layers: int
    dropout: float
    positions: int
    propensity_path: Optional[str] = None
    loss_fn: Callable = rax.pointwise_sigmoid_loss


@dataclass
class RegressionEMOutput:
    loss: Array
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

        loss = regression_em(
            examination=examination,
            relevance=relevance,
            labels=batch["click"],
            where=batch["mask"],
            loss_fn=self.config.loss_fn,
        )

        return RegressionEMOutput(
            loss=loss,
            click=click,
            examination=examination,
            relevance=relevance,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
