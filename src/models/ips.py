from typing import Callable, Dict

from flax import linen as nn
from flax.struct import dataclass
from jax import Array

from src.loss import inverse_propensity_weighting
from src.models.base import (
    RelevanceModel,
    PretrainedExaminationModel,
)


@dataclass
class IPSConfig:
    loss_fn: Callable
    dims: int
    layers: int
    dropout: float
    clip: float
    propensity_path: str


@dataclass
class IPSOutput:
    loss: Array
    examination: Array
    relevance: Array


class IPSModel(nn.Module):
    config: IPSConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.examination_model = PretrainedExaminationModel(file=config.propensity_path)
        self.max_weight = 1 / config.clip

    def __call__(self, batch: Dict, training: bool) -> IPSOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)

        loss = inverse_propensity_weighting(
            examination=examination,
            relevance=relevance,
            labels=batch["click"],
            where=batch["mask"],
            max_weight=self.max_weight,
        )

        return IPSOutput(
            loss=loss,
            examination=examination,
            relevance=relevance,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
