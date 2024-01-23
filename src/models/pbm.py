from typing import Optional, Callable, Dict

from flax import linen as nn
from flax.struct import dataclass
from jax import Array

from src.models.base import (
    RelevanceModel,
    PretrainedExaminationModel,
    ExaminationModel,
)


@dataclass
class PBMConfig:
    loss_fn: Callable
    dims: int
    layers: int
    dropout: float
    positions: int
    propensity_path: Optional[str] = None


@dataclass
class PBMOutput:
    loss: Array
    click: Array
    examination: Array
    relevance: Array


class PositionBasedModel(nn.Module):
    config: PBMConfig

    def setup(self):
        self.relevance_model = RelevanceModel(self.config)

        if self.config.propensity_path is not None:
            self.examination_model = PretrainedExaminationModel(
                file=self.config.propensity_path,
            )
        else:
            self.examination_model = ExaminationModel(
                positions=self.config.positions,
            )

    def __call__(self, batch: Dict, training: bool) -> PBMOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)
        click = examination + relevance

        loss = self.config.loss_fn(click, batch["click"], where=batch["mask"])

        return PBMOutput(
            loss=loss,
            click=click,
            examination=examination,
            relevance=relevance,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
