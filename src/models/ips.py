from typing import Callable, Dict

from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from jax.scipy.special import logit
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.loss import listwise_softmax_ips
from src.models.base import (
    RelevanceModel,
    PretrainedExaminationModel,
)
from src.util import reduce_per_query


@dataclass
class IPSConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float
    clip: float
    positions: int
    propensity_path: str
    loss_fn: Callable = listwise_softmax_ips
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class IPSOutput:
    click: Array
    examination: Array
    relevance: Array


class IPSModel(nn.Module):
    config: IPSConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.examination_model = PretrainedExaminationModel(
            file=config.propensity_path,
            positions=config.positions,
        )

    def __call__(self, batch: Dict, training: bool) -> IPSOutput:
        examination = self.predict_examination(batch, training=training)
        relevance = self.predict_relevance(batch, training=training)

        # Compute log-odds of click for NLL evaluation.
        # Convert relevance logit to probability and return click log odds.
        # Assumes examination is already a probability:
        click = logit(examination * nn.sigmoid(relevance))

        return IPSOutput(
            click=click,
            examination=examination,
            relevance=relevance,
        )

    def get_loss(self, output: IPSOutput, batch: Dict) -> Array:
        max_weight = 1 / self.config.clip

        return self.config.loss_fn(
            examination=output.examination,
            relevance=output.relevance,
            labels=batch["click"],
            where=batch["mask"],
            reduce_fn=self.config.reduce_fn,
            max_weight=max_weight,
        )

    def predict_examination(self, batch: Dict, training: bool = False) -> Array:
        return self.examination_model(batch, training=training)

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
