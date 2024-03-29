from typing import Dict, Callable

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.loss import pairwise_debiasing
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)
from src.util import reduce_per_query


@dataclass
class PairwiseDebiasConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float
    positions: int
    clip: float
    l_norm: int = 1
    loss_fn: Callable = rax.pairwise_logistic_loss
    lambdaweight_fn: Callable = rax.dcg_lambdaweight
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class PairwiseDebiasOutput:
    ratio_positive: Array
    ratio_negative: Array
    relevance: Array


class PairwiseDebiasModel(nn.Module):
    """
    With document d, query q, at position k:
    P(C = 1 | d, q, k) = t+(k) x P(R = 1 | d, q)
    P(C = 0 | d, q, k) = t-(k) x P(R = 0 | d, q)
    """

    config: PairwiseDebiasConfig

    def setup(self):
        config = self.config
        self.relevance_model = RelevanceModel(config)
        self.bias_model_positive = ExaminationModel(positions=config.positions)
        self.bias_model_negative = ExaminationModel(positions=config.positions)

    def __call__(self, batch: Dict, training: bool) -> PairwiseDebiasOutput:
        relevance = self.predict_relevance(batch, training=training)
        ratio_positive = nn.softplus(self.bias_model_positive(batch, training=training))
        ratio_negative = nn.softplus(self.bias_model_negative(batch, training=training))

        return PairwiseDebiasOutput(
            ratio_positive=ratio_positive,
            ratio_negative=ratio_negative,
            relevance=relevance,
        )

    def get_loss(self, output: PairwiseDebiasOutput, batch: Dict) -> Array:
        max_weight = 1 / self.config.clip

        return pairwise_debiasing(
            ratio_positive=output.ratio_positive,
            ratio_negative=output.ratio_negative,
            relevance=output.relevance,
            labels=batch["click"],
            where=batch["mask"],
            loss_fn=self.config.loss_fn,
            reduce_fn=self.config.reduce_fn,
            lambdaweight_fn=self.config.lambdaweight_fn,
            max_weight=max_weight,
            l_norm=self.config.l_norm,
        )

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
