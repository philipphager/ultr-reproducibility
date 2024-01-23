from typing import Dict, Callable

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array

from src.loss import pairwise_debiasing
from src.models.base import (
    RelevanceModel,
    ExaminationModel,
)


@dataclass
class PairwiseDebiasConfig:
    dims: int
    layers: int
    dropout: float
    positions: int
    clip: float
    l_norm: int = 1
    loss_fn: Callable = rax.pairwise_logistic_loss
    lambdaweight_fn: Callable = rax.dcg_lambdaweight


@dataclass
class PairwiseDebiasOutput:
    loss: Array
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
        self.max_weight = 1 / config.clip

    def __call__(self, batch: Dict, training: bool) -> PairwiseDebiasOutput:
        relevance = self.predict_relevance(batch, training=training)
        ratio_positive = self.bias_model_positive(batch, training=training)
        ratio_negative = self.bias_model_negative(batch, training=training)

        loss = pairwise_debiasing(
            ratio_positive=ratio_positive,
            ratio_negative=ratio_negative,
            relevance=relevance,
            labels=batch["click"],
            where=batch["mask"],
            loss_fn=self.config.loss_fn,
            lambdaweight_fn=self.config.lambdaweight_fn,
            max_weight=self.max_weight,
            l_norm=self.config.l_norm,
        )

        return PairwiseDebiasOutput(
            loss=loss,
            ratio_positive=ratio_positive,
            ratio_negative=ratio_negative,
            relevance=relevance,
        )

    def predict_relevance(self, batch: Dict, training: bool = False) -> Array:
        return self.relevance_model(batch, training=training)
