from typing import Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn

from src.data import FeatureType
from src.util import reduce_per_query


@dataclass
class BM25Config:
    features: FeatureType
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class BM25Output:
    click: Array
    relevance: Array


class BM25(nn.Module):
    config: BM25Config

    def __call__(self, batch, training: bool = False) -> BM25Output:
        relevance = self.predict_relevance(batch, training)

        return BM25Output(
            click=relevance,
            relevance=relevance,
        )

    def get_loss(self, output: BM25Output, batch: Dict) -> Array:
        # Note that this model does not actually learn.
        # Returning NLL to comply with the trainer setup.
        return rax.pointwise_sigmoid_loss(
            scores=output.click,
            labels=batch["click"],
            where=batch["mask"],
            reduce_fn=self.config.reduce_fn,
        )

    def predict_relevance(
        self,
        batch,
        training: bool = False,
    ) -> Array:
        return batch["bm25"]
