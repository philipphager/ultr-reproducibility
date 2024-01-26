from typing import Dict

import rax
from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from jax import random
from rax._src.types import ReduceFn

from src.util import reduce_per_query


@dataclass
class RandomConfig:
    reduce_fn: ReduceFn = reduce_per_query


@dataclass
class RandomOutput:
    click: Array
    relevance: Array


class RandomModel(nn.Module):
    config: RandomConfig
    rng_collection: str = "random_model"

    def __call__(self, batch, training: bool = False) -> RandomOutput:
        relevance = self.predict_relevance(batch, training)

        return RandomOutput(
            click=relevance,
            relevance=relevance,
        )

    def get_loss(self, output: RandomOutput, batch: Dict) -> Array:
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
        rng = self.make_rng(self.rng_collection)

        # Shape: batch x n_documents
        shape = batch["query_document_embedding"].shape[:2]

        return random.uniform(rng, shape=shape)
