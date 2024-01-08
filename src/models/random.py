from typing import Tuple

from flax import linen as nn
from jax import Array
from jax import random


class RandomModel(nn.Module):
    rng_collection: str = "random_model"

    def __call__(self, batch, training: bool = False) -> Tuple[Array, Array, None]:
        relevance = self.predict_relevance(batch, training)
        return relevance, relevance, None

    def predict_relevance(
        self,
        batch,
        training: bool = False,
    ) -> Array:
        rng = self.make_rng(self.rng_collection)

        # Shape: batch x n_documents
        shape = batch["query_document_embedding"].shape[:2]

        return random.uniform(rng, shape=shape)
