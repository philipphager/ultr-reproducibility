from flax import linen as nn
from flax.struct import dataclass
from jax import Array
from jax import random


@dataclass
class RandomOutput:
    loss: float
    click: Array
    relevance: Array


class RandomModel(nn.Module):
    rng_collection: str = "random_model"

    def __call__(self, batch, training: bool = False) -> RandomOutput:
        relevance = self.predict_relevance(batch, training)

        return RandomOutput(
            loss=0,
            click=relevance,
            relevance=relevance,
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
