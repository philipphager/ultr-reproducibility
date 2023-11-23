from typing import Union, Tuple, List

from flax import linen as nn
from jax import Array

from src.models.base import Tower


class NaiveModel(nn.Module):
    relevance_layers: List[int]
    relevance_dropouts: List[float]

    def setup(self) -> None:
        self.relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )

    def __call__(self, batch, training: bool = False) -> Tuple[Array, Array, None]:
        relevance = self.predict_relevance(batch, training)
        return relevance, relevance, None

    def predict_relevance(self, batch, training: bool = False) -> Array:
        x = batch["query_document_embedding"]
        return self.relevance_model(x, training).squeeze()
