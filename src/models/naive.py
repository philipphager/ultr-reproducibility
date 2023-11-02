from typing import Union, Tuple, List

from flax import linen as nn
from jax import Array

from src.models.base import Tower


class NaiveModel(nn.Module):
    relevance_layers: List[int]
    relevance_dropouts: List[float]

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )
        relevance = relevance_model(batch["query_document_embedding"], training)
        return relevance.squeeze()
