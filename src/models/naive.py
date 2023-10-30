from typing import Union, Tuple

from flax import linen as nn
from jax import Array

from src.models.base import Tower


class NaiveModel(nn.Module):
    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(layers=[16, 16])
        relevance = relevance_model(batch["query_document_embedding"])
        return relevance.squeeze()
