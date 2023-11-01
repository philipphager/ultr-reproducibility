from typing import Union, Tuple

from flax import linen as nn
from jax import Array

from src.models.base import Tower


class PositionBasedModel(nn.Module):
    """
    With document d, query q, at position k:
    P(C = 1 | d, q, k) = P(E = 1 | k) x P(R = 1 | d, q)
    """

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(layers=[16, 16])
        examination_model = nn.Embed(num_embeddings=50, features=1)
        relevance = relevance_model(batch["query_document_embedding"]).squeeze()

        if training:
            examination = examination_model(batch["position"]).squeeze()
            return examination + relevance
        else:
            return relevance
