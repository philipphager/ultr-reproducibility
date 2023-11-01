from typing import Union, Tuple, List

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from src.models.base import Tower


class BiasTower(nn.Module):
    layers: List[int]
    dropouts: List[float]

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        position = nn.Embed(num_embeddings=50, features=8)
        media_type = nn.Embed(num_embeddings=1_000, features=8)
        serp_height = nn.Embed(num_embeddings=18, features=8)
        displayed_time = nn.Embed(num_embeddings=18, features=8)
        slipoff_count = nn.Embed(num_embeddings=18, features=8)

        x = jnp.concatenate(
            [
                position(batch["position"]),
                media_type(batch["media_type"]),
                displayed_time(batch["displayed_time"]),
                serp_height(batch["serp_height"]),
                slipoff_count(batch["slipoff_count_after_click"]),
            ],
            axis=-1,
        )
        examination = Tower(layers=self.layers, dropouts=self.dropouts)
        return examination(x)


class TwoTowerModel(nn.Module):
    """
    With document d, query q, and bias features b:
    P(C = 1 | d, q, b) = P(E = 1 | b) x P(R = 1 | d, q)
    """

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(layers=[16, 16], dropouts=[0.5, 0.5])
        relevance = relevance_model(
            batch["query_document_embedding"], training
        ).squeeze()
        examination_model = BiasTower(layers=[16, 16], dropouts=[0.5, 0.5])

        if training:
            examination = examination_model(batch, training).squeeze()
            return examination + relevance
        else:
            return relevance
