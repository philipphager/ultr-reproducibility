import enum
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

        bias_model = Tower(layers=self.layers, dropouts=self.dropouts)
        return bias_model(x, training)


class TowerCombination(enum.Enum):
    NONE = "NONE"
    ADDITIVE = "ADDITIVE"


def combine_towers(
    examination: Array,
    relevance: Array,
    combination: Union[TowerCombination, str],
) -> Union[Array, Tuple[Array, Array]]:
    combination = TowerCombination[combination]

    if combination == TowerCombination.NONE:
        return examination, relevance
    elif combination == TowerCombination.ADDITIVE:
        return examination + relevance
    else:
        raise ValueError(f"Unknown tower combination: {combination}")


class TwoTowerModel(nn.Module):
    """
    With document d, query q, and bias features b:
    P(C = 1 | d, q, b) = P(E = 1 | b) x P(R = 1 | d, q)
    """

    bias_layers: List[int]
    bias_dropouts: List[float]
    relevance_layers: List[int]
    relevance_dropouts: List[float]
    tower_combination: TowerCombination

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )

        relevance = relevance_model(batch["query_document_embedding"], training)

        if training:
            bias_model = BiasTower(
                layers=self.bias_layers,
                dropouts=self.bias_dropouts,
            )
            bias = bias_model(batch, training)

            return combine_towers(
                bias.squeeze(),
                relevance.squeeze(),
                self.tower_combination,
            )
        else:
            return relevance.squeeze()
