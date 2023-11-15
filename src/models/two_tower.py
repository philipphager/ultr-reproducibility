import enum
from typing import Union, Tuple, List, Dict

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from src.models.base import Tower


class BiasTower(nn.Module):
    layers: List[int]
    dropouts: List[float]
    column_embeddings: Dict[str, int]
    embedding_dims: int

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        embeddings = []

        for column, num_embeddings in self.column_embeddings.items():
            layer = nn.Embed(
                num_embeddings=num_embeddings,
                features=self.embedding_dims,
            )
            embedding = layer(batch[column])
            embeddings.append(embedding)

        x = jnp.concatenate(embeddings, axis=-1)
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
    bias_embeddings: Dict[str, int]
    relevance_layers: List[int]
    relevance_dropouts: List[float]
    tower_combination: TowerCombination

    def setup(self) -> None:
        self.relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )
        self.bias_model = BiasTower(
            layers=self.bias_layers,
            dropouts=self.bias_dropouts,
            column_embeddings=self.bias_embeddings,
            embedding_dims=8,
        )

    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance = self.predict_relevance(batch, training)
        examination = self.predict_examination(batch, training)
        return combine_towers(examination, relevance, self.tower_combination)

    def predict_relevance(self, batch, training: bool = False) -> Array:
        x = batch["query_document_embedding"]
        return self.relevance_model(x, training).squeeze()

    def predict_examination(self, batch, training: bool = False) -> Array:
        return self.bias_model(batch, training).squeeze()
