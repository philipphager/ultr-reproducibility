from typing import Union, Tuple

from flax import linen as nn
from jax import Array
import jax.numpy as jnp
import numpy as np

from src.models.base import Tower
from src.models.two_tower import TowerCombination, combine_towers, BiasTower


class PairPositionBasedModel(nn.Module):
    """
    With document d, query q, at position k:
    P(C = 1 | d, q, k) = t+(k) x P(R = 1 | d, q)
    P(C = 0 | d, q, k) = t-(k) x P(R = 0 | d, q)
    """

    bias_dims: int
    bias_layers: int
    bias_dropout: float
    relevance_dims: int
    relevance_layers: int
    relevance_dropout: float
    tower_combination: TowerCombination
    propensities_path: str | None = None

    def setup(self) -> None:
        self.relevance_model = Tower(
            dims=self.relevance_dims,
            layers=self.relevance_layers,
            dropout=self.relevance_dropout,
        )
        if self.propensities_path is None:
            self.bias_model_positive = BiasTower(
                dims=self.bias_dims,
                layers=self.bias_layers,
                dropout=self.bias_dropout,
                embeddings={
                    "position": nn.Embed(num_embeddings=50, features=8),
                },
            )
            self.bias_model_negative = BiasTower(
                dims=self.bias_dims,
                layers=self.bias_layers,
                dropout=self.bias_dropout,
                embeddings={
                    "position": nn.Embed(num_embeddings=50, features=8),
                },
            )
        else:
            propensities = jnp.asarray(np.genfromtxt(self.propensities_path, delimiter=',')[1])
            self.bias_model_positive = lambda batch: propensities[0, batch["position"]]
            self.bias_model_negative = lambda batch: propensities[1, batch["position"]]

    def __call__(
        self, batch, training: bool = False
    ) -> Tuple[Array | Tuple[Array | Tuple[Array, Array], Array], Array, Tuple[Array, Array]]:
        relevance = self.predict_relevance(batch, training)
        examination = self.predict_examination(batch, training)
        return combine_towers(examination, relevance, combination = "NONE"), relevance, examination

    def predict_relevance(self, batch, training: bool = False) -> Array:
        x = batch["query_document_embedding"]
        return self.relevance_model(x, training).squeeze()

    def predict_examination(self, batch, training: bool = False) -> Tuple[Array, Array]:
        return self.bias_model_positive(batch).squeeze(), self.bias_model_negative(batch).squeeze()
