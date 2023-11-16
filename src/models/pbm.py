from typing import Union, Tuple, List

from flax import linen as nn
from jax import Array

from src.models.base import Tower
from src.models.two_tower import TowerCombination, combine_towers, BiasTower


class PositionBasedModel(nn.Module):
    """
    With document d, query q, at position k:
    P(C = 1 | d, q, k) = P(E = 1 | k) x P(R = 1 | d, q)
    """

    bias_layers: List[int]
    bias_dropouts: List[float]
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
            embeddings={
                "position": nn.Embed(num_embeddings=50, features=8),
            },
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
        return self.bias_model(batch).squeeze()
