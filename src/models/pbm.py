from typing import Union, Tuple, List

from flax import linen as nn
from jax import Array

from src.models.base import Tower
from src.models.two_tower import TowerCombination, combine_towers


class PositionBasedModel(nn.Module):
    """
    With document d, query q, at position k:
    P(C = 1 | d, q, k) = P(E = 1 | k) x P(R = 1 | d, q)
    """

    relevance_layers: List[int]
    relevance_dropouts: List[float]
    tower_combination: Union[TowerCombination, str]
    n_positions: int

    def setup(self) -> None:
        self.relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )
        self.examination_model = nn.Embed(
            num_embeddings=self.n_positions,
            features=1,
        )

    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance = self.predict_relevance(batch, training)
        examination = self.predict_examination(batch, training)
        return combine_towers(examination, relevance, self.tower_combination)

    def predict_relevance(self, batch, training: bool = False) -> Array:
        return self.relevance_model(
            batch["query_document_embedding"], training
        ).squeeze()

    def predict_examination(self, batch, training: bool = False) -> Array:
        return self.examination_model(
            batch["position"],
        ).squeeze()
