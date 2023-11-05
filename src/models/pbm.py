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
    n_positions: int
    tower_combination: Union[TowerCombination, str]

    @nn.compact
    def __call__(
        self, batch, training: bool = False
    ) -> Union[Array, Tuple[Array, Array]]:
        relevance_model = Tower(
            layers=self.relevance_layers,
            dropouts=self.relevance_dropouts,
        )
        examination_model = nn.Embed(
            num_embeddings=self.n_positions,
            features=1,
        )

        relevance = relevance_model(batch["query_document_embedding"], training)

        if training:
            examination = examination_model(batch["position"])
            return combine_towers(
                examination.squeeze(),
                relevance.squeeze(),
                self.tower_combination,
            )
        else:
            return relevance.squeeze()
