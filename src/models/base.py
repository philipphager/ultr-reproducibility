from typing import List

from flax import linen as nn
from jax import Array


class Tower(nn.Module):
    layers: List[int]
    dropouts: List[float]

    @nn.compact
    def __call__(self, x, training: bool) -> Array:
        modules = []

        for layer, dropout in zip(self.layers, self.dropouts):
            modules.append(nn.Dense(features=layer))
            modules.append(nn.elu)
            modules.append(nn.Dropout(rate=dropout, deterministic=not training))

        modules.append(nn.Dense(features=1))
        model = nn.Sequential(modules)
        return model(x)
