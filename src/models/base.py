from typing import List

from flax import linen as nn
from jax import Array


class Tower(nn.Module):
    layers: List[int]

    @nn.compact
    def __call__(self, x) -> Array:
        modules = []

        for layer in self.layers:
            modules.append(nn.Dense(features=layer))
            modules.append(nn.elu)

        modules.append(nn.Dense(features=1))
        model = nn.Sequential(modules)
        return model(x)
