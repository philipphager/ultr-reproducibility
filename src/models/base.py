from flax import linen as nn
from jax import Array


class Tower(nn.Module):
    dims: int
    layers: int
    dropout: float

    @nn.compact
    def __call__(self, x, training: bool) -> Array:
        modules = []

        for i in range(self.layers):
            modules.append(nn.Dense(features=self.dims))
            modules.append(nn.elu)
            modules.append(nn.Dropout(rate=self.dropout, deterministic=not training))

        modules.append(nn.Dense(features=1))
        model = nn.Sequential(modules)
        return model(x)
