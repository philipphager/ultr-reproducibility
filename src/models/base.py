from pathlib import Path
from typing import Dict

import flax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import Array


@flax.struct.dataclass
class RelevanceConfig:
    dims: int
    layers: int
    dropout: float


class RelevanceModel(nn.Module):
    config: RelevanceConfig

    @nn.compact
    def __call__(self, batch: Dict, training: bool) -> Array:
        x = batch["query_document_embedding"]

        model = self.get_sequential(training)
        return model(x).squeeze()

    def concat_features(self, batch: Dict) -> Array:
        x = [jnp.atleast_3d(batch[f]) for f in [self.config.feature]]
        return jnp.concatenate(x, axis=-1)

    def get_sequential(self, training: bool) -> nn.Module:
        config = self.config
        modules = []

        for i in range(config.layers):
            modules.append(nn.Dense(features=config.dims))
            modules.append(nn.elu)
            modules.append(nn.Dropout(rate=config.dropout, deterministic=not training))

        modules.append(nn.Dense(features=1))
        return nn.Sequential(modules)


class ExaminationModel(nn.Module):
    positions: int

    @nn.compact
    def __call__(self, batch: Dict, training: bool) -> Array:
        model = nn.Embed(num_embeddings=self.positions, features=1)
        return model(batch["position"]).squeeze()


class PretrainedExaminationModel(nn.Module):
    file: str

    def setup(self):
        print(f"Loading propensities from file: {self.file}")
        assert Path(self.file).exists()

        data = np.genfromtxt(self.file, delimiter=",")
        propensities = jnp.asarray(data[1])
        self.model = lambda x: propensities[x]

    def __call__(self, batch: Dict, training: bool) -> Array:
        return self.model(batch["position"])
