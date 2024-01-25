import logging
from pathlib import Path
from typing import Dict

import flax
import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
from jax import Array

from src.data import FeatureType, filter_features


@flax.struct.dataclass
class RelevanceConfig:
    features: FeatureType
    dims: int
    layers: int
    dropout: float


class RelevanceModel(nn.Module):
    config: RelevanceConfig

    @nn.compact
    def __call__(self, batch: Dict, training: bool) -> Array:
        x = self.concat_features(batch)
        model = self.get_sequential(training)
        return model(x).squeeze()

    def concat_features(self, batch: Dict) -> Array:
        features = filter_features(self.config.features)
        x = [jnp.atleast_3d(batch[f]) for f in features]
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
    positions: int

    def setup(self):
        logging.info(f"Loading propensities from file: {self.file}")
        assert Path(self.file).exists()

        df = pd.read_csv(self.file)
        model = jnp.zeros(self.positions)
        # Load propensities, position 0 is used for padding and has propensity 0:
        positions = df["position"].values
        propensities = df.iloc[:, 1].values
        self.model = model.at[positions].set(propensities)

    def __call__(self, batch: Dict, training: bool) -> Array:
        return self.model[batch["position"]]
