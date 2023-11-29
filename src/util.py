from pathlib import Path
from typing import Dict, List

import jax
import numpy as np
import pandas as pd
from flax.training import early_stopping, checkpoints
from jax import Array


class EarlyStopping:
    def __init__(
        self,
        metric: str,
        patience: int = 0,
        maximize: bool = True,
        min_delta: float = 0,
    ):
        self.metric = metric
        self.maximize = maximize
        self.state = early_stopping.EarlyStopping(
            patience=patience, min_delta=min_delta
        )

    def update(self, metrics: Dict):
        metric = metrics[self.metric]
        sign = -1 if self.maximize else 1
        self.state = self.state.update(sign * metric)
        return self.state.has_improved, self.state.should_stop

    def should_stop(self) -> bool:
        return self.state.should_stop

    def has_improved(self) -> bool:
        return self.state.has_improved


def collect_metrics(results: List[Dict[str, Array]]) -> pd.DataFrame:
    """
    Collects batches of metrics into a single pandas DataFrame:
    [
        {"ndcg": [0.8, 0.3], "MRR": [0.9, 0.2]},
        {"ndcg": [0.2, 0.1], "MRR": [0.1, 0.02]},
        ...
    ]
    """
    # Convert Jax Arrays to numpy:
    np_results = [dict_to_numpy(r) for r in results]
    # Unroll values in batches into individual rows:
    df = pd.DataFrame(np_results)
    return df.explode(column=list(df.columns)).reset_index(drop=True)


def aggregate_metrics(df: pd.DataFrame, ignore_columns=["query_id", "frequency_bucket"]) -> Dict[str, float]:
    df = df.drop(columns=ignore_columns, errors = "ignore")
    return df.mean(axis=0).to_dict()


def dict_to_numpy(_dict: Dict[str, Array]) -> Dict[str, np.ndarray]:
    return {k: jax.device_get(v) for k, v in _dict.items()}


def save_state(state, directory: Path, name: str = "best_state"):
    checkpoints.save_checkpoint(directory / name, state, step=0, overwrite=True)
