from collections import defaultdict
from typing import Dict, List, Callable

import numpy as np
from flax.training import early_stopping


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


def aggregate_metrics(results: List[Dict], reduce_fn: Callable = np.mean):
    """
    Aggregates a list of metric dicts into a single dict:
    [
        {"ndcg": 0.8, "MRR": 0.9},
        {"ndcg": 0.3, "MRR": 0.2},
        ...
    ]
    """
    agg_metrics = defaultdict(lambda: [])

    for result in results:
        for name, metric in result.items():
            agg_metrics[name].append(metric)

    return {name: reduce_fn(metric) for name, metric in agg_metrics.items()}
