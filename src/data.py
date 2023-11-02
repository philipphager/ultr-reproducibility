from collections import defaultdict
from typing import List, Dict

import numpy as np

PADDED_CLICK_COLUMNS = [
    "position",
    "query_document_embedding",
    "media_type",
    "displayed_time",
    "serp_height",
    "slipoff_count_after_click",
    "click",
]

PADDED_ANNOTATION_COLUMNS = [
    "query_document_embedding",
    "label",
]


def collate_clicks(samples: List[Dict[str, np.ndarray]]):
    """
    Collate function for training clicks from the Baidu-ULTR-606k dataset:
    https://huggingface.co/datasets/philipphager/baidu-ultr-606k/blob/main/baidu-ultr-606k.py
    """
    batch = defaultdict(lambda: [])
    max_n = int(max([sample["n"] for sample in samples]))

    for sample in samples:
        batch["query_id"].append(sample["query_id"])
        batch["mask"].append(pad(np.ones(sample["n"]), max_n))

        for column in PADDED_CLICK_COLUMNS:
            batch[column].append(pad(sample[column], max_n))

    return {column: np.array(features, dtype=int) for column, features in batch.items()}


def collate_annotations(samples: List):
    """
    Pad a batch of queries to the size of the query with the most documents.
    """
    batch = defaultdict(lambda: [])
    max_n = int(max([sample["n"] for sample in samples]))

    for sample in samples:
        batch["query_id"].append(sample["query_id"])
        batch["mask"].append(pad(np.ones(sample["n"]), max_n))

        for column in PADDED_ANNOTATION_COLUMNS:
            batch[column].append(pad(sample[column], max_n))

    return {column: np.array(features, dtype=int) for column, features in batch.items()}


def pad(x: np.ndarray, max_n: int):
    """
    Pads first (batch) dimension with zeros.

    E.g.: x = np.array([5, 4, 3]), n = 5
    -> np.array([5, 4, 3, 0, 0])

    E.g.: x = np.array([[5, 4, 3], [1, 2, 3]]), n = 4
    -> np.array([[5, 4, 3], [1, 2, 3], [0, 0, 0], [0, 0, 0]])
    """
    padding = max(max_n - x.shape[0], 0)
    pad_width = [(0, padding)]

    for i in range(x.ndim - 1):
        pad_width.append((0, 0))

    return np.pad(x, pad_width, mode="constant")


class LabelEncoder:
    def __init__(self):
        self.value2id = {}
        self.max_id = 1

    def __call__(self, x):
        return np.array(list(map(self.encode, x)))

    def encode(self, value):
        if value not in self.value2id:
            self.value2id[value] = self.max_id
            self.max_id += 1

        return self.value2id[value]

    def __len__(self):
        return len(self.value2id)


class Discretize:
    def __init__(self, low: float, high: float, buckets: int):
        """
        Bucket a continuous variable into n buckets. Indexing starts at 1 to avoid
        confusion with the padding value 0.
        """
        self.low = low
        self.high = high
        self.buckets = buckets
        self.boundaries = np.linspace(low, high, num=buckets + 1)

    def __call__(self, x):
        return np.digitize(x, self.boundaries, right=False)
