from collections import defaultdict
from typing import List, Dict, Optional

import mmh3
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

COLUMNS = {
    "query_document_embedding": {"padded": True, "dtype": float},
    "position": {"padded": True, "dtype": int},
    "mask": {"padded": True, "dtype": int},
    "n": {"padded": False, "dtype": int},
    "click": {"padded": True, "dtype": int},
    "label": {"padded": True, "dtype": int},
    "frequency_bucket": {"padded": False, "dtype": int},
}


def collate_fn(samples: List[Dict[str, np.ndarray]]):
    """
    Collate function for training clicks / labels from the Baidu-ULTR-606k dataset:
    https://huggingface.co/datasets/philipphager/baidu-ultr-606k/blob/main/baidu-ultr-606k.py

    The function parses all available features, pads queries to the same numer of
    documents, and converts datatypes.
    """
    batch = defaultdict(lambda: [])
    max_n = int(max([sample["n"] for sample in samples]))

    for sample in samples:
        for column, x in sample.items():
            if column in COLUMNS:
                x = pad(x, max_n) if COLUMNS[column]["padded"] else x
                batch[column].append(x)

        batch["mask"].append(pad(np.ones(sample["n"]), max_n))

    return {
        column: np.array(features, dtype=COLUMNS[column]["dtype"])
        for column, features in batch.items()
    }


def random_split(
    dataset: Dataset,
    shuffle: bool,
    random_state: int,
    test_size: float,
    stratify: Optional[str] = None,
):
    """
    Stratify a train/test split of a Huggingface dataset.
    While huggingface implements stratification, this function enables stratification
    on all columns, not only the dataset's class label.
    """
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        idx,
        stratify=dataset[stratify] if stratify else None,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state,
    )
    return dataset.select(train_idx), dataset.select(test_idx)


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


def hash_labels(x: np.ndarray, buckets: int, random_state: int = 0) -> np.ndarray:
    """
    Use a fast and robust non-cryptographic hash function to map class labels to
    a fixed number of buckets into the range(1, buckets + 1).
    E.g.: np.array([1301239102, 12039, 12309]) -> np.array([5, 1, 20])
    """

    def hash(i: int) -> int:
        hash_value = mmh3.hash(str(i), seed=random_state)
        bucket = hash_value % buckets
        return bucket + 1

    return np.array(list(map(hash, x)))


def discretize(x: np.ndarray, low: float, high: float, buckets: int):
    """
    Bucket a continuous variable into n buckets. Indexing starts at 1 to avoid
    confusion with the padding value 0.
    """
    boundaries = np.linspace(low, high, num=buckets + 1)
    return np.digitize(x, boundaries, right=False)
