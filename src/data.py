from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

CLICK_COLUMNS = [
    "query_id",
    "position",
    "query_document_embedding",
    "media_type",
    "displayed_time",
    "serp_height",
    "slipoff_count_after_click",
    "click",
]

ANNOTATION_COLUMNS = [
    "query_id",
    "query_document_embedding",
    "label",
]


def collate_clicks(samples: List[Dict[str, torch.Tensor]]):
    """
    Collate function for training clicks from the Baidu-ULTR-606k dataset:
    https://huggingface.co/datasets/philipphager/baidu-ultr-606k/blob/main/baidu-ultr-606k.py
    """
    batch = defaultdict(lambda: [])

    for sample in samples:
        for column in CLICK_COLUMNS:
            batch[column].append(sample[column])

        batch["mask"].append(torch.ones(sample["n"]))

    collated_batch = {"query_id": torch.tensor(batch["query_id"]).numpy()}

    for column, values in batch.items():
        if column != "query_id":
            collated_batch[column] = (
                pad_sequence(values, batch_first=True).int().numpy()
            )

    return collated_batch


def collate_annotations(samples: List):
    """
    Pad a batch of queries to the size of the query with the most documents.
    """
    batch = defaultdict(lambda: [])

    for sample in samples:
        for column in ANNOTATION_COLUMNS:
            batch[column].append(sample[column])

        batch["mask"].append(torch.ones(sample["n"]))

    collated_batch = {"query_id": torch.tensor(batch["query_id"]).numpy()}

    for column, values in batch.items():
        if column != "query_id":
            collated_batch[column] = (
                pad_sequence(values, batch_first=True).int().numpy()
            )

    return collated_batch


class LabelEncoder:
    def __init__(self):
        self.value2id = {}
        self.max_id = 1

    def __call__(self, x):
        return x.apply_(self.encode)

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
        self.boundaries = torch.linspace(low, high, steps=buckets + 1)

    def __call__(self, x):
        return torch.bucketize(x, self.boundaries, right=True).int()
