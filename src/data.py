from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_clicks(samples: List[Dict[str, torch.Tensor]]):
    """
    Collate function for training clicks from the Baidu-ULTR-606k dataset:
    https://huggingface.co/datasets/philipphager/baidu-ultr-606k/blob/main/baidu-ultr-606k.py
    """
    batch = defaultdict(lambda: [])

    for sample in samples:
        batch["query_id"].append(sample["query_id"])
        batch["position"].append(sample["position"])
        batch["query_document_embedding"].append(sample["query_document_embedding"])
        batch["click"].append(sample["click"])
        batch["mask"].append(torch.ones(sample["n"]))

    return {
        "query_id": np.array(batch["query_id"]),
        "position": np.array(pad_sequence(batch["position"], batch_first=True)),
        "query_document_embedding": np.array(
            pad_sequence(batch["query_document_embedding"], batch_first=True)
        ),
        "mask": np.array(pad_sequence(batch["mask"], batch_first=True)),
        "click": np.array(pad_sequence(batch["click"], batch_first=True)),
    }


def collate_annotations(samples: List):
    """
    Pad a batch of queries to the size of the query with the most documents.
    """
    batch = defaultdict(lambda: [])

    for sample in samples:
        # Available are: ["query_id", "label", "n", "query_document_embedding", "frequency_bucket"]
        batch["query_id"].append(sample["query_id"])
        batch["query_document_embedding"].append(sample["query_document_embedding"])
        batch["label"].append(sample["label"])
        batch["mask"].append(torch.ones(sample["n"]))

    return {
        "query_id": np.array(torch.tensor(batch["query_id"])),
        "query_document_embedding": np.array(
            pad_sequence(batch["query_document_embedding"], batch_first=True)
        ),
        "label": np.array(pad_sequence(batch["label"], batch_first=True)),
        "mask": np.array(pad_sequence(batch["mask"], batch_first=True)),
    }
