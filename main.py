import logging
from functools import partial

import hydra
import jax
import optax
import rax
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader, random_split

from src.data import LabelEncoder, Discretize, collate_fn
from src.trainer import Trainer
from src.util import EarlyStopping

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            tracebacks_suppress=[jax],
            console=Console(width=800),
        )
    ],
)

BAIDU_DATASET = "philipphager/baidu-ultr-1m"


def load_train_data():
    train_dataset = load_dataset(BAIDU_DATASET, name="clicks", split="train")
    train_dataset.set_format("numpy")

    encode_media_type = LabelEncoder()
    encode_serp_height = Discretize(0, 1024, 16)
    encode_displayed_time = Discretize(0, 128, 16)
    encode_slipoff = Discretize(0, 10, 10)

    def encode_bias(batch):
        batch["media_type"] = encode_media_type(batch["media_type"])
        batch["displayed_time"] = encode_displayed_time(batch["displayed_time"])
        batch["serp_height"] = encode_serp_height(batch["serp_height"])
        batch["slipoff_count_after_click"] = encode_slipoff(
            batch["slipoff_count_after_click"]
        )
        return batch

    return train_dataset.map(encode_bias)


def load_val_data():
    val_dataset = load_dataset(BAIDU_DATASET, name="annotations", split="validation")
    val_dataset.set_format("numpy")
    return val_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(config: DictConfig):
    torch.manual_seed(config.random_state)

    train_dataset = load_train_data()
    val_dataset = load_val_data()
    val_dataset, test_dataset = random_split(val_dataset, config.val_test_split)

    trainer_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=1,
    )

    model = instantiate(config.model)
    criterion = instantiate(config.loss)

    trainer = Trainer(
        random_state=0,
        optimizer=optax.adam(learning_rate=0.0001),
        criterion=criterion,
        metric_fns={
            "ndcg@10": partial(rax.ndcg_metric, topn=10),
            "mrr@10": partial(rax.mrr_metric, topn=10),
            "dcg@01": partial(rax.dcg_metric, topn=1),
            "dcg@03": partial(rax.dcg_metric, topn=3),
            "dcg@05": partial(rax.dcg_metric, topn=5),
            "dcg@10": partial(rax.dcg_metric, topn=10),
        },
        epochs=25,
        early_stopping=EarlyStopping(metric="dcg@10", patience=2),
    )
    best_state = trainer.train(model, trainer_loader, val_loader)

    val_df = trainer.test(model, best_state, val_loader, "Validation")
    val_df.to_parquet("val.parquet")

    test_df = trainer.test(model, best_state, test_loader, "Testing")
    test_df.to_parquet("test.parquet")


if __name__ == "__main__":
    main()
