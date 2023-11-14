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

from src.data import collate_fn, hash_labels, discretize
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

BAIDU_DATASET = "philipphager/baidu-ultr"


def load_train_data(cache_dir: str):
    train_dataset = load_dataset(
        BAIDU_DATASET, name="clicks", split="train", cache_dir=cache_dir
    )
    train_dataset.set_format("numpy")

    def encode_bias(batch):
        batch["media_type"] = hash_labels(batch["media_type"], 10_000)
        batch["displayed_time"] = discretize(batch["displayed_time"], 0, 1024, 16)
        batch["serp_height"] = discretize(batch["serp_height"], 0, 128, 16)
        batch["slipoff_count_after_click"] = discretize(
            batch["slipoff_count_after_click"], 0, 10, 10
        )
        return batch

    return train_dataset.map(encode_bias, num_proc=8)


def load_val_data(cache_dir: str):
    val_dataset = load_dataset(
        BAIDU_DATASET, name="annotations", split="validation", cache_dir=cache_dir
    )
    val_dataset.set_format("numpy")
    return val_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(config: DictConfig):
    torch.manual_seed(config.random_state)

    train_dataset = load_train_data(cache_dir=config.cache_dir)
    val_dataset = load_val_data(cache_dir=config.cache_dir)
    val_dataset, test_dataset = random_split(val_dataset, config.val_test_split)

    trainer_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        num_workers=config.num_workers,
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
