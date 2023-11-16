import logging
from functools import partial
import pyarrow_hotfix; pyarrow_hotfix.uninstall()

import hydra
import jax
import optax
import pyarrow
import rax
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader
import wandb
import time

from src.data import collate_fn, hash_labels, discretize, stratified_split
from src.trainer import Trainer
from src.util import EarlyStopping, aggregate_metrics

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            tracebacks_suppress=[jax, pyarrow],
            console=Console(width=800),
        )
    ],
)

BAIDU_DATASET = "philipphager/baidu-ultr"


def load_train_data(cache_dir: str, num_workers: int):
    train_dataset = load_dataset(
        BAIDU_DATASET, name="clicks-1p", split="train", cache_dir=cache_dir
    )
    train_dataset.set_format("numpy")

    def encode_bias(batch):
        batch["media_type"] = hash_labels(batch["media_type"], 10_000)
        batch["displayed_time"] = discretize(batch["displayed_time"], 0, 128, 16)
        batch["serp_height"] = discretize(batch["serp_height"], 0, 1024, 16)
        batch["slipoff_count_after_click"] = discretize(
            batch["slipoff_count_after_click"], 0, 10, 10
        )
        return batch

    return train_dataset.map(encode_bias, num_proc=num_workers)


def load_val_data(cache_dir: str):
    val_dataset = load_dataset(
        BAIDU_DATASET, name="annotations", split="validation", cache_dir=cache_dir
    )
    val_dataset.set_format("numpy")
    return val_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(config: DictConfig):
    torch.manual_seed(config.random_state)
    print(OmegaConf.to_yaml(config))

    run_name = f"{config.model._target_.split('.')[-1]}__{config.loss._target_.split('.')[-1]}__{config.random_state}__{int(time.time())}"
    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        config=vars(config)["_content"],
        name=run_name,
    )

    train_dataset = load_train_data(config.cache_dir, config.num_workers)
    val_dataset = load_val_data(config.cache_dir)
    val_dataset, test_dataset = stratified_split(
        val_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
        stratify="frequency_bucket",
    )

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
        random_state=config.random_state,
        optimizer=optax.adam(learning_rate=config.trainer.learning_rate),
        criterion=criterion,
        metric_fns={
            "ndcg@10": partial(rax.ndcg_metric, topn=10),
            "mrr@10": partial(rax.mrr_metric, topn=10),
            "dcg@01": partial(rax.dcg_metric, topn=1),
            "dcg@03": partial(rax.dcg_metric, topn=3),
            "dcg@05": partial(rax.dcg_metric, topn=5),
            "dcg@10": partial(rax.dcg_metric, topn=10),
        },
        epochs=config.trainer.epochs,
        early_stopping=EarlyStopping(
            metric=config.trainer.val_metric,
            patience=config.trainer.patience,
        ),
    )
    best_state = trainer.train(model, trainer_loader, val_loader)

    val_df = trainer.test(model, best_state, val_loader, "Validation")
    val_df.to_parquet("val.parquet")

    test_df = trainer.test(model, best_state, test_loader, "Testing")
    test_df.to_parquet("test.parquet")

    # Return best val metric for hyperparameter tuning using Optuna
    best_val_metrics = aggregate_metrics(val_df)
    return best_val_metrics[config.trainer.val_metric]


if __name__ == "__main__":
    main()
