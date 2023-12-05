import logging
from functools import partial

import hydra
import jax
import optax
import pyarrow
import rax
import torch
import wandb
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from src.data import collate_fn, hash_labels, discretize, random_split
from src.log import get_wandb_run_name
from src.trainer import Trainer, Stage
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

def load_train_data(config: DictConfig):
    train_dataset = load_dataset(
        config.data.name,
        name="clicks",
        split="train",
        cache_dir=config.cache_dir,
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

    return train_dataset.map(encode_bias, num_proc=1)


def load_val_data(config: DictConfig):
    val_dataset = load_dataset(
        config.data.name,
        name="annotations",
        split="validation",
        cache_dir=config.cache_dir,
    )
    val_dataset.set_format("numpy")
    return val_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(config: DictConfig):
    torch.manual_seed(config.random_state)
    print(OmegaConf.to_yaml(config))

    if config.logging:
        run_name = get_wandb_run_name(config)
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            save_code=True,
        )

    train_click_dataset = load_train_data(config)
    train_click_dataset, test_click_dataset = random_split(
        train_click_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.2,
    )
    val_click_dataset, test_click_dataset = random_split(
        test_click_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
    )

    val_rel_dataset = load_val_data(config)
    val_rel_dataset, test_rel_dataset = random_split(
        val_rel_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
        stratify="frequency_bucket",
    )

    train_loader = DataLoader(
        train_click_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_click_loader = DataLoader(
        val_click_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_rel_loader = DataLoader(
        val_rel_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_click_loader = DataLoader(
        test_click_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
    )
    test_rel_loader = DataLoader(
        test_rel_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
    )

    model = instantiate(config.model)
    criterion = instantiate(config.loss)

    trainer = Trainer(
        random_state=config.random_state,
        optimizer=optax.adam(learning_rate=config.lr),
        criterion=criterion,
        metric_fns={
            "ndcg@10": partial(rax.ndcg_metric, topn=10),
            "mrr@10": partial(rax.mrr_metric, topn=10),
            "dcg@01": partial(rax.dcg_metric, topn=1),
            "dcg@03": partial(rax.dcg_metric, topn=3),
            "dcg@05": partial(rax.dcg_metric, topn=5),
            "dcg@10": partial(rax.dcg_metric, topn=10),
        },
        epochs=config.max_epochs,
        early_stopping=EarlyStopping(
            metric=config.es_metric,
            patience=config.es_patience,
        ),
        save_checkpoints=config.checkpoints,
        log_metrics=config.logging,
    )
    best_state = trainer.train(
        model,
        train_loader,
        val_click_loader,
        val_rel_loader,
    )

    _, val_rel_df = trainer.eval(
        model,
        best_state,
        val_click_loader,
        val_rel_loader,
        stage=Stage.VAL,
    )
    val_rel_df.to_parquet("val.parquet")

    _, test_rel_df = trainer.eval(
        model,
        best_state,
        test_click_loader,
        test_rel_loader,
        stage=Stage.TEST,
    )
    test_rel_df.to_parquet("test.parquet")

    if config.logging:
        wandb.finish()

    # Return best val metric for hyperparameter tuning using Optuna
    best_val_metrics = aggregate_metrics(val_rel_df)
    return best_val_metrics[config.es_metric]


if __name__ == "__main__":
    main()
