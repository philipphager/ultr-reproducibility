import logging
from functools import partial

import hydra
import jax
import optax
import pyarrow
import pyarrow_hotfix
import rax
import torch
import wandb
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from src.data import collate_fn, random_split, LabelEncoder
from src.log import get_wandb_run_name
from src.trainer import Trainer, Stage
from src.util import EarlyStopping, aggregate_metrics

pyarrow_hotfix.uninstall()

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


def load_clicks(config: DictConfig, split: str):
    encode_query = LabelEncoder()

    def preprocess(batch):
        batch["query_id"] = encode_query(batch["query_id"])
        return batch

    dataset = load_dataset(
        config.data.name,
        name="clicks",
        split=split,
        cache_dir=config.cache_dir,
    )
    dataset.set_format("numpy")

    return dataset.map(preprocess)


def load_annotations(config: DictConfig, split="test"):
    dataset = load_dataset(
        config.data.name,
        name="annotations",
        split=split,
        cache_dir=config.cache_dir,
    )
    dataset.set_format("numpy")
    return dataset


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

    train_clicks = load_clicks(config, split="train")
    test_clicks = load_clicks(config, split="test")
    test_rels = load_annotations(config)

    val_clicks, test_clicks = random_split(
        test_clicks,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
    )

    val_rels, test_rels = random_split(
        test_rels,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
        stratify="frequency_bucket",
    )

    train_click_loader = DataLoader(
        train_clicks,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_click_loader = DataLoader(
        val_clicks,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_click_loader = DataLoader(
        test_clicks,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
    )
    val_rel_loader = DataLoader(
        val_rels,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_rel_loader = DataLoader(
        test_rels,
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
        progress_bar=config.progress_bar,
    )
    best_state, history_df = trainer.train(
        model,
        train_click_loader,
        val_click_loader,
        val_rel_loader,
    )
    history_df.to_parquet("history.parquet")

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
