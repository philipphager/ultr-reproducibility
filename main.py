import logging
from functools import partial

import hydra
import optax
import pyarrow
import pyarrow_hotfix
import rax
import torch
import wandb
from datasets import load_dataset, Dataset
from flax.training.early_stopping import EarlyStopping
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import collate_fn, random_split, LabelEncoder
from src.log import get_wandb_run_name
from src.metrics import negative_log_likelihood
from src.trainer import Trainer, Stage

logging.basicConfig(level=logging.INFO)

pyarrow_hotfix.uninstall()
pyarrow.PyExtensionType.set_auto_load(True)


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
    encode_query = LabelEncoder()

    def preprocess(batch):
        batch["query_id"] = encode_query(batch["query_id"])
        return batch

    dataset = load_dataset(
        config.data.name,
        name="annotations",
        split=split,
        cache_dir=config.cache_dir,
    )
    dataset.set_format("numpy")

    return dataset.map(preprocess)


def get_loader(config: DictConfig, dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )


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
    annotations = load_annotations(config)

    ### FIXME REMOVE
    train_clicks, _ = random_split(
        train_clicks,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.99,
    )

    test_clicks, _ = random_split(
        test_clicks,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.99,
    )
    ### FIXME REMOVE

    val_clicks, test_clicks = random_split(
        test_clicks,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
    )

    train_loader = get_loader(config, train_clicks)
    val_loader = get_loader(config, val_clicks)
    test_click_loader = get_loader(config, test_clicks)
    test_rel_loader = get_loader(config, annotations)

    model = instantiate(config.model)

    trainer = Trainer(
        random_state=config.random_state,
        optimizer=optax.adam(learning_rate=config.lr),
        metric_fns={
            "ndcg@10": partial(rax.ndcg_metric, topn=10),
            "mrr@10": partial(rax.mrr_metric, topn=10),
            "dcg@01": partial(rax.dcg_metric, topn=1),
            "dcg@03": partial(rax.dcg_metric, topn=3),
            "dcg@05": partial(rax.dcg_metric, topn=5),
            "dcg@10": partial(rax.dcg_metric, topn=10),
        },
        click_metric_fns={
            "nll": negative_log_likelihood
        },
        epochs=config.max_epochs,
        early_stopping=EarlyStopping(patience=config.es_patience),
        save_checkpoints=config.checkpoints,
        log_metrics=config.logging,
        progress_bar=config.progress_bar,
    )

    best_state, history_df = trainer.train(
        model,
        train_loader,
        val_loader,
    )

    val_df = trainer.test_clicks(
        model,
        best_state,
        val_loader,
    )

    test_click_df = trainer.test_clicks(
        model,
        best_state,
        test_click_loader,
        log_stage=Stage.TEST
    )

    test_rel_df = trainer.test_relevance(
        model,
        best_state,
        test_rel_loader,
        log_stage=Stage.TEST
    )

    history_df.to_parquet("history.parquet")
    val_df.to_parquet("val.parquet")
    test_click_df.to_parquet("test_click.parquet")
    test_rel_df.to_parquet("test_rel.parquet")

    if config.logging:
        wandb.finish()


if __name__ == "__main__":
    main()
