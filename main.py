import logging

import jax
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from src.data import collate_clicks, collate_annotations, LabelEncoder, Discretize
from src.models.pbm import PositionBasedModel
from src.trainer import Trainer

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


def main():
    train_dataset = load_dataset(
        "philipphager/baidu-ultr-606k", name="clicks", split="train"
    )
    val_dataset = load_dataset(
        "philipphager/baidu-ultr-606k", name="annotations", split="validation[:50%]"
    )
    test_dataset = load_dataset(
        "philipphager/baidu-ultr-606k", name="annotations", split="validation[50%:]"
    )
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

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

    train_dataset = train_dataset.map(encode_bias)

    trainer_loader = DataLoader(
        train_dataset,
        collate_fn=collate_clicks,
        batch_size=16,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_annotations,
        batch_size=16,
        num_workers=4,
    )
    test_loader = DataLoader(
        val_dataset,
        collate_fn=collate_annotations,
        batch_size=16,
        num_workers=1,
    )

    model = PositionBasedModel()
    trainer = Trainer()
    best_state = trainer.train(model, trainer_loader, val_loader)
    trainer.test(best_state, test_loader)


if __name__ == "__main__":
    main()
