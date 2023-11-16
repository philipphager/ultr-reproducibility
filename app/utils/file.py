from pathlib import Path


def has_checkpoint(path: Path) -> bool:
    state_path = path / "best_state"
    config_path = path / ".hydra/config.yaml"
    return config_path.exists() and state_path.exists()


def has_metrics(path: Path) -> bool:
    val_path = path / "val.parquet"
    test_path = path / "test.parquet"
    return val_path.exists() and test_path.exists()


def parse_model_name(path: Path):
    directory = path.name
    options = {}

    for option in directory.split(","):
        k, v = option.split("=")
        options[k] = v

    return options
