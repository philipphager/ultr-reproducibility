from pathlib import Path
from typing import List


def is_model_directory(path: Path) -> bool:
    state_path = path / "best_state"
    config_path = path / ".hydra/config.yaml"
    return config_path.exists() and state_path.exists()


def get_model_directories(path: Path) -> List[Path]:
    return list(filter(is_model_directory, map(Path, path.glob("*/"))))


def parse_model_name(path: Path):
    directory = path.name
    options = {}

    for option in directory.split(","):
        k, v = option.split("=")
        options[k] = v

    return options
