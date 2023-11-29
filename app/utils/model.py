from pathlib import Path

from flax.training import checkpoints


def load_state(path: Path, name: str = "best_state"):
    return checkpoints.restore_checkpoint(path.resolve() / name, target=None)
