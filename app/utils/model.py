from pathlib import Path

from orbax.checkpoint import PyTreeCheckpointer


def load_state(path: Path, name: str = "best_state"):
    checkpointer = PyTreeCheckpointer()
    return checkpointer.restore(path.resolve() / name)
