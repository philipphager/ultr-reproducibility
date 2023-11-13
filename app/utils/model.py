from pathlib import Path

from orbax.checkpoint import PyTreeCheckpointer


def load_state(path: Path, name: str = "best_state"):
    checkpointer = PyTreeCheckpointer()
    return checkpointer.restore(path.resolve() / name)


def load_model(path: Path):
    streamlit.write(path)

    with initialize(version_base=None, config_path=str(path / ".hydra")):
        config = compose(config_name="config.yaml")
        return instantiate(config.model)