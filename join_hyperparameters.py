import os
from pathlib import Path
from typing import List, Annotated, Optional

import pandas as pd
import typer
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm


def parse_config(path: Path) -> pd.DataFrame:
    """
    Loads Hydra config and flattens nested YAML to dataframe.
    """
    assert path.exists()
    config = OmegaConf.load(path)
    config = OmegaConf.to_object(config)
    return pd.json_normalize(config, sep="/")


def cross_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1["key"] = 0
    df2["key"] = 0
    df = df1.merge(df2, on="key", how="outer")
    return df.drop(columns=["key"])


def parse_remote(entity: str, project: str, run_name: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"config.config.run_name": run_name})
    dfs = []
    incomplete_runs = 0

    for run in tqdm(runs, desc=f"Loading {run_name} from W&B"):
        history_df = run.history(pandas=True)

        if len(history_df) == 0:
            incomplete_runs += 1
            continue

        # Ignore test columns, as test metrics were only computed based on DCG@10
        # and might be misleading:
        columns = [c for c in history_df.columns if "test" not in c.lower()]
        history_df = history_df[columns]

        # Rename columns:
        history_df.columns = history_df.columns.str.replace(".", "")
        history_df = history_df.rename(columns={"_step": "Misc/Epoch"})

        # Parse run config, flatten to dataframe, and ignore selected columns:
        config_df = pd.json_normalize(run.config, sep="/")
        columns = [c for c in config_df.columns if c not in ["_runtime", "_timestamp"]]
        config_df = config_df[columns]

        df = cross_join(config_df, history_df)
        dfs.append(df)

    if incomplete_runs > 0:
        print(f"Found {incomplete_runs} incomplete runs from W&B")

    return pd.concat(dfs)


def parse_local(run_path: Path) -> pd.DataFrame:
    dfs = []
    incomplete_runs = 0

    for path in tqdm(run_path.iterdir(), desc=f"Loading {run_path.name} locally"):
        path = Path(path)
        config_path = path / ".hydra/config.yaml"
        history_path = path / "history.parquet"

        if not path.is_dir() or path.name.startswith("."):
            # Ignore misc Hydra files and hidden directories
            continue

        if config_path.exists() and history_path.exists():
            config_df = parse_config(config_path)
            history_df = pd.read_parquet(history_path)
            df = cross_join(config_df, history_df)
            dfs.append(df)
        else:
            incomplete_runs += 1

    if incomplete_runs > 0:
        print(f"Found {incomplete_runs} incomplete local runs")

    return pd.concat(dfs)


def main(
    local_directory: Annotated[
        str, typer.Option(help="Local directory with Hydra runs")
    ] = "multirun/",
    output_file: Annotated[
        str, typer.Option(help="Path to output .parquet file with hyperparameters")
    ] = "multirun/hyperparameters.parquet",
    wandb_run: Annotated[
        Optional[List[str]], typer.Option(help="Runs to load from W&B")
    ] = None,
    wandb_entity: Annotated[
        str, typer.Option(help="W&B entity name")
    ] = "cm-offline-metrics",
    wandb_project: Annotated[
        str, typer.Option(help="W&B project name")
    ] = "baidu-reproducibility",
):
    local_directory = Path(local_directory)
    dfs = []
    wandb_run = wandb_run if wandb_run is not None else []

    for run in wandb_run:
        dfs.append(parse_remote(wandb_entity, wandb_project, run))

    for run_directory in local_directory.iterdir():
        run_directory = Path(run_directory)

        if run_directory.is_dir():
            dfs.append(parse_local(run_directory))

    df = pd.concat(dfs)
    df.to_parquet(output_file)
    print(f"Saved hyperparameters to {output_file}")


if __name__ == "__main__":
    typer.run(main)
