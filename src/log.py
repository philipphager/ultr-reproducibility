import time
from typing import Dict

from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table


def print_metric_table(metrics: Dict, title=""):
    console = Console()
    table = Table(show_header=True, header_style="bold", title=title, min_width=32)

    for name in metrics.keys():
        table.add_column(name)

    table.add_row(*list(map(str, metrics.values())))
    console.print(table)


def get_wandb_run_name(config: DictConfig) -> str:
    model = config.model._target_.split('.')[-1]
    random_state = config.random_state
    timestamp = int(time.time())

    return f"{model}__{random_state}__{timestamp}"
