import time

from omegaconf import DictConfig


def get_wandb_run_name(config: DictConfig) -> str:
    model = config.model._target_.split('.')[-1]
    random_state = config.random_state
    timestamp = int(time.time())

    return f"{model}__{random_state}__{timestamp}"
