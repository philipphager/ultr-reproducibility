from pathlib import Path

import pandas as pd
import torch
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import AllPairsEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator

FEATURE_URL = "https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-mlm-ctr/blob/main/parts/train-features.feather?download=true"


def main(
    cache_directory: str = "/beegfs/scratch/user/rdeffaye/baidu-bert/features/",
    random_state: int = 2024,
):
    # For the cross-entopy maximization in AllPairs
    torch.manual_seed(random_state)

    cache_directory = Path(cache_directory).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    feature_path = cache_directory / "train-features.feather"

    if not feature_path.exists():
        print("Downloading Baidu features from huggingface...")
        df = pd.read_feather(FEATURE_URL, columns = ["query_md5", "text_md5", "position", "click"])
        df.to_feather(feature_path)
    else:
        df = pd.read_feather(feature_path, columns = ["query_md5", "text_md5", "position", "click"])

    estimators = {
        "ctr": NaiveCtrEstimator(),
        "pivot_one": PivotEstimator(pivot_rank=1),
        "adjacent_chain": AdjacentChainEstimator(),
        "global_all_pairs": AllPairsEstimator(),
    }
    examination_dfs = []

    for i, (name, estimator) in enumerate(estimators.items()):
        print(f"{name} ({i+1}/{len(estimators)})")
        examination_df = estimator(df, query_col="query_md5", doc_col="text_md5")
        examination_df["estimator"] = name
        examination_dfs.append(examination_df)

    examination_df = pd.concat(examination_dfs)
    examination_df = examination_df.pivot_table(
        values="examination",
        index="position",
        columns="estimator",
    )
    print(examination_df.head(10))

    Path("propensities").mkdir(parents=True, exist_ok=True)
    for name in estimators.keys():
        examination_df[name].to_csv(f"propensities/{name}.csv")


if __name__ == "__main__":
    main()
