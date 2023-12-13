from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from ultr_bias_toolkit.bias.intervention_harvesting import AdjacentChainEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import AllPairsEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator
from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator

FEATURE_URL = "https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-base-12L/resolve/main/features/features-part-0.csv?download=true"


def main(
    cache_directory: str = "~/.cache/huggingface/features/",
    random_state: int = 2024,
    ignored_docs: List[str] = [
        "876a3c0014c4ca2c534b3f0c8adc8b1c",
        "e86e1992b76bde5e803c912f9964e1a6",
    ],
):
    # For the cross-entopy maximization in AllPairs
    torch.manual_seed(random_state)

    cache_directory = Path(cache_directory).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    feature_path = cache_directory / "features-part-0.csv"

    if not feature_path.exists():
        print("Downloading Baidu features from huggingface...")
        df = pd.read_csv(FEATURE_URL)
        df.to_csv(feature_path)
    else:
        df = pd.read_csv(feature_path)

    # Remove problematic documents
    df = df[~df["text_md5"].isin(ignored_docs)]
    df = df[["query_md5", "text_md5", "position", "click"]]

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
    print(len(examination_df))

    for name in estimators.keys():
        np.save(f"propensities/{name}_nofake.npy", examination_df[name].to_numpy())


if __name__ == "__main__":
    main()
