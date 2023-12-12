import pandas as pd
import numpy as np
import torch

from ultr_bias_toolkit.bias.naive import NaiveCtrEstimator
from ultr_bias_toolkit.bias.intervention_harvesting import PivotEstimator, AdjacentChainEstimator, AllPairsEstimator

torch.manual_seed(2024) # For the cross-entopy maximization in AllPairs

df = pd.read_csv("~/.cache/huggingface/features/features-part-0.csv")
df = df[~df["text_md5"].isin(["876a3c0014c4ca2c534b3f0c8adc8b1c", "e86e1992b76bde5e803c912f9964e1a6"])] # Remove the two problematic docs
df = df[["query_md5", "text_md5", "position", "click"]].rename(columns={"query_md5": "query_id", "text_md5": "doc_id"})

estimators = {
    "ctr": NaiveCtrEstimator(),
    "pivot_one": PivotEstimator(pivot_rank=1),
    "adjacent_chain": AdjacentChainEstimator(),
    "global_all_pairs": AllPairsEstimator(),
}
examination_dfs = []

for i, (name, estimator) in enumerate(estimators.items()):
    print(f"{name} ({i+1}/{len(estimators)})")
    examination_df = estimator(df)
    examination_df["estimator"] = name
    examination_dfs.append(examination_df)

examination_df = pd.concat(examination_dfs).pivot_table(values="examination", index="position", columns="estimator")
print(examination_df.head(10))
print(len(examination_df))

for name in estimators.keys():
    np.save("propensities/"+name+"_nofake.npy", examination_df[name].to_numpy())

