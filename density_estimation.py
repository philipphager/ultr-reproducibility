'''
    Multivariate kernel density estimation of the logging policy.
'''

from functools import partial

import hydra
import numpy as np
from sklearn.decomposition import PCA
from KDEpy import TreeKDE
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
import time

from src.data import random_split

def load_train_data(config: DictConfig):
    train_dataset = load_dataset(
        config.data.name,
        name="clicks",
        split="train",
        cache_dir=config.cache_dir,
    )
    train_dataset = train_dataset.select_columns(["query_document_embedding"])
    train_dataset.set_format("numpy")

    def encode_embedds(batch):
        batch["query_document_embedding"] = batch["query_document_embedding"].mean(axis=0)
        return batch

    return train_dataset.map(encode_embedds, num_proc=1)


def load_val_data(config: DictConfig):
    val_dataset = load_dataset(
        config.data.name,
        name="annotations",
        split="validation",
        cache_dir=config.cache_dir,
    )
    val_dataset.set_format("numpy")
    return val_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="kde")
def main(config: DictConfig):
    np.random.seed(config.random_state)

    ### Load the appropriate data
    train_click_dataset = load_train_data(config)
    train_click_dataset, test_click_dataset = random_split(
        train_click_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.2,
    )
    qdoc_train = train_click_dataset["query_document_embedding"]
    qdoc_test = test_click_dataset["query_document_embedding"]

    ### Reduce the dimensionality of features
    pca = PCA(n_components=config.pca_dim, whiten=False)
    pca.fit(qdoc_train)
    qdoc_train = pca.transform(qdoc_train)
    qdoc_test = pca.transform(qdoc_test)

    ### Apply KDE
    start_time = time.time()
    kde = TreeKDE(kernel=config.kde_kernel, bw=config.kde_bw).fit(qdoc_train)
    likelihood_train = kde.evaluate(qdoc_train)
    likelihood_test = kde.evaluate(qdoc_test)
    mll_train = np.log(likelihood_train).mean()
    mll_test = np.log(likelihood_test).mean()

    print(f"Train log-likelihood: {mll_train}, test log-likelihood: {mll_test}")
    print(f"KDE runtime: {time.time() - start_time}")

    return mll_train, mll_test


if __name__ == "__main__":
    main()
