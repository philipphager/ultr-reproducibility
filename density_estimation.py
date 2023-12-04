'''
    Multivariate kernel density estimation of the logging policy.
'''

from functools import partial

import hydra
import numpy as np
from sklearn.decomposition import PCA
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
import time
from csv import writer

from src.data import random_split
from src.distribution import kde, gaussian_model, kl_divergence

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

    def encode_embedds(batch):
        batch["query_document_embedding"] = batch["query_document_embedding"].mean(axis=0)
        return batch
    
    return val_dataset.map(encode_embedds, num_proc=1)


@hydra.main(version_base="1.2", config_path="config", config_name="kde")
def main(config: DictConfig):
    np.random.seed(config.random_state)

    start_time = time.time()
    train_click_dataset = load_train_data(config)
    train_click_dataset, test_click_dataset = random_split(
        train_click_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.2,
    )
    test_rel_dataset = load_val_data(config)
    test_rel_dataset, _ = random_split(
        test_rel_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.01,
        stratify="frequency_bucket",
    )

    qdoc_train = train_click_dataset["query_document_embedding"]
    qdoc_test = test_click_dataset["query_document_embedding"]
    qdoc_test_rel = test_rel_dataset["query_document_embedding"]
    print(f"Loaded the data ({time.time() - start_time:.3f}s)")

    start_time = time.time()
    pca = PCA(n_components=config.pca_dim, whiten=False)
    pca.fit(qdoc_train)
    qdoc_train = pca.transform(qdoc_train)
    qdoc_test = pca.transform(qdoc_test)
    qdoc_test_rel = pca.transform(qdoc_test_rel)
    print(f"Applied PCA ({time.time() - start_time:.3f}s)")

    start_time = time.time()
    kde_train, [kde_test, kde_test_rel] = kde(
        qdoc_train, 
        [qdoc_test, qdoc_test_rel], 
        config.kde_kernel, 
        config.kde_bw,
        )
    print(f"KDE estimation ({time.time() - start_time:.3f}s):")
    print(f" train: {kde_train:.3f}, sessions: {kde_test:.3f}, annotations: {kde_test_rel:.3f}")

    start_time = time.time()
    gaussian_train, [gaussian_test, gaussian_test_rel] = gaussian_model(
        qdoc_train, 
        [qdoc_test, qdoc_test_rel], 
        )
    print(f"Gaussian estimation ({time.time() - start_time:.3f}s):")
    print(f" train: {gaussian_train:.3f}, sessions: {gaussian_test:.3f}, annotations: {gaussian_test_rel:.3f}")

    start_time = time.time()
    kl_test, kl_test_rel = kl_divergence(
        qdoc_train, 
        [qdoc_test, qdoc_test_rel], 
        )
    print(f"KL estimation ({time.time() - start_time:.3f}s):")
    print(f" train|sessions: {kl_test:.3f}, train|annotations: {kl_test_rel:.3f}")

    return 

if __name__ == "__main__":
    main()
