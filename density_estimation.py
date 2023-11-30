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
from csv import writer


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
    val_click_dataset, test_click_dataset = random_split(
        test_click_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
    )
    val_rel_dataset = load_val_data(config)
    val_rel_dataset, test_rel_dataset = random_split(
        val_rel_dataset,
        shuffle=True,
        random_state=config.random_state,
        test_size=0.5,
        stratify="frequency_bucket",
    )

    qdoc_train = train_click_dataset["query_document_embedding"]
    qdoc_val = val_click_dataset["query_document_embedding"]
    qdoc_test = test_click_dataset["query_document_embedding"]
    qdoc_val_rel = val_rel_dataset["query_document_embedding"]
    qdoc_test_rel = test_rel_dataset["query_document_embedding"]
    print(f"Loaded the data ({time.time() - start_time:.3f}s)")
    start_time = time.time()

    pca = PCA(n_components=config.pca_dim, whiten=False)
    pca.fit(qdoc_train)
    qdoc_train = pca.transform(qdoc_train)
    qdoc_val = pca.transform(qdoc_val)
    qdoc_test = pca.transform(qdoc_test)
    qdoc_val_rel = pca.transform(qdoc_val_rel)
    qdoc_test_rel = pca.transform(qdoc_test_rel)
    print(f"Applied PCA ({time.time() - start_time:.3f}s)")

    start_time = time.time()
    kde = TreeKDE(kernel=config.kde_kernel, bw=config.kde_bw).fit(qdoc_train)
    fitting_duration = time.time() - start_time
    print(f"Fitted KDE ({fitting_duration:.3f}s)")

    start_time = time.time()
    likelihood_train = kde.evaluate(qdoc_train)
    likelihood_val = kde.evaluate(qdoc_val)
    likelihood_test = kde.evaluate(qdoc_test)
    mll_train = np.log(likelihood_train).mean()
    mll_val = np.log(likelihood_val).mean()
    mll_test = np.log(likelihood_test).mean()
    eval_duration = time.time() - start_time
    print(f"KDE evaluation on sessions ({eval_duration:.3f}s)")
    print(f"Log-likelihood: \n train: {mll_train:.3f}, val: {mll_val:.3f}, test: {mll_test:.3f}")

    start_time = time.time()
    likelihood_val_rel = kde.evaluate(qdoc_val_rel)
    likelihood_test_rel = kde.evaluate(qdoc_test_rel)
    mll_val_rel = np.log(likelihood_val_rel).mean()
    mll_test_rel = np.log(likelihood_test_rel).mean()
    print(f"KDE evaluation on annotated queries ({time.time() - start_time:.3f}s)")
    print(f"Log-likelihood : \n val: {mll_val_rel:.3f}, test: {mll_test_rel:.3f}")

    if config.logging:
        with open(config.cache_dir + 'tencent_KDE_gridsearch.csv', 'a') as f: 
            csvwriter = writer(f)
            csvwriter.writerow([config.kde_kernel, config.kde_bw, config.pca_dim, fitting_duration, 
                                    eval_duration, mll_train, mll_val, mll_test, mll_val_rel, mll_test_rel])
            f.close()

    return mll_train, mll_val, mll_test, mll_val_rel, mll_test_rel

if __name__ == "__main__":
    main()
