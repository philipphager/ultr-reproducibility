# Baidu-ULTR Reproducibility
Repository for the work `Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset` under submission at the SIGIR 24 reproducibility track.

## Overview
### Repositories
| Dataset                   | Description |
|---------------------------|-------------------------|
| [Baidu-ULTR Reprocibility](https://github.com/philipphager/ultr-reproducibility/edit/main/README.md) | **This repository**, containing the code to tune, train, and evaluate all reranking methods including reference implementations of [ULTR methods in Jax and Rax](https://github.com/philipphager/ultr-reproducibility/blob/main/src/loss.py). |
| [Baidu BERT models](https://github.com/philipphager/baidu-bert-model) | Repository containing the code to train flax-based MonoBERT models from scratch (optionally with ULTR). This dataset will be included into the main repository in the near future. |
| [Reranking datasets](https://github.com/philipphager/baidu-ultr) | Code to preprocess and publish the two reranking datasets to Huggingface (see below). |
| [ULTR Bias Toolkit](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr) | Reference implementation of Intervention Harvesting methods. RegressionEM as used in our work was implemented in this dataset. |

### Datasets

| Dataset                   | Description |
|---------------------------|-------------------------|
| [Language modeling dataset](https://huggingface.co/datasets/philipphager/baidu-ultr-pretrain/tree/main) | Subset of the original Baidu-ULTR used in our work to train and evaluate MonoBERT cross-encoders. |
| [Reranking dataset (Baidu BERT)](https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-mlm-ctr) | The first four partition of Baidu-ULTR with query-document embeddings produced by the official MonoBERT cross-encoder released by Baidu plus additional LTR features computed by us. |
| [Reranking dataset (our BERT)](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr) | The first four partition of Baidu-ULTR with query-document embeddings produced by our naive MonoBERT cross-encoder and additional LTR features. |

### Hyperparameters

#### Base Language Models
We list all hyperparameters used to train our BERT models [here](https://github.com/philipphager/baidu-bert-model/blob/main/config/config.yaml).

#### Reranking models
We train small feed-forward networks with ReLU activation on fixed query-document embeddings and LTR vectors to compare ULTR objectives. We tune the model architecture per dataset and lr and dropout regularization per method/dataset combination. We list all final hyperparameters of the reranking models under [`config/hyperparameters/`](https://github.com/philipphager/ultr-reproducibility/tree/main/config/hyperparameter).

##: Installation
* If [Poetry](https://python-poetry.org/docs/cli/) is available, you can install all dependencies by running: `poetry install`.
* If Conda or [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) is available, you can use `mamba env create --file environment.yaml` which supports CUDA 11.8.

### Position Bias


## Usage
Select a model `["naive", "pbm", "two-tower"]` and loss `["pointwise", "listwise"]`:
```bash
 python main.py -m model=naive loss=pointwise
```

You can also train the PBM and Two Tower models using the DLA `["pointwise-dla", "listwise-dla"]` and RegressionEM `["pointwise-em", "listwise-em"]` loss. However, they expect that relevance and examination are not combined into a click prediction:  
```bash
 python main.py -m model=two-towers loss=listwise-dla model.tower_combination=NONE
```

## Dashboards
You can run streamlit apps to inspect test metrics:
```bash
streamlit run 1_plot_metrics.py
```
Or plot the learned position bias of models:
```bash
streamlit run 2_plot_bias.py
```
