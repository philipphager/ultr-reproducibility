# Baidu-ULTR Reproducibility
Repository for the work `Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset` under submission at the SIGIR 24 reproducibility track.

## Overview
### Repositories
| Dataset                   | Description |
|---------------------------|-------------------------|
| [Baidu-ULTR reprocibility](https://github.com/philipphager/ultr-reproducibility/edit/main/README.md) | **This repository**, containing the code to tune, train, and evaluate all reranking methods including reference implementations of [ULTR methods in Jax and Rax](https://github.com/philipphager/ultr-reproducibility/blob/main/src/loss.py). |
| [Baidu-ULTR MonoBERT models](https://github.com/philipphager/baidu-bert-model) | Repository containing the code to train flax-based MonoBERT models from scratch (optionally with ULTR). |
| [Reranking datasets](https://github.com/philipphager/baidu-ultr) | Code to preprocess and publish the two reranking datasets to Huggingface (see below). |
| [ULTR bias toolkit](https://github.com/philipphager/ultr-bias-toolkit) | Reference implementation of Intervention Harvesting methods. RegressionEM as used in our work was implemented in this dataset. |

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

### Position Bias
<p align="center">
 <img src='https://github.com/philipphager/ultr-reproducibility/assets/9155371/c1bb9d2d-9c82-4c3f-a09d-dae7ce10c8f4' width='600'>
</p>

Position bias as estimated with the [ULTR Bias Toolkit](https://github.com/philipphager/ultr-bias-toolkit) on partitions 1-3 of Baidu-ULTR.

### Installation
* If [Poetry](https://python-poetry.org/docs/cli/) is available, you can install all dependencies by running: `poetry install`.
* If [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) is available, you can use `mamba env create --file environment.yaml` which supports CUDA 11.8.

### Usage
Select a dataset `["baidu", "uva", "ltr"]` and model/loss combination, e.g.,: `["naive-pointwise", "regression-em", "dla", "pairwise-debias"]` and run:
```bash
 python main.py data=baidu model=naive-pointwise

### Reference
```
@inproceedings{Hager2024BaiduULTR,
  author = {Philipp Hager and Romain Deffayet and Jean-Michel Renders and Onno Zoeter and Maarten de Rijke},
  title = {Unbiased Learning to Rank Meets Reality: Lessons from Baidu’s Large-Scale Search Dataset},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`24)},
  organization = {ACM},
  year = {2024},
}
```
