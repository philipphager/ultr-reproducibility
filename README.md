# ULTR Reproducibility [WIP name]
Reproducing the Baidu ULTR experiments in Jax.

## WIP: Installation
* If [Poetry](https://python-poetry.org/docs/cli/) is available, you can install all dependencies by running: `poetry install`.
* If Conda or [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) is available, you can use `mamba env create --file environment.yaml` which supports CUDA 11.8.

## Usage
Select a model `["naive", "pbm", "two-tower"]` and loss `["pointwise", "listwise"]`:
```bash
 python main.py -m model=naive loss=pointwise
```

You can also train the PBM and Two Tower models using the DLA `["pointwise-dla", "listwise-dla"]` and RegressionEM `["pointwise-em", "listwise-em"]` loss. However, they expect that relevance and examination are not combined into a click prediction:  
```bash
 python main.py -m model=two-towers loss=listwise-dla model.tower_combination=NONE
```
