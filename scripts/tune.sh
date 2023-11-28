#!/bin/bash

source ${HOME}/.bashrc
conda activate ultr-reproducibility

python main.py -m model=naive loss=pointwise +tune=naive $@
python main.py -m model=pbm loss=pointwise +tune=naive $@
python main.py -m model=two-towers loss=pointwise +tune=naive $@
