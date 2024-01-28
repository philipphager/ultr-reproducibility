#!/bin/bash

source ${HOME}/.bashrc
mamba activate ultr-reproducibility

python main.py -m \
  checkpoints=False\
  logging=False\
  data=baidu-mlm-ctr \
  model=naive-pointwise \
  model.config.features=bert \
  model.config.dims=128,256,512 \
  model.config.layers=3,4,5 \
  model.config.dropout=0.0,0.3,0.5 \
  random_state=816,1906,4269,5707,9057 \
  lr=0.01,0.001,0.001
