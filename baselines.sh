#!/usr/bin/env bash
rm -rf multirun
python main.py -m model=naive loss=pointwise,listwise
python main.py -m model=pbm,two-towers model.tower_combination=ADDITIVE loss=pointwise,listwise
python main.py -m model=pbm,two-towers model.tower_combination=NONE loss=listwise-dla,pointwise-em,listwise-em
