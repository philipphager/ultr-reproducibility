#!/usr/bin/env bash
python main.py -m model=naive loss=pointwise,listwise
python main.py -m model=pbm,two-towers model.tower_combination=ADDITIVE loss=pointwise,listwise
python main.py -m model=pbm,two-towers model.tower_combination=NONE loss=pointwise-dla,listwise-dla,pointwise-em,listwise-em
