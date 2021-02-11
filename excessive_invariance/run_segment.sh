#!/bin/bash

DATA_DIR=$1
FRAC=$2
GPU=$3

for run in 1 2 3 4 5
do
  CUDA_VISIBLE_DEVICES=$GPU python3 train.py --save_name mnist_model_${FRAC}_${run}_with_dot --frac $FRAC --add_dot
done