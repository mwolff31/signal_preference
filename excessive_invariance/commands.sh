#!/bin/bash

MNIST_PATH=$1
GPU=$2

# make train/val splits and save image files
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --first
# train an MNIST model 
# we're going to use the model to make deviation segments
CUDA_VISIBLE_DEVICES=$GPU python3 train.py --mnist_path $MNIST_PATH

# creates segments of the training data with in frac deviation
# according to the trained model's latent space
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.75
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.5
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.25
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.1
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.05
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --mnist_path $MNIST_PATH --frac 0.01

# train 5 models on each segment with a dot next to the digit
sh run_segment.sh $MNIST_PATH 1 $GPU
sh run_segment.sh $MNIST_PATH 0.75 $GPU
sh run_segment.sh $MNIST_PATH 0.5 $GPU
sh run_segment.sh $MNIST_PATH 0.25 $GPU
sh run_segment.sh $MNIST_PATH 0.1 $GPU
sh run_segment.sh $MNIST_PATH 0.05 $GPU
sh run_segment.sh $MNIST_PATH 0.01 $GPU

# aggregate results and make graphs
python3 main.py --mnist_path $MNIST_PATH --mode graph
python3 main.py --mnist_path $MNIST_PATH --mode graph --add_dot
python3 main.py --mnist_path $MNIST_PATH --mode graph --just_dot