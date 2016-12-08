#!/bin/bash
#PBS -m bea
#PBS -M oerpli@outlook.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd submissions
cd assignment2
python3 softmax_classify_hog_tinycifar10.py
python3 opt_softmax_classify_hog_tinycifar10.py
