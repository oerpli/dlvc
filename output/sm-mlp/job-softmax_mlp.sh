#!/bin/bash
#PBS -m bea

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd submissions
cd assignment2
python3 opt_mlp_classify_hog_tinycifar10.py
