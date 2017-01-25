#!/bin/bash
#PBS -m bea

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd submissions
cd assignment3
python3 train_best_model.py
