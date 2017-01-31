#!/bin/bash
#PBS -m bea

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd test
cd test_ah
python3 train_best_model_a2-k.py
