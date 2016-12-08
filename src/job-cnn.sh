#!/bin/bash
#PBS -m bea
#PBS -M oerpli@outlook.com

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd submissions
cd assignment2
python3 cnn_classify_cifar10.py
