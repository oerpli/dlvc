#!/bin/bash
#PBS -m bea

# Request free GPU
export CUDA_VISIBLE_DEVICES=$(getFreeGPU)

cd group1
cd submissions
cd assignment2
python3 cnn_classify_image.py --means 125.31 122.91 113.8 --stds 63.05 62.16 66.74 --model model_cnn_best.h5 --image cat.jpg
python3 cnn_classify_image.py --means 125.31 122.91 113.8 --stds 63.05 62.16 66.74 --model model_cnn_best.h5 --image cat_new.jpg
