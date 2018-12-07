#! /bin/bash

PYTHON=python3
GPU_ID=3


$PYTHON src/main.py $@

#  
# train:
#     run.sh --mode train --gpu_id 3 --epochs 500
# test:
#     run.sh --gpu_id 3 --load_model --model_name best
# 
