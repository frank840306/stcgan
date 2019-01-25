#! /bin/bash


./run.sh --mode train --gpu_id 0 --epochs 50 --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size 10 --dataset_name DSRD_extend_aug --model_step 1
python src/eval_all_model.py --root_dir /media/yslin/SSD_DATA/research/stcgan --batch_size_test 5 --lrG 0.001 --lrD 0.001 --dataset_name DSRD_extend_aug