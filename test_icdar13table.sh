#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --dataroot ./datasets/icdar13table --dataset_mode tbrec_icdar13table --model tbrec --batch_size 1 --name icdar13table_overall --num_rows 58 --num_cols 13 --epoch best
