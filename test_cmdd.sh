#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env test.py --dataroot ./datasets/cmdd --dataset_mode tbrec_cmdd --model tbrec --batch_size 1 --name cmdd_overall --num_rows 25 --num_cols 6 --epoch best
