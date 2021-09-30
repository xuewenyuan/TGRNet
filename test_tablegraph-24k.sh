#! /bin/bash
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port 2351 --use_env test.py --dataroot ./datasets/tablegraph24k --dataset_mode tbrec_tablegraph24k --model tbrec --batch_size 1 --name tablegraph24k_overall --num_rows 38 --num_cols 22 --epoch best
