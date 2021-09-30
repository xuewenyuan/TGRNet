#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1351 --use_env test.py --dataroot ./datasets/icdar19_ctdar --dataset_mode tbrec_icdar19ctdar_lloc --model tbrec_llocpre --batch_size 1 --name icdar19_lloc --num_rows 88 --num_cols 44 --load_height 800 --load_width 800 --epoch best
