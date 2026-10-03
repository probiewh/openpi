#!/bin/bash

#CUDA_VISIBLE_DEVICES=6,7,8,9 uv run scripts/compute_norm_stats.py --config-name pi05_lora_g1_130_old > train_norm_g1_130_old.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py first_task_4cam_armonly --exp-name=first_task_4cam_armonly --overwrite > train_first_task_4cam_armonly.log 2>&1
    
