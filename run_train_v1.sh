#!/bin/bash

# #CUDA_VISIBLE_DEVICES=6,7,8,9 uv run scripts/compute_norm_stats.py --config-name pi05_lora_g1_130_old > train_norm_g1_130_old.log 2>&1

# CUDA_VISIBLE_DEVICES=6,7,8,9 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py first_task_4cam_lora32 --exp-name=first_task_4cam_lora32 --overwrite > train_first_task_4cam_lora32.log 2>&1
   
# HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=6,7,8,9 uv run scripts/compute_norm_stats.py --config-name third_task_4cam_lora32 > /root/data/xxy/openpi/logs/norm_third_task_4cam_lora32.log 2>&1

# rsync -aH --partial --info=progress2 /root/data1/xxy/openpi/assets/third_task_4cam_lora32 /root/data/xxy/openpi/assets/

HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=6,7,8,9 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py third_task_4cam_lora32 --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints --exp-name=third_task_4cam_lora32 --overwrite > /root/data/xxy/openpi/logs/train_third_task_4cam_lora32.log 2>&1
