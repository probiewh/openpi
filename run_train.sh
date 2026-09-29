#!/bin/bash

#set -e

#CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/compute_norm_stats.py --config-name second_task > ./logs/norm_second_task.log 2>&1

#HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py second_task --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints --exp-name=second_task --overwrite > /root/data/xxy/openpi/logs/train_second_task.log 2>&1

#HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py second_task --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints --exp-name=second_task --overwrite > /root/data/xxy/openpi/logs/train_second_task.log 2>&1

HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/compute_norm_stats.py --config-name third_task > /root/data/xxy/openpi/logs/norm_third_task.log 2>&1

#HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=6,7,8,9 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py second_task_4cam --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints --exp-name=second_task_4cam --overwrite > /root/data/xxy/openpi/logs/train_second_task_4cam.log 2>&1

HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py third_task --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints --exp-name=third_task --overwrite > /root/data/xxy/openpi/logs/train_third_task.log 2>&1
