#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run scripts/compute_norm_stats.py --config-name pi05_lora_g1_130_old > train_norm_g1_130_old.log 2>&1


CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lora_g1_130_old --exp-name=pi05_lora_g1_130_old --overwrite > train_g1_130_old.log 2>&1
    
