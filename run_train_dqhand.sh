#!/usr/bin/env bash
set -euo pipefail

cd /root/data1/xxy/openpi

uv run scripts/train_dqhand_tokenizer.py
uv run scripts/compute_norm_stats.py --config-name first_task_4cam_dqhand

echo "DQ-Hand tokenizer and normalization statistics are ready."
echo "Start Pi0.5 training with:"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py first_task_4cam_dqhand --exp-name=<name> --overwrite"
