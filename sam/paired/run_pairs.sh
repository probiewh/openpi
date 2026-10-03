#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/data1/xxy/openpi/sam
cd "$ROOT/paired"
exec 9>"$ROOT/paired/processing.lock"
if ! flock -n 9; then echo 'Paired pipeline is already running.' >&2; exit 1; fi
export HF_HOME=/root/data/xxy/openpi/sam_runtime/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
"$ROOT/.venv/bin/python" -u run_pair_pipeline.py --gpus "${SAM_PAIR_GPUS:-4,5,6,7}"
