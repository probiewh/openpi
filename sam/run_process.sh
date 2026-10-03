#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/data1/xxy/openpi/sam
cd "$ROOT"
exec 9>"$ROOT/processing.lock"
if ! flock -n 9; then echo 'Another SAM processing job is already running.' >&2; exit 1; fi
export HF_HOME=/root/data/xxy/openpi/sam_runtime/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
"$ROOT/.venv/bin/python" -u "$ROOT/dataset_driver.py" --config "$ROOT/settings.json" --gpus "${SAM_GPUS:-0,1,2,3}"
