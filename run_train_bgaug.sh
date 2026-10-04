#!/usr/bin/env bash
set -euo pipefail
cd /root/data1/xxy/openpi

CONFIG=first_task_4cam_bgaug_lora32
EXP=first_task_4cam_bgaug_lora32
BG_TRAIN_GPUS="${BG_TRAIN_GPUS:-6,7,8,9}"
export HF_LEROBOT_HOME=/root/data/xxy/openpi/datasets

# Background edits do not alter the state/action distribution. Baseline norm
# stats have already been copied into this config's own asset directory.
# Do not repeat compute_norm_stats or the old data1 -> data assets rsync here.
test -f "$HF_LEROBOT_HOME/competition/cup_four_cameras_bgaug32_sam2_pairs400/PROCESSING_COMPLETE.json"
test -f /root/data/xxy/openpi/assets/first_task_4cam_bgaug_lora32/competition/cup_four_cameras_bgaug32_sam2_pairs400/norm_stats.json
if [[ "${1:-}" == "--check-only" ]]; then exit 0; fi
if [[ $# -gt 0 ]]; then echo 'Usage: bash run_train_bgaug.sh [--check-only]' >&2; exit 2; fi

CHECKPOINT_DIR="/root/data/xxy/openpi/checkpoints/$CONFIG/$EXP"
if [[ -d "$CHECKPOINT_DIR" ]] && [[ -n "$(ls -A "$CHECKPOINT_DIR")" ]]; then
    echo "Existing checkpoint directory: $CHECKPOINT_DIR; refusing overwrite. Resume explicitly if needed." >&2
    exit 1
fi
mkdir -p /root/data/xxy/openpi/logs
printf 'Training config=%s GPUs=%s; log=/root/data/xxy/openpi/logs/train_%s.log\n' "$CONFIG" "$BG_TRAIN_GPUS" "$CONFIG"
BG_UV_BIN="$(command -v uv || true)"
if [[ -z "$BG_UV_BIN" ]]; then BG_UV_BIN=/root/.local/bin/uv; fi
if [[ ! -x "$BG_UV_BIN" ]]; then echo "uv executable not found: $BG_UV_BIN" >&2; exit 1; fi
CUDA_VISIBLE_DEVICES="$BG_TRAIN_GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    "$BG_UV_BIN" run scripts/train.py "$CONFIG" \
    --checkpoint-base-dir=/root/data/xxy/openpi/checkpoints \
    --exp-name="$EXP" \
    > "/root/data/xxy/openpi/logs/train_$CONFIG.log" 2>&1
