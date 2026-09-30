#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/bak:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_FILE="${CONFIG_FILE:-configs/libero_history_future_noise.yaml}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data/libero_dataset}"
CACHE_DIR="${CACHE_DIR:-$DATASET_DIR/uniwam_cache}"
WAN_PATH="${WAN_PATH:-$PROJECT_ROOT/pretrained_models}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
RUN_NAME="${RUN_NAME:-uniwam_libero_history_future_noise}"

[[ -f "$CONFIG_FILE" && -d "$DATASET_DIR" ]] || { echo "Missing LIBERO config or dataset" >&2; exit 1; }
(( NPROC_PER_NODE > 0 )) || { echo "No visible GPU" >&2; exit 1; }

DATASET_DIR="$DATASET_DIR" CACHE_DIR="$CACHE_DIR" WAN_PATH="$WAN_PATH" \
    bash scripts/libero/prepare_libero_cache.sh

exec torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" train/train.py \
    --deepspeed configs/zero2.json --config "$CONFIG_FILE" \
    --dataset_dir "$DATASET_DIR" --cache_dir "$CACHE_DIR" \
    --run_name "$RUN_NAME" --report_to tensorboard
