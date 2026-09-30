#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/bak:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data/libero_dataset}"
CACHE_DIR="${CACHE_DIR:-$DATASET_DIR/uniwam_cache}"
WAN_PATH="${WAN_PATH:-$PROJECT_ROOT/pretrained_models}"

python -m data.libero.generate_action_stats --dataset-dir "$DATASET_DIR" --output "$CACHE_DIR/raw_osc_action_stats.json"
python -m data.libero.generate_language_action --dataset-dir "$DATASET_DIR" --cache-dir "$CACHE_DIR" --horizon 16 --lap-subdir language_actions_h16
python -m data.libero.generate_t5_embeddings --dataset-dir "$DATASET_DIR" --cache-dir "$CACHE_DIR" --wan-path "$WAN_PATH" --device cuda
