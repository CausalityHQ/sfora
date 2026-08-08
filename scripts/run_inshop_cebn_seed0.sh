#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/riomus/group-learning}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}"
LOG="${LOG:-logs/pass183_cebn_seed0.log}"
REPORT="reports/generated/pass183_cebn_seed0.json"
CHECKPOINT="reports/checkpoints/pass183_cebn_seed0.pt"

cd "$PROJECT_DIR"
for path in "$REPORT" "$CHECKPOINT"; do
  [[ ! -e "$path" ]] || { echo "refusing pre-existing artifact: $path" >&2; exit 2; }
done
mkdir -p "$(dirname "$REPORT")" "$(dirname "$CHECKPOINT")" "$(dirname "$LOG")"

PYTHONPATH=src .venv/bin/sfora image-end-to-end \
  --dataset-name inshop \
  --dataset-root "$DATASET_ROOT" \
  --objectives proxy_anchor \
  --recipe pa_cebn \
  --num-workers 8 \
  --seed 0 \
  --save-model-path "$CHECKPOINT" \
  --output "$REPORT" >"$LOG" 2>&1
