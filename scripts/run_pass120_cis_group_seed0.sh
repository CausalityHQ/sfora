#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/riomus/group-learning}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}"
REPORT_DIR="${REPORT_DIR:-reports/generated/pass120_cis}"
LOG_DIR="${LOG_DIR:-logs/pass120_cis}"
cd "$PROJECT_DIR"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

for method in pa_coalition_single pa_coalition_complementary pa_coalition_dropout pa_coalition; do
  report="$REPORT_DIR/pass120_inshop.${method}.seed0.json"
  checkpoint="reports/checkpoints/pass120_inshop.${method}.seed0.pt"
  [[ ! -e "$report" ]] || { echo "refusing pre-existing artifact: $report" >&2; exit 2; }
  [[ ! -e "$checkpoint" ]] || { echo "refusing pre-existing artifact: $checkpoint" >&2; exit 2; }
  PYTHONPATH=src .venv/bin/sfora image-end-to-end \
    --dataset-name inshop --dataset-root "$DATASET_ROOT" \
    --objectives proxy_anchor --recipe "$method" --num-workers 8 --seed 0 \
    --save-model-path "$checkpoint" --output "$report" \
    >"$LOG_DIR/${method}.seed0.log" 2>&1
done
