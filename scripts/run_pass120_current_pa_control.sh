#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/riomus/group-learning}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}"
REPORT="${REPORT:-reports/generated/pass120_cis/pass120_inshop.proxy_anchor.seed0.current.json}"
CHECKPOINT="${CHECKPOINT:-reports/checkpoints/pass120_inshop.proxy_anchor.seed0.current.pt}"
LOG="${LOG:-logs/pass120_cis/proxy_anchor.seed0.current.log}"

cd "$PROJECT_DIR"
for arm in \
  pa_coalition_single \
  pa_coalition_complementary \
  pa_coalition_dropout \
  pa_coalition; do
  required="reports/generated/pass120_cis/pass120_inshop.${arm}.seed0.json"
  [[ -s "$required" ]] || {
    echo "refusing current-code PA control: missing completed CIS artifact $required" >&2
    exit 3
  }
done

[[ ! -e "$REPORT" ]] || { echo "refusing pre-existing artifact: $REPORT" >&2; exit 2; }
[[ ! -e "$CHECKPOINT" ]] || {
  echo "refusing pre-existing artifact: $CHECKPOINT" >&2
  exit 2
}

PYTHONPATH=src .venv/bin/sfora image-end-to-end \
  --dataset-name inshop --dataset-root "$DATASET_ROOT" \
  --objectives proxy_anchor --recipe auto --num-workers 8 --seed 0 \
  --save-model-path "$CHECKPOINT" --output "$REPORT" \
  >"$LOG" 2>&1
