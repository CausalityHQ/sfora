#!/usr/bin/env bash
# Frozen serial continuation after seed 179023 bank and float have completed.
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=f6af548085c2c7578d1dfe059bd3348675e6a68bc668e595346a6748182ddb83
PREFLIGHT_SHA=3102f89577583bfda526865d5a34b1712df0787c376e6e699cc8ff83333d814b
COST_SHA=6fb8c5acb616e97cf7d739641a01ed87daa7a67fbe6374ac43c66cb1ed92d12c
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

test "$(sha256sum "$RUNS/train_inshop_siglip2_compact.py" | cut -d' ' -f1)" = "$TRAINER_SHA"
test "$(sha256sum "$RUNS/inshop-coverage-preflight-v1.json" | cut -d' ' -f1)" = "$PREFLIGHT_SHA"
test "$(sha256sum "$RUNS/inshop-member-bank-cost-v2.json" | cut -d' ' -f1)" = "$COST_SHA"
test -f "$RUNS/sfora-inshop-siglip2-bank-179023-v1/receipt.json"
test -f "$RUNS/sfora-inshop-siglip2-float-179023-v1/receipt.json"
for seed in 179024 179025; do
  for arm in bank float; do
    test ! -e "$RUNS/sfora-inshop-siglip2-$arm-$seed-v1"
  done
done
for seed in 179024 179025; do
  for arm in bank float; do
    output="$RUNS/sfora-inshop-siglip2-$arm-$seed-v1"
    test ! -e "$output"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "START arm=$arm seed=$seed" >&2
    "$PYTHON" "$RUNS/train_inshop_siglip2_compact.py" \
      --dataset-root /home/riomus/datasets/inshop_official_standard \
      --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
      --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
      --preflight "$RUNS/inshop-coverage-preflight-v1.json" \
      --expected-preflight-sha256 "$PREFLIGHT_SHA" \
      --bank-cost "$RUNS/inshop-member-bank-cost-v2.json" \
      --expected-bank-cost-sha256 "$COST_SHA" \
      --output-dir "$output" --arm "$arm" --seed "$seed" --updates 1000
    test -f "$output/receipt.json"
    echo "DONE arm=$arm seed=$seed receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
  done
done
