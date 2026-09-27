#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-multidepth-head-v1"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
test "$(sha256sum "$WORK/probe_inshop_multidepth_head.py" | cut -d' ' -f1)" = cd097269ad6f99c5ca674b1c3e9654ea83a94cadde29191cb3f610b9f8926dce
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
export PYTHONPATH="$WORK:$CHECKOUT/src:$CHECKOUT/scripts"
/home/riomus/group-learning/.venv/bin/python "$WORK/probe_inshop_multidepth_head.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --short-cache "$RUNS/sfora-inshop-depth22-features-v1" \
  --full-cache "$RUNS/sfora-inshop-siglip2-train-features-v1" \
  --baseline "$WORK/baseline.json" \
  --output "$WORK/receipt.json"
echo "DONE receipt_sha256=$(sha256sum "$WORK/receipt.json" | cut -d' ' -f1)"
