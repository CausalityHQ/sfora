#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-resolution-probe-224-179026-v1"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
script="$WORK/probe_inshop_siglip2_resolution.py"
test "$(sha256sum "$script" | cut -d' ' -f1)" = 79fb303fceb8d3a72b18326c99a53f8ba20ca5c6e8a277364a2964721408ab81
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
export PYTHONPATH="$WORK:$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
/home/riomus/group-learning/.venv/bin/python "$script" \
  --candidate-size 224 \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --checkpoint "$RUNS/sfora-inshop-valid-anchor-confirm-freeze_emb-179026-v1/checkpoint.pt" \
  --baseline-receipt "$RUNS/sfora-inshop-valid-anchor-confirm-freeze_emb-179026-v1/receipt.json" \
  --preflight "$RUNS/sfora-valid-anchor-confirmation-179026-27/preflight-179026.json" \
  --output "$WORK/report.json"
echo "DONE receipt_sha256=$(sha256sum "$WORK/report.json" | cut -d' ' -f1)"
