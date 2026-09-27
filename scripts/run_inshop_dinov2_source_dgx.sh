#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-dinov2-source-v1"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
test "$(sha256sum "$WORK/probe_inshop_dinov2_source.py" | cut -d' ' -f1)" = 9e1e7e0ba009fedc5e91d4f8221e208f7842bde00d2c0682987826c589425519
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
export PYTHONPATH="$WORK:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
/home/riomus/group-learning/.venv/bin/python "$WORK/probe_inshop_dinov2_source.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --output-dir "$WORK/output"
echo "DONE receipt_sha256=$(sha256sum "$WORK/output/receipt.json" | cut -d' ' -f1)"
