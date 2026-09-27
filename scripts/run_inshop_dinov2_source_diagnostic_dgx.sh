#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-dinov2-source-v1"
test "$(sha256sum "$WORK/diagnose_inshop_dinov2_source.py" | cut -d' ' -f1)" = 60c990142b55992aaf5b383ae3424a9cc61e5b2f35afdbac6d0cd22f381934da
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
export PYTHONPATH="$WORK:/home/riomus/sfora-siglip2-deployed-batch-v1/src"
/home/riomus/group-learning/.venv/bin/python "$WORK/diagnose_inshop_dinov2_source.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --run-dir "$WORK/output" \
  --output "$WORK/raw-diagnostic.json"
echo "DONE receipt_sha256=$(sha256sum "$WORK/raw-diagnostic.json" | cut -d' ' -f1)"
