#!/usr/bin/env bash
set -euo pipefail
run=/home/riomus/runs/sfora-inshop-moda-source-v1
export PYTHONPATH="$run/src:$run/scripts:/home/riomus/runs/moda-openclip-3.3.0"
export HF_HUB_OFFLINE=1
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$run/scripts/probe_inshop_moda_source.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/runs/moda-fashion-vision-fp16-9f3358c2 \
  --baseline /home/riomus/runs/sfora-inshop-cross-depth-head-v1/result/receipt.json \
  --output-dir "$run/result"
