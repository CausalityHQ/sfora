#!/usr/bin/env bash
set -euo pipefail
base=/home/riomus/runs/sfora-inshop-spatial-parts-v1
export PYTHONPATH="$base/src:$base"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
exec flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /home/riomus/group-learning/.venv/bin/python "$base/probe_inshop_spatial_parts.py" \
  --dataset-root '/home/riomus/datasets/In-shop Clothes Retrieval Benchmark' \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --checkpoint /home/riomus/runs/sfora-inshop-valid-anchor-confirm-freeze_emb-179026-v1/checkpoint.pt \
  --misses "$base/misses.json" \
  --output "$base/receipt.json"
