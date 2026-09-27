#!/usr/bin/env bash
set -euo pipefail
base=/home/riomus/runs/sfora-inshop-sop-product-prior-v1
export PYTHONPATH="$base/src:$base"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
exec flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /home/riomus/group-learning/.venv/bin/python "$base/probe_inshop_sop_product_prior.py" \
  --dataset-root '/home/riomus/datasets/In-shop Clothes Retrieval Benchmark' \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --checkpoint /home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1/checkpoint.pt \
  --checkpoint-receipt /home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1/receipt.json \
  --source-cache /home/riomus/runs/sfora-inshop-siglip2-train-features-v1/train_features.npy \
  --source-rank-receipt "$base/source-rank-receipt.json" \
  --output "$base/receipt.json"
