#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
cd /home/riomus/runs/sfora-inshop-description-alignment-v1
export PYTHONPATH="$PWD/src:$PWD:/home/riomus/runs/sfora-inshop-sop-warmstart-smoke-v1"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
exec timeout 120s /home/riomus/group-learning/.venv/bin/python probe_inshop_description_alignment.py \
  --partition '/home/riomus/datasets/In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition.txt' \
  --metadata '/home/riomus/datasets/In-shop Clothes Retrieval Benchmark/Anno/list_description_inshop.json' \
  --eligibility "$PWD/inshop-description-eligibility-v2.json" \
  --feature-cache /home/riomus/runs/sfora-inshop-siglip2-train-features-v1/train_features.npy \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --output "$PWD/receipt.json"
