#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:"$PATH"
export PYTHONPATH="$root/src:$root/scripts"

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/probe_sop_siglip2_unseen_gallery.py" \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --control-receipt "$run_base/sfora-siglip2-member-bank-control-1000-v1/receipt.json" \
  --control-embeddings "$run_base/sfora-siglip2-member-bank-control-1000-v1/train_embeddings.npy" \
  --bank-receipt "$run_base/sfora-siglip2-member-bank-treatment-1000-v1/receipt.json" \
  --bank-embeddings "$run_base/sfora-siglip2-member-bank-treatment-1000-v1/train_embeddings.npy" \
  --output "$run_base/sfora-siglip2-unseen-gallery-v1.json"
