#!/usr/bin/env bash
set -euo pipefail

export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:"$PATH"
export PYTHONPATH=/home/riomus/sfora-siglip2-train-matched-v10-55040a7/src:/home/riomus/sfora-siglip2-train-matched-v10-55040a7/scripts

run=/home/riomus/runs/sfora-siglip2-arcface-1000-v10-55040a7
exec /home/riomus/group-learning/.venv/bin/python \
  /home/riomus/sfora-siglip2-live-trained-v1/diagnose_sop_siglip2_fit_errors.py \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --source-features /home/riomus/runs/sfora-siglip2-train-28e19203/train_features.npy \
  --training-receipt "$run/receipt.json" \
  --expected-training-receipt-sha256 3ff998f70900ea24c4c6c33d1cee866ca8650b629b799d58d060b2063624ecca \
  --training-checkpoint "$run/checkpoint.pt" \
  --train-embeddings "$run/train_embeddings.npy" \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --output "$run/fit_error_census_v1.json"
