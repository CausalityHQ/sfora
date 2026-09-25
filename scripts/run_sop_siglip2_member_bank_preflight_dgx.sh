#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-train-matched-v10-55040a7
run=/home/riomus/runs/sfora-siglip2-arcface-1000-v10-55040a7
export PYTHONPATH="$root/src:$root/scripts"

exec /home/riomus/group-learning/.venv/bin/python \
  "$run/probe_sop_siglip2_member_bank.py" \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --source-features /home/riomus/runs/sfora-siglip2-train-28e19203/train_features.npy \
  --training-receipt "$run/receipt.json" \
  --training-checkpoint "$run/checkpoint.pt" \
  --train-embeddings "$run/train_embeddings.npy" \
  --fit-census "$run/fit_error_census_v1.json" \
  --output "$run/member_bank_preflight_v1.json" \
  --expected-training-receipt-sha256 3ff998f70900ea24c4c6c33d1cee866ca8650b629b799d58d060b2063624ecca \
  --expected-fit-census-sha256 4947bb6ae8e70153018e07ed05a26777ab0e8286a9c640cdab36f219ffb64f79
