#!/usr/bin/env bash
set -euo pipefail

# Pass 113: one preregistered corrected In-Shop BEP screen.  The paired
# Proxy Anchor reference is already frozen in the current corpus; this script
# deliberately uses the same sampler, seed, and deployment recipe.
project=${PROJECT:-/home/riomus/group-learning}
dataset_root=${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}
seed=${SEED:-0}
cd "$project"
mkdir -p logs/pass113_bep reports/generated reports/checkpoints
exec env PYTHONPATH=src .venv/bin/sfora image-end-to-end \
  --dataset-name inshop \
  --dataset-root "$dataset_root" \
  --objectives proxy_anchor \
  --recipe pa_bep \
  --num-workers 8 \
  --seed "$seed" \
  --save-model-path "reports/checkpoints/pass113_bep_seed${seed}.pt" \
  --output "reports/generated/pass113_bep_seed${seed}.json"
