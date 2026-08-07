#!/usr/bin/env bash
set -euo pipefail
project=${PROJECT:-/home/riomus/group-learning}
pid=${RCC_PID:?RCC_PID is required}
cd "$project"
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
if [[ ! -s reports/generated/pass111_rcc_memory_seed0.json || ! -s reports/checkpoints/pass111_rcc_memory_seed0.pt ]]; then
  echo "RCC exited without report and checkpoint" >&2
  exit 2
fi
PYTHONPATH=src .venv/bin/python scripts/export_final_inshop_embeddings.py \
  --checkpoint reports/checkpoints/pass111_rcc_memory_seed0.pt \
  --report reports/generated/pass111_rcc_memory_seed0.json \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --query-output reports/generated/pass111_rcc_memory_seed0_query.npz \
  --gallery-output reports/generated/pass111_rcc_memory_seed0_gallery.npz \
  --retrieval-output reports/generated/pass111_rcc_memory_seed0_retrieval.json \
  --num-workers 8 >logs/pass111_rcc/rcc_export.log 2>&1
