#!/usr/bin/env bash
set -euo pipefail

# Durable controller for Pass 111. It never starts RCC until the matched
# baseline report and checkpoint exist, so the arms cannot overlap or diverge
# in sampler/seed configuration.
project=${PROJECT:-/home/riomus/group-learning}
cd "$project"
baseline_pid=${BASELINE_PID:?BASELINE_PID is required}
log_root=${LOG_ROOT:-logs/pass111_rcc}
mkdir -p "$log_root"
while kill -0 "$baseline_pid" 2>/dev/null; do
  sleep 30
done
if [[ ! -s reports/generated/pass111_rcc_pa_full_seed0.json || ! -s reports/checkpoints/pass111_rcc_pa_full_seed0.pt ]]; then
  echo "baseline exited without both report and checkpoint" >&2
  exit 2
fi

PYTHONPATH=src .venv/bin/python scripts/export_final_inshop_embeddings.py \
  --checkpoint reports/checkpoints/pass111_rcc_pa_full_seed0.pt \
  --report reports/generated/pass111_rcc_pa_full_seed0.json \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --query-output reports/generated/pass111_rcc_pa_full_seed0_query.npz \
  --gallery-output reports/generated/pass111_rcc_pa_full_seed0_gallery.npz \
  --retrieval-output reports/generated/pass111_rcc_pa_full_seed0_retrieval.json \
  --num-workers 8 >>"$log_root/controller.log" 2>&1

exec env PYTHONPATH=src .venv/bin/sfora image-end-to-end \
  --dataset-name inshop \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --objectives proxy_anchor \
  --recipe pa_rcc \
  --num-workers 8 \
  --seed 0 \
  --save-model-path reports/checkpoints/pass111_rcc_memory_seed0.pt \
  --output reports/generated/pass111_rcc_memory_seed0.json \
  >"$log_root/rcc_seed0.log" 2>&1
