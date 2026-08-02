#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
CARS_PATTERN="image_end_to_end.*cars.*recall_at_k_surrogate"

cd "${PROJECT}"
mkdir -p "${LOG_ROOT}" reports/emb reports/generated reports/checkpoints

while pgrep -f "${CARS_PATTERN}" >/dev/null; do
  sleep 60
done

for seed in 1 2; do
  pack="reports/emb/inshop_pa_epoch10_operating_seed${seed}.train.npz"
  report="reports/generated/inshop_pa_epoch10_operating_seed${seed}.json"
  checkpoint="reports/checkpoints/inshop_pa_epoch10_operating_seed${seed}.pt"
  log="${LOG_ROOT}/inshop.pa.epoch10.seed${seed}.log"
  if [[ ! -s "${pack}" ]]; then
    .venv/bin/sfora image-end-to-end \
      --dataset-name inshop \
      --objectives proxy_anchor \
      --recipe auto \
      --num-workers 8 \
      --train-epochs 10 \
      --eval-test-interval-epochs 0 \
      --save-train-embeddings "${pack}" \
      --save-model-path "${checkpoint}" \
      --seed "${seed}" \
      --output "${report}" \
      >"${log}" 2>&1
  fi
  .venv/bin/python scripts/measure_spectral_class_connectivity.py "${pack}" \
    | tee "reports/generated/inshop_fragmentation_epoch10_seed${seed}.json"
  sha256sum "${pack}" "${report}" "${checkpoint}" \
    | tee "reports/generated/inshop_fragmentation_epoch10_seed${seed}.sha256"
done
