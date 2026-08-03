#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}"
REFERENCE="${REFERENCE:-reports/generated/inshop_corrected_pa_seed0.json}"
REFERENCE_FINAL="${REFERENCE_FINAL:-reports/generated/inshop_corrected_pa_seed0_final_retrieval.json}"
REFERENCE_GEOMETRY="${REFERENCE_GEOMETRY:-reports/generated/inshop_corrected_pa_seed0_training_geometry.json}"
REPORT="${REPORT:-reports/generated/inshop_corrected_pa_epoch10_seed0.json}"
CHECKPOINT="${CHECKPOINT:-reports/checkpoints/inshop_corrected_pa_epoch10_seed0.pt}"
PACK="${PACK:-reports/emb/inshop_corrected_pa_epoch10_seed0.train.npz}"
RESULT="${RESULT:-reports/generated/inshop_corrected_pa_epoch10_seed0_rspg_gate.txt}"
LOG="${LOG:-logs/inshop_corrected_pa_epoch10_seed0.log}"

cd "${PROJECT}"
while [[ ! -s "${REFERENCE}" || ! -s "${REFERENCE_FINAL}" || ! -s "${REFERENCE_GEOMETRY}" ]]; do
  sleep 60
done

.venv/bin/python - "${REFERENCE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
method = payload["methods"][next(iter(payload["methods"]))]
score = float(method["best_test_recall_at_1"])
if not 0.907 <= score <= 0.929:
    raise SystemExit(f"corrected reference outside registered interval: {score}")
PY

for path in "${REPORT}" "${CHECKPOINT}" "${PACK}" "${RESULT}"; do
  [[ ! -e "${path}" ]] || {
    echo "refusing pre-existing corrected RSPG gate artifact: ${path}" >&2
    exit 2
  }
done

PYTHONPATH=src .venv/bin/sfora image-end-to-end \
  --dataset-name inshop \
  --dataset-root "${DATASET_ROOT}" \
  --objectives proxy_anchor \
  --recipe auto \
  --num-workers 8 \
  --train-epochs 10 \
  --eval-test-interval-epochs 0 \
  --save-train-embeddings "${PACK}" \
  --save-model-path "${CHECKPOINT}" \
  --seed 0 \
  --output "${REPORT}" \
  >"${LOG}" 2>&1

temporary="${RESULT}.tmp"
set +e
PYTHONPATH=src .venv/bin/python scripts/diagnose_rspg_graph.py "${PACK}" \
  >"${temporary}" 2>&1
status=$?
set -e
printf 'RSPG_CPU_GATE_EXIT=%s\n' "${status}" >>"${temporary}"
mv "${temporary}" "${RESULT}"
exit "${status}"
