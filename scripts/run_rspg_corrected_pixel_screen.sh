#!/usr/bin/env bash
# Prospectively registered corrected-standard-pixel RSPG seeds 0 and 1.
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
DATASET_ROOT="${DATASET_ROOT:-/home/riomus/datasets/inshop_official_standard}"
LOG_ROOT="${LOG_ROOT:-logs/rspg_corrected_pixel_screen}"
THRESHOLD="${THRESHOLD:-0.9175}"

cd "${PROJECT}"
mkdir -p "${LOG_ROOT}" reports/generated reports/checkpoints reports/emb

for seed in 0 1; do
  stem="inshop_corrected_rspg_seed${seed}"
  report="reports/generated/${stem}.json"
  checkpoint="reports/checkpoints/${stem}.pt"
  final="reports/generated/${stem}_final_retrieval.json"
  query="reports/emb/${stem}_query_final.npz"
  gallery="reports/emb/${stem}_gallery_final.npz"
  log="${LOG_ROOT}/seed${seed}.log"

  for path in "${report}" "${checkpoint}" "${final}" "${query}" "${gallery}"; do
    [[ ! -e "${path}" ]] || {
      echo "refusing pre-existing corrected RSPG artifact: ${path}" >&2
      exit 2
    }
  done

  PYTHONPATH=src .venv/bin/sfora image-end-to-end \
    --dataset-name inshop \
    --dataset-root "${DATASET_ROOT}" \
    --objectives proxy_anchor \
    --recipe rspg \
    --num-workers 8 \
    --seed "${seed}" \
    --save-model-path "${checkpoint}" \
    --output "${report}" \
    >"${log}" 2>&1

  PYTHONPATH=src .venv/bin/python scripts/export_final_inshop_embeddings.py \
    --checkpoint "${checkpoint}" \
    --report "${report}" \
    --dataset-root "${DATASET_ROOT}" \
    --query-output "${query}" \
    --gallery-output "${gallery}" \
    --retrieval-output "${final}" \
    --num-workers 8 \
    >>"${log}" 2>&1

  PYTHONPATH=src .venv/bin/python - "${report}" "${final}" "${THRESHOLD}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
final_path = Path(sys.argv[2])
threshold = float(sys.argv[3])
report = json.loads(report_path.read_text(encoding="utf-8"))
method = report["methods"][next(iter(report["methods"]))]
final = json.loads(final_path.read_text(encoding="utf-8"))
raw = float(method["best_test_recall_at_1"])
frozen = float(final["independent_recall_at_1"])
print(
    f"RSPG_CORRECTED_RESULT seed={report['config']['seed']} "
    f"raw_best={raw:.8f} final={frozen:.8f} "
    f"clears={int(raw >= threshold)} threshold={threshold:.4f}"
)
PY
done
