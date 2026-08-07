#!/usr/bin/env bash
# Pass 119 Gate 4: corrected In-Shop ECT-R and preregistered controls.
set -u

PROJECT="${PROJECT:-/home/riomus/group-learning}"
ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop_official_standard}"
REPORT_ROOT="${PROJECT}/reports/generated"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/pass119-ectr}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT}" || exit 2

run() {
  local method="$1"
  local out="${REPORT_ROOT}/pass119_inshop.${method}.seed0.json"
  local log="${LOG_ROOT}/${method}.seed0.log"
  if [[ -s "${out}" ]]; then
    echo "SKIP ${method} (artifact exists)"
    return 0
  fi
  echo "START ${method}"
  .venv/bin/sfora image-end-to-end \
    --dataset-name inshop --dataset-root "${ROOT}" \
    --objectives proxy_anchor --recipe "${method}" --seed 0 \
    --num-workers 8 --output "${out}" >"${log}" 2>&1
  echo "DONE ${method} status=$?"
}

# A0 is the exact paired reference; the five derived arms share its recipe.
run auto
run ectr_soft
run ectr_random
run ectr
run ectr_plateau
run ectr_area
