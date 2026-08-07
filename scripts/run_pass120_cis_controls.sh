#!/usr/bin/env bash
# Pass 120 Gate 4: CIS and its preregistered same-path controls.
set -u
PROJECT="${PROJECT:-/home/riomus/emafactorial-cis-active}"
ROOT="${INSHOP_ROOT:-/home/riomus/datasets/inshop_official_standard}"
REPORT_ROOT="${REPORT_ROOT:-${PROJECT}/reports/generated/pass120-cis}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/pass120-cis}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT}" || exit 2

run() {
  local method="$1"
  local out="${REPORT_ROOT}/pass120_inshop.${method}.seed0.json"
  local log="${LOG_ROOT}/${method}.seed0.log"
  if [[ -s "${out}" ]]; then
    echo "SKIP ${method} (artifact exists)"
    return 0
  fi
  echo "START ${method}"
  PYTHONPATH="${PROJECT}/src" .venv/bin/python -m sfora.cli image-end-to-end \
    --dataset-name inshop --dataset-root "${ROOT}" \
    --objectives proxy_anchor --recipe "${method}" --seed 0 \
    --num-workers 8 --output "${out}" >"${log}" 2>&1
  echo "DONE ${method} status=$?"
}

run pa_coalition_single
run pa_coalition_dropout
run pa_coalition
