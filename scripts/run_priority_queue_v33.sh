#!/usr/bin/env bash
# Conditional RSPG ablation continuation. The two full-method runs were admitted
# through a contaminated diagnostic path, so both must clear the preregistered
# raw threshold before any control receives GPU time.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
V32_PID="${V32_PID:-2810470}"
THRESHOLD="${THRESHOLD:-0.9085}"

cd "${PROJECT}" || exit 2
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v33 armed: wait for v32 PID ${V32_PID}"
while kill -0 "${V32_PID}" 2>/dev/null; do sleep 60; done

seed0="$(arm_best inshop rspg proxy_anchor rspg 0)"
seed1="$(arm_best inshop rspg proxy_anchor rspg 1)"
log "RSPG contaminated-path gate: seed0=${seed0} seed1=${seed1} threshold=${THRESHOLD}"

if [[ "${seed0}" == "-" || "${seed1}" == "-" ]]; then
  log "RSPG gate STOP: one or both completed artifacts are unavailable"
  exit 1
fi
if ! .venv/bin/python - "${seed0}" "${seed1}" "${THRESHOLD}" <<'PY'
import sys
seed0, seed1, threshold = map(float, sys.argv[1:])
raise SystemExit(0 if seed0 >= threshold and seed1 >= threshold else 1)
PY
then
  log "RSPG gate DEAD: both seeds did not strictly clear ${THRESHOLD}; no ablations"
  exit 0
fi

log "RSPG gate LIVE: both seeds strictly clear ${THRESHOLD}; run three mandatory controls"
run_arm inshop rspg_soft_js proxy_anchor rspg_soft_js 0
run_arm inshop rspg_distance_gate proxy_anchor rspg_distance_gate 0
run_arm inshop rspg_instance_gate proxy_anchor rspg_instance_gate 0
log "priority queue v33 complete: stop for raw and selection-corrected four-arm judgement"
