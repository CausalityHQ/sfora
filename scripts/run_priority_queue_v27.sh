#!/usr/bin/env bash
# Priority queue v27: user-authoritative ordering.
#
# Do not overwrite a running queue script. This wrapper composes the already
# deployed, digest-cached queues in the required order:
#
#   1. finish the momentum-matched EMA 2x2;
#   2. run paired corrected Cars196;
#   3. run the pre-registered averaging momentum sweep at three seeds per cell.
#
# v24 performs (1) and (2), skipping completed current-digest artifacts. v25 then
# performs (3). Predictions and falsification conditions were frozen before any
# sweep run in docs/ema_averaging_momentum_preregistration.md.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"

cd "${PROJECT}" || exit 2

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${CONTROLLER_LOG}"
}

log "priority queue v27 start: finish 2x2 -> Cars196 -> momentum sweep"

if ! bash scripts/run_priority_queue_v24.sh; then
  log "FAIL priority queue v27: v24 phase returned nonzero; sweep not started"
  exit 1
fi

if ! bash scripts/run_priority_queue_v25.sh; then
  log "FAIL priority queue v27: v25 sweep phase returned nonzero"
  exit 1
fi

log "priority queue v27 complete"
