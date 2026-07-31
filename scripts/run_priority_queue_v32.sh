#!/usr/bin/env bash
# Contaminated-decision confirmation: RSPG In-Shop seeds 0 and 1 only.
# Both are required before interpretation. No ablations or second dataset.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v32 start: contaminated RSPG In-Shop seeds 0,1 only"
run_arm inshop rspg proxy_anchor rspg 0
run_arm inshop rspg proxy_anchor rspg 1
log "priority queue v32 complete: stop for two-seed RSPG judgement"
