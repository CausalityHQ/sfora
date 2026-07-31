#!/usr/bin/env bash
# Candidate 8 Gate 4: RSPG on In-Shop seed 0 only.
#
# The implementation aborts after epoch 10 if the preregistered training-graph
# diagnostic fails. Never substitute CUB and never queue EMA/momentum work.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v31 start: RSPG In-Shop seed-0 protocol screen only"
run_arm inshop rspg proxy_anchor rspg 0
log "priority queue v31 complete: stop for registered RSPG judgement"
