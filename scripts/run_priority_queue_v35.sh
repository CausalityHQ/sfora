#!/usr/bin/env bash
# Candidate 136 Gate 4: matched IPC=4 control and spectral-connectivity arm.
# The control must run first because balanced sampling is a material intervention.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v35 start: candidate 136 In-Shop matched screen"
run_arm inshop pa_ipc4 proxy_anchor pa_ipc4 0
run_arm inshop pa_fiedler proxy_anchor pa_fiedler 0
log "priority queue v35 complete: stop for registered candidate-136 judgement"
