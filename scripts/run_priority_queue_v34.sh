#!/usr/bin/env bash
# Candidate 52 Gate 4: preregistered TIRD In-Shop seed-0 screen only.
#
# Stop after this arm.  Raw R@1 below 0.9085 kills the candidate; no extra seed,
# ablation, or second dataset may be launched before protocol judgement.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v34 start: TIRD In-Shop seed-0 protocol screen only"
run_arm inshop tird proxy_anchor tird 0
log "priority queue v34 complete: stop for registered TIRD judgement"
