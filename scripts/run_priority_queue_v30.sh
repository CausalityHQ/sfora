#!/usr/bin/env bash
# In-Shop confirmation of BN-correct weight averaging.
#
# Run only seeds 1 and 2, then stop. Do not start dual EMA, Cars, or the
# averaging-momentum sweep until the three-seed averaging result is judged.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import v24's digest validation and runner without executing its queue body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v30 start: In-Shop BN-correct averaging confirmation only"

run_arm inshop pa_ema_avg_bnfix proxy_anchor pa_ema_avg_bnfix 1
run_arm inshop pa_ema_avg_bnfix proxy_anchor pa_ema_avg_bnfix 2

log "priority queue v30 complete: stop for three-seed averaging judgement"
