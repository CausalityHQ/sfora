#!/usr/bin/env bash
# Candidate 1 gate-4 screen: BN-correct In-Shop averaging control and dual EMA.
#
# Stop after the two screening arms. The result must be judged and committed
# under docs/search_protocol.md before any confirmation or second dataset.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"

cd "${PROJECT}" || exit 2

# Import v24's digest validation and runner without executing its queue body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v29 start: candidate 1 In-Shop gate-4 screen"

run_arm inshop pa_ema_avg_bnfix proxy_anchor pa_ema_avg_bnfix 0
run_arm inshop pa_dual_ema_bnfix proxy_anchor pa_dual_ema_bnfix 0

log "priority queue v29 complete: stop for protocol judgement"
