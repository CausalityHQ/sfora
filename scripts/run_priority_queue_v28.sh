#!/usr/bin/env bash
# Priority queue v28: finish CUB 2x2 -> In-Shop averaging -> Cars -> CUB sweep.
#
# The run implementation is imported from v24 only up to (but not including) its
# queue body. This preserves its digest checks, logging, and In-Shop dataset root
# handling without overwriting or executing the old queue order.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
LOG_ROOT="${LOG_ROOT:-/home/riomus/experiment-logs/reference-matrix}"
CONTROLLER_LOG="${LOG_ROOT}/controller.log"

cd "${PROJECT}" || exit 2

# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

log "priority queue v28 start: finish CUB 2x2 -> In-Shop averaging -> Cars196 -> sweep"

# Complete every momentum-matched 0.99 cell at three seeds. Current-digest
# artifacts are validated and skipped.
for seed in 0 1 2; do
  run_arm cub pa_distill_avg proxy_anchor pa_distill_avg "${seed}"
  run_arm cub pa_distill_fast proxy_anchor pa_distill_fast "${seed}"
done
for seed in 1 2; do
  run_arm cub pa_ema_avg_fast proxy_anchor pa_ema_avg_fast "${seed}"
done

# Trainable BatchNorm requires averaged buffers as well as averaged weights.
# Never substitute pa_ema_avg_fast on In-Shop.
run_arm inshop pa_ema_avg_bnfix proxy_anchor pa_ema_avg_bnfix 0

for seed in 0 1 2; do
  run_arm cars proxy_anchor proxy_anchor auto "${seed}"
  run_arm cars pa_distill proxy_anchor pa_distill "${seed}"
done

# Fixed three-seed sweep; predictions were frozen before either arm ran.
for seed in 0 1 2; do
  run_arm cub pa_ema_avg_m95 proxy_anchor pa_ema_avg_m95 "${seed}"
  run_arm cub pa_ema_avg_m90 proxy_anchor pa_ema_avg_m90 "${seed}"
done

log "priority queue v28 complete"
