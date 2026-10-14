#!/usr/bin/env bash
# Parent starts this staged driver with nohup setsid. Fixed order, no retries.
# A So400 scorer failure after the Large official read is HOLD. No re-score,
# including a byte-identical re-score, is allowed without a new decision.
set -euo pipefail
cd "$(dirname "$0")"
TOOL=$PWD/stage_siglip2_substrate_full.py
say() { echo "FULL $(date -u +%FT%TZ) $*"; }
step() { python3 -B "$TOOL" "$@" || { say "HOLD TOOL_FAIL $*"; exit 1; }; }
unit() {
  local cmd rc current
  current=$(python3 -B "$TOOL" environment) || { say "HOLD environment unreadable before $*"; exit 1; }
  say "ENVIRONMENT before $* $current"
  [[ "$current" == "$baseline" ]] || { say "HOLD driver/kernel changed before $*"; exit 1; }
  cmd=$(python3 -B "$TOOL" launch "$@") || { say "HOLD LAUNCH_FAIL $*"; exit 1; }
  say "START $*"
  if bash -c "$cmd"; then rc=0; else rc=$?; fi
  say "UNIT_RC $* rc=$rc"
  (( rc == 0 )) || { say "HOLD after $*"; exit "$rc"; }
  step admit "$@"
}
step lifetime
baseline=$(python3 -B "$TOOL" environment) || { say "HOLD initial environment unreadable"; exit 1; }
say "BEGIN pid=$$ order=large-100..2000,so400-100..2000,large-score,so400-score environment=$baseline"
for arm in large so400; do
  for ((end=100; end<=2000; end+=100)); do
    unit train "$arm" "$end"
  done
done
unit score large
unit score so400
step decision
