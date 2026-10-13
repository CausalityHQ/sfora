#!/usr/bin/env bash
# ONE native256-v9 paired So400m/Large TRAIN100 campaign (run on DGX from sfora-native256-v9-tools, under nohup setsid).
# Order: stage-checked root must exist; mechanics Large, So400m; then TRAIN in the exporter's frozen ORDER.
# Stops at the first failure or watchdog stop; never retries (no duplicate under the authorization).
set -u
cd "$(dirname "$0")"
TOOL=$PWD/stage_native256_v9.py
say() { echo "CAMPAIGN $(date -u +%FT%TZ) $*"; }
unit() {
  local cmd rc
  python3 -B "$TOOL" authority "$@" || { say "AUTHORITY_FAIL $*"; exit 1; }
  cmd=$(python3 -B "$TOOL" launch "$@") || { say "LAUNCH_FAIL $*"; exit 1; }
  say "START $*"
  bash -c "$cmd"; rc=$?
  say "UNIT_RC $* rc=$rc"
  (( rc == 0 )) || { say "HOLD after $*"; exit "$rc"; }
}
say "BEGIN pid=$$"
unit mechanics large
unit mechanics so400
unit train large 179032
unit train so400 179032
unit train so400 179041
unit train large 179041
say "DONE all six units rc=0"
