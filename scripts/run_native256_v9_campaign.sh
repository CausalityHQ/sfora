#!/usr/bin/env bash
# ONE native256-v9 paired So400m/Large TRAIN100 campaign (run on DGX from sfora-native256-v9-tools, under nohup setsid).
# Order: stage-checked root must exist; mechanics Large, So400m; then TRAIN in the exporter's frozen ORDER.
# Then stages the held root from src/ and runs the held read (run_native256_v9_held.sh).
# Stops at the first failure or watchdog stop; never retries (no duplicate under the authorization).
# Attempt names come from SFORA_NATIVE256_ATTEMPT (e.g. r1); earlier attempts are never touched.
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
python3 -B "$PWD/stage_native256_v9_held.py" stage "$PWD/src" || { say "HELD_STAGE_FAIL"; exit 1; }
python3 -B "$PWD/stage_native256_v9_held.py" authority || { say "HELD_AUTHORITY_FAIL"; exit 1; }
bash "$PWD/run_native256_v9_held.sh"; rc=$?
say "HELD_RC rc=$rc"
exit "$rc"
