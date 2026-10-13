#!/usr/bin/env bash
# Held quality read for the ONE native256-v9 campaign (run on DGX from the held tools dir, under nohup setsid).
# Precondition: held root staged and native256-held-authority.json written from the four passed TRAIN100 units.
# Per endpoint in frozen ORDER: CPU proof -> standalone CPU terminal -> export; then wires -> score.
# Stops at the first failure or watchdog stop; never retries.
set -u
cd "$(dirname "$0")"
TOOL=$PWD/stage_native256_v9_held.py
say() { echo "HELD $(date -u +%FT%TZ) $*"; }
step() { python3 -B "$TOOL" "$@" || { say "TOOL_FAIL $*"; exit 1; }; }
unit() {
  local cmd rc
  cmd=$(python3 -B "$TOOL" launch "$@") || { say "LAUNCH_FAIL $*"; exit 1; }
  say "START $*"
  bash -c "$cmd"; rc=$?
  say "UNIT_RC $* rc=$rc"
  (( rc == 0 )) || { say "HOLD after $*"; exit "$rc"; }
}
say "BEGIN pid=$$"
for endpoint in "179032 large" "179032 so400" "179041 so400" "179041 large"; do
  unit cpu $endpoint
  step cpu-terminal $endpoint
  unit export $endpoint
done
step wires
unit score
say "DONE held cpu/export x4 + score rc=0"
