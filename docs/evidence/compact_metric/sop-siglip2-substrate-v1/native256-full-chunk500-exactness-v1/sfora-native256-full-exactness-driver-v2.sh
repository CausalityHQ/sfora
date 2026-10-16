#!/usr/bin/env bash
# One-unit chunk500 exactness driver: Large v2 0->500 only; no So400, no score, no retry.
set -euo pipefail
cd /home/riomus/runs/sfora-native256-full-v2
TOOL=$PWD/stage_siglip2_substrate_full_v2.py
say() { echo "EXACT $(date -u +%FT%TZ) $*"; }
step() { python3 -B "$TOOL" "$@" || { say "HOLD TOOL_FAIL $*"; exit 1; }; }
step lifetime
baseline=$(python3 -B "$TOOL" environment) || { say "HOLD environment unreadable"; exit 1; }
say "BEGIN pid=$$ unit=train-large-500 environment=$baseline"
cmd=$(python3 -B "$TOOL" launch train large 500) || { say "HOLD LAUNCH_FAIL"; exit 1; }
say "START train large 500"
if bash -c "$cmd"; then rc=0; else rc=$?; fi
say "UNIT_RC rc=$rc"
(( rc == 0 )) || { say "HOLD after unit"; exit "$rc"; }
step admit train large 500
say "DONE"
