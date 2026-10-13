#!/usr/bin/env bash
# Smallest checks for run_with_progress_watchdog.sh: alive-but-quiet, real-dead, single stop, exit collection.
set -u
W=$(dirname "$0")/run_with_progress_watchdog.sh
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
export STALL_SECONDS=3 POLL_SECONDS=1 GRACE_SECONDS=3
fail() { echo "FAIL: $*"; exit 1; }

# 1. Alive but quiet: no output for 8s (> stall) while burning CPU; CPU-time liveness keeps it alive.
LIVENESS_CMD='ps -o times= -g "$WATCHDOG_PID"' "$W" "$d/quiet.log" -- bash -c 'end=$((SECONDS+8)); while ((SECONDS<end)); do :; done; exit 0'
rc=$?; [ $rc -eq 0 ] || fail "quiet-but-alive killed (rc $rc)"
# Same child without a liveness signal is (correctly) treated as stalled: proves the signal matters.
"$W" "$d/quiet2.log" -- bash -c 'end=$((SECONDS+8)); while ((SECONDS<end)); do :; done'
[ $? -eq 124 ] || fail "log-only mode did not stop silent child"

# 2. Really dead (blocked, no CPU, no output): stopped once, rc 124, STOP_CMD exactly once, no orphan.
LIVENESS_CMD='ps -o times= -g "$WATCHDOG_PID"' STOP_CMD="echo stop >> $d/stops" "$W" "$d/dead.log" -- sleep 317
rc=$?; [ $rc -eq 124 ] || fail "dead child rc $rc"
[ "$(wc -l < "$d/stops")" -eq 1 ] || fail "STOP_CMD ran $(wc -l < "$d/stops") times"
[ "$(grep -c '^watchdog:' "$d/dead.log")" -eq 1 ] || fail "duplicate stop message"
ps -eo args | grep -qx 'sleep 317' && fail "orphan left"

# 3. Exit collection after STOP_CMD: a child that exits by itself within grace keeps its summary in the log.
STOP_CMD="touch $d/stopnow" "$W" "$d/grace.log" -- bash -c "while [ ! -e $d/stopnow ]; do sleep 0.2; done; echo 'Finished with result: timeout'; exit 7"
rc=$?; [ $rc -eq 124 ] || fail "grace rc $rc"
grep -q '^Finished with result: timeout$' "$d/grace.log" || fail "original summary not collected"

# 4. Normal progress and own exit status pass through untouched.
"$W" "$d/ok.log" -- bash -c 'for i in 1 2 3 4 5 6; do echo $i; sleep 1; done; exit 3'
[ $? -eq 3 ] || fail "exit status not propagated"
grep -q '^watchdog:' "$d/ok.log" && fail "healthy run stopped"
echo "PASS: quiet-alive kept, log-only stops silent, dead stopped once with no orphan, summary collected, exit status propagated"
