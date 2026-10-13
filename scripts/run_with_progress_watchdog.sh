#!/usr/bin/env bash
# Stuck-process guard instead of a wall-time gate. CMD is stopped only when NOTHING shows progress for
# STALL_SECONDS (default 600): LOG size and the output of LIVENESS_CMD (e.g. the unit's cgroup cpu.stat,
# so quiet compile/export/load phases that still use CPU count as alive). BACKSTOP_SECONDS (default 3600)
# is a pure hang backstop. On a stall STOP_CMD (e.g. "systemctl --user stop UNIT") runs exactly once, then CMD gets
# GRACE_SECONDS (default 60) to exit by itself so its original exit/summary is collected before any signal.
# Usage: run_with_progress_watchdog.sh LOG -- CMD [ARGS...]
# Exit: CMD's own status, or 124 when stopped (always an invalid run; any '^watchdog:' log line invalidates it).
# Liveness is any change, so CPU-spinning hangs are caught only by the backstop; refuses an existing LOG.
# LIVENESS_CMD runs via bash with WATCHDOG_PID set to CMD's pid (= its process group).
set -u
log=$1; shift; [ "$1" = -- ] && shift
stall=${STALL_SECONDS:-600}; backstop=${BACKSTOP_SECONDS:-3600}; grace=${GRACE_SECONDS:-60}; poll=${POLL_SECONDS:-5}
[ -e "$log" ] && { echo "watchdog: refusing existing log $log" >&2; exit 2; }
setsid "$@" >>"$log" 2>&1 &
pid=$!; export WATCHDOG_PID=$pid
start=$SECONDS; last=; last_change=$SECONDS
token() { stat -c %s "$log" 2>/dev/null; [ -n "${LIVENESS_CMD:-}" ] && timeout 30 bash -c "$LIVENESS_CMD" 2>/dev/null; }
while kill -0 "$pid" 2>/dev/null; do
  now=$(token)
  [ "$now" != "$last" ] && { last=$now; last_change=$SECONDS; }
  if (( SECONDS - last_change >= stall || SECONDS - start >= backstop )); then
    reason=stall; (( SECONDS - start >= backstop )) && reason=backstop
    echo "watchdog: $reason (no progress $((SECONDS - last_change))s, stall ${stall}s, backstop ${backstop}s); stopping $pid" >>"$log"
    [ -n "${STOP_CMD:-}" ] && { timeout 120 bash -c "$STOP_CMD" >>"$log" 2>&1; echo "watchdog: stop rc=$?" >>"$log"; }
    for _ in $(seq "$grace"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -TERM -- "-$pid" 2>/dev/null; sleep 2; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null; exit 124
  fi
  sleep "$poll"
done
wait "$pid"
