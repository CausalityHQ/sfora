#!/usr/bin/env bash
# Stuck-process guard instead of a wall-time gate: kill CMD only when LOG stops growing
# for STALL_SECONDS (default 600) or the hang backstop BACKSTOP_SECONDS (default 3600) passes.
# Usage: run_with_progress_watchdog.sh LOG -- CMD [ARGS...]   (exit 124 = killed as stuck)
# STOP_CMD (optional) runs first on a stall, e.g. "systemctl --user stop UNIT" when CMD is systemd-run --wait.
set -u
log=$1; shift; [ "$1" = -- ] && shift
stall=${STALL_SECONDS:-600}; backstop=${BACKSTOP_SECONDS:-3600}
setsid "$@" >>"$log" 2>&1 &
pid=$!; start=$SECONDS; last_size=-1; last_change=$SECONDS
while kill -0 "$pid" 2>/dev/null; do
  size=$(stat -c %s "$log" 2>/dev/null || echo 0)
  [ "$size" != "$last_size" ] && { last_size=$size; last_change=$SECONDS; }
  if (( SECONDS - last_change >= stall || SECONDS - start >= backstop )); then
    echo "watchdog: stuck (no log progress ${stall}s or backstop ${backstop}s); killing $pid" >>"$log"
    [ -n "${STOP_CMD:-}" ] && bash -c "$STOP_CMD" >>"$log" 2>&1
    kill -TERM -- "-$pid" 2>/dev/null; sleep 10; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null; exit 124
  fi
  sleep "${POLL_SECONDS:-5}"
done
wait "$pid"
