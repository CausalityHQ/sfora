#!/usr/bin/env bash
# Prospective budget amendment (orchestrator 1791993379765870637-4031287) for the ONE full-scale campaign:
# cap <= 8 DGX GPU-hours from driver launch 2026-10-14T15:44:23Z -> deadline 2026-10-14T23:44:23Z.
# Watches ONLY the original driver (pid 1644374 + its exact start time). At the cap, stops the active
# sfora-native256-full-* unit so the driver records its own HOLD (no retry). Exits when the driver is terminal.
set -u
DRIVER=1644374
START='Wed Oct 14 17:44:23 2026'
DEADLINE=$(date -u -d '2026-10-14T23:44:23Z' +%s)
original() { [ "$(ps -p "$DRIVER" -o lstart= 2>/dev/null | sed 's/^ *//')" = "$START" ] && ps -p "$DRIVER" -o args= | grep -q 'run_siglip2_substrate_full.sh'; }
echo "GUARD $(date -u +%FT%TZ) BEGIN driver=$DRIVER start='$START' deadline=2026-10-14T23:44:23Z"
original || { echo "GUARD $(date -u +%FT%TZ) driver not original/live at guard start; exit"; exit 1; }
while original; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    echo "GUARD $(date -u +%FT%TZ) BUDGET_CAP_REACHED (8 DGX GPU-h); stopping active campaign unit(s)"
    for unit in $(systemctl --user list-units 'sfora-native256-full-*' --state=active --no-legend --plain | awk '{print $1}'); do
      echo "GUARD stop $unit"; systemctl --user stop "$unit"; echo "GUARD stop rc=$?"
    done
    exit 0
  fi
  sleep 30
done
echo "GUARD $(date -u +%FT%TZ) driver terminal before cap; exit"
