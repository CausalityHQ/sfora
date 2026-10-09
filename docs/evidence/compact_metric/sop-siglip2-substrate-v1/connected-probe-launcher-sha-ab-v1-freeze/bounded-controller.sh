#!/bin/bash
set -u
set -o noclobber
/usr/bin/python3 -I -B -S /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/residency_sampler.py --cgroup /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-connected-probe-launcher-sha-bounded-v1.service --log /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-original.log --output /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-memory.jsonl > /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-observer.log 2>&1 &
observer_pid=$!
trap 'kill -TERM "$observer_pid" 2>/dev/null || true; wait "$observer_pid" || true' EXIT
/bin/bash /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-launch.sh > /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-original.log 2>&1
status=$?
printf '%s\n' "$status" > /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/bounded-terminal-status.txt
exit "$status"
