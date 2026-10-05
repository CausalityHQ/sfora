#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-identity-diversity-train-source-v3
printf '%s  %s\n' 53409094fbbffd51eb8d7f38b47e6248d60b9e6346ed1ec184cf5a1574639922 residency-sampler-v1.py b91ef430314109c1640fca31d67d923ef5dd0c6d58e92c93d81d0c1f908ba018 cpu-residency-diag-v1-launch.sh | sha256sum -c
test ! -e cpu-residency-diag-v1.log
test ! -e cpu-residency-diag-v1-memory.jsonl
test ! -e /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-so400-identity-diversity-cpu-residency-diag-v1.service
observer_pid=''
cleanup() { if [[ -n "$observer_pid" ]]; then kill -TERM "$observer_pid" 2>/dev/null || true; wait "$observer_pid" || true; fi; }
trap cleanup EXIT
/usr/bin/python3 /home/riomus/runs/sfora-so400-identity-diversity-train-source-v3/residency-sampler-v1.py --cgroup /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-so400-identity-diversity-cpu-residency-diag-v1.service --log /home/riomus/runs/sfora-so400-identity-diversity-train-source-v3/cpu-residency-diag-v1.log --output /home/riomus/runs/sfora-so400-identity-diversity-train-source-v3/cpu-residency-diag-v1-memory.jsonl > cpu-residency-diag-v1-observer.log 2>&1 &
observer_pid=$!
set +e
/bin/bash cpu-residency-diag-v1-launch.sh > cpu-residency-diag-v1.log 2>&1
native_status=$?
kill -TERM "$observer_pid" 2>/dev/null
wait "$observer_pid"
observer_status=$?
observer_pid=''
printf 'DIAG_CONTROLLER native_status=%s observer_status=%s\n' "$native_status" "$observer_status"
exit "$native_status"
