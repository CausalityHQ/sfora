#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-identity-diversity-train-source-v5
printf '%s  %s\n' 53409094fbbffd51eb8d7f38b47e6248d60b9e6346ed1ec184cf5a1574639922 residency-sampler-v1.py fbc6b5309ddafcb411336bc06b660eb27ec80aa225386c4690ac1a6c536b7bce cpu-v5-launch.sh | sha256sum -c
test ! -e cpu-v5.log
test ! -e cpu-v5-memory.jsonl
test ! -e /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-so400-identity-diversity-cpu-v5.service
observer_pid=''
cleanup() { if [[ -n "$observer_pid" ]]; then kill -TERM "$observer_pid" 2>/dev/null || true; wait "$observer_pid" || true; fi; }
trap cleanup EXIT
/usr/bin/python3 /home/riomus/runs/sfora-so400-identity-diversity-train-source-v5/residency-sampler-v1.py --cgroup /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-so400-identity-diversity-cpu-v5.service --log /home/riomus/runs/sfora-so400-identity-diversity-train-source-v5/cpu-v5.log --output /home/riomus/runs/sfora-so400-identity-diversity-train-source-v5/cpu-v5-memory.jsonl > cpu-v5-observer.log 2>&1 &
observer_pid=$!
set +e
/bin/bash cpu-v5-launch.sh > cpu-v5.log 2>&1
native_status=$?
kill -TERM "$observer_pid" 2>/dev/null
wait "$observer_pid"
observer_status=$?
observer_pid=''
printf 'DIAG_CONTROLLER native_status=%s observer_status=%s\n' "$native_status" "$observer_status"
exit "$native_status"
