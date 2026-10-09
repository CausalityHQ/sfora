#!/bin/bash
set -u
set -o noclobber
printf '%s  %s\n' c11363abb8719207ddf75b5ed67e9728f9849395ea3faca57213491324fbac70 /home/riomus/runs/sfora-asymmetric-exit-residency-observer-source-v2/observe_asymmetric_exit_residency.py | sha256sum -c || exit 88
/usr/bin/prlimit --as=268435456 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -I -S -B /home/riomus/runs/sfora-asymmetric-exit-residency-observer-source-v2/observe_asymmetric_exit_residency.py --cgroup /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/sfora-connected-asymmetric-cache-bounded-exit-179061-v1.service --log /home/riomus/runs/sfora-connected-asymmetric-cache-source-v3/original.log --output /home/riomus/runs/sfora-asymmetric-exit-residency-observer-source-v2/samples.jsonl > /home/riomus/runs/sfora-asymmetric-exit-residency-observer-source-v2/original.log 2>&1 &
sfora_observer_pid=$!
finish_observer() {
    sfora_controller_status=$?
    trap - EXIT
    set +e
    if kill -0 "$sfora_observer_pid" 2>/dev/null; then kill -TERM "$sfora_observer_pid"; fi
    wait "$sfora_observer_pid"
    sfora_observer_status=$?
    printf '%s\n' "$sfora_observer_status" > /home/riomus/runs/sfora-asymmetric-exit-residency-observer-source-v2/terminal-status.txt
    exit "$sfora_controller_status"
}
trap finish_observer EXIT
/bin/bash /home/riomus/runs/sfora-connected-asymmetric-cache-source-v3/launch.sh > /home/riomus/runs/sfora-connected-asymmetric-cache-source-v3/original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-connected-asymmetric-cache-source-v3/terminal-status.txt
exit "$sfora_launch_status"
