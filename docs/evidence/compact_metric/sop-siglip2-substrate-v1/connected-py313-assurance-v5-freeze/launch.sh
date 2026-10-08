#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-connected-py313-assurance-v5
printf '%s\n' 'd2bbe55f22477eb754a54d0e55fcebdba150eacad324a27aa64eecce77433d9c  inputs.tar.gz' '661962da5112945c19fc3edc0f49bd221a8dcaa96f6a64e3c7d2726b69607322  inventory.json' 'f092054d4dd1706b12d463c9d234a0e22536b185bde90fd62f3808c53c1941ab  run.py' | sha256sum -c
 test ! -e original.log
 test -z "$(systemctl --user list-units --state=running --no-legend | grep sfora || true)"
 test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
tar -xzf inputs.tar.gz
set +e
systemd-run --user --unit=sfora-connected-py313-assurance-v5 --wait --pipe --collect --property=RuntimeMaxSec=120 --property=MemoryMax=1073741824 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash -c '/usr/bin/prlimit --as=1073741824 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -B /home/riomus/runs/sfora-connected-py313-assurance-v5/run.py --layer && /usr/bin/prlimit --as=1073741824 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -B /home/riomus/runs/sfora-connected-py313-assurance-v5/run.py' > original.log 2>&1
status=$?
( set -o noclobber; printf '%s\n' "$status" > terminal-status.txt )
exit "$status"
