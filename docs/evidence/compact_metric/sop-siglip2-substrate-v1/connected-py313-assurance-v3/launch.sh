#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-connected-py313-assurance-v3
printf '%s\n' 'e0b7a22460a6280ff6c38df3d34d5cf95b3857dcf596b4c75909609d6a045f50  inputs.tar.gz' '00ab54e7397f14b787230210a5980bdd47a415e5fe2d79f692e2b2a0a0d0894b  inventory.json' 'dc1efd271d0da140dbbea37078178ac2ba4092b60d4ba00c7c4ac029683151f6  run.py' | sha256sum -c
 test ! -e original.log
 test -z "$(systemctl --user list-units --state=running --no-legend | grep sfora || true)"
 test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
tar -xzf inputs.tar.gz
set +e
systemd-run --user --unit=sfora-connected-py313-assurance-v3 --wait --pipe --collect --property=RuntimeMaxSec=120 --property=MemoryMax=1073741824 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /usr/bin/prlimit --as=1073741824 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -B /home/riomus/runs/sfora-connected-py313-assurance-v3/run.py > original.log 2>&1
status=$?
( set -o noclobber; printf '%s\n' "$status" > terminal-status.txt )
exit "$status"
