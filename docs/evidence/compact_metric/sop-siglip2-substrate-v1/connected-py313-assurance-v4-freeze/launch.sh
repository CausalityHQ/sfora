#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-connected-py313-assurance-v4
printf '%s\n' '7b5e9f44bacc849caa375455e013b8d7741fea3bf95fcddb43922c100770e294  inputs.tar.gz' 'b2bdb9d192a8a3a7711a89328be347204523a6ec7ebb167a5b289000bc431af8  inventory.json' 'dc1efd271d0da140dbbea37078178ac2ba4092b60d4ba00c7c4ac029683151f6  run.py' | sha256sum -c
 test ! -e original.log
 test -z "$(systemctl --user list-units --state=running --no-legend | grep sfora || true)"
 test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
tar -xzf inputs.tar.gz
set +e
systemd-run --user --unit=sfora-connected-py313-assurance-v4 --wait --pipe --collect --property=RuntimeMaxSec=120 --property=MemoryMax=1073741824 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /usr/bin/prlimit --as=1073741824 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -B /home/riomus/runs/sfora-connected-py313-assurance-v4/run.py > original.log 2>&1
status=$?
( set -o noclobber; printf '%s\n' "$status" > terminal-status.txt )
exit "$status"
