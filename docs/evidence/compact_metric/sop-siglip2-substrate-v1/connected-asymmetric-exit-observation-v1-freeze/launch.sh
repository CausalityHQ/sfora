#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-connected-asymmetric-cache-exit-observation-179061-v1
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
printf '%s  %s\n' 8556b6804056da7e1cbd5d7190a7b5dfbe17ad2bd7556f55e46fd06a01ffa60b /home/riomus/runs/sfora-connected-asymmetric-cache-source-v2/command.sh | sha256sum -c
bash -n /home/riomus/runs/sfora-connected-asymmetric-cache-source-v2/command.sh
systemd-run --user --unit=sfora-connected-asymmetric-cache-first-179061-v1 --wait --pipe --collect --property=RuntimeMaxSec=700 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-connected-asymmetric-cache-source-v2/command.sh
