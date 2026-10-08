#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-connected-installed-control-serving-diagnostic-v2
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
printf '%s  %s\n' 97b9dea7bed947a3b23b615d8fbdd14830e9ae30a23efdafab1c89e1fabac3ca /home/riomus/runs/sfora-connected-installed-control-serving-source-v2/command.sh | sha256sum -c
bash -n /home/riomus/runs/sfora-connected-installed-control-serving-source-v2/command.sh
systemd-run --user --unit=sfora-connected-installed-control-serving-diagnostic-v2 --wait --pipe --collect --property=RuntimeMaxSec=1500 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-connected-installed-control-serving-source-v2/command.sh
