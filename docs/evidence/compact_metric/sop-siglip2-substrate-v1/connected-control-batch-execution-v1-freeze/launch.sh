#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-connected-control-batch-execution-observation-v1
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
printf '%s  %s\n' bf40ecb64340bbe0f2d441a5e8cdad2a7c96f5e21b51406079507456b25297a6 /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/command.sh | sha256sum -c
bash -n /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/command.sh
systemd-run --user --unit=sfora-connected-control-batch-execution-observation-v1 --wait --pipe --collect --property=RuntimeMaxSec=700 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/command.sh
