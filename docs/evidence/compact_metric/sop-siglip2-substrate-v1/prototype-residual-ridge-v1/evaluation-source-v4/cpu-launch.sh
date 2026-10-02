#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-so400-prototype-residual-evaluation-cpu-v4
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
bash -n /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v4/cpu-command.sh
printf '%s  %s\n' f4fd7c0d7388ef7ce60e0b0cba87d5d43fdefd03d864a0b7b4dd004e5b75b706 /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v4/cpu-command.sh | sha256sum -c
sudo -n /bin/sync
sudo -n /bin/sh -c 'echo 3 > /proc/sys/vm/drop_caches'
systemd-run --user --unit=sfora-so400-prototype-residual-evaluation-cpu-v4 --wait --pipe --collect --property=RuntimeMaxSec=120 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v4/cpu-command.sh
