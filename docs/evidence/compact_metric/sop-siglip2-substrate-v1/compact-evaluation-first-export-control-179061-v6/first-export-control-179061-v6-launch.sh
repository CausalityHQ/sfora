#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-export-control-179061-v6
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
bash -n /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v22/first-export-control-179061-v6-command.sh
printf '%s  %s\n' 62f095136ff4e0f1e79a56aaedfb3b2fa568bc7a17e1d790c749154936d648e8 /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v22/first-export-control-179061-v6-command.sh | sha256sum -c
systemd-run --user --unit=sfora-so400-compact-ranking-evaluation-first-export-control-179061-v6 --wait --pipe --collect --property=RuntimeMaxSec=300 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash -c 'set -e; sudo -n /bin/sync; sudo -n /bin/sh -c "echo 3 > /proc/sys/vm/drop_caches"; exec /bin/bash /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v22/first-export-control-179061-v6-command.sh'
