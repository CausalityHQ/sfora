#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-connected-probe-mechanics-control-179061-v1
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
bash -n /home/riomus/runs/sfora-connected-probe-train-source-v2/mechanics-control-179061-v1-command.sh
printf '%s  %s\n' caba2dc6f3ea6750793ac028188a88468c5c5ab0f9e87c8b7ce3bc3ac44be680 /home/riomus/runs/sfora-connected-probe-train-source-v2/mechanics-control-179061-v1-command.sh | sha256sum -c
systemd-run --user --unit=sfora-connected-probe-mechanics-control-179061-v1 --wait --pipe --collect --property=RuntimeMaxSec=1200 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash -c 'set -e; sudo -n /bin/sync; sudo -n /bin/sh -c "echo 3 > /proc/sys/vm/drop_caches"; exec /bin/bash /home/riomus/runs/sfora-connected-probe-train-source-v2/mechanics-control-179061-v1-command.sh'
