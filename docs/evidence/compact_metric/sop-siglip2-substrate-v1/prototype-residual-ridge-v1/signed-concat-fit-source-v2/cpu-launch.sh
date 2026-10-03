#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-so400-signed-concat-cpu-v2
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
bash -n /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/cpu-command.sh
printf '%s  %s\n' e2813b859fc37f86f5cd36e754bfa8a75282e7f47c4759f3b67cd26946add31d /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/cpu-command.sh | sha256sum -c
systemd-run --user --unit=sfora-so400-signed-concat-cpu-v2 --wait --pipe --collect --property=RuntimeMaxSec=300 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash -c 'set -e; sudo -n /bin/sync; sudo -n /bin/sh -c "echo 3 > /proc/sys/vm/drop_caches"; exec /bin/bash /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/cpu-command.sh'
