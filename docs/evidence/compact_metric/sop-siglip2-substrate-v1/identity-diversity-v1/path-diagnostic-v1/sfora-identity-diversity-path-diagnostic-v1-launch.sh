#!/bin/bash
set -euo pipefail
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
printf '%s  %s\n' 85da8eb450e9c1a93deee7750395576c10855bae62fae29808762247e349a428 /home/riomus/runs/sfora-identity-diversity-path-diagnostic-v1-command.sh | sha256sum -c
systemd-run --user --unit=sfora-identity-diversity-path-diagnostic-v1 --wait --pipe --collect --property=RuntimeMaxSec=300 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop '--property=ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-identity-diversity-path-diagnostic-v1-command.sh
