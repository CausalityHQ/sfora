#!/bin/bash
set -euo pipefail
test ! -e /home/riomus/runs/sfora-connected-exact-torch-census-v1.json
if systemctl --user list-units --state=running --no-legend | grep -q sfora; then exit 90; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock /bin/true
printf '%s  %s\n' 9f81f05dbae02882e2f98b28b07be564eca766d0afaf291cc009fb53210f5b92 /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/command.sh 8ced9e4ea0119b3b12e7d4f7633d0b4769dd9b57bed5571916d2f5ec3e9c068e /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/authority.json | sha256sum -c
systemd-run --user --unit=sfora-connected-exact-torch-census-v1 --wait --pipe --collect --property=RuntimeMaxSec=900 --property=MemoryMax=8589934592 --property=MemorySwapMax=0 --property=KillMode=control-group --property=OOMPolicy=stop --property='ExecStopPost=/usr/bin/python3 /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py' --setenv=CUDA_VISIBLE_DEVICES= --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=TOKENIZERS_PARALLELISM=false /usr/bin/time -v /bin/bash /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/command.sh 8ced9e4ea0119b3b12e7d4f7633d0b4769dd9b57bed5571916d2f5ec3e9c068e
