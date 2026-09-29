#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-optimization-gpu-v2
output=/home/riomus/runs/sfora-large-optimization-mechanics-v2
test ! -e "$root/mechanics-v2.log"
test ! -e "$output"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
systemd-run --user --unit=sfora-large-optimization-mechanics-v2 --wait --pipe \
  -p RuntimeMaxSec=299 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/mechanics-v2-time.txt" \
  /home/riomus/group-learning/.venv/bin/python -u "$root/train_pe_large_optimization.py" \
  --execution-sha256 "${1:?execution SHA required}" --output "$output" \
  > "$root/mechanics-v2.log" 2>&1
