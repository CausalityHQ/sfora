#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-final-two-v1
test ! -e "$root/siglip-final-two-native-gpu-v1.log"
test ! -e /home/riomus/runs/sfora-large-final-two-gpu-v1
systemd-run --user --unit=sfora-large-siglip-final-two-native-gpu-v1 --wait --pipe \
  -p RuntimeMaxSec=119 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/siglip-final-two-native-gpu-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/qualify_pe_large_final_two_gpu.py" \
  --execution-sha256 "${1:?execution SHA required}" --output /home/riomus/runs/sfora-large-final-two-gpu-v1 \
  > "$root/siglip-final-two-native-gpu-v1.log" 2>&1
