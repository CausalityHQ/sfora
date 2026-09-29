#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-final-block-v1
output=/home/riomus/runs/sfora-large-final-block-mechanics-v1
test ! -e "$root/final-block-mechanics-v1.log"
test ! -e "$root/final-block-mechanics-v1-time.txt"
test ! -e "$output"
systemd-run --user --unit=sfora-large-final-block-mechanics-v1 --wait --pipe \
  -p RuntimeMaxSec=119 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/final-block-mechanics-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/train_pe_large_final_block.py" \
  --execution-sha256 "${1:?execution SHA required}" --output "$output" \
  > "$root/final-block-mechanics-v1.log" 2>&1
