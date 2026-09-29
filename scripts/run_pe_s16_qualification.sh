#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-pe-s16-source-v1
cd "$root"
test ! -e s16-gpu-v1.log
test ! -e s16-gpu-v1-time.txt
test ! -e /home/riomus/runs/sfora-pe-s16-gpu-v1
systemd-run --user --unit=sfora-pe-s16-gpu-v1 --wait --pipe \
  -p RuntimeMaxSec=119 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/s16-gpu-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/qualify_pe_s16_gpu.py" \
  --cpu-preflight-sha256 "${1:?CPU authority SHA required}" --execution-sha256 "${2:?execution authority SHA required}" \
  > "$root/s16-gpu-v1.log" 2>&1
