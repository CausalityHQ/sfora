#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-pool-export-overlap-v1
cd "$root"
test ! -e export-measurement-v1.log
test ! -e export-measurement-v1-time.txt
test ! -e /home/riomus/runs/sfora-large-pool-export-measurement-v1
systemd-run --user --unit=sfora-large-pool-export-measurement-v1 --wait --pipe \
  -p RuntimeMaxSec=119 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/export-measurement-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/measure_pe_large_export_overlap.py" \
  --execution-sha256 "${1:?execution SHA required}" --output /home/riomus/runs/sfora-large-pool-export-measurement-v1 \
  > "$root/export-measurement-v1.log" 2>&1
