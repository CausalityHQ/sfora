#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-pe-lowrank-v1
cd "$root"
test ! -e pilot-100-v1.log
test ! -e pilot-100-v1-time.txt
test ! -e /home/riomus/runs/sfora-pe-lowrank-pilot-100-v1
systemd-run --user --unit=sfora-pe-lowrank-pilot-100-v1 --wait --pipe \
  -p RuntimeMaxSec=299 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/pilot-100-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/train_pe_lowrank_100.py" \
  --authority "$root/pilot-authority.json" \
  --output /home/riomus/runs/sfora-pe-lowrank-pilot-100-v1 \
  > "$root/pilot-100-v1.log" 2>&1
