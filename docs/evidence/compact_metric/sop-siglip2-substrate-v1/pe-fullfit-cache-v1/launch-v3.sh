#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-pe-fp16-mechanics-v1
cache_root=/home/riomus/runs/sfora-pe-fullfit-cache-v3
venv_python=/home/riomus/group-learning/.venv/bin/python
cd "$run_root"
test ! -e "$cache_root/receipt.json"
test ! -e "$cache_root/large.fit.npy"
test ! -e "$cache_root/pe.fit.npy"
test ! -e "$run_root/fullfit-cache-v3.log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units 'sfora*' --state=running --no-legend --no-pager)"
flock -n /home/riomus/.sfora-siglip2-gpu.lock true
systemd-run --user --unit=sfora-pe-fullfit-cache-v3 \
  --property=RuntimeMaxSec=480 --property=MemoryMax=8G \
  --property=KillMode=control-group --property="WorkingDirectory=$run_root" \
  --property="StandardOutput=append:$run_root/fullfit-cache-v3.log" \
  --property="StandardError=append:$run_root/fullfit-cache-v3.log" \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv="PYTHONPATH=$run_root:$run_root/src:$run_root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$run_root/fullfit-cache-v3-time.txt" \
  "$venv_python" "$run_root/export_inshop_pe_fit_features.py" \
  --root "$run_root" --output "$cache_root" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --large-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --mechanics-dir /home/riomus/runs/sfora-pe-fp16-smoke-v1 \
  --preflight-sha256 4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff
systemctl --user show sfora-pe-fullfit-cache-v3 --property=InvocationID,ActiveState,SubState,RuntimeMaxUSec,MemoryMax,KillMode
