#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-pe-augmented-pair-v1
out_root=/home/riomus/runs/sfora-pe-compression-v1
cd "$run_root"
test ! -e "$out_root"
test ! -e "$run_root/compression-v1.log"
systemd-run --user --unit=sfora-pe-compression-v1 --wait \
 --property=RuntimeMaxSec=180 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$run_root" \
 --property="StandardOutput=append:$run_root/compression-v1.log" \
 --property="StandardError=append:$run_root/compression-v1.log" \
 --setenv=CUDA_VISIBLE_DEVICES= \
 --setenv="PYTHONPATH=$run_root:$run_root/src:$run_root/isolated-deps" \
 /usr/bin/time -v -o "$run_root/compression-v1-time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$run_root/diagnose_inshop_pe_compression.py" \
 --source-cache /home/riomus/runs/sfora-pe-fullfit-cache-v3 \
 --training-dir /home/riomus/runs/sfora-pe-augmented-100-v2 \
 --dataset-root /home/riomus/datasets/inshop_official_standard --output "$out_root"
