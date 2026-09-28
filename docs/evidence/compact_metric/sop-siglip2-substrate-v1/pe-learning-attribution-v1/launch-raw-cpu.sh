#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-pe-augmented-pair-v1
out_root=/home/riomus/runs/sfora-pe-raw-attribution-v1
test ! -e "$out_root"
test ! -e "$run_root/attribution-raw-v1.log"
systemd-run --user --unit=sfora-pe-attribution-raw-v1 --wait \
 --property=RuntimeMaxSec=180 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$run_root" \
 --property="StandardOutput=append:$run_root/attribution-raw-v1.log" \
 --property="StandardError=append:$run_root/attribution-raw-v1.log" \
 --setenv=CUDA_VISIBLE_DEVICES= \
 --setenv="PYTHONPATH=$run_root:$run_root/src:$run_root/isolated-deps" \
 /usr/bin/time -v -o "$run_root/attribution-raw-v1-time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$run_root/score_inshop_pe_raw_attribution.py" \
 --attribution-dir /home/riomus/runs/sfora-pe-learning-attribution-v1 \
 --training-dir /home/riomus/runs/sfora-pe-augmented-100-v2 --output "$out_root" \
 --receipt-sha256 63499ecfcf189595610ab0c20ae688f8350e1a21e1bb6f6e312051766203d33c
