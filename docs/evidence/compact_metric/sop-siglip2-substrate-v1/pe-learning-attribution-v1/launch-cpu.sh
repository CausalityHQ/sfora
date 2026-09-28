#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-pe-augmented-pair-v1
out_root=/home/riomus/runs/sfora-pe-learning-attribution-v1
test ! -e "$out_root"
systemd-run --user --unit=sfora-pe-attribution-cpu-v1 --wait \
 --property=RuntimeMaxSec=180 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$run_root" \
 --property="StandardOutput=append:$run_root/attribution-cpu-v1.log" \
 --property="StandardError=append:$run_root/attribution-cpu-v1.log" \
 --setenv=CUDA_VISIBLE_DEVICES= \
 --setenv="PYTHONPATH=$run_root:$run_root/src:$run_root/isolated-deps" \
 /usr/bin/time -v -o "$run_root/attribution-cpu-v1-time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$run_root/diagnose_inshop_pe_learning_attribution.py" \
 --root "$run_root" --cache /home/riomus/runs/sfora-pe-fullfit-cache-v3 \
 --training-dir /home/riomus/runs/sfora-pe-augmented-100-v2 --output "$out_root" \
 --dataset-root /home/riomus/datasets/inshop_official_standard \
 --mechanics-dir /home/riomus/runs/sfora-pe-fp16-smoke-v1 \
 --large-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
 --preflight-only
