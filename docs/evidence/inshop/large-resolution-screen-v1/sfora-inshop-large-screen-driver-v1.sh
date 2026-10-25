#!/usr/bin/env bash
# Detached driver (session scope keeps user@1000 alive): In-Shop pretrained Large 384 vs 256 screen.
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-inshop-res-screen-src-v1; L=$R/sfora-inshop-large-screen-v1.log
echo "DRIVER $(date -u +%FT%TZ) pid=$$ cgroup=$(cat /proc/$$/cgroup)"
test ! -e $L
systemd-run --user --wait --collect --unit=sfora-inshop-large-screen-v1 -p StandardOutput=append:$L -p StandardError=append:$L -p WorkingDirectory=$D/scripts -p MemoryMax=24G -p MemorySwapMax=0 -p RuntimeMaxSec=7200 --setenv=PYTHONPATH=$D/scripts:$D/src --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=HF_HUB_OFFLINE=1 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v /home/riomus/group-learning/.venv/bin/python -B $D/scripts/screen_inshop_large_resolution_v1.py --dataset-root /home/riomus/datasets/inshop_official_standard --large-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c --output-dir $R/sfora-inshop-large-screen-v1 && echo "DONE $(date -u +%FT%TZ)" || echo "FAIL rc=$? $(date -u +%FT%TZ)"
