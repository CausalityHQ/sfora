#!/usr/bin/env bash
# Detached driver: its session scope keeps user@1000 alive while the unit runs (Linger=no).
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-inshop-res-screen-src-v1; L=$R/sfora-inshop-res-screen-v1-a3.log
echo "DRIVER $(date -u +%FT%TZ) pid=$$ cgroup=$(cat /proc/$$/cgroup)"
test ! -e $L
systemd-run --user --wait --collect --unit=sfora-inshop-res-screen-v1-a3 -p StandardOutput=append:$L -p StandardError=append:$L \
  -p WorkingDirectory=$D/scripts -p MemoryMax=24G -p MemorySwapMax=0 -p RuntimeMaxSec=7200 \
  --setenv=PYTHONPATH=$D/scripts:$D/src --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=HF_HUB_OFFLINE=1 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v /home/riomus/group-learning/.venv/bin/python -B $D/scripts/screen_inshop_so400m_resolution_v1.py \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --so400-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  --output-dir $R/sfora-inshop-res-screen-v1 \
  && echo "DONE $(date -u +%FT%TZ)" || echo "FAIL rc=$? $(date -u +%FT%TZ)"
