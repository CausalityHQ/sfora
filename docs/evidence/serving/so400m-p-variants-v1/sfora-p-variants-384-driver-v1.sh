#!/usr/bin/env bash
# Detached driver: its session scope keeps user@1000 alive while the unit runs (Linger=no).
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-p-variants-src-v1; L=$R/sfora-p-variants-384-v1.log
echo "DRIVER $(date -u +%FT%TZ) pid=$$ cgroup=$(cat /proc/$$/cgroup)"
test ! -e $L
systemd-run --user --wait --collect --unit=sfora-p-variants-384-v1 -p StandardOutput=append:$L -p StandardError=append:$L \
  -p WorkingDirectory=$D -p MemoryMax=48G -p MemorySwapMax=0 -p RuntimeMaxSec=10800 \
  --setenv=PYTHONPATH=$D/src:$D/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v /home/riomus/group-learning/.venv/bin/python $D/scripts/check_so400m_serving_variants_v1.py \
  --run-dir $R/sfora-sop-384d512-so400-179024-1000-v1 --resolution 384 \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  --archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --output $R/sfora-p-variants-384-v1.json \
  && echo "DONE $(date -u +%FT%TZ)" || echo "FAIL rc=$? $(date -u +%FT%TZ)"
