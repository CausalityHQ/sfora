#!/usr/bin/env bash
# Detached driver: its session scope keeps user@1000 alive while the unit runs (Linger=no).
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-sop-so400m-screen-src-v1; L=$R/sfora-sop-so400m-screen-v1-a2.log
echo "DRIVER $(date -u +%FT%TZ) pid=$$ cgroup=$(cat /proc/$$/cgroup)"
test ! -e $L
systemd-run --user --wait --collect --unit=sfora-sop-so400m-screen-v1-a2 -p StandardOutput=append:$L -p StandardError=append:$L \
  -p WorkingDirectory=$D/scripts -p MemoryMax=16G -p MemorySwapMax=0 -p RuntimeMaxSec=5400 \
  --setenv=PYTHONPATH=$D/scripts:$D/src --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /usr/bin/flock -n $R/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v /home/riomus/group-learning/.venv/bin/python -B $D/scripts/screen_sop_so400m_pretrained_v1.py \
  --unicom-train-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --so400-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  --large-features $R/sfora-siglip2-train-28e19203/train_features.npy --output-dir $R/sfora-sop-so400m-screen-v1-a2 \
  && echo "DONE $(date -u +%FT%TZ)" || echo "FAIL rc=$? $(date -u +%FT%TZ)"
