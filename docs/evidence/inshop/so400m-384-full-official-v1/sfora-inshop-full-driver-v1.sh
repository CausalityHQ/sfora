#!/usr/bin/env bash
# Detached driver: In-Shop So400m@384/512-D full-TRAIN + ONE official read (prereg 72e6fd19, audit 4482f8469eb44fe2, source eefebb57). No retries.
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-inshop-pilot-src-v1; N=sfora-inshop-so400m-full-c384-179024-1000-v1
echo "FULL $(date -u +%FT%TZ) BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) wrapper=$(sha256sum $D/scripts/train_inshop_so400m_pilot.py | cut -c1-64)"
test ! -e $R/$N
systemd-run --user --wait --collect --unit=$N -p StandardOutput=append:$R/$N.log -p StandardError=append:$R/$N.log \
  -p WorkingDirectory=$D/scripts -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
  --setenv=PYTHONPATH=$D/src:$D/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
  /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v /home/riomus/group-learning/.venv/bin/python $D/scripts/train_inshop_so400m_pilot.py --crop 384 --seed 179024 --updates 1000 --workers 2 --full-train \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f \
  --pretrained-features $R/sfora-inshop-res-screen-v1/r384_train_features.npy --output-dir $R/$N \
  && echo "FULL $(date -u +%FT%TZ) DONE" || echo "FULL $(date -u +%FT%TZ) HOLD rc=$?"
