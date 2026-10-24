#!/usr/bin/env bash
# Detached driver (prereg 13c41e5f): shipped In-Shop serving gate, then seed-179025 full-TRAIN replicate. No retries.
set -euo pipefail
R=/home/riomus/runs; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
DS=/home/riomus/datasets/inshop_official_standard; M=$R/sfora-inshop-so400m-full-c384-179024-1000-v1
say() { echo "FOLLOWUP $(date -u +%FT%TZ) $*"; }
unit() { local n=$1 wd=$2; shift 2
  systemd-run --user --wait --collect --unit=$n -p StandardOutput=append:$R/$n.log -p StandardError=append:$R/$n.log \
    -p WorkingDirectory=$wd -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$wd/../src:$wd --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY "$@"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup)"
S=$R/sfora-inshop-serving-src-v1/scripts
unit sfora-inshop-serving-v1 $S $S/check_inshop_serving_v1.py --run-dir $M --model-snapshot $SNAP --dataset-root $DS \
  --output $R/sfora-inshop-serving-v1.json && say "DONE serving" || say "HOLD serving"
P=$R/sfora-inshop-pilot-src-v1/scripts; N=sfora-inshop-so400m-full-c384-179025-1000-v1
test ! -e $R/$N
unit $N $P $P/train_inshop_so400m_pilot.py --crop 384 --seed 179025 --updates 1000 --workers 2 --full-train \
  --dataset-root $DS --model-snapshot $SNAP --pretrained-features $R/sfora-inshop-res-screen-v1/r384_train_features.npy \
  --output-dir $R/$N && say "DONE seed179025" || say "HOLD seed179025"
say "ALLDONE"
