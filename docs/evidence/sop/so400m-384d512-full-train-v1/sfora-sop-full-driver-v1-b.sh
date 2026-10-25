#!/usr/bin/env bash
# Detached driver (prereg docs/sop_full_train_384d512_2026-10-25.md): scorer parity, then SOP full-TRAIN + one official read. No retries.
set -euo pipefail
R=/home/riomus/runs; P=$R/sfora-inshop-pilot-src-v1/scripts; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
A=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz; DS=/home/riomus/datasets/Stanford_Online_Products
F=$R/sfora-sop-so400m-train-features-v1/train_features.npy
say() { echo "SOPFULL $(date -u +%FT%TZ) $*"; }
unit() { local n=$1; shift
  systemd-run --user --wait --collect --unit=$n -p StandardOutput=append:$R/$n.log -p StandardError=append:$R/$n.log \
    -p WorkingDirectory=$P -p MemoryMax=40G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$P/../src:$P --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY "$@"; }
COMMON="--dataset sop --sop-archive $A --substrate so400 --crop 384 --seed 179024 --updates 1000 --workers 2 --full-train --dataset-root $DS --model-snapshot $SNAP --pretrained-features $F"
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) wrapper=$(sha256sum $P/train_inshop_so400m_pilot.py | cut -c1-16)"
N1=sfora-sop-full-c384-179024-1000-v1; test ! -e $R/$N1
unit $N1 $P/train_inshop_so400m_pilot.py $COMMON --output-dir $R/$N1 || { say "HOLD full"; exit 1; }
say "OFFICIAL $(python3 -c "import json;q=json.load(open('$R/$N1/receipt.json'))['quality'];print(q['recall_at_1'],q['map_at_r'])")"
say "ALLDONE"
