#!/usr/bin/env bash
# Detached driver (prereg 3b289e72): In-Shop Large@384 512-D pilot -> paired screen vs So400m@384 pilot. Smoke first. No retries.
set -euo pipefail
R=/home/riomus/runs; P=$R/sfora-inshop-pilot-src-v1/scripts; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
DS=/home/riomus/datasets/inshop_official_standard; F=$R/sfora-inshop-res-screen-v1/r384_train_features.npy
say() { echo "EXPO500 $(date -u +%FT%TZ) $*"; }
unit() { local n=$1; shift
  systemd-run --user --wait --collect --unit=$n -p StandardOutput=append:$R/$n.log -p StandardError=append:$R/$n.log \
    -p WorkingDirectory=$P -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$P/../src:$P --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY "$@"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) wrapper=$(sha256sum $P/train_inshop_so400m_pilot.py | cut -c1-16)"
for u in 500; do
  N=sfora-inshop-expo500-c384-179024-$u-v1; test ! -e $R/$N
  unit $N $P/train_inshop_so400m_pilot.py --substrate so400 --crop 384 --seed 179024 --updates $u --workers 2 \
    --dataset-root $DS --model-snapshot $SNAP --pretrained-features $F --output-dir $R/$N || { say "HOLD $u"; exit 1; }
  say "DONE $u"
done
cd $P; CUDA_VISIBLE_DEVICES= PYTHONPATH=../src:. $PY screen_inshop_pilot_pair_v1.py --candidate $R/sfora-inshop-so400m-c384-179024-1000-v1 \
  --control $R/sfora-inshop-expo500-c384-179024-500-v1 --dataset-root $DS > $R/sfora-inshop-expo500-screen.json
say "SCREEN $(python3 -c "import json;r=json.load(open('$R/sfora-inshop-expo500-screen.json'));print(r['r1'], r['map_at_r'])")"
say "ALLDONE"
