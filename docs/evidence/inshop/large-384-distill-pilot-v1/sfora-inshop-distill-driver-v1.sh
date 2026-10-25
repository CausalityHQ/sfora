#!/usr/bin/env bash
# Detached driver (prereg 3b289e72): In-Shop Large@384 512-D pilot -> paired screen vs So400m@384 pilot. Smoke first. No retries.
set -euo pipefail
R=/home/riomus/runs; P=$R/sfora-inshop-pilot-src-v1/scripts; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
DS=/home/riomus/datasets/inshop_official_standard; F=$R/sfora-inshop-large-screen-v1/large_r384_train_features.npy
say() { echo "DISTILL $(date -u +%FT%TZ) $*"; }
unit() { local n=$1; shift
  systemd-run --user --wait --collect --unit=$n -p StandardOutput=append:$R/$n.log -p StandardError=append:$R/$n.log \
    -p WorkingDirectory=$P -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$P/../src:$P --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY "$@"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) wrapper=$(sha256sum $P/train_inshop_so400m_pilot.py | cut -c1-16)"
for u in 17 1000; do
  N=sfora-inshop-large-distill-c384-179024-$u-v1; test ! -e $R/$N
  unit $N $P/train_inshop_so400m_pilot.py --substrate large --crop 384 --seed 179024 --updates $u --workers 2 \
    --dataset-root $DS --model-snapshot $SNAP --pretrained-features $F --teacher-embeddings $R/sfora-inshop-so400m-c384-179024-1000-v1/train_embeddings.npy --distill-weight 1.0 --output-dir $R/$N || { say "HOLD $u"; exit 1; }
  say "DONE $u"
done
cd $P; CUDA_VISIBLE_DEVICES= PYTHONPATH=../src:. $PY screen_inshop_pilot_pair_v1.py --candidate $R/sfora-inshop-large-distill-c384-179024-1000-v1 \
  --control $R/sfora-inshop-so400m-c384-179024-1000-v1 --dataset-root $DS > $R/sfora-inshop-distill384-screen.json
say "SCREEN $(python3 -c "import json;r=json.load(open('$R/sfora-inshop-distill384-screen.json'));print(r['r1'], r['map_at_r'])")"
say "ALLDONE"
