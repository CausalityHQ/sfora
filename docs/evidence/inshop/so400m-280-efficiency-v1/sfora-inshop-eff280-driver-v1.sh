#!/usr/bin/env bash
# Detached driver (prereg 735d57ed/c97b630e): In-Shop 280 pilot -> paired screen vs 384 pilot -> (ADVANCE) full-TRAIN 280 + one official read -> serving gate @280. No retries.
set -euo pipefail
R=/home/riomus/runs; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
DS=/home/riomus/datasets/inshop_official_standard; F=$R/sfora-inshop-res-screen-v1/r384_train_features.npy
P=$R/sfora-inshop-pilot-src-v1/scripts; S=$R/sfora-inshop-serving-src-v1/scripts
say() { echo "EFF280 $(date -u +%FT%TZ) $*"; }
unit() { local n=$1 wd=$2; shift 2
  systemd-run --user --wait --collect --unit=$n -p StandardOutput=append:$R/$n.log -p StandardError=append:$R/$n.log \
    -p WorkingDirectory=$wd -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$wd/../src:$wd --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY "$@"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) wrapper=$(sha256sum $P/train_inshop_so400m_pilot.py | cut -c1-16)"
N=sfora-inshop-so400m-c280-179024-1000-v1
unit $N $P $P/train_inshop_so400m_pilot.py --crop 280 --seed 179024 --updates 1000 --workers 2 \
  --dataset-root $DS --model-snapshot $SNAP --pretrained-features $F --output-dir $R/$N || { say "HOLD pilot"; exit 1; }
cd $P; CUDA_VISIBLE_DEVICES= PYTHONPATH=../src:. $PY screen_inshop_pilot_pair_v1.py --candidate $R/$N \
  --control $R/sfora-inshop-so400m-c384-179024-1000-v1 --dataset-root $DS > $R/sfora-inshop-eff280-screen.json
LOW=$(python3 -c "import json;print(json.load(open('$R/sfora-inshop-eff280-screen.json'))['r1']['lower95'])")
say "SCREEN lower95=$LOW"
if python3 -c "import sys; sys.exit(0 if $LOW > -0.30 else 1)"; then say "ADVANCE"; else say "KILL"; say "ALLDONE"; exit 0; fi
M=sfora-inshop-so400m-full-c280-179024-1000-v1
unit $M $P $P/train_inshop_so400m_pilot.py --crop 280 --seed 179024 --updates 1000 --workers 2 --full-train \
  --dataset-root $DS --model-snapshot $SNAP --pretrained-features $F --output-dir $R/$M || { say "HOLD full"; exit 1; }
say "OFFICIAL $(python3 -c "import json;q=json.load(open('$R/$M/receipt.json'))['quality'];print(q['recall_at_1'],q['map_at_r'])")"
unit sfora-inshop-serving-280-v1 $S $S/check_inshop_serving_v1.py --run-dir $R/$M --model-snapshot $SNAP --dataset-root $DS \
  --resolution 280 --output $R/sfora-inshop-serving-280-v1.json && say "DONE serving" || say "HOLD serving"
say "ALLDONE"
