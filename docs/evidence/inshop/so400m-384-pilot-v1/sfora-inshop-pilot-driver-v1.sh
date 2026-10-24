#!/usr/bin/env bash
# Detached driver (session scope keeps user@1000 alive). In-Shop So400m 384 vs 256 pilot, prereg feeb1e24.
# Smokes (17 updates) at both crops must succeed before the 1000-update arms. No retries.
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-inshop-pilot-src-v1; PY=/home/riomus/group-learning/.venv/bin/python
SNAP=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
say() { echo "INSHOPPILOT $(date -u +%FT%TZ) $*"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup)"
run() {  # crop updates
  local c=$1 u=$2 name=sfora-inshop-so400m-c$1-179024-$2-v1
  test ! -e $R/$name
  say "START $c $u"
  systemd-run --user --wait --collect --unit=$name -p StandardOutput=append:$R/$name.log -p StandardError=append:$R/$name.log \
    -p WorkingDirectory=$D/scripts -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=14400 \
    --setenv=PYTHONPATH=$D/src:$D/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY $D/scripts/train_inshop_so400m_pilot.py --crop $c --seed 179024 --updates $u --workers 2 \
    --dataset-root /home/riomus/datasets/inshop_official_standard --model-snapshot $SNAP \
    --pretrained-features $R/sfora-inshop-res-screen-v1/r${c}_train_features.npy --output-dir $R/$name \
    || { say "HOLD $c $u failed"; exit 1; }
  say "DONE $c $u receipt=$(sha256sum $R/$name/receipt.json | cut -c1-64)"
}
run 384 17
run 256 17
run 384 1000
run 256 1000
say "ALLDONE"
