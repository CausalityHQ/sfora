#!/usr/bin/env bash
# Detached driver: P1 serving parity vs training export (01694072), 280/512 then 384/512. Waits for the GPU lock. No retries.
set -euo pipefail
R=/home/riomus/runs; S=$R/sfora-p1-serving-src-v1; PY=/home/riomus/group-learning/.venv/bin/python
MODEL=/home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
say() { echo "P1PARITY $(date -u +%FT%TZ) $*"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) module=$(sha256sum $S/src/sfora/so400m_compact_serving.py | cut -c1-64)"
run() {  # resolution run-dir
  local r=$1 dir=$2 name=sfora-p1-parity-$1-v1
  test ! -e $R/$name.json
  say "START $r"
  systemd-run --user --wait --collect --unit=$name -p StandardOutput=append:$R/$name.log -p StandardError=append:$R/$name.log \
    -p WorkingDirectory=$S -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=7200 \
    --setenv=PYTHONPATH=$S/src:$S/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY $S/scripts/check_so400m_serving_parity_v1.py --run-dir $dir --resolution $r \
    --model-snapshot $MODEL --archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products --output $R/$name.json || { say "HOLD $r parity failed or differs"; exit 1; }
  say "DONE $r result=$(sha256sum $R/$name.json | cut -c1-64)"
}
run 280 $R/sfora-sop-res280d512-so400_280-179024-1000-v1
run 384 $R/sfora-sop-384d512-so400-179024-1000-v1
say "ALLDONE"
