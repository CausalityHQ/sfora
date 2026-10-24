#!/usr/bin/env bash
# Detached session driver: ONE official SOP TEST read per pair arm (evaluator e88fffcc, pre-registered). No retries.
set -euo pipefail
R=/home/riomus/runs; D=$R/sfora-sop-so400m-pair-src-v3; PY=/home/riomus/group-learning/.venv/bin/python
HUB=/home/riomus/.cache/huggingface/hub
declare -A MODEL=([large]=$HUB/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
                  [so400_280]=$HUB/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f)
TILE=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
say() { echo "OFFICIAL512 $(date -u +%FT%TZ) $*"; }
say "BEGIN pid=$$ cgroup=$(cat /proc/$$/cgroup) evaluator=$(sha256sum $D/scripts/evaluate_sop_so400m_res280_dim512_official.py | cut -c1-64)"
run() {
  local s=$1 name=sfora-sop-so400m-official-res280d512-179024-v1 train=$R/sfora-sop-res280d512-$1-179024-1000-v1
  test ! -e $R/$name
  say "START $s"
  systemd-run --user --wait --collect --unit=$name -p StandardOutput=append:$R/$name.log -p StandardError=append:$R/$name.log \
    -p WorkingDirectory=$D/scripts -p MemoryMax=24G -p MemorySwapMax=0 -p RuntimeMaxSec=7200 \
    --setenv=PYTHONPATH=$D/src:$D/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=CUTILE_TILEIRAS_PATH=$TILE \
    --setenv=PATH=$(dirname $TILE):/usr/local/bin:/usr/bin:/bin --setenv=CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/flock -n $R/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
    /usr/bin/time -v $PY $D/scripts/evaluate_sop_so400m_res280_dim512_official.py --substrate $s \
    --training-receipt $train/receipt.json --training-checkpoint $train/checkpoint.pt --training-source-root $D \
    --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
    --test-image-manifest $R/sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products --model-snapshot ${MODEL[$s]} \
    --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
    --output-dir $R/$name || { say "HOLD $s failed"; exit 1; }
  say "DONE $s receipt=$(sha256sum $R/$name/receipt.json | cut -c1-64)"
}
run so400_280
say "ALLDONE"
