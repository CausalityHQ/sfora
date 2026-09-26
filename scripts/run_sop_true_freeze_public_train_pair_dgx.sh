#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
precision="${2:?precision required}"
case "$seed" in 179026|179027) ;; *) exit 2 ;; esac
case "$precision" in fp32_autocast) short=fp32 ;; fp16_native) short=fp16 ;; *) exit 2 ;; esac
runs=/home/riomus/runs
source_root=/home/riomus/sfora-sop-true-freeze-official-v1
script="$runs/evaluate_sop_true_freeze_public_train.py"
export PYTHONPATH="$source_root/src"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
test "$(sha256sum "$script" | cut -d' ' -f1)" = b835aa81b39f2a5a25fcc1902e817b36e1794a1a30671bf08feb1b99739e2a8b
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$runs/.sfora-siglip2-gpu.lock"
flock -n 9
for arm in control freeze; do
  test ! -e "$runs/sfora-sop-true-freeze-public-train-$seed-$arm-$short-v1.json"
done
for arm in control freeze; do
  echo "START public TRAIN seed=$seed arm=$arm precision=$precision" >&2
  /home/riomus/group-learning/.venv/bin/python "$script" \
    --seed "$seed" --arm "$arm" --precision "$precision" \
    --decision "$runs/sfora-sop-true-freeze-decision-$seed-v1.json" \
    --training-run "$runs/sfora-sop-true-freeze-$arm-$seed-1000-v1" \
    --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
    --output "$runs/sfora-sop-true-freeze-public-train-$seed-$arm-$short-v1.json"
  echo "DONE public TRAIN seed=$seed arm=$arm precision=$precision" >&2
done
