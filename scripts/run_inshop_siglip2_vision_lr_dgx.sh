#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=d616513d3850b0268fa80ee28f2c16d4d852c0d25b889376650df72c5cb01290
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

seed="${1:?seed required}"
updates="${2:?updates required}"
case "$seed" in
  179024) preflight_sha=f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 ;;
  179025) preflight_sha=9dc46a73d7000996028551600ae84a6e6fe1d8d3e113d6503547a2fc8ca927f7 ;;
  *) exit 2 ;;
esac
case "$updates" in 17|1000) ;; *) exit 2 ;; esac
preflight="$RUNS/inshop-unseen-gallery-preflight-$seed-v2.json"
trainer="$RUNS/train_inshop_siglip2_unseen_gallery_vision_lr.py"
test "$(sha256sum "$trainer" | cut -d' ' -f1)" = "$TRAINER_SHA"
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = "$preflight_sha"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
for arm in control lr3x; do
  test ! -e "$RUNS/sfora-inshop-vision-lr-$arm-$seed-$updates-v1"
done
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for arm in control lr3x; do
  vision_lr=1e-5
  if [[ "$arm" == lr3x ]]; then vision_lr=3e-5; fi
  output="$RUNS/sfora-inshop-vision-lr-$arm-$seed-$updates-v1"
  echo "START seed=$seed updates=$updates arm=$arm vision_lr=$vision_lr" >&2
  "$PYTHON" "$trainer" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --output-dir "$output" --seed "$seed" --arm control --updates "$updates" \
    --vision-lr "$vision_lr"
  test -s "$output/receipt.json"
  echo "DONE seed=$seed updates=$updates arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
