#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=a01f3cbc6754393bbdbc2aa3b89cdd3c852d703adffea4c22933fbad4759cb0e
PREFLIGHT_SHA=f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

mode="${1:?smoke or full required}"
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
trainer="$RUNS/train_inshop_siglip2_unseen_gallery_true_freeze.py"
preflight="$RUNS/inshop-unseen-gallery-preflight-179024-v2.json"
test "$(sha256sum "$trainer" | cut -d' ' -f1)" = "$TRAINER_SHA"
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = "$PREFLIGHT_SHA"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
arms=(freeze_emb)
updates=1000
if [[ "$mode" == smoke ]]; then
  arms=(freeze freeze_emb)
  updates=17
fi
for arm in "${arms[@]}"; do
  output="$RUNS/sfora-inshop-true-freeze-$arm-179024-$updates-v1"
  test ! -e "$output"
  echo "START mode=$mode arm=$arm updates=$updates" >&2
  "$PYTHON" "$trainer" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$preflight" --preflight-sha256 "$PREFLIGHT_SHA" \
    --output-dir "$output" --seed 179024 --arm "$arm" --updates "$updates"
  echo "DONE arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
