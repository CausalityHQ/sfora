#!/usr/bin/env bash
# One frozen serial GPU gate: control -> freeze -> budget.
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=7acea65835a5c079c4b8ad625e0144f41943cf24506ac3265444d77c540f25e3
PREFLIGHT_SHA=d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

test "$(sha256sum "$RUNS/train_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = "$TRAINER_SHA"
test "$(sha256sum "$RUNS/inshop-unseen-gallery-preflight-v1.json" | cut -d' ' -f1)" = "$PREFLIGHT_SHA"
test -f "$RUNS/sfora-inshop-unseen-control-smoke17-v1/receipt.json"
test -f "$RUNS/sfora-inshop-unseen-freeze-smoke17-v1/receipt.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for arm in control freeze budget; do
  test ! -e "$RUNS/sfora-inshop-unseen-$arm-179023-v1"
done
for arm in control freeze budget; do
  updates=1000
  if [[ "$arm" == budget ]]; then updates=3000; fi
  output="$RUNS/sfora-inshop-unseen-$arm-179023-v1"
  echo "START unseen-gallery arm=$arm updates=$updates" >&2
  "$PYTHON" "$RUNS/train_inshop_siglip2_unseen_gallery.py" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$RUNS/inshop-unseen-gallery-preflight-v1.json" \
    --output-dir "$output" --arm "$arm" --updates "$updates"
  test -s "$output/receipt.json"
  echo "DONE unseen-gallery arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
