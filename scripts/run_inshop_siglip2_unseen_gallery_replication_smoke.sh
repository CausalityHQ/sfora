#!/usr/bin/env bash
# Four serial 17-update paired smokes before the frozen seed replication.
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=1edb0bf0fe51003719f893a50975d17be80283747f93ef980ad45550bee621dd
PROTOCOL_SHA=6e88cd6dade40cfa6a0fb563d645b7657f8fa0083907ce5a7cc9b5f98473d98b
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

test "$(sha256sum "$RUNS/train_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = "$TRAINER_SHA"
test "$(sha256sum "$RUNS/inshop-unseen-gallery-seed-replication-protocol-v1.json" | cut -d' ' -f1)" = "$PROTOCOL_SHA"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for seed in 179024 179025; do
  preflight_sha=f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
  if [[ "$seed" == 179025 ]]; then
    preflight_sha=9dc46a73d7000996028551600ae84a6e6fe1d8d3e113d6503547a2fc8ca927f7
  fi
  preflight="$RUNS/inshop-unseen-gallery-preflight-$seed-v2.json"
  test "$(sha256sum "$preflight" | cut -d' ' -f1)" = "$preflight_sha"
  for arm in control freeze; do
    output="$RUNS/sfora-inshop-unseen-$arm-$seed-smoke17-v2"
    test ! -e "$output"
    echo "START unseen-gallery replication smoke seed=$seed arm=$arm" >&2
    "$PYTHON" "$RUNS/train_inshop_siglip2_unseen_gallery.py" \
      --dataset-root /home/riomus/datasets/inshop_official_standard \
      --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
      --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
      --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
      --output-dir "$output" --seed "$seed" --arm "$arm" --updates 17
    test -s "$output/receipt.json"
    echo "DONE unseen-gallery replication smoke seed=$seed arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
  done
done
