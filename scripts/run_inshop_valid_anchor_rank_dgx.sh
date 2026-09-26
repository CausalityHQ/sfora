#!/usr/bin/env bash
set -euo pipefail

mode="${1:?smoke or full required}"
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
trainer="$RUNS/train_inshop_siglip2_unseen_gallery_valid_anchor_57981249.py"
preflight="$RUNS/inshop-unseen-gallery-preflight-179024-v2.json"
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
test "$(sha256sum "$trainer" | cut -d' ' -f1)" = 57981249fda8ee2d7950d631db444c0d28e18fc3f23a7d38ff24ccc2a7b54388
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
arms=(freeze_emb_rank)
updates=1000
if [[ "$mode" == smoke ]]; then
  arms=(freeze_emb freeze_emb_rank)
  updates=17
fi
for arm in "${arms[@]}"; do
  output="$RUNS/sfora-inshop-valid-anchor-$arm-179024-$updates-v2"
  test ! -e "$output"
  /home/riomus/group-learning/.venv/bin/python "$trainer" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$preflight" --preflight-sha256 f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 \
    --output-dir "$output" --seed 179024 --arm "$arm" --updates "$updates"
  echo "DONE arm=$arm updates=$updates receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
