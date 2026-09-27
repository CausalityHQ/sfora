#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-live-head-179024-v1"
PYTHON=/home/riomus/group-learning/.venv/bin/python
PREFLIGHT="$RUNS/inshop-unseen-gallery-preflight-179024-v2.json"
export PYTHONPATH="$WORK:$RUNS:$RUNS/sfora-inshop-mapr-freeze16-179024-v1/src:/home/riomus/sfora-siglip2-deployed-batch-v1/scripts"
export HF_HUB_OFFLINE=1

case "${1:?smoke, control100, or live100 required}" in
  smoke) arm=freeze_emb_live; updates=17 ;;
  control100) arm=freeze_emb; updates=100 ;;
  live100) arm=freeze_emb_live; updates=100 ;;
  *) exit 2 ;;
esac
test "$(sha256sum "$WORK/train_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = 2bbcf55ff4e07ad63f5052ddcbb1fa706b8e239b71ca96c627cb2690393d6799
test "$(sha256sum "$WORK/train_sop_siglip2_compact.py" | cut -d' ' -f1)" = ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495
test "$(sha256sum "$RUNS/sfora-inshop-mapr-freeze16-179024-v1/src/sfora/live_head_bank.py" | cut -d' ' -f1)" = bb3b2bb69d0fb25af308d255cf9c1d728a8119bd65f8bf75f6eade040a4fc13f
test "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" = f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9

output="$RUNS/sfora-inshop-live-head-179024-$arm-$updates-v1"
test ! -e "$output"
"$PYTHON" "$WORK/train_inshop_siglip2_unseen_gallery.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
  --preflight "$PREFLIGHT" --preflight-sha256 f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 \
  --output-dir "$output" --seed 179024 --arm "$arm" \
  --freeze-first-blocks 16 --updates "$updates"
echo "DONE arm=$arm updates=$updates receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
