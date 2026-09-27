#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-mapr-freeze16-179024-v1"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
PREFLIGHT="$RUNS/inshop-unseen-gallery-preflight-179024-v2.json"
export PYTHONPATH="$WORK:$RUNS:$WORK/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

mode="${1:?smoke or full required}"
case "$mode" in
  smoke) updates=17 ;;
  full) updates=1000 ;;
  *) exit 2 ;;
esac
test "$(sha256sum "$WORK/train_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = f92f792c52165350c16e59943fe8f06749f7dd0e8b5e7b3006ded9f683dd772e
test "$(sha256sum "$WORK/train_sop_siglip2_compact.py" | cut -d' ' -f1)" = ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495
test "$(sha256sum "$WORK/src/sfora/deployed_code_rank.py" | cut -d' ' -f1)" = 435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870
test "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" = f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9

output="$RUNS/sfora-inshop-mapr-freeze16-179024-$updates-v1"
test ! -e "$output"
"$PYTHON" "$WORK/train_inshop_siglip2_unseen_gallery.py" \
  --dataset-root /home/riomus/datasets/inshop_official_standard \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
  --preflight "$PREFLIGHT" --preflight-sha256 f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 \
  --output-dir "$output" --seed 179024 --arm freeze_emb_mapr \
  --freeze-first-blocks 16 --updates "$updates"
echo "DONE updates=$updates receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
