#!/usr/bin/env bash
set -euo pipefail

RUNS=/home/riomus/runs
WORK="$RUNS/sfora-inshop-depth22-179026-v1"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PRE="$RUNS/sfora-valid-anchor-confirmation-179026-27/preflight-179026.json"
TRAIN="$WORK/train_inshop_siglip2_unseen_gallery.py"
export PYTHONPATH="$WORK:$RUNS:$RUNS/sfora-valid-anchor-confirmation-179026-27:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
test "$(sha256sum "$TRAIN" | cut -d' ' -f1)" = 92e60e603a88951a071b778ba1f6b2d1c362af9a81fffd78bac117e6c7a823a9
test "$(sha256sum "$RUNS/train_sop_siglip2_compact.py" | cut -d' ' -f1)" = e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6
test "$(sha256sum "$PRE" | cut -d' ' -f1)" = 4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254

for depth in 0 2; do
  if test "$depth" = 0; then
    features="$RUNS/sfora-inshop-siglip2-train-features-v1"
    output="$RUNS/sfora-inshop-depth22-control-179026-100-v1"
  else
    features="$RUNS/sfora-inshop-depth22-features-v1"
    output="$RUNS/sfora-inshop-depth22-treatment-179026-100-v1"
  fi
  test ! -e "$output"
  /home/riomus/group-learning/.venv/bin/python "$TRAIN" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$features" --preflight "$PRE" \
    --preflight-sha256 4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254 \
    --output-dir "$output" --seed 179026 --arm freeze_emb --updates 100 \
    --tail-blocks-to-drop "$depth"
  echo "DONE depth=$depth receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
