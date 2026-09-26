#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
RUNS=/home/riomus/runs
WORK="$RUNS/sfora-valid-anchor-confirmation-179026-27"
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
trainer="$WORK/train_inshop_siglip2_unseen_gallery.py"
preflight="$WORK/preflight-$seed.json"
case "$seed" in
  179026) preflight_sha=4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254; arms=(freeze_emb freeze_emb_rank) ;;
  179027) preflight_sha=45d9fb0cd46844dccc9b25673634f047475f62991d42b3e61069d3823574d80b; arms=(freeze_emb_rank freeze_emb) ;;
  *) exit 2 ;;
esac
export PYTHONPATH="$WORK:$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
test "$(sha256sum "$trainer" | cut -d' ' -f1)" = 76e20c328df6e632b387486e4761fb26994b74eff4c0a944c23400a40c4985cc
test "$(sha256sum "$WORK/preflight_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = b6556df2d2011cb23c6b5ce25458c54679646b1ea7989fc47cb67e6a5142906e
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = "$preflight_sha"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for arm in "${arms[@]}"; do
  output="$RUNS/sfora-inshop-valid-anchor-confirm-$arm-$seed-v1"
  test ! -e "$output"
  /home/riomus/group-learning/.venv/bin/python "$trainer" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
    --output-dir "$output" --seed "$seed" --arm "$arm" --updates 1000
  echo "DONE seed=$seed arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
