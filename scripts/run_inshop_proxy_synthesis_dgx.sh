#!/usr/bin/env bash
set -euo pipefail

mode="${1:?smoke or full required}"
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
trainer="$RUNS/train_inshop_siglip2_unseen_gallery_proxy_441b5bfe.py"
preflight="$RUNS/inshop-unseen-gallery-preflight-179024-v2.json"
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1
test "$(sha256sum "$trainer" | cut -d' ' -f1)" = a9172a0b439238cd943d9d9e0aaed614c5c508bb94db4b1190b6aa87ad8cf2a6
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
arms=(proxy_synthesis)
updates=1000
if [[ "$mode" == smoke ]]; then
  arms=(control proxy_synthesis)
  updates=17
fi
for arm in "${arms[@]}"; do
  output="$RUNS/sfora-inshop-proxy-synthesis-$arm-179024-$updates-v1"
  test ! -e "$output"
  /home/riomus/group-learning/.venv/bin/python "$trainer" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir "$RUNS/sfora-inshop-siglip2-train-features-v1" \
    --preflight "$preflight" --preflight-sha256 f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 \
    --output-dir "$output" --seed 179024 --arm "$arm" --updates "$updates"
  echo "DONE arm=$arm updates=$updates receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
