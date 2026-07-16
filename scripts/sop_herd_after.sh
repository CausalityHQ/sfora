#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for i in $(seq 1 400); do
  [ -f reports/generated/sop_pa_distill.json ] && break
  sleep 60
done
SFORA=.venv/bin/sfora
SOP=(--protocol proxy-anchor-resnet50-512 --dataset-name sop --samples-per-class 3
  --warmup-epochs 5 --lr-step-epochs 10 --train-epochs 60 --eval-test-interval-epochs 10 --seed 0)
echo "=== [$(date +%H:%M:%S)] sop:herd_hist (solo, after pa_distill) ==="
"$SFORA" image-end-to-end "${SOP[@]}" --objectives hist --proxy-count-per-class 0 \
  --embedding-layer-norm --hist-lr-ds 0.05 --ema-distill-weight 1.0 --ema-momentum 0.999 \
  --ema-distill-tau 0.1 --output reports/generated/sop_herd_hist.json > logs/sop_herd_hist.log 2>&1
echo "=== [$(date +%H:%M:%S)] sop:herd_hist DONE rc=$? ==="
echo "[sop2] ALL DONE"
