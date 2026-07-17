#!/usr/bin/env bash
# SOP tuning toward 0.8+ (PA+distill; my baseline run got 0.7119 at batch 120).
# Highest-impact PA-SOP levers vs that baseline: bigger batch (more in-batch negatives
# at 11k classes), sharper/less-frequent LR decay (step 20, gamma 0.25), short warmup.
# Runs ONE job at a time (large batch = memory-heavy; the box just thrashed at 2x).
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
COMMON=(--protocol proxy-anchor-resnet50-512 --dataset-name sop --objectives proxy_anchor
  --proxy-count-per-class 1 --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
  --samples-per-class 4 --warmup-epochs 1 --lr-schedule step --lr-gamma 0.25 --lr-step-epochs 20
  --train-epochs 60 --eval-test-interval-epochs 10 --seed 0)
mkdir -p reports/generated logs
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] soptune:$name ==="
  "$SFORA" image-end-to-end "${COMMON[@]}" "$@" --output "reports/generated/soptune_${name}.json" > "logs/soptune_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] soptune:$name DONE rc=$? ==="; }
# sequential, one at a time
run b180        --batch-size 180
run b256_lr2    --batch-size 256 --learning-rate 2e-4
echo "[soptune] ALL DONE"
