#!/usr/bin/env bash
# SOP (Stanford Online Products) — the biggest gap in the beyond-CUB story (undertuned
# at 0.698 vs reported ~0.796). Base-adaptive check at 11k-class scale: is PA or HIST the
# stronger SOP base? Both + distillation, samples-per-class 3 (SOP has ~6 images/class,
# so 4 excludes many). This is a FIRST improved attempt, not a fully-tuned SOP recipe.
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
SOP=(--protocol proxy-anchor-resnet50-512 --dataset-name sop --samples-per-class 3
  --warmup-epochs 5 --lr-step-epochs 10 --train-epochs 60 --eval-test-interval-epochs 10 --seed 0)
mkdir -p reports/generated logs
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] sop:$name ==="
  "$SFORA" image-end-to-end "${SOP[@]}" "$@" --output "reports/generated/sop_${name}.json" > "logs/sop_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] sop:$name DONE rc=$? ==="; }
DISTILL=(--ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1)
run pa_distill  --objectives proxy_anchor --proxy-count-per-class 1 "${DISTILL[@]}" &
run herd_hist   --objectives hist --proxy-count-per-class 0 --embedding-layer-norm --hist-lr-ds 0.05 "${DISTILL[@]}" &
wait
echo "[sop] ALL DONE"
