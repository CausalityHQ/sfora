#!/usr/bin/env bash
# Distillation universality across MORE bases (bulletproof "improves any base").
# For each base, run plain vs +EMA-teacher distillation on CUB, seed 0. If distill
# lifts every base, universality is evidenced on {HIST, PA} + these three = 5 bases.
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
BASE=(--protocol proxy-anchor-resnet50-512 --dataset-name cub --proxy-count-per-class 0
  --samples-per-class 4 --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60
  --eval-test-interval-epochs 5 --seed 0)
mkdir -p reports/generated logs
run() { # name objective extra...
  local name="$1" obj="$2"; shift 2
  echo "=== [$(date +%H:%M:%S)] univ:$name ==="
  "$SFORA" image-end-to-end "${BASE[@]}" --objectives "$obj" "$@" \
    --output "reports/generated/univ_${name}.json" > "logs/univ_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] univ:$name DONE rc=$? ==="
}
DISTILL=(--ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1)
# 2-at-a-time waves (unified-memory GPU OOMs at 4). Wave = plain+distill of one base.
run supcon_plain     supcon              & run supcon_distill     supcon              "${DISTILL[@]}" & wait
run triplet_plain    batch_hard_triplet  & run triplet_distill    batch_hard_triplet  "${DISTILL[@]}" & wait
run vtriplet_plain   triplet             & run vtriplet_distill    triplet             "${DISTILL[@]}" & wait
echo "[univ] ALL DONE"
