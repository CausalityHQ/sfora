#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
CARS=(--protocol proxy-anchor-resnet50-512 --dataset-name cars --samples-per-class 4
  --warmup-epochs 1 --lr-step-epochs 20 --train-epochs 60 --eval-test-interval-epochs 5 --seed 0)
TEACHER=reports/emb/cars_pa_teacher.pt
HERD=(--objectives hist --proxy-count-per-class 0 --embedding-layer-norm
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 --teacher-checkpoint "$TEACHER")
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] xteach:$name ==="
  "$SFORA" image-end-to-end "${CARS[@]}" "$@" --output "reports/generated/xteach_${name}.json" > "logs/xteach_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] xteach:$name DONE rc=$? ==="; }
# SEQUENTIAL: each student holds student+EMA+PA-teacher (3 nets); 2-parallel OOMs on unified RAM
run herd_pateach_w05 "${HERD[@]}" --teacher-similarity-weight 0.5
run herd_pateach_w10 "${HERD[@]}" --teacher-similarity-weight 1.0
echo "[xteach2] ALL DONE"
