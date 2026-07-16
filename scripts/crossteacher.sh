#!/usr/bin/env bash
# Cross-teacher distillation (item-2 long-shot): can a *trained PA teacher* pull a
# HIST-based student past PA on Cars, where HIST's own EMA self-teacher could not?
# 1) train a PA teacher on Cars, save weights. 2) train HERD (HIST+is_norm+EMA) students
# that ALSO distill the frozen PA teacher's pairwise similarities (teacher_similarity_weight).
# Compare to plain HIST 0.871, HERD 0.884, PA 0.888.
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
CARS=(--protocol proxy-anchor-resnet50-512 --dataset-name cars --samples-per-class 4
  --warmup-epochs 1 --lr-step-epochs 20 --train-epochs 60 --eval-test-interval-epochs 5 --seed 0)
TEACHER=reports/emb/cars_pa_teacher.pt
mkdir -p reports/generated reports/emb logs
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] xteach:$name ==="
  "$SFORA" image-end-to-end "${CARS[@]}" "$@" --output "reports/generated/xteach_${name}.json" > "logs/xteach_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] xteach:$name DONE rc=$? ==="; }

# 1. PA teacher (must finish before students) — save its weights
run pa_teacher --objectives proxy_anchor --proxy-count-per-class 1 --save-model-path "$TEACHER"

# 2. HERD students + frozen-PA-teacher relational distillation, two teacher weights (2-job wave)
HERD=(--objectives hist --proxy-count-per-class 0 --embedding-layer-norm
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
  --teacher-checkpoint "$TEACHER")
run herd_pateach_w05 "${HERD[@]}" --teacher-similarity-weight 0.5 &
run herd_pateach_w10 "${HERD[@]}" --teacher-similarity-weight 1.0 &
wait
echo "[xteach] ALL DONE"
