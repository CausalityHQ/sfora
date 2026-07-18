#!/usr/bin/env bash
# SOP gap fix (codex trace): my PA-SOP runs all MISSED --embedding-layer-norm (is_norm),
# which the code's own note calls "the main architectural source of our ~2pt offset from
# published numbers". Plus SOP is under-trained (60 vs paper's 90 epochs; 11k proxies seen
# 113x less/step). Test is_norm@60ep (fast signal) then is_norm@90ep. Plain PA (reported
# setup). spc2 (all classes), batch 256, lr 2e-4, gentle decay (my finding: less decay wins).
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
BASE=(--protocol proxy-anchor-resnet50-512 --dataset-name sop --objectives proxy_anchor
  --proxy-count-per-class 1 --embedding-layer-norm --samples-per-class 2 --batch-size 256
  --learning-rate 2e-4 --warmup-epochs 1 --lr-schedule step --lr-gamma 0.5
  --eval-test-interval-epochs 10 --seed 0)
mkdir -p reports/generated logs
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] sopisn:$name ==="
  "$SFORA" image-end-to-end "${BASE[@]}" "$@" --output "reports/generated/sopisn_${name}.json" > "logs/sopisn_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] sopisn:$name DONE rc=$? ==="; }
run isnorm60 --train-epochs 60 --lr-step-epochs 20
run isnorm90 --train-epochs 90 --lr-step-epochs 30
echo "[sopisn] ALL DONE"
