#!/usr/bin/env bash
# SOP fix: samples-per-class=2 sees ALL 11317 classes (spc4 excluded 36%, spc3 19% —
# the real gap; PA proxies for excluded classes never trained). Batch 256, lr 2e-4,
# gentler decay (gamma 0.5) so late epochs keep improving. PA+distill and plain PA.
set -uo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
COMMON=(--protocol proxy-anchor-resnet50-512 --dataset-name sop --objectives proxy_anchor
  --proxy-count-per-class 1 --samples-per-class 2 --batch-size 256 --learning-rate 2e-4
  --warmup-epochs 1 --lr-schedule step --lr-gamma 0.5 --lr-step-epochs 20
  --train-epochs 60 --eval-test-interval-epochs 10 --seed 0)
mkdir -p reports/generated logs
run() { local name="$1"; shift; echo "=== [$(date +%H:%M:%S)] sopfix:$name ==="
  "$SFORA" image-end-to-end "${COMMON[@]}" "$@" --output "reports/generated/sopfix_${name}.json" > "logs/sopfix_${name}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] sopfix:$name DONE rc=$? ==="; }
run spc2_padistill --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
run spc2_pa_plain
echo "[sopfix] ALL DONE"
