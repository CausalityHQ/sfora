#!/usr/bin/env bash
# Generate the two Cars196 test-embedding sets the separation viz is missing, so
# Cars can show the same five spaces as CUB: frozen -> proxy -> HIST -> HERD -> SFORA.
#
# The Cars viz currently only has proxy/HERD/SFORA because frozen and plain-HIST
# embeddings for Cars were never produced (baseline_frozen_* and hist_only_* are
# CUB). This fills the gap:
#   1. frozen ImageNet ResNet-50 features on the Cars test set (no training)
#   2. plain HIST on Cars (no is_norm head, no EMA teacher) — same recipe as the
#      CUB hist_only baseline, just --dataset-name cars
# Both save best/only test embeddings that share the Cars test ordering, so
# make_projection.py aligns points across panels by index.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

echo "=== [$(date +%H:%M:%S)] Cars frozen ImageNet features ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name cars \
  --objectives frozen_pretrained --train-epochs 1 \
  --save-test-embeddings reports/emb/baseline_frozen_cars.npz \
  --output reports/generated/frozen_cars.json 2>&1 | tee logs/frozen_cars.log

echo "=== [$(date +%H:%M:%S)] Cars plain HIST (no is_norm, no EMA) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name cars \
  --objectives hist --proxy-count-per-class 0 \
  --samples-per-class 8 --hist-lr-ds 0.03 --lr-step-epochs 10 \
  --warmup-epochs 1 --train-epochs 60 --eval-test-interval-epochs 5 --seed 0 \
  --save-test-embeddings reports/emb/hist_cars_seed0.npz \
  --output reports/generated/hist_cars_seed0.json 2>&1 | tee logs/hist_cars_seed0.log

echo "=== [$(date +%H:%M:%S)] Cars panel embeddings DONE ==="
ls -la reports/emb/baseline_frozen_cars.npz reports/emb/hist_cars_seed0.npz
