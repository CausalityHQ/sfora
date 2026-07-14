#!/usr/bin/env bash
# SOP (Stanford Online Products) — validation run for the standard end-to-end
# benchmark (ResNet-50/512, full ~11.3k-class zero-shot split). The old SOP runs
# here were a 100-class subset on frozen DINOv2/CLIP features, NOT this. This does
# ONE HERD run to prove the pipeline scales to 60k images / 11.3k classes and to
# get a first honest R@1 before committing the full HERD+PA+HIST multi-seed headline.
#
# SOP has ~5 images/class, so samples-per-class 4 (vs 8 on CUB) and min-per-class 2.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

echo "=== [$(date +%H:%M:%S)] SOP HERD validation (seed 0) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name sop \
  --objectives hist --proxy-count-per-class 0 --embedding-layer-norm \
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
  --samples-per-class 4 --min-per-class 2 --hist-lr-ds 0.03 \
  --warmup-epochs 1 --lr-step-epochs 15 --train-epochs 40 \
  --eval-test-interval-epochs 5 --seed 0 \
  --save-test-embeddings reports/emb/sop_herd_seed0.npz \
  --output reports/generated/sop_herd_seed0.json 2>&1 | tee logs/sop_herd_seed0.log

echo "=== [$(date +%H:%M:%S)] SOP HERD validation DONE ==="
.venv/bin/python -c "
import json
d=json.load(open('reports/generated/sop_herd_seed0.json'))
def f(o,k):
    if isinstance(o,dict):
        for kk,v in o.items():
            if kk==k and isinstance(v,(int,float)): return v
            r=f(v,k)
            if r is not None: return r
    return None
print('SOP HERD seed0 best R@1 =', f(d,'best_test_recall_at_1'), '(reported PA/HIST SOP ~0.79)')
"
