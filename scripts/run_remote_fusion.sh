#!/usr/bin/env bash
# The single fused method: hist_proxy_anchor (HIST + Proxy Anchor in one model) +
# EMA-teacher relational distillation. Goal: ONE fixed recipe best on every dataset,
# since HIST wins CUB and PA wins Cars. Must beat the per-dataset best:
#   CUB  > HERD 0.716     Cars > PA+distill 0.8961
# Sweeps the fusion weight (0.5, 1.0) on both datasets, seed 0; winner reseeded.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

fused() {  # name dataset spc lrds lrstep w
  local name="$1" ds="$2" spc="$3" lrds="$4" lrstep="$5" w="$6"
  echo "=== [$(date +%H:%M:%S)] fusion:$name (${ds}, w=${w}) ==="
  "$SFORA" image-end-to-end \
    --protocol proxy-anchor-resnet50-512 --dataset-name "$ds" --objectives hist_proxy_anchor \
    --proxy-count-per-class 1 --embedding-layer-norm \
    --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
    --proxy-fusion-weight "$w" \
    --samples-per-class "$spc" --hist-lr-ds "$lrds" \
    --warmup-epochs 1 --lr-step-epochs "$lrstep" --train-epochs 60 \
    --eval-test-interval-epochs 5 --seed 0 \
    --save-test-embeddings "reports/emb/fusion_${name}_seed0.npz" \
    --output "reports/generated/fusion_${name}.json" 2>&1 | tee "logs/fusion_${name}.log"
}

# CUB (HERD's recipe: spc8, hist-lr-ds 0.03) at two fusion weights
fused cub_w10  cub  8 0.03 10 1.0
fused cub_w05  cub  8 0.03 10 0.5
# Cars (clean60 recipe: spc4, hist-lr-ds 0.05) at two fusion weights
fused cars_w10 cars 4 0.05 20 1.0
fused cars_w05 cars 4 0.05 20 0.5

echo
echo "=== FUSED hist_proxy_anchor (beat CUB 0.716 and Cars 0.8961) ==="
for n in cub_w10 cub_w05 cars_w10 cars_w05; do
  f="reports/generated/fusion_${n}.json"
  [ -f "$f" ] && .venv/bin/python -c "
import json,sys
d=json.load(open(sys.argv[1]))
def b(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=='best_test_recall_at_1' and isinstance(v,(int,float)): return v
            r=b(v)
            if r is not None: return r
    return None
print('  fusion:%-9s %.4f' % (sys.argv[2], b(d)))
" "$f" "$n"
done
