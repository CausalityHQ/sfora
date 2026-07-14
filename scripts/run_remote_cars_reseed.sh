#!/usr/bin/env bash
# Cars round-3: fair mean-over-seeds comparison to settle "is single HERD > PA?".
#
# Round-2 winner clean60 (HERD) = 0.8844 @ seed 0; PA (our repro) = 0.8857 @ seed 0.
# Both are single best-over-training runs 0.13 pt apart — inside seed noise. To make
# a defensible "HERD beats PA by a little" claim we reseed BOTH methods' best recipe
# (seeds 1, 2; seed 0 already exists) and compare the MEANS with error bars. HERD is
# adopted as the Cars winner only if its mean clears PA's mean. No seed-picking.
#
# The var-floor base change was a null/negative result (vfloor2 0.6837 < control
# 0.6917), so HERD uses the plain clean60 recipe. HERD runs save embeddings so the
# better recipe can also refresh the Cars SFORA pack.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

# HERD clean60 recipe (is_norm + EMA, spc4, clean 60-ep schedule).
herd() {
  local s="$1"
  echo "=== [$(date +%H:%M:%S)] HERD clean60 seed $s ==="
  "$SFORA" image-end-to-end \
    --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives hist \
    --proxy-count-per-class 0 --embedding-layer-norm \
    --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
    --samples-per-class 4 --hist-lr-ds 0.05 \
    --warmup-epochs 1 --lr-step-epochs 20 --train-epochs 60 \
    --eval-test-interval-epochs 5 --seed "$s" \
    --save-test-embeddings "reports/emb/cars_herd_clean60_seed$s.npz" \
    --output "reports/generated/cars_herd_clean60_seed$s.json" \
    2>&1 | tee "logs/cars_herd_clean60_seed$s.log"
}

# Proxy Anchor Cars recipe (faithful to our 0.8857 run: proxy-count 1, spc4, no EMA/isnorm).
pa() {
  local s="$1"
  echo "=== [$(date +%H:%M:%S)] Proxy Anchor seed $s ==="
  "$SFORA" image-end-to-end \
    --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives proxy_anchor \
    --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1 \
    --warmup-epochs 5 --lr-step-epochs 10 --train-epochs 60 \
    --eval-test-interval-epochs 5 --seed "$s" \
    --output "reports/generated/cars_pa_seed$s.json" \
    2>&1 | tee "logs/cars_pa_seed$s.log"
}

herd 1
herd 2
pa 1
pa 2

echo
echo "=== Cars mean-over-seeds: HERD vs Proxy Anchor ==="
.venv/bin/python - <<'PY'
import json, os
def best(f):
    if not os.path.exists(f): return None
    d=json.load(open(f))
    def b(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if k=="best_test_recall_at_1" and isinstance(v,(int,float)): return v
                r=b(v)
                if r is not None: return r
        return None
    return b(d)
herd=[0.8844]  # seed 0 from carstune2 clean60
for s in (1,2):
    r=best(f"reports/generated/cars_herd_clean60_seed{s}.json")
    if r: herd.append(r)
pa=[0.8857]  # seed 0 from cars_pa
for s in (1,2):
    r=best(f"reports/generated/cars_pa_seed{s}.json")
    if r: pa.append(r)
import statistics as st
hm=st.mean(herd); pm=st.mean(pa)
def sd(x): return st.pstdev(x) if len(x)>1 else 0.0
print(f"HERD clean60: seeds={[round(x,4) for x in herd]} mean={hm:.4f} ±{sd(herd):.4f}")
print(f"Proxy Anchor: seeds={[round(x,4) for x in pa]} mean={pm:.4f} ±{sd(pa):.4f}")
print(f"=> HERD {'BEATS' if hm>pm else 'does NOT beat'} PA on Cars by {hm-pm:+.4f} (mean)")
PY
