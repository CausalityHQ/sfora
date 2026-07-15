#!/usr/bin/env bash
# Cars: add HERD's core innovation — the EMA-teacher RELATIONAL distillation — on top
# of the base that is actually strong on Cars (Proxy Anchor), instead of forcing the
# HIST base past PA. The distillation is additive and ungated by objective, so this is
# "PA + our relational self-distillation". If it beats plain PA (mean 0.8879) it shows
# our procedure improvement lifts ANY base above its plain version — the honest,
# general claim. Winner reseeded before adoption; embeddings saved for the ensemble.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
# PA's own Cars recipe (proxy-count 1, spc4, hist-lr-ds 0.1, warmup 5) that reaches 0.8879.
PA=(
  --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives proxy_anchor
  --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1
  --warmup-epochs 5 --lr-step-epochs 10 --train-epochs 60
  --eval-test-interval-epochs 5 --seed 0
)
mkdir -p reports/emb reports/generated logs
run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] paema:$name ==="
  "$SFORA" image-end-to-end "${PA[@]}" "$@" \
    --save-test-embeddings "reports/emb/cars_paema_${name}_seed0.npz" \
    --output "reports/generated/paema_${name}.json" 2>&1 | tee "logs/paema_${name}.log"
}

# PA + EMA-teacher relational distillation, full weight (as HERD uses on HIST).
run w10       --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
# gentler distillation, in case full weight perturbs the PA proxies.
run w05       --ema-distill-weight 0.5 --ema-momentum 0.999 --ema-distill-tau 0.1
# also add the is_norm head (the full HERD-style augmentation on a PA base).
run w10_isnorm --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 --embedding-layer-norm

echo
echo "=== PA + EMA-distill on Cars (plain PA mean=0.8879, seed0=0.8857) ==="
for n in w10 w05 w10_isnorm; do
  f="reports/generated/paema_${n}.json"
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
x=b(d); print('  paema:%-11s %.4f %s'%(sys.argv[2],x,'BEATS PA seed0' if x and x>0.8857 else ''))
" "$f" "$n"
done
