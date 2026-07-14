#!/usr/bin/env bash
# HIST base-loss ablation on CUB: does relaxing the class-variance floor help?
#
# Our HIST is a FAITHFUL port of ljin0429/HIST — relu6(log_vars) floors every
# class variance at 1 (exp(0)), so a class can never cluster tighter than unit
# variance on the L2-normalised sphere. That is plausibly a ceiling for fine-
# grained retrieval. hist_var_floor generalises relu6's lower clamp (0.0 = the
# faithful default); a negative floor lets classes tighten. We test it on the full
# HERD recipe (CUB), seed 0, against the faithful control. If a floor beats the
# control robustly (reseed before adopting), it lifts HERD on EVERY dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
COMMON=(
  --protocol proxy-anchor-resnet50-512 --dataset-name cub --objectives hist
  --proxy-count-per-class 0 --embedding-layer-norm
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
  --samples-per-class 8 --hist-lr-ds 0.03
  --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60
  --eval-test-interval-epochs 5 --seed 0
)
mkdir -p reports/emb reports/generated logs

run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] varfloor:$name ==="
  "$SFORA" image-end-to-end "${COMMON[@]}" "$@" \
    --output "reports/generated/varfloor_${name}.json" \
    2>&1 | tee "logs/varfloor_${name}.log"
}

run control  --hist-var-floor 0.0     # faithful relu6 (variance >= 1) — reference
run vfloor2  --hist-var-floor -2.0    # variance >= ~0.135
run vfloor3  --hist-var-floor -3.0    # variance >= ~0.050
run vfloor3_tau64 --hist-var-floor -3.0 --hist-tau 64.0  # tighter + sharper softmax

echo
echo "=== HIST var-floor ablation (CUB, HERD recipe, seed 0) ==="
for name in control vfloor2 vfloor3 vfloor3_tau64; do
  f="reports/generated/varfloor_${name}.json"
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
print('%-14s best R@1 = %.4f' % (sys.argv[2], b(d)))
" "$f" "$name"
done
echo "(control = faithful HERD; a floor that clearly beats it is worth reseeding.)"
