#!/usr/bin/env bash
# Cars round-4 — chase Proxy Anchor (mean 0.8879) with single HERD. Round-2/3 landed
# at 0.8835 mean via spc4 + clean schedule; var-floor was null. This tries the levers
# NOT yet explored, on the clean60 base, seed 0. Any variant that clearly clears PA's
# seed-0 (0.8857) gets reseeded; nothing is adopted without a mean that beats PA.
# Honesty guardrail: if none robustly clears PA, we stop and report parity/PA-ahead.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
BASE=(
  --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives hist
  --proxy-count-per-class 0
  --ema-distill-weight 1.0 --ema-distill-tau 0.1
  --samples-per-class 4 --hist-lr-ds 0.05 --warmup-epochs 1 --seed 0
  --eval-test-interval-epochs 5
)
mkdir -p reports/emb reports/generated logs
run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] carstune3:$name ==="
  "$SFORA" image-end-to-end "${BASE[@]}" "$@" \
    --output "reports/generated/carstune3_${name}.json" 2>&1 | tee "logs/carstune3_${name}.log"
}

# slower EMA teacher (more stable relational targets)
run mom9995  --embedding-layer-norm --ema-momentum 0.9995 --lr-step-epochs 20 --train-epochs 60
# faster EMA teacher (fresher targets)
run mom99    --embedding-layer-norm --ema-momentum 0.99   --lr-step-epochs 20 --train-epochs 60
# no is_norm head (plain HIST + EMA) — is the LayerNorm head hurting on Cars?
run noisnorm --no-embedding-layer-norm --ema-momentum 0.999 --lr-step-epochs 20 --train-epochs 60
# longer schedule
run long80   --embedding-layer-norm --ema-momentum 0.999  --lr-step-epochs 25 --train-epochs 80

echo
echo "=== carstune3 (chase PA 0.8879 mean / 0.8857 seed0) ==="
for n in mom9995 mom99 noisnorm long80; do
  f="reports/generated/carstune3_${n}.json"
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
x=b(d); print('%-9s %.4f %s'%(sys.argv[2],x,'CLEARS PA seed0' if x and x>0.8857 else ''))
" "$f" "$n"
done
