#!/usr/bin/env bash
# Cars HERD, round-3 — the HIST-internal levers not yet tried, a genuine last attempt
# to get the HIST-based HERD past Proxy Anchor (mean 0.8879) on Cars. Prior rounds
# tuned samples-per-class (4 wins), schedule (clean60), tau, ema weight/momentum, and
# the variance floor — all plateaued at ~0.8835. This varies the hypergraph CE weight
# (lambda_s), the incidence sharpness (alpha), and the HGNN width (hidden), on the
# clean60 base, seed 0. Any variant that clears PA's seed-0 (0.8857) gets reseeded;
# adopt only if its mean beats PA's. If none clears, HIST is genuinely the weaker base.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
BASE=(
  --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives hist
  --proxy-count-per-class 0 --embedding-layer-norm
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1
  --samples-per-class 4 --hist-lr-ds 0.05 --warmup-epochs 1 --lr-step-epochs 20
  --train-epochs 60 --eval-test-interval-epochs 5 --seed 0
)
mkdir -p reports/generated logs
run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] carstune_hist:$name ==="
  "$SFORA" image-end-to-end "${BASE[@]}" "$@" \
    --output "reports/generated/carstune_hist_${name}.json" 2>&1 | tee "logs/carstune_hist_${name}.log"
}

run lambda2   --hist-lambda-s 2.0     # stronger hypergraph cross-entropy
run lambda05  --hist-lambda-s 0.5     # weaker
run alpha15   --hist-alpha 1.5        # sharper soft incidence (default 0.9)
run hidden1024 --hist-hidden 1024     # wider HGNN (default 512)

echo
echo "=== carstune_hist (clear PA 0.8857 seed0 / 0.8879 mean) ==="
for n in lambda2 lambda05 alpha15 hidden1024; do
  f="reports/generated/carstune_hist_${n}.json"
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
x=b(d); print('  %-10s %.4f%s'%(sys.argv[2],x,'  CLEARS PA' if x and x>0.8857 else ''))
" "$f" "$n"
done
echo "(clean60 baseline was 0.8844 seed0 / 0.8835 mean; PA 0.8879 mean.)"
