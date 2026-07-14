#!/usr/bin/env bash
# SOP baselines (Proxy Anchor + plain HIST) to complete the in-harness matrix vs
# HERD on the full 11.3k-class split, so "HERD best on SOP" is provable (or falsified)
# with same-code, same-settings runs. Matches the HERD-SOP knobs (spc4, min-per-class
# 2, 40 epochs) for a fair comparison. Runs only if the HERD-SOP validation actually
# trained (guard), so we do not burn GPU on baselines if the pipeline was broken.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

# Guard: require a sane HERD-SOP validation number first.
GUARD=$(.venv/bin/python - <<'PY'
import json, os
f = "reports/generated/sop_herd_seed0.json"
if not os.path.exists(f):
    print("MISSING"); raise SystemExit
d = json.load(open(f))
def b(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "best_test_recall_at_1" and isinstance(v, (int, float)): return v
            r = b(v)
            if r is not None: return r
    return None
r = b(d)
print("OK %.4f" % r if (r or 0) > 0.5 else "LOW %s" % r)
PY
)
echo "[sop-baselines] HERD-SOP validation guard: $GUARD"
case "$GUARD" in
  OK*) : ;;
  *) echo "[sop-baselines] HERD-SOP validation not sane ($GUARD); skipping baselines."; exit 0 ;;
esac

# Proxy Anchor on SOP (proxy-count 1, spc4, no is_norm/EMA).
echo "=== [$(date +%H:%M:%S)] SOP Proxy Anchor (seed 0) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name sop --objectives proxy_anchor \
  --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1 \
  --warmup-epochs 1 --lr-step-epochs 15 --train-epochs 40 \
  --eval-test-interval-epochs 5 --seed 0 \
  --output reports/generated/sop_pa_seed0.json 2>&1 | tee logs/sop_pa_seed0.log

# Plain HIST on SOP (no is_norm, no EMA).
echo "=== [$(date +%H:%M:%S)] SOP plain HIST (seed 0) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name sop --objectives hist \
  --proxy-count-per-class 0 --samples-per-class 4 --hist-lr-ds 0.03 \
  --warmup-epochs 1 --lr-step-epochs 15 --train-epochs 40 \
  --eval-test-interval-epochs 5 --seed 0 \
  --output reports/generated/sop_hist_seed0.json 2>&1 | tee logs/sop_hist_seed0.log

echo "=== [$(date +%H:%M:%S)] SOP baselines DONE ==="
