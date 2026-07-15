#!/usr/bin/env bash
# Confirm whether "PA + EMA-teacher relational distillation" is the single method
# best on every dataset. On Cars seed0 it already beats PA (0.8944 vs 0.8879 mean).
# This adds:
#   - CUB: PA+distill vs plain PA (in-harness) and vs HERD (0.716) — is it best on CUB too?
#   - Cars: reseed PA+distill (seeds 1,2) so its MEAN can be compared to PA's 0.8879.
# Adopt as the headline method only if it beats every baseline's mean per dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
mkdir -p reports/emb reports/generated logs

# --- CUB: plain PA (in-harness baseline for this recipe) ---
echo "=== [$(date +%H:%M:%S)] CUB plain PA (seed 0) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name cub --objectives proxy_anchor \
  --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1 \
  --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60 \
  --eval-test-interval-epochs 5 --seed 0 \
  --output reports/generated/cub_pa_seed0.json 2>&1 | tee logs/cub_pa_seed0.log

# --- CUB: PA + EMA-distill (the candidate unified method) ---
echo "=== [$(date +%H:%M:%S)] CUB PA + EMA-distill (seed 0) ==="
"$SFORA" image-end-to-end \
  --protocol proxy-anchor-resnet50-512 --dataset-name cub --objectives proxy_anchor \
  --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1 \
  --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
  --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60 \
  --eval-test-interval-epochs 5 --seed 0 \
  --save-test-embeddings reports/emb/cub_paema_seed0.npz \
  --output reports/generated/cub_paema_seed0.json 2>&1 | tee logs/cub_paema_seed0.log

# --- Cars: reseed the winning PA+distill (w10) at seeds 1 and 2 ---
for s in 1 2; do
  echo "=== [$(date +%H:%M:%S)] Cars PA + EMA-distill seed $s ==="
  "$SFORA" image-end-to-end \
    --protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives proxy_anchor \
    --proxy-count-per-class 1 --samples-per-class 4 --hist-lr-ds 0.1 \
    --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
    --warmup-epochs 5 --lr-step-epochs 10 --train-epochs 60 \
    --eval-test-interval-epochs 5 --seed "$s" \
    --save-test-embeddings "reports/emb/cars_paema_w10_seed$s.npz" \
    --output "reports/generated/cars_paema_w10_seed$s.json" 2>&1 | tee "logs/cars_paema_w10_seed$s.log"
done

echo
echo "=== CONFIRM SUMMARY ==="
.venv/bin/python - <<'PY'
import json, os, statistics as st
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
print("CUB: plain PA =", best("reports/generated/cub_pa_seed0.json"),
      "| PA+distill =", best("reports/generated/cub_paema_seed0.json"), "| HERD ref = 0.716")
cars=[0.8944]+[x for s in (1,2) if (x:=best(f"reports/generated/cars_paema_w10_seed{s}.json"))]
print("Cars PA+distill seeds:", [round(x,4) for x in cars], "mean=%.4f"%st.mean(cars), "| PA mean=0.8879")
PY
