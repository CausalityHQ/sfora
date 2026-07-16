#!/usr/bin/env bash
# Overnight autonomous chain. Runs AFTER SOP finishes (waits for sop_herd_hist.json).
# A) regenerate 3 HERD-Cars train+test packs; B) projection-generalization sweep on Cars
#    (does the uncentered 100%-at-reduced-dim finding hold beyond CUB?);
# C) reseed universality at seed 1 (de-risk the single-seed distillation result).
# Not set -e: one failure must not abort the rest. 2-job GPU max (unified-mem OOM).
cd "$(dirname "$0")/.."
SFORA=.venv/bin/sfora
log(){ echo "=== [$(date +%m-%d\ %H:%M:%S)] $* ==="; }
mkdir -p reports/generated reports/emb logs

log "overnight: waiting for SOP (sop_herd_hist.json)"
for i in $(seq 1 600); do [ -f reports/generated/sop_herd_hist.json ] && break; sleep 60; done
log "overnight: SOP done, starting phase A (Cars packs)"

CARS=(--protocol proxy-anchor-resnet50-512 --dataset-name cars --objectives hist
  --proxy-count-per-class 0 --embedding-layer-norm --ema-distill-weight 1.0
  --ema-momentum 0.999 --ema-distill-tau 0.1 --samples-per-class 4 --hist-lr-ds 0.05
  --warmup-epochs 1 --lr-step-epochs 20 --train-epochs 60 --eval-test-interval-epochs 5)
cars_seed(){ local s="$1"
  "$SFORA" image-end-to-end "${CARS[@]}" --seed "$s" \
    --save-train-embeddings "reports/emb/herd_cars_tt_seed${s}.train.npz" \
    --save-test-embeddings  "reports/emb/herd_cars_tt_seed${s}.test.npz" \
    --output "reports/generated/herd_cars_tt_seed${s}.json" > "logs/herd_cars_tt_seed${s}.log" 2>&1
  log "cars pack seed$s rc=$?"; }
cars_seed 0 & cars_seed 1 & wait
cars_seed 2 & wait

log "phase B: projection-generalization on Cars"
.venv/bin/python scripts/explore_trainclean_projection.py \
  --train 'reports/emb/herd_cars_tt_seed*.train.npz' \
  --test  'reports/emb/herd_cars_tt_seed*.test.npz' > logs/projgen_cars.log 2>&1
log "projgen Cars rc=$? -> logs/projgen_cars.log"

log "phase C: universality reseed (seed 1)"
UB=(--protocol proxy-anchor-resnet50-512 --dataset-name cub --proxy-count-per-class 0
  --samples-per-class 4 --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60
  --eval-test-interval-epochs 5 --seed 1)
DIS=(--ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1)
ur(){ local name="$1" obj="$2"; shift 2
  "$SFORA" image-end-to-end "${UB[@]}" --objectives "$obj" "$@" \
    --output "reports/generated/univ_${name}_s1.json" > "logs/univ_${name}_s1.log" 2>&1; log "univ $name s1 rc=$?"; }
ur supcon_plain    supcon             & ur supcon_distill    supcon             "${DIS[@]}" & wait
ur triplet_plain   batch_hard_triplet & ur triplet_distill   batch_hard_triplet "${DIS[@]}" & wait
ur vtriplet_plain  triplet            & ur vtriplet_distill  triplet            "${DIS[@]}" & wait

log "[overnight] ALL DONE"
