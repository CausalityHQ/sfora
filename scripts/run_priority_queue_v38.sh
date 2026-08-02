#!/usr/bin/env bash
# Corrected Cars RS@k reference (implementation commit 5e79bc1), followed only
# after a complete artifact by candidate-174's blinded training-only Gate 1.
set -uo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
cd "${PROJECT}" || exit 2

# Import the digest-validated runner without executing v24's historical body.
# shellcheck disable=SC1090
source <(sed '/^log "priority queue v24 start"/,$d' scripts/run_priority_queue_v24.sh)

cars_artifact="$PROJECT/reports/generated/image_end_to_end_cars.rsatk.recall_at_k_surrogate.cars.official-ed05202.0c02a7e6e2a0_seed0.json"
oapf_output="$PROJECT/reports/generated/oapf_gate1_seed0.json"

log "priority queue v38 start: corrected RS@k retrieved-count cap commit=5e79bc1"
run_arm cars rsatk recall_at_k_surrogate auto 0

if [[ ! -s "$cars_artifact" ]]; then
  log "v38 REFUSED OAPF: corrected Cars RS@k exited without registered artifact"
  exit 1
fi
if [[ -s "$oapf_output" ]]; then
  log "v38 skip: OAPF Gate-1 artifact already exists"
  exit 0
fi

log "START inshop/OAPF/Gate1 training-only diagnostic after corrected Cars artifact"
.venv/bin/python scripts/diagnose_oapf.py \
  --checkpoint reports/checkpoints/arcg_inshop_pa_epoch10_seed0.pt \
  --training-report reports/generated/arcg_inshop_pa_epoch10_seed0.json \
  --dataset-root /home/riomus/datasets/inshop \
  --output "$oapf_output" \
  --view-metadata-output reports/generated/oapf_gate1_seed0.views.json \
  --embedding-cache reports/generated/oapf_gate1_seed0.embeddings.npz \
  --batch-size 128 \
  --distance-chunk-size 512 \
  --num-workers 8 \
  >/home/riomus/experiment-logs/reference-matrix/oapf_gate1_seed0.log 2>&1
log "DONE inshop/OAPF/Gate1 artifact=$oapf_output"
