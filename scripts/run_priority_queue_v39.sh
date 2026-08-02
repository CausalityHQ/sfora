#!/usr/bin/env bash
# Candidate 174: execute only the blinded, training-only Gate-1 diagnostic.
# Stop for registered adjudication; this script never trains OAPF.
set -euo pipefail

PROJECT="${PROJECT:-/home/riomus/group-learning}"
log_dir=/home/riomus/experiment-logs/reference-matrix
output="$PROJECT/reports/generated/oapf_gate1_seed0.json"
cd "$PROJECT"

printf '%s priority queue v39 start: OAPF blinded Gate 1 only\n' \
  "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"
if [[ -s "$output" ]]; then
  printf '%s v39 skip: OAPF artifact already exists\n' \
    "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"
  exit 0
fi

.venv/bin/python scripts/diagnose_oapf.py \
  --checkpoint reports/checkpoints/arcg_inshop_pa_epoch10_seed0.pt \
  --training-report reports/generated/arcg_inshop_pa_epoch10_seed0.json \
  --dataset-root /home/riomus/datasets/inshop \
  --output "$output" \
  --view-metadata-output reports/generated/oapf_gate1_seed0.views.json \
  --embedding-cache reports/generated/oapf_gate1_seed0.embeddings.npz \
  --batch-size 128 \
  --distance-chunk-size 512 \
  --num-workers 8 \
  >"$log_dir/oapf_gate1_seed0.log" 2>&1
printf '%s priority queue v39 DONE: OAPF Gate 1 artifact=%s\n' \
  "$(date --iso-8601=seconds)" "$output" >>"$log_dir/controller.log"
