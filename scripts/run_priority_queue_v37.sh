#!/usr/bin/env bash
set -euo pipefail

# Candidate 174 is a training-only provenance diagnostic, not an OAPF training
# run.  Wait for the preregistered Cars RS@k reference to release the GPU, then
# execute exactly the digest-bound diagnostic committed before its data exist.

root=/home/riomus/group-learning
log_dir=/home/riomus/experiment-logs/reference-matrix
cars_artifact="$root/reports/generated/image_end_to_end_cars.rsatk.recall_at_k_surrogate.cars.official-ed05202.0c02a7e6e2a0_seed0.json"
output="$root/reports/generated/oapf_gate1_seed0.json"
metadata="$root/reports/generated/oapf_gate1_seed0.views.json"
cache="$root/reports/generated/oapf_gate1_seed0.embeddings.npz"

mkdir -p "$log_dir"
printf '%s priority queue v37 armed: wait for Cars RS@k, then OAPF Gate 1 only\n' \
  "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"

while pgrep -f '[s]fora image-end-to-end --dataset-name cars --objectives recall_at_k_surrogate' \
  >/dev/null; do
  sleep 300
done

if [[ ! -s "$cars_artifact" ]]; then
  printf '%s v37 REFUSED: Cars RS@k exited without its registered artifact\n' \
    "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"
  exit 1
fi

if [[ -s "$output" ]]; then
  printf '%s v37 skip: OAPF diagnostic artifact already exists\n' \
    "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"
  exit 0
fi

printf '%s START inshop/OAPF/Gate1 training-only diagnostic\n' \
  "$(date --iso-8601=seconds)" >>"$log_dir/controller.log"
cd "$root"
.venv/bin/python scripts/diagnose_oapf.py \
  --checkpoint reports/checkpoints/arcg_inshop_pa_epoch10_seed0.pt \
  --training-report reports/generated/arcg_inshop_pa_epoch10_seed0.json \
  --dataset-root /home/riomus/datasets/inshop \
  --output "$output" \
  --view-metadata-output "$metadata" \
  --embedding-cache "$cache" \
  --batch-size 128 \
  --distance-chunk-size 512 \
  --num-workers 8 \
  >"$log_dir/oapf_gate1_seed0.log" 2>&1
printf '%s DONE inshop/OAPF/Gate1 artifact=%s\n' \
  "$(date --iso-8601=seconds)" "$output" >>"$log_dir/controller.log"
