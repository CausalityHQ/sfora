#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

while pgrep -f '[s]fora image-end-to-end.*sop_official_pa_seed0' >/dev/null; do
  sleep 300
done

test -f reports/generated/sop_official_pa_seed0.json
test -f reports/checkpoints/sop_official_pa_seed0.pt

.venv/bin/python /tmp/export_final_sop_train_embeddings_49e9fbc.py \
  --checkpoint reports/checkpoints/sop_official_pa_seed0.pt \
  --report reports/generated/sop_official_pa_seed0.json \
  --output reports/emb/sop_official_pa_seed0_train_final.npz \
  > logs/sop_official_final_export.log 2>&1

.venv/bin/python /tmp/analyze_sop_official_structure_aedffa8.py \
  --embeddings reports/emb/sop_official_pa_seed0_train_final.npz \
  --metadata /tmp/Ebay_train.txt \
  --output reports/generated/sop_official_structure_seed0_final.json \
  > logs/sop_official_structure_seed0_final.log 2>&1
