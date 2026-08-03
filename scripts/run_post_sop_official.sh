#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

while pgrep -f '[s]fora image-end-to-end.*sop_official_bninc_pa_seed0' >/dev/null; do
  sleep 300
done

test -f reports/generated/sop_official_bninc_pa_seed0.json
test -f reports/checkpoints/sop_official_bninc_pa_seed0.pt

.venv/bin/python /tmp/export_final_sop_embeddings.py \
  --checkpoint reports/checkpoints/sop_official_bninc_pa_seed0.pt \
  --report reports/generated/sop_official_bninc_pa_seed0.json \
  --split train \
  --output reports/emb/sop_official_bninc_pa_seed0_train_final.npz \
  > logs/sop_official_final_export.log 2>&1

.venv/bin/python /tmp/export_final_sop_embeddings.py \
  --checkpoint reports/checkpoints/sop_official_bninc_pa_seed0.pt \
  --report reports/generated/sop_official_bninc_pa_seed0.json \
  --split test \
  --output reports/emb/sop_official_bninc_pa_seed0_test_final.npz \
  > logs/sop_official_final_test_export.log 2>&1

.venv/bin/python /tmp/analyze_sop_official_structure.py \
  --embeddings reports/emb/sop_official_bninc_pa_seed0_train_final.npz \
  --metadata /tmp/Ebay_train.txt \
  --output reports/generated/sop_official_structure_seed0_final.json \
  > logs/sop_official_structure_seed0_final.log 2>&1

.venv/bin/python /tmp/analyze_sop_official_structure.py \
  --embeddings reports/emb/sop_official_bninc_pa_seed0_test_final.npz \
  --metadata /tmp/Ebay_test.txt \
  --output reports/generated/sop_official_test_verification_seed0_final.json \
  > logs/sop_official_test_verification_seed0_final.log 2>&1

.venv/bin/python /tmp/analyze_sop_proxy_clock.py \
  --checkpoint reports/checkpoints/sop_official_bninc_pa_seed0.pt \
  --embeddings reports/emb/sop_official_bninc_pa_seed0_train_final.npz \
  --output reports/generated/sop_official_proxy_clock_seed0_final.json \
  > logs/sop_official_proxy_clock_seed0_final.log 2>&1
