#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

REPORT=reports/generated/inshop_official_pa_repaired_seed0.json
CHECKPOINT=reports/checkpoints/inshop_official_pa_repaired_seed0.pt

while pgrep -f '[s]fora image-end-to-end.*inshop_official_pa_repaired_seed0' >/dev/null; do
  sleep 300
done

test -f "${REPORT}"
test -f "${CHECKPOINT}"

.venv/bin/python /tmp/export_final_inshop_embeddings.py \
  --checkpoint "${CHECKPOINT}" \
  --report "${REPORT}" \
  --dataset-root /home/riomus/datasets/inshop \
  --query-output reports/emb/inshop_official_pa_repaired_seed0_query_final.npz \
  --gallery-output reports/emb/inshop_official_pa_repaired_seed0_gallery_final.npz \
  --retrieval-output reports/generated/inshop_official_pa_repaired_seed0_final_retrieval.json \
  > logs/inshop_official_pa_repaired_seed0_final_export.log 2>&1
