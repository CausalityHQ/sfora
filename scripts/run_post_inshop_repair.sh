#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

REPORT=reports/generated/inshop_official_pa_repaired_seed0.json
CHECKPOINT=reports/checkpoints/inshop_official_pa_repaired_seed0.pt

while [[ ! -f "${REPORT}" || ! -f "${CHECKPOINT}" ]]; do
  if ! pgrep -f '[r]un_inshop_after_sop_repair.sh' >/dev/null; then
    echo "In-Shop repair queue exited without required artifacts" >&2
    exit 1
  fi
  sleep 300
done

.venv/bin/python /tmp/export_final_inshop_embeddings.py \
  --checkpoint "${CHECKPOINT}" \
  --report "${REPORT}" \
  --dataset-root /home/riomus/datasets/inshop \
  --query-output reports/emb/inshop_official_pa_repaired_seed0_query_final.npz \
  --gallery-output reports/emb/inshop_official_pa_repaired_seed0_gallery_final.npz \
  --retrieval-output reports/generated/inshop_official_pa_repaired_seed0_final_retrieval.json
