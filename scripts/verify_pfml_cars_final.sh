#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-riomus@100.104.199.68}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/riomus/sfora-cem-run}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/riomus/group-learning/.venv/bin/python}"

REPORT_REL="reports/generated/image_end_to_end_cars.pfml_alpha3_seed0.json"
CHECKPOINT_REL="reports/generated/image_end_to_end_cars.pfml_alpha3_seed0_final.pt"
TRAIN_PACK_REL="reports/emb/pfml_cars_alpha3_seed0_final.train.npz"
TEST_PACK_REL="reports/emb/pfml_cars_alpha3_seed0_final.test.npz"
FIELD_REL="reports/generated/pfml_cars_alpha3_seed0_final_field.json"
SCALAR_REL="reports/generated/pfml_cars_alpha3_seed0_scalar_analysis.json"

if ssh "$REMOTE_HOST" "pgrep -af '[s]fora image-end-to-end.*pfml.*alpha3_seed0'"; then
  echo "refusing final verification while the PFML training process is active" >&2
  exit 1
fi

ssh "$REMOTE_HOST" "test -s '$REMOTE_PROJECT/$REPORT_REL' && test -s '$REMOTE_PROJECT/$CHECKPOINT_REL'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/scripts' '$REMOTE_PROJECT/reports/emb'"
rsync -a \
  scripts/export_final_cars_embeddings.py \
  scripts/analyze_pfml_final_field.py \
  scripts/analyze_pfml_reference.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/"

ssh "$REMOTE_HOST" "cd '$REMOTE_PROJECT' && '$REMOTE_PYTHON' \
  scripts/analyze_pfml_reference.py \
  --report '$REPORT_REL' \
  --output '$SCALAR_REL'"

ssh "$REMOTE_HOST" "cd '$REMOTE_PROJECT' && env PYTHONPATH=src '$REMOTE_PYTHON' \
  scripts/export_final_cars_embeddings.py \
  --checkpoint '$CHECKPOINT_REL' \
  --report '$REPORT_REL' \
  --split test \
  --output '$TEST_PACK_REL' \
  --batch-size 128 \
  --num-workers 8"

ssh "$REMOTE_HOST" "cd '$REMOTE_PROJECT' && env PYTHONPATH=src '$REMOTE_PYTHON' \
  scripts/export_final_cars_embeddings.py \
  --checkpoint '$CHECKPOINT_REL' \
  --report '$REPORT_REL' \
  --split train \
  --output '$TRAIN_PACK_REL' \
  --batch-size 128 \
  --num-workers 8"

ssh "$REMOTE_HOST" "cd '$REMOTE_PROJECT' && '$REMOTE_PYTHON' \
  scripts/analyze_pfml_final_field.py \
  --checkpoint '$CHECKPOINT_REL' \
  --report '$REPORT_REL' \
  --train-pack '$TRAIN_PACK_REL' \
  --output '$FIELD_REL'"

mkdir -p reports/generated reports/emb
scp "$REMOTE_HOST:$REMOTE_PROJECT/$REPORT_REL" "reports/generated/"
scp "$REMOTE_HOST:$REMOTE_PROJECT/$FIELD_REL" "reports/generated/"
scp "$REMOTE_HOST:$REMOTE_PROJECT/$SCALAR_REL" "reports/generated/"
scp "$REMOTE_HOST:$REMOTE_PROJECT/$TRAIN_PACK_REL" "reports/emb/"
scp "$REMOTE_HOST:$REMOTE_PROJECT/$TEST_PACK_REL" "reports/emb/"

ssh "$REMOTE_HOST" "sha256sum \
  '$REMOTE_PROJECT/$REPORT_REL' \
  '$REMOTE_PROJECT/$CHECKPOINT_REL' \
  '$REMOTE_PROJECT/$TRAIN_PACK_REL' \
  '$REMOTE_PROJECT/$TEST_PACK_REL' \
  '$REMOTE_PROJECT/$FIELD_REL' \
  '$REMOTE_PROJECT/$SCALAR_REL'"
