#!/usr/bin/env bash
# Gate 1: Proxy Anchor reproduction on CUB under the repaired protocol.
# Acceptance: R@1 >= 0.685 (published Proxy Anchor ResNet-50/512: 69.9).
# Below 0.685 -> debug protocol (see plan Background table) before any GSI run.
# The artifact also carries interference diagnostics used to calibrate gsi_floor.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-researcher@gpu.example.com}"
REMOTE_DIR="${REMOTE_DIR:-/home/researcher/group-learning}"
DATASET="${DATASET:-cub}"
PROTOCOL="${PROTOCOL:-proxy-anchor-resnet50-512}"
OBJECTIVES="${OBJECTIVES:-frozen_pretrained,proxy_anchor}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-.proxy_anchor_repro_60e}"

OUTPUT_FILE="reports/generated/image_end_to_end_${DATASET}${OUTPUT_SUFFIX}.json"

rsync -az --delete \
  --exclude .venv \
  --exclude .git \
  --exclude data \
  --exclude reports/generated \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && uv sync --group dev --extra research"

# Skip only when the artifact contains every configured objective; the training
# loop writes partial JSON after each objective, so a bare existence check would
# treat an interrupted run as complete.
artifact_complete() {
  ssh "${REMOTE}" "python3 - ${REMOTE_DIR}/${OUTPUT_FILE} '${OBJECTIVES}' <<'PY'
import json, sys
path, objectives = sys.argv[1], sys.argv[2].split(\",\")
try:
    methods = json.load(open(path)).get(\"methods\", {})
except Exception:
    sys.exit(1)
missing = [o for o in objectives if not any(k.startswith(o + \"_end_to_end\") for k in methods)]
sys.exit(1 if missing else 0)
PY"
}

if [[ "${FORCE_RERUN}" != "1" ]] && artifact_complete; then
  echo "Skipping; ${OUTPUT_FILE} already contains all objectives."
else
  ssh "${REMOTE}" "cd ${REMOTE_DIR} && (uv run --group dev --extra research sfora image-end-to-end \
    --dataset-name ${DATASET} \
    --protocol ${PROTOCOL} \
    --objectives ${OBJECTIVES} \
    --train-epochs ${TRAIN_EPOCHS} \
    --checkpoint-selection-interval 0 \
    --seed ${SEED} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE}) || \
    ([ -x .venv/bin/sfora ] && .venv/bin/sfora image-end-to-end \
    --dataset-name ${DATASET} \
    --protocol ${PROTOCOL} \
    --objectives ${OBJECTIVES} \
    --train-epochs ${TRAIN_EPOCHS} \
    --checkpoint-selection-interval 0 \
    --seed ${SEED} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE})"
fi

rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" "${LOCAL_DIR}/reports/generated/"
