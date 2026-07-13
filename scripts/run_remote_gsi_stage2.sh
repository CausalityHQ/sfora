#!/usr/bin/env bash
# Stage 2 (run ONLY after Gate 1 passes: proxy_anchor CUB R@1 >= 0.685).
#
# Gate 2: PFML reproduction (accept >= 0.72; paper: 73.4 at 200 epochs).
#   PROTOCOL=pfml-resnet50-512 OBJECTIVES=frozen_pretrained,pfml OUTPUT_SUFFIX=.pfml_repro_100e ./scripts/run_remote_gsi_stage2.sh
# Gate 3: paired GSI arm on the SAME seed as its base run (accept mean >= +0.5 over paired seeds):
#   PROTOCOL=proxy-anchor-resnet50-512 OBJECTIVES=proxy_anchor,proxy_anchor_gsi OUTPUT_SUFFIX=.pa_gsi_60e ./scripts/run_remote_gsi_stage2.sh
#   PROTOCOL=pfml-resnet50-512 OBJECTIVES=pfml,pfml_gsi OUTPUT_SUFFIX=.pfml_gsi_100e ./scripts/run_remote_gsi_stage2.sh
# BGSI discriminator:
#   PROTOCOL=proxy-anchor-resnet50-512 OBJECTIVES=proxy_anchor,proxy_anchor_bgsi TRAIN_EPOCHS=60 BGSI_WEIGHT=0.3 BGSI_FLOOR=0.0 OUTPUT_SUFFIX=.pa_bgsi_pair_w03_60e ./scripts/run_remote_gsi_stage2.sh
# Ablation controls (pre-registered falsifier c): GSI_AXIS_MODE=random / GSI_AXIS_MODE=global.
# Calibrate GSI_FLOOR from the Gate-1 artifact's interference diagnostics before Gate 3.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-researcher@gpu.example.com}"
REMOTE_DIR="${REMOTE_DIR:-/home/researcher/group-learning}"
DATASET="${DATASET:-cub}"
PROTOCOL="${PROTOCOL:-pfml-resnet50-512}"
OBJECTIVES="${OBJECTIVES:-frozen_pretrained,pfml}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GSI_WEIGHT="${GSI_WEIGHT:-}"
GSI_FLOOR="${GSI_FLOOR:-}"
GSI_AXIS_MODE="${GSI_AXIS_MODE:-}"
BGSI_WEIGHT="${BGSI_WEIGHT:-}"
BGSI_FLOOR="${BGSI_FLOOR:-}"
BGSI_TOP_K="${BGSI_TOP_K:-}"
BGSI_TEMPERATURE="${BGSI_TEMPERATURE:-}"
BGSI_START_EPOCH="${BGSI_START_EPOCH:-}"
BGSI_MIN_GROUP_SIZE="${BGSI_MIN_GROUP_SIZE:-}"
BGSI_VARIANCE_FLOOR="${BGSI_VARIANCE_FLOOR:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-.stage2}"

EPOCH_ARGS=""
if [[ -n "${TRAIN_EPOCHS}" ]]; then
  EPOCH_ARGS="--train-epochs ${TRAIN_EPOCHS}"
fi
GSI_ARGS=""
if [[ -n "${GSI_WEIGHT}" ]]; then
  GSI_ARGS="${GSI_ARGS} --gsi-weight ${GSI_WEIGHT}"
fi
if [[ -n "${GSI_FLOOR}" ]]; then
  GSI_ARGS="${GSI_ARGS} --gsi-floor ${GSI_FLOOR}"
fi
if [[ -n "${GSI_AXIS_MODE}" ]]; then
  GSI_ARGS="${GSI_ARGS} --gsi-axis-mode ${GSI_AXIS_MODE}"
fi
BGSI_ARGS=""
if [[ -n "${BGSI_WEIGHT}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-weight ${BGSI_WEIGHT}"
fi
if [[ -n "${BGSI_FLOOR}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-floor ${BGSI_FLOOR}"
fi
if [[ -n "${BGSI_TOP_K}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-top-k ${BGSI_TOP_K}"
fi
if [[ -n "${BGSI_TEMPERATURE}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-temperature ${BGSI_TEMPERATURE}"
fi
if [[ -n "${BGSI_START_EPOCH}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-start-epoch ${BGSI_START_EPOCH}"
fi
if [[ -n "${BGSI_MIN_GROUP_SIZE}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-min-group-size ${BGSI_MIN_GROUP_SIZE}"
fi
if [[ -n "${BGSI_VARIANCE_FLOOR}" ]]; then
  BGSI_ARGS="${BGSI_ARGS} --bgsi-variance-floor ${BGSI_VARIANCE_FLOOR}"
fi

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
    ${EPOCH_ARGS} \
    ${GSI_ARGS} \
    ${BGSI_ARGS} \
    --checkpoint-selection-interval 0 \
    --seed ${SEED} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE}) || \
    ([ -x .venv/bin/sfora ] && .venv/bin/sfora image-end-to-end \
    --dataset-name ${DATASET} \
    --protocol ${PROTOCOL} \
    --objectives ${OBJECTIVES} \
    ${EPOCH_ARGS} \
    ${GSI_ARGS} \
    ${BGSI_ARGS} \
    --checkpoint-selection-interval 0 \
    --seed ${SEED} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE})"
fi

rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" "${LOCAL_DIR}/reports/generated/"
