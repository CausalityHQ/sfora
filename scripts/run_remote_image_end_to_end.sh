#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-riomus@192.168.1.35}"
REMOTE_DIR="${REMOTE_DIR:-/home/riomus/group-learning}"
DATASETS="${DATASETS:-cub cars sop}"
PROTOCOL="${PROTOCOL:-sota-resnet50-512}"
OBJECTIVES="${OBJECTIVES:-frozen_pretrained,group_supcon_xbm_radius}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
LIMIT_PER_CLASS="${LIMIT_PER_CLASS:-}"
MAX_CLASSES="${MAX_CLASSES:-}"
GROUP_SIZE="${GROUP_SIZE:-4}"
XBM_MEMORY_SIZE="${XBM_MEMORY_SIZE:-4096}"
XBM_WEIGHT="${XBM_WEIGHT:-0.25}"
RADIUS_WEIGHT="${RADIUS_WEIGHT:-0.01}"
RADIUS_TARGET="${RADIUS_TARGET:-0.0}"
RETRIEVAL_QUERY_LIMIT="${RETRIEVAL_QUERY_LIMIT:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"

LIMIT_ARGS=""
if [[ -n "${LIMIT_PER_CLASS}" ]]; then
  LIMIT_ARGS="--limit-per-class ${LIMIT_PER_CLASS}"
fi
MAX_CLASS_ARGS=""
if [[ -n "${MAX_CLASSES}" ]]; then
  MAX_CLASS_ARGS="--max-classes ${MAX_CLASSES}"
fi
BATCH_ARGS=""
if [[ -n "${BATCH_SIZE}" ]]; then
  BATCH_ARGS="--batch-size ${BATCH_SIZE}"
fi
LR_ARGS=""
if [[ -n "${LEARNING_RATE}" ]]; then
  LR_ARGS="--learning-rate ${LEARNING_RATE}"
fi
STEP_ARGS=""
if [[ -n "${TRAIN_STEPS}" ]]; then
  STEP_ARGS="--train-steps ${TRAIN_STEPS}"
fi
EPOCH_ARGS=""
if [[ -n "${TRAIN_EPOCHS}" ]]; then
  EPOCH_ARGS="--train-epochs ${TRAIN_EPOCHS}"
fi
RETRIEVAL_ARGS=""
if [[ -n "${RETRIEVAL_QUERY_LIMIT}" ]]; then
  RETRIEVAL_ARGS="--retrieval-query-limit ${RETRIEVAL_QUERY_LIMIT}"
fi

rsync -az --delete \
  --exclude .venv \
  --exclude .git \
  --exclude data \
  --exclude reports/generated \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && uv sync --group dev --extra research"

for DATASET in ${DATASETS}; do
  OUTPUT_FILE="reports/generated/image_end_to_end_${DATASET}${OUTPUT_SUFFIX}.json"
  if [[ "${FORCE_RERUN}" != "1" ]] && ssh "${REMOTE}" "test -s ${REMOTE_DIR}/${OUTPUT_FILE}"; then
    echo "Skipping ${DATASET}; ${OUTPUT_FILE} already exists."
    continue
  fi
  ssh "${REMOTE}" "cd ${REMOTE_DIR} && (uv run --group dev --extra research group-learning image-end-to-end \
    --dataset-name ${DATASET} \
    --protocol ${PROTOCOL} \
    --objectives ${OBJECTIVES} \
    ${STEP_ARGS} \
    ${EPOCH_ARGS} \
    ${BATCH_ARGS} \
    ${LR_ARGS} \
    ${LIMIT_ARGS} \
    ${MAX_CLASS_ARGS} \
    --group-size ${GROUP_SIZE} \
    --xbm-memory-size ${XBM_MEMORY_SIZE} \
    --xbm-weight ${XBM_WEIGHT} \
    --radius-weight ${RADIUS_WEIGHT} \
    --radius-target ${RADIUS_TARGET} \
    ${RETRIEVAL_ARGS} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE}) || \
    ([ -x .venv/bin/group-learning ] && .venv/bin/group-learning image-end-to-end \
    --dataset-name ${DATASET} \
    --protocol ${PROTOCOL} \
    --objectives ${OBJECTIVES} \
    ${STEP_ARGS} \
    ${EPOCH_ARGS} \
    ${BATCH_ARGS} \
    ${LR_ARGS} \
    ${LIMIT_ARGS} \
    ${MAX_CLASS_ARGS} \
    --group-size ${GROUP_SIZE} \
    --xbm-memory-size ${XBM_MEMORY_SIZE} \
    --xbm-weight ${XBM_WEIGHT} \
    --radius-weight ${RADIUS_WEIGHT} \
    --radius-target ${RADIUS_TARGET} \
    ${RETRIEVAL_ARGS} \
    --num-workers ${NUM_WORKERS} \
    --output ${OUTPUT_FILE})"
done

rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" "${LOCAL_DIR}/reports/generated/"
