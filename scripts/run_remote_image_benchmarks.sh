#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="researcher@gpu.example.com"
REMOTE_DIR="/home/researcher/group-learning"
DATASETS="${DATASETS:-cub cars sop}"
MODELS="${MODELS:-facebook/dinov2-small,openai/clip-vit-base-patch32,google/siglip-base-patch16-224}"
OBJECTIVES="${OBJECTIVES:-triplet,batch_hard_triplet,group,hard_group,supcon,group_supcon,proxy_nca,proxy_anchor,cosface,arcface,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius,group_supcon_xbm_radius}"
LIMIT_PER_CLASS="${LIMIT_PER_CLASS-8}"
MAX_CLASSES="${MAX_CLASSES-100}"
MIN_PER_CLASS="${MIN_PER_CLASS-}"
TRAIN_STEPS="${TRAIN_STEPS-160}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT-1.0}"
GROUP_WEIGHT="${GROUP_WEIGHT-1.0}"
HARD_WEIGHT="${HARD_WEIGHT-0.5}"
SPREAD_WEIGHT="${SPREAD_WEIGHT-0.1}"
WEIGHT_ARGS="--triplet-weight ${TRIPLET_WEIGHT} --group-weight ${GROUP_WEIGHT} --hard-weight ${HARD_WEIGHT} --spread-weight ${SPREAD_WEIGHT}"
OUTPUT_DIMENSIONS="${OUTPUT_DIMENSIONS-}"
DIMENSION_ARGS=""
if [[ -n "${OUTPUT_DIMENSIONS}" ]]; then
  DIMENSION_ARGS="--output-dimensions ${OUTPUT_DIMENSIONS}"
fi
XBM_MEMORY_SIZE="${XBM_MEMORY_SIZE-1024}"
XBM_WEIGHT="${XBM_WEIGHT-0.25}"
XBM_ARGS="--xbm-memory-size ${XBM_MEMORY_SIZE} --xbm-weight ${XBM_WEIGHT}"
RADIUS_WEIGHT="${RADIUS_WEIGHT-0.01}"
RADIUS_TARGET="${RADIUS_TARGET-0.0}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT-0.01}"
RADIUS_ARGS="--radius-weight ${RADIUS_WEIGHT} --radius-target ${RADIUS_TARGET} --variance-weight ${VARIANCE_WEIGHT}"
SHUFFLE_GROUPS_EACH_STEP="${SHUFFLE_GROUPS_EACH_STEP-1}"
SHUFFLE_ARGS=""
if [[ "${SHUFFLE_GROUPS_EACH_STEP}" == "1" ]]; then
  SHUFFLE_ARGS="--shuffle-groups-each-step"
fi
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR-data/image_embeddings_cache}"
CACHE_ARGS="--embedding-cache-dir ${EMBEDDING_CACHE_DIR}"
FORCE_RERUN="${FORCE_RERUN-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX-}"
PROJECTION_TRAIN_LIMIT="${PROJECTION_TRAIN_LIMIT-}"
PROJECTION_ARGS=""
if [[ -n "${PROJECTION_TRAIN_LIMIT}" ]]; then
  PROJECTION_ARGS="--projection-train-limit ${PROJECTION_TRAIN_LIMIT}"
fi
LIMIT_ARGS=""
if [[ -n "${LIMIT_PER_CLASS}" ]]; then
  LIMIT_ARGS="--limit-per-class ${LIMIT_PER_CLASS}"
fi
MAX_CLASS_ARGS=""
if [[ -n "${MAX_CLASSES}" ]]; then
  MAX_CLASS_ARGS="--max-classes ${MAX_CLASSES}"
fi
MIN_CLASS_ARGS=""
if [[ -n "${MIN_PER_CLASS}" ]]; then
  MIN_CLASS_ARGS="--min-per-class ${MIN_PER_CLASS}"
fi
RETRIEVAL_QUERY_LIMIT="${RETRIEVAL_QUERY_LIMIT-}"
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
  OUTPUT_FILE="reports/generated/image_retrieval_${DATASET}${OUTPUT_SUFFIX}.json"
  if [[ "${FORCE_RERUN}" != "1" ]] && ssh "${REMOTE}" "test -s ${REMOTE_DIR}/${OUTPUT_FILE}"; then
    echo "Skipping ${DATASET}; ${OUTPUT_FILE} already exists."
    continue
  fi
  ssh "${REMOTE}" "cd ${REMOTE_DIR} && (uv run --group dev --extra research sfora image-benchmark \
    --dataset-name ${DATASET} \
    --model-names ${MODELS} \
    --objectives ${OBJECTIVES} \
    ${LIMIT_ARGS} \
    ${MIN_CLASS_ARGS} \
    ${MAX_CLASS_ARGS} \
    --group-size 4 \
    --batch-size 64 \
    --train-steps ${TRAIN_STEPS} \
    ${WEIGHT_ARGS} \
    ${DIMENSION_ARGS} \
    ${XBM_ARGS} \
    ${RADIUS_ARGS} \
    ${SHUFFLE_ARGS} \
    ${CACHE_ARGS} \
    ${PROJECTION_ARGS} \
    ${RETRIEVAL_ARGS} \
    --output ${OUTPUT_FILE}) || \
    ([ -x .venv/bin/sfora ] && .venv/bin/sfora image-benchmark \
    --dataset-name ${DATASET} \
    --model-names ${MODELS} \
    --objectives ${OBJECTIVES} \
    ${LIMIT_ARGS} \
    ${MIN_CLASS_ARGS} \
    ${MAX_CLASS_ARGS} \
    --group-size 4 \
    --batch-size 64 \
    --train-steps ${TRAIN_STEPS} \
    ${WEIGHT_ARGS} \
    ${DIMENSION_ARGS} \
    ${XBM_ARGS} \
    ${RADIUS_ARGS} \
    ${SHUFFLE_ARGS} \
    ${CACHE_ARGS} \
    ${PROJECTION_ARGS} \
    ${RETRIEVAL_ARGS} \
    --output ${OUTPUT_FILE})"
done

rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" "${LOCAL_DIR}/reports/generated/"
