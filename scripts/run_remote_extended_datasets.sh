#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-riomus@192.168.1.35}"
REMOTE_DIR="${REMOTE_DIR:-/home/riomus/group-learning}"
INSHOP_ROOT="${INSHOP_ROOT:?Set INSHOP_ROOT to the official DeepFashion In-Shop root on the remote host}"
INAT2018_ROOT="${INAT2018_ROOT:?Set INAT2018_ROOT to the iNaturalist 2018 root on the remote host}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-120}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-0}"

rsync -az --delete \
  --exclude .venv \
  --exclude .git \
  --exclude data \
  --exclude reports/generated \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd '${REMOTE_DIR}' && uv sync --group dev --extra research"

run_method() {
  METHOD="$1"
  OBJECTIVE="$2"
  shift 2
  OUTPUT_FILE="reports/generated/image_end_to_end_${DATASET}.${METHOD}_seed${SEED}.json"
  if [[ "${FORCE_RERUN}" != "1" ]] && \
    ssh "${REMOTE}" "test -s '${REMOTE_DIR}/${OUTPUT_FILE}'"; then
    echo "Skipping ${DATASET}/${METHOD}/seed${SEED}; artifact already exists."
    return
  fi
  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && .venv/bin/sfora image-end-to-end \
    --dataset-name '${DATASET}' \
    --dataset-root '${DATASET_ROOT}' \
    --protocol proxy-anchor-resnet50-512 \
    --objectives '${OBJECTIVE}' \
    --train-epochs '${TRAIN_EPOCHS}' \
    --batch-size '${BATCH_SIZE}' \
    --samples-per-class 4 \
    --warmup-epochs 5 \
    --lr-step-epochs 10 \
    --eval-test-interval-epochs 5 \
    --num-workers '${NUM_WORKERS}' \
    --seed '${SEED}' \
    $* \
    --output '${OUTPUT_FILE}'"
}

for DATASET in inshop inat2018; do
  if [[ "${DATASET}" == "inshop" ]]; then
    DATASET_ROOT="${INSHOP_ROOT}"
  else
    DATASET_ROOT="${INAT2018_ROOT}"
  fi

  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && .venv/bin/sfora image-dataset-preflight \
    --dataset-name '${DATASET}' --dataset-root '${DATASET_ROOT}'"

  for SEED in 0 1 2; do
    run_method "proxy_anchor" "proxy_anchor" \
      --no-embedding-layer-norm --ema-distill-weight 0.0
    run_method "pa_distill" "proxy_anchor" \
      --no-embedding-layer-norm --ema-distill-weight 1.0 \
      --ema-momentum 0.999 --ema-distill-tau 0.1
    run_method "hist" "hist" \
      --proxy-count-per-class 0 --no-embedding-layer-norm \
      --ema-distill-weight 0.0 --hist-lr-ds 0.03
    run_method "herd" "hist" \
      --proxy-count-per-class 0 --embedding-layer-norm \
      --ema-distill-weight 1.0 --ema-momentum 0.999 \
      --ema-distill-tau 0.1 --hist-lr-ds 0.03
  done
done

rsync -az "${REMOTE}:${REMOTE_DIR}/reports/generated/" \
  "${LOCAL_DIR}/reports/generated/"
