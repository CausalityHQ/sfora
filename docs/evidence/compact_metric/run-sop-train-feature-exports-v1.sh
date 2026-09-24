#!/usr/bin/env bash
set -euo pipefail

# One-time DGX controller. Keep the source snapshot and existing GPU queue fixed.
readonly RUN=/home/riomus/runs/sfora-sop-reference-b8f85611-179019
readonly SOURCE=/home/riomus/sfora-trained-gallery-export-src-89a4a096
readonly DATASET=/home/riomus/datasets/Stanford_Online_Products
readonly UNICOM=/home/riomus/unicom-d71992e
readonly SOURCE_CHECKPOINT=${UNICOM}/checkpoints/FP16-ViT-B-16.pt
readonly PYTHON=/home/riomus/group-learning/.venv/bin/python
readonly PRIOR_STATUS=${RUN}/sop-stage-split-v2.watcher-status
readonly STATUS=${RUN}/sop-train-feature-exports-v1.watcher-status
readonly LOG=${RUN}/sop-train-feature-exports-v1.log
readonly STAGE_PID=3692634
readonly STAGE_START_TICKS=211690406
readonly EXPECTED_EXPORTER_SHA=a62e4c0ccb1fca50392256cf276c969e1c0ba3b36df8a0be088ada6c49e1d8c9

exec 9>"${RUN}/.sop-train-feature-exports-v1.lock"
flock -n 9 || { echo 'SOP trained-feature exporter is already queued' >&2; exit 2; }
test ! -e "${STATUS}"
test ! -e "${RUN}/.sop-train-feature-exports-v1-invoked"
exec >>"${LOG}" 2>&1

on_exit() {
  local rc=$?
  printf 'exit=%d\n' "${rc}" > "${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

test "$(sha256sum "${SOURCE}/scripts/export_sop_trained_train_features.py" | cut -d' ' -f1)" = \
  "${EXPECTED_EXPORTER_SHA}"
echo "Waiting for stage-split watcher PID ${STAGE_PID}" >&2
while [[ ! -s "${PRIOR_STATUS}" ]]; do
  if [[ ! -r "/proc/${STAGE_PID}/stat" ]] \
    || [[ "$(awk '{print $22}' "/proc/${STAGE_PID}/stat")" != "${STAGE_START_TICKS}" ]]; then
    [[ -s "${PRIOR_STATUS}" ]] && break
    echo 'Stage-split watcher exited without terminal status' >&2
    exit 3
  fi
  sleep 30
done
grep -qx 'exit=0' "${PRIOR_STATUS}"
for _ in $(seq 1 120); do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [[ -z "${gpu_pids}" ]] && break
  sleep 30
done
test -z "${gpu_pids}"
mkdir "${RUN}/.sop-train-feature-exports-v1-invoked"

cd "${SOURCE}"
export PYTHONPATH="${SOURCE}/src:${SOURCE}/scripts"

export_arm() {
  local name=$1
  local stem=$2
  local output="${RUN}/sop-${name}-step48000-train-features-v1.npz"
  test -s "${RUN}/${stem}.step48000.pt"
  test -s "${RUN}/${stem}.step48000.json"
  test ! -e "${output}"
  echo "Exporting ${name} step48000 train features" >&2
  "${PYTHON}" scripts/export_sop_trained_train_features.py \
    --dataset-root "${DATASET}" \
    --unicom-checkout "${UNICOM}" \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --trained-checkpoint "${RUN}/${stem}.step48000.pt" \
    --training-receipt "${RUN}/${stem}.step48000.json" \
    --output "${output}" \
    --workers 4 \
    --execute-after-gpu-idle
  test -s "${output}"
  sha256sum "${output}" >&2
}

export_arm compact arcface-seed179019-53760
export_arm fullwidth arcface-seed179019-fullwidth768-53760
echo 'Both authenticated SOP train-feature exports completed' >&2
