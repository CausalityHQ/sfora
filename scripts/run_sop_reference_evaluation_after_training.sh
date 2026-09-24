#!/usr/bin/env bash
set -euo pipefail

# One-off DGX controller for the already-running seed-179019 reference run.
# The evaluator itself owns the atomic official-test claim.
readonly TRAIN_PID=3495844
readonly TRAIN_START_TICKS=208007190
readonly RUN=/home/riomus/runs/sfora-sop-reference-b8f85611-179019
readonly STEM=arcface-seed179019-53760
readonly EVALUATOR=/home/riomus/sfora-sop-evaluator-src-bb9fbaf2
readonly TRAIN_SOURCE=/home/riomus/sfora-sop-reference-src-b8f85611
readonly DATASET=/home/riomus/datasets/Stanford_Online_Products
readonly CHECKPOINT=/home/riomus/unicom-d71992e/checkpoints/FP16-ViT-B-16.pt
readonly PYTHON=/home/riomus/group-learning/.venv/bin/python
readonly OUTPUT="${RUN}/${STEM}.official-test.json"
readonly STATUS="${RUN}/${STEM}.official-test.watcher-status"
readonly LOG="${RUN}/${STEM}.official-test.watcher.log"

exec 9>"${RUN}/.official-test.watcher.lock"
if ! flock -n 9; then
  echo 'Another SOP official-test watcher is already running' >&2
  exit 2
fi
exec >>"${LOG}" 2>&1
on_exit() {
  local rc=$?
  printf '%s exit=%d\n' "$(date -u +%FT%TZ)" "${rc}" > "${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}
trap on_exit EXIT

if [[ ! -f "${RUN}/sop-test-image-sha256.bin" ]] \
    || [[ ! -f "${EVALUATOR}/scripts/evaluate_sop_reference_checkpoint.py" ]] \
    || [[ ! -d "${TRAIN_SOURCE}" ]] \
    || [[ -e "${OUTPUT}" ]]; then
  echo 'SOP post-training evaluation authority differs' >&2
  exit 2
fi

# Pin every source file the evaluator records as loaded. These bytes come from
# the reviewed bb9fbaf2 evaluator snapshot, before the one-time test claim.
if ! (cd "${EVALUATOR}" && sha256sum --quiet --check -) <<'SHA256'; then
db1a1c470738f3b23bd05c87db2a1badfecf4a8777ce59b78ff6f49d31b30410  scripts/train_sop_compact_backbone.py
8967844e48dc45bb5f0eff3692caa6079d40301e5e0905ffd17bf6f896cfec26  scripts/export_unicom_sop_embeddings.py
1f60a8d5ad4a8779be3f078f3e748120d79ab77e990c309a898e744063582cb6  scripts/sop_teacher_anchored_runtime.py
888bb8a71a81b8cdbd098fa97e67546edb83481e19f8bbfa98c85130133f6152  src/sfora/sop_compact_training.py
5670a4dcb72eda7f31ef84d5907b6905ad25a4095926d705c794d7ba135478fe  src/sfora/deployed_code_rank.py
a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272  src/sfora/unicom_training.py
b77c119aab09dae0e7ba6bab5dbffbcb9b1695d85c7b544771adaf54b0cce253  src/sfora/unicom_rank_finish.py
1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c  src/sfora/representation_ceiling.py
eda764023c8a767d2fe46daa12d87dacdc0a38774e983cf445e6a4cb26293037  src/sfora/sop_evaluation.py
663a78daeb1b8bacfa211878cbf5608b9a4952d1a049bda60757189b0dc25644  src/sfora/sop_reference_recipe.py
4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67  src/sfora/joint_relational_compaction.py
bef633b19bd9b8162d67e600e31080f5241746916a3b92f0cf29c19aaf859d39  src/sfora/sop_reference_selection.py
af66d5e38ef722688307ed6255e4db9def415baaae18401bb5acf1afaea9e28c  scripts/evaluate_sop_reference_checkpoint.py
SHA256
  echo 'Evaluator snapshot differs from bb9fbaf2' >&2
  exit 2
fi

echo "Watching original trainer PID ${TRAIN_PID}, start tick ${TRAIN_START_TICKS}" >&2
while [[ -e "/proc/${TRAIN_PID}" ]]; do
  if ! proc_fields=$(awk '{print $3, $22}' "/proc/${TRAIN_PID}/stat" 2>/dev/null); then
    if [[ ! -e "/proc/${TRAIN_PID}" ]]; then
      break
    fi
    echo 'Original SOP trainer status unreadable; refusing evaluation' >&2
    exit 2
  fi
  read -r proc_state current_start <<< "${proc_fields}"
  if [[ "${current_start}" != "${TRAIN_START_TICKS}" ]]; then
    echo 'Original SOP trainer PID was reused; refusing evaluation' >&2
    exit 2
  fi
  if [[ "${proc_state}" == Z ]]; then
    break
  fi
  sleep 30
done

for step in 4000 8000 16000 32000 48000; do
  test -s "${RUN}/${STEM}.step${step}.pt"
  test -s "${RUN}/${STEM}.step${step}.json"
done
test -s "${RUN}/${STEM}.pt"
test -s "${RUN}/${STEM}.json"
test ! -e "${OUTPUT}"

if ! gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed; refusing evaluation' >&2
  exit 2
fi
if [[ -n "${gpu_pids}" ]]; then
  echo 'Another GPU compute process is active; refusing contended evaluation' >&2
  exit 2
fi

cd "${EVALUATOR}"
export PYTHONPATH="${EVALUATOR}/src:${EVALUATOR}/scripts"
echo 'Running train-selected official SOP evaluation once' >&2
if ! mkdir "${RUN}/.official-evaluator-invoked"; then
  echo 'SOP official evaluator invocation already reserved; refusing repeat' >&2
  exit 2
fi
"${PYTHON}" scripts/evaluate_sop_reference_checkpoint.py \
  --dataset-root "${DATASET}" \
  --unicom-checkout /home/riomus/unicom-d71992e \
  --source-checkpoint "${CHECKPOINT}" \
  --training-source-root "${TRAIN_SOURCE}" \
  --test-image-manifest "${RUN}/sop-test-image-sha256.bin" \
  --candidate "${RUN}/${STEM}.step4000.pt" "${RUN}/${STEM}.step4000.json" \
  --candidate "${RUN}/${STEM}.step8000.pt" "${RUN}/${STEM}.step8000.json" \
  --candidate "${RUN}/${STEM}.step16000.pt" "${RUN}/${STEM}.step16000.json" \
  --candidate "${RUN}/${STEM}.step32000.pt" "${RUN}/${STEM}.step32000.json" \
  --candidate "${RUN}/${STEM}.step48000.pt" "${RUN}/${STEM}.step48000.json" \
  --candidate "${RUN}/${STEM}.pt" "${RUN}/${STEM}.json" \
  --output "${OUTPUT}" \
  --workers 8 \
  --execute-official-sop-reference-evaluation
echo "Official SOP evaluation finished: ${OUTPUT}" >&2
