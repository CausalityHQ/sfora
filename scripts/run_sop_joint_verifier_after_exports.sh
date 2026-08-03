#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

TRAIN_PACK=reports/emb/sop_official_bninc_pa_seed0_train_final.npz
TEST_PACK=reports/emb/sop_official_bninc_pa_seed0_test_final.npz
CHECKPOINT=reports/checkpoints/sop_official_bninc_pa_seed0.pt
REPORT=reports/generated/sop_official_bninc_pa_seed0.json
OUTPUT=reports/generated/sop_official_joint_artifact_verification_seed0_final.json

test ! -e "${OUTPUT}"
while [[ ! -f "${TRAIN_PACK}" || ! -f "${TEST_PACK}" ]]; do
  if ! pgrep -f '[s]fora image-end-to-end.*sop_official_bninc_pa_seed0' >/dev/null \
      && ! pgrep -f '[r]un_post_sop_official.*\.sh' >/dev/null; then
    echo "SOP producer chain exited without both final embedding packs" >&2
    exit 1
  fi
  sleep 300
done

test -f "${CHECKPOINT}"
test -f "${REPORT}"
.venv/bin/python /tmp/verify_sop_final_artifacts.py \
  --train "${TRAIN_PACK}" \
  --test "${TEST_PACK}" \
  --checkpoint "${CHECKPOINT}" \
  --report "${REPORT}" \
  --output "${OUTPUT}"
