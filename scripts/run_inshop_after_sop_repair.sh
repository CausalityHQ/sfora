#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

SOP_TRAIN_REPORT=reports/generated/sop_official_bninc_pa_seed0.json
SOP_TEST_AUDIT=reports/generated/sop_official_test_verification_seed0_final.json
SOP_JOINT_AUDIT=reports/generated/sop_official_joint_artifact_verification_seed0_final.json
INSHOP_REPORT=reports/generated/inshop_official_pa_repaired_seed0.json
INSHOP_CHECKPOINT=reports/checkpoints/inshop_official_pa_repaired_seed0.pt

while [[ ! -f "${SOP_TEST_AUDIT}" || ! -f "${SOP_JOINT_AUDIT}" ]]; do
  if ! pgrep -f '[s]fora image-end-to-end.*sop_official_bninc_pa_seed0' >/dev/null \
      && ! pgrep -f '[r]un_post_sop_official.*\.sh' >/dev/null; then
    echo "SOP run/post-controller exited without the required test audit" >&2
    exit 1
  fi
  sleep 300
done

test -f "${SOP_TRAIN_REPORT}"
.venv/bin/python - "${SOP_JOINT_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "verified":
    raise SystemExit("SOP joint artifact audit did not report verified")
PY
test ! -e "${INSHOP_REPORT}"
test ! -e "${INSHOP_CHECKPOINT}"

.venv/bin/sfora image-end-to-end \
  --dataset-name inshop \
  --dataset-root /home/riomus/datasets/inshop \
  --objectives proxy_anchor \
  --recipe proxy_anchor.inshop.official-51db570 \
  --num-workers 8 \
  --seed 0 \
  --save-model-path "${INSHOP_CHECKPOINT}" \
  --output "${INSHOP_REPORT}"
