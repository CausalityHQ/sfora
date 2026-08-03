#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

JOINT_AUDIT=reports/generated/sop_official_joint_artifact_verification_seed0_final.json
TEST_AUDIT=reports/generated/sop_official_test_verification_seed0_final.json
TRAIN_PACK=reports/emb/sop_official_bninc_pa_seed0_train_final.npz
OUTPUT=reports/generated/sop_official_fragmentation_seed0_final.json
TEMPORARY="${OUTPUT}.tmp.$$"

test ! -e "${OUTPUT}"
test ! -e "${TEMPORARY}"
while [[ ! -f "${JOINT_AUDIT}" || ! -f "${TEST_AUDIT}" ]]; do
  if ! pgrep -f '[s]fora image-end-to-end.*sop_official_bninc_pa_seed0' >/dev/null \
      && ! pgrep -f '[r]un_post_sop_official.*\.sh' >/dev/null \
      && ! pgrep -f '[v]erify_sop_final_artifacts\.py' >/dev/null; then
    echo "SOP producers exited without required audits" >&2
    exit 1
  fi
  sleep 300
done

.venv/bin/python - "${JOINT_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "verified":
    raise SystemExit("SOP joint artifact audit did not report verified")
PY
test -f "${TRAIN_PACK}"

cleanup() {
  rm -f -- "${TEMPORARY}"
}
trap cleanup EXIT
.venv/bin/python /tmp/measure_spectral_class_connectivity_275.py \
  "${TRAIN_PACK}" \
  --temperature 0.1 \
  > "${TEMPORARY}"
.venv/bin/python - "${TEMPORARY}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "eligible_classes",
    "one_nn_fragmented_count",
    "one_nn_fragmented_fraction",
    "size_matched_fragmented_minus_connected_top1_points",
}
missing = required - payload.keys()
if missing:
    raise SystemExit(f"fragmentation result lacks required fields: {sorted(missing)}")
PY
mv -- "${TEMPORARY}" "${OUTPUT}"
trap - EXIT
