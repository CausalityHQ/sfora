#!/usr/bin/env bash
set -euo pipefail

cd /home/riomus/group-learning

SOP_TRAIN_REPORT=reports/generated/sop_official_bninc_pa_seed0.json
SOP_TEST_AUDIT=reports/generated/sop_official_test_verification_seed0_final.json
SOP_JOINT_AUDIT=reports/generated/sop_official_joint_artifact_verification_seed0_final.json
SOP_FRAGMENTATION=reports/generated/sop_official_fragmentation_seed0_final.json
SOP_FRAGMENTATION_AUDIT=reports/generated/sop_official_fragmentation_gate_verification_seed0_final.json
INSHOP_REPORT=reports/generated/inshop_official_pa_repaired_seed0.json
INSHOP_CHECKPOINT=reports/checkpoints/inshop_official_pa_repaired_seed0.pt

required=(
  "${SOP_TEST_AUDIT}"
  "${SOP_JOINT_AUDIT}"
  "${SOP_FRAGMENTATION}"
  "${SOP_FRAGMENTATION_AUDIT}"
)
while true; do
  missing=0
  for artifact in "${required[@]}"; do
    if [[ ! -f "${artifact}" ]]; then
      missing=1
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  if ! pgrep -f '[s]fora image-end-to-end.*sop_official_bninc_pa_seed0' >/dev/null \
      && ! pgrep -f '[r]un_post_sop_official.*\.sh' >/dev/null \
      && ! pgrep -f '[m]easure_spectral_class_connectivity_275\.py' >/dev/null; then
    # The independent verifier is legitimate liveness only after its input result
    # exists. Its waiting controller must not mask a failed fragmentation producer.
    if [[ ! -f "${SOP_FRAGMENTATION}" ]] \
        || ! pgrep -f '[v]erify_spectral_fragmentation_gate\.py' >/dev/null; then
      echo "SOP evidence chain exited without all required artifacts" >&2
      exit 1
    fi
  fi
  sleep 300
done

test -f "${SOP_TRAIN_REPORT}"
.venv/bin/python - "${SOP_JOINT_AUDIT}" "${SOP_FRAGMENTATION_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

joint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
fragmentation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if joint.get("status") != "verified":
    raise SystemExit("SOP joint artifact audit did not report verified")
if fragmentation.get("status") != "verified":
    raise SystemExit("SOP fragmentation audit did not report verified")
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
