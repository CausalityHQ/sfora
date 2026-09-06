#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_spatial=${REMOTE_SPATIAL:-/home/riomus/sfora-spatial-tail-recovery/498a10d34b9095fa92a7f45f8e0af6031b33344d/spatial-tail.safetensors}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-coverage-calibration}
revision=$(git rev-parse HEAD)
remote_source=$remote_root/$revision
remote_output=$remote_output_root/$revision
local_result=${LOCAL_RESULT:-/tmp/sfora-coverage-calibration-$revision.json}
local_artifact=${LOCAL_ARTIFACT:-/tmp/sfora-coverage-calibration-$revision.safetensors}
local_fit_receipt=${LOCAL_FIT_RECEIPT:-/tmp/sfora-coverage-calibration-$revision.fit.json}
local_execution=${LOCAL_EXECUTION:-/tmp/sfora-coverage-calibration-$revision.execution.json}
spatial_sha=cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9
spatial_bytes=67973944

source_files=(
  scripts/landlock_exec.c
  scripts/deploy_siglip_coverage_calibration_v1.sh
  scripts/prepare_siglip_coverage_calibration.py
  scripts/probe_siglip_coverage_calibration.py
  scripts/run_siglip_coverage_calibration_two_phase.sh
  src/sfora/siglip_coverage_calibration.py
  tests/test_deploy_siglip_coverage_calibration.py
  tests/test_probe_siglip_coverage_calibration.py
  tests/test_prepare_siglip_coverage_calibration.py
  tests/test_run_siglip_coverage_calibration_two_phase.py
  tests/test_siglip_coverage_calibration.py
)
git diff --quiet HEAD -- src/sfora scripts \
  ':(exclude)src/sfora/qwen_geometry_control.py' \
  ':(exclude)scripts/run_qwen_geometry_control.py'
git diff --cached --quiet -- src/sfora scripts \
  ':(exclude)src/sfora/qwen_geometry_control.py' \
  ':(exclude)scripts/run_qwen_geometry_control.py'
git diff --quiet HEAD -- "${source_files[@]}"
git diff --cached --quiet -- "${source_files[@]}"
for path in "${source_files[@]}"; do git ls-files --error-unmatch "$path" >/dev/null; done
test ! -e "$local_result"
test ! -e "$local_artifact"
test ! -e "$local_fit_receipt"
test ! -e "$local_execution"

scratch=$(mktemp -d /tmp/sfora-coverage-calibration-deploy.XXXXXX)
bundle=$scratch/source.bundle
cleanup() {
  test ! -e "$bundle" || unlink "$bundle"
  rmdir "$scratch"
}
trap cleanup EXIT INT TERM
git bundle create "$bundle" HEAD
remote_bundle=/tmp/sfora-coverage-calibration-source-$revision.bundle

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_spatial" \
  "$remote_bundle" "$spatial_sha" "$spatial_bytes" <<'PREFLIGHT'
set -euo pipefail
output_parent=$(dirname "$2")
mkdir -p "$output_parent"
test -d "$output_parent"
pgrep -f '[p]robe_siglip_coverage_calibration.py' >/dev/null && {
  echo 'coverage calibration process is already active' >&2; exit 75;
}
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$gpu_processes"
test -f "$3/control.receipt.json"
test -f "$3/seed-017.receipt.json"
test -f "$3/seed-029.receipt.json"
test -f "$3/seed-043.receipt.json"
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt"
test -f "$3/seed-029/checkpoints/seed-029-epoch-060.pt"
test -f "$3/seed-043/checkpoints/seed-043-epoch-060.pt"
test -f "$4"
test "$(stat -c%s "$4")" = "$7"
test "$(sha256sum "$4" | awk '{print $1}')" = "$6"
test ! -e "$1"; test ! -e "$2"; test ! -e "$2.partial"
test ! -e "$2.execution.json"; test ! -e "$2.execution.json.partial"
test ! -e "$2.inputs.partial"; test ! -e "$5"
PREFLIGHT
rsync -a -- "$bundle" "$remote_host:$remote_bundle"

set +e
ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_spatial" \
  "$revision" "$remote_bundle" "$spatial_sha" <<'REMOTE'
set -euo pipefail
source_dir=$1; output=$2; control=$3; spatial=$4; revision=$5; bundle=$6; spatial_sha=$7
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output
input_staging=$output.inputs.partial
execution_receipt=$output.execution.json
phase1=$output/phase1
phase2=$output/phase2
authority=$output/authority
optimization_images=$input_staging/optimization-images
support_images=$input_staging/support-images
heldout_images=$input_staging/heldout-images
child= group= run_owned=0 source_owned=0 source_checkout_complete=0 staging_owned=0
input_staging_owned=0 stop_reason= receipt_written=0 drain_attempted=0
group_drained=true
write_execution_receipt() {
  local receipt_status=$1 failed_phase=$2 reason_code=$3 exit_code=$4 drained=$5
  local map_path=$phase1/maps.safetensors
  local fit_path=$phase1/fit-receipt.json
  local result_path=$phase2/result.json
  "$python" -B - "$execution_receipt" "$revision" "$receipt_status" "$failed_phase" \
    "$reason_code" "$exit_code" "$drained" "$map_path" "$fit_path" "$result_path" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

def identity(raw_path):
    path = pathlib.Path(raw_path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw)}

target = pathlib.Path(sys.argv[1])
value = {
    "schema": "sfora-siglip-coverage-execution-v1",
    "claim_eligible": False,
    "execution_source_commit": sys.argv[2],
    "status": sys.argv[3],
    "failed_phase": None if sys.argv[4] == "none" else sys.argv[4],
    "reason_code": sys.argv[5],
    "exit_code": int(sys.argv[6]),
    "group_drained": sys.argv[7] == "true",
    "verified_artifacts": {
        "map_artifact": identity(sys.argv[8]),
        "fit_receipt": identity(sys.argv[9]),
        "result": identity(sys.argv[10]),
    },
}
raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
partial = target.with_name(target.name + ".partial")
with partial.open("xb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
os.link(partial, target)
partial.unlink()
PY
  receipt_written=1
}
drain_group() {
  local target_group=${group:-}
  if [[ "$drain_attempted" = 1 ]]; then return; fi
  drain_attempted=1
  group_drained=true
  if [[ -z $target_group ]]; then return; fi
  kill -TERM -- "-$target_group" 2>/dev/null || true
  if [[ -n ${child:-} ]]; then wait "$child" 2>/dev/null || true; fi
  for _ in 1 2 3 4 5; do
    pgrep -g "$target_group" >/dev/null 2>&1 || break
    sleep 1
  done
  if pgrep -g "$target_group" >/dev/null 2>&1; then
    kill -KILL -- "-$target_group" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      pgrep -g "$target_group" >/dev/null 2>&1 || break
      sleep 1
    done
  fi
  if pgrep -g "$target_group" >/dev/null 2>&1; then
    group_drained=false
  else
    child= group=
  fi
}
cleanup_remote() {
  original_status=$?
  set +e
  if [[ "$run_owned" != 1 ]]; then return; fi
  drain_group
  if [[ "$receipt_written" = 0 ]]; then
    failed_phase=preparation
    test ! -e "$staging/preparation.complete" || failed_phase=fit
    test ! -s "$phase1/maps.safetensors" || failed_phase=seal
    test ! -s "$phase1/fit-receipt.json" || failed_phase=evaluation
    reason_code=${stop_reason:-process-exit}
    write_execution_receipt failed "$failed_phase" "$reason_code" "$original_status" \
      "$group_drained"
  fi
  test ! -e "$bundle" || unlink "$bundle"
  if [[ "$group_drained" = false ]]; then return; fi
  test "$staging" = "$output" || exit 99
  test "$input_staging" = "$output.inputs.partial" || exit 99
  if [[ "$staging_owned" = 1 && -e "$staging" ]]; then rm -rf -- "$staging"; fi
  if [[ "$input_staging_owned" = 1 && -e "$input_staging" ]]; then
    rm -rf -- "$input_staging"
  fi
  if [[ "$source_owned" = 1 && ! -e "$output" && -e "$source_dir" ]]; then
    test ! -L "$source_dir"
    test "${source_dir##*/}" = "$revision"
    test "$(dirname "$source_dir")" != /
    if [[ "$source_checkout_complete" = 1 ]]; then
      test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision" || exit 99
    fi
    rm -rf -- "$source_dir"
  fi
}
trap cleanup_remote EXIT
trap 'stop_reason=signal-int; exit 130' INT
trap 'stop_reason=signal-term; exit 143' TERM

test ! -e "$source_dir"
mkdir "$source_dir"
source_owned=1
run_owned=1
mkdir "$staging"
staging_owned=1
mkdir "$input_staging"
input_staging_owned=1
git clone --quiet --no-checkout "$bundle" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$revision"
unlink "$bundle"
cd "$source_dir"
test "$(git rev-parse HEAD)" = "$revision"
source_checkout_complete=1
test -z "$(git status --porcelain --untracked-files=no)"
printf '%s\n' "$revision" >SOURCE_REVISION
: >SOURCE_MANIFEST.sha256
git ls-files -z | while IFS= read -r -d '' relative; do
  printf '%s  %s\n' "$(sha256sum "$relative" | awk '{print $1}')" "$relative" \
    >>SOURCE_MANIFEST.sha256
done
printf '%s  SOURCE_REVISION\n' "$(sha256sum SOURCE_REVISION | awk '{print $1}')" \
  >>SOURCE_MANIFEST.sha256
LC_ALL=C sort -k2 -o SOURCE_MANIFEST.sha256 SOURCE_MANIFEST.sha256
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null

pgrep -f '[p]robe_siglip_coverage_calibration.py' >/dev/null && {
  echo 'coverage calibration process is already active' >&2; exit 75;
}
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
test -z "$gpu_processes"
test -x "$python"; test -f "$spatial"
test ! -e "$execution_receipt"; test ! -e "$execution_receipt.partial"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

binding=$authority/control-binding.json
optimization_manifest=$authority/optimization-manifest.json
evaluation_manifest=$authority/evaluation-manifest.json
private_tmp1=$staging/private-tmp-1
private_tmp2=$staging/private-tmp-2
export HF_HOME=/home/riomus/.cache/huggingface
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)

setsid timeout --signal=TERM --kill-after=30s 5400s \
  scripts/run_siglip_coverage_calibration_two_phase.sh \
  "$python" "$staging/landlock-exec" "$source_dir" "$control" "$spatial" \
  "$binding" "$control/seed-017/checkpoints/seed-017-epoch-060.pt" \
  "$optimization_manifest" "$optimization_images" \
  "$evaluation_manifest" "$support_images" "$heldout_images" \
  "$spatial_sha" "$revision" "$staging" "$phase1" "$phase2" \
  "$private_tmp1" "$private_tmp2" &
child=$!
group=$child
stop_reason=; psi_hits=0; progress_gap=0; last_cpu=
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  cpu=$(ps -o time= -g "$child" 2>/dev/null || true)
  gpu_rows=
  if ! gpu_rows=$(nvidia-smi --query-compute-apps=used_memory \
    --format=csv,noheader,nounits); then
    stop_reason=gpu-telemetry
  elif [[ -n $gpu_rows ]] && grep -qvE '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"$gpu_rows"; then
    stop_reason=gpu-telemetry
  fi
  gpu_mib=$(awk '{s+=$1}END{printf "%.0f",s}' <<<"$gpu_rows")
  test -n "$gpu_mib" || gpu_mib=0
  ((rss <= 118111600640)) || stop_reason=rss-cap
  ((gpu_mib <= 98304)) || stop_reason=gpu-memory-cap
  if awk -v x="$psi" 'BEGIN{exit !(x>=0.50)}'; then ((psi_hits+=1)); else psi_hits=0; fi
  awk -v x="$psi" 'BEGIN{exit !(x>=0.79)}' && stop_reason=psi-immediate || true
  ((psi_hits < 3)) || stop_reason=psi-sustained
  ((swap <= swap0)) || stop_reason=swap-delta
  if [[ $cpu = "$last_cpu" ]]; then ((progress_gap+=5)); else progress_gap=0; last_cpu=$cpu; fi
  ((progress_gap < 300)) || stop_reason=progress-gap
  if [[ -n $stop_reason ]]; then kill -TERM -- "-$child" 2>/dev/null || true; break; fi
  sleep 5
done
set +e; wait "$child"; status=$?; set -e; child=
drain_group
test "$group_drained" = true
if [[ -n $stop_reason ]]; then echo "STOP:$stop_reason" >&2; exit 125; fi
((status == 0)) || exit "$status"
test -s "$phase2/result.json"; test -s "$phase1/maps.safetensors"
test -s "$phase1/fit-receipt.json"
map_sha=$(sha256sum "$phase1/maps.safetensors" | awk '{print $1}')
"$python" -B - "$phase2/result.json" "$revision" "$map_sha" \
  "$phase1/maps.safetensors" "$optimization_manifest" "$phase1/fit-receipt.json" <<'PY'
import json
import pathlib
import sys
from safetensors import safe_open
from probe_siglip_coverage_calibration import _artifact_metadata, _ids_sha256
from sfora.siglip_coverage_calibration import (
    validate_coverage_calibration_result_bytes,
    validate_coverage_fit_receipt_bytes,
)

value = validate_coverage_calibration_result_bytes(pathlib.Path(sys.argv[1]).read_bytes())
assert value["claim_eligible"] is False
assert value["execution_source_commit"] == sys.argv[2]
assert value["inputs"]["map_artifact"]["sha256"] == sys.argv[3]
fit_raw = pathlib.Path(sys.argv[6]).read_bytes()
fit = validate_coverage_fit_receipt_bytes(fit_raw)
assert value["inputs"]["fit_receipt"]["sha256"] == __import__("hashlib").sha256(fit_raw).hexdigest()
assert fit["inputs"]["map_artifact"]["sha256"] == sys.argv[3]
assert value["image_namespaces"]["support"]["sha256"]
assert value["image_namespaces"]["evaluation"]["sha256"]
assert len(value["support_ids"]) == 264
assert set(value["support_labels"]) == set(range(49, 82))
for solver in value["solver"].values():
    assert solver["dimensions"] == 512
    assert solver["rank"] == 513
optimization = json.loads(pathlib.Path(sys.argv[5]).read_bytes())
base_ids = tuple(row["example_id"] for row in optimization["examples"])
support_ids = tuple(value["support_ids"])
fitting_ids = tuple(sorted((*base_ids, *support_ids)))
with safe_open(sys.argv[4], framework="pt", device="cpu") as stream:
    metadata = stream.metadata()
expected_metadata = _artifact_metadata(
    support_ids=support_ids,
    fitting_ids=fitting_ids,
    checkpoint_sha256=value["inputs"]["checkpoint"]["sha256"],
    control_binding_sha256=value["inputs"]["control_binding"]["sha256"],
    optimization_manifest_sha256=value["inputs"]["optimization_manifest"]["sha256"],
    evaluation_manifest_sha256=value["inputs"]["evaluation_manifest"]["sha256"],
    spatial_artifact_sha256=value["inputs"]["spatial_artifact"]["sha256"],
    optimization_images_sha256=value["image_namespaces"]["optimization"]["sha256"],
    support_images_sha256=value["image_namespaces"]["support"]["sha256"],
)
assert metadata == expected_metadata
assert metadata["optimization_images_sha256"] == value["image_namespaces"]["optimization"]["sha256"]
assert metadata["support_images_sha256"] == value["image_namespaces"]["support"]["sha256"]
assert metadata["support_ids_sha256"] == _ids_sha256(
    support_ids, domain=b"sfora-coverage-map-support-v1\0"
)
PY
unlink "$staging/landlock-exec"
unlink "$staging/preparation.complete"
rm -rf -- "$private_tmp1" "$private_tmp2" "$optimization_images" "$support_images" "$heldout_images"
test ! -e "$input_staging" || rm -rf -- "$input_staging"
write_execution_receipt complete none complete 0 true
trap - EXIT INT TERM
REMOTE
remote_status=$?
set -e

rsync -a -- "$remote_host:$remote_output.execution.json" "$local_execution"
uv run python - "$local_execution" "$revision" "$remote_status" <<'PY'
import pathlib
import sys

from sfora.siglip_coverage_calibration import validate_coverage_execution_receipt_bytes

receipt = validate_coverage_execution_receipt_bytes(pathlib.Path(sys.argv[1]).read_bytes())
assert receipt["execution_source_commit"] == sys.argv[2]
expected = "complete" if int(sys.argv[3]) == 0 else "failed"
assert receipt["status"] == expected
PY
((remote_status == 0)) || exit "$remote_status"

rsync -a -- "$remote_host:$remote_output/phase2/result.json" "$local_result"
rsync -a -- "$remote_host:$remote_output/phase1/maps.safetensors" "$local_artifact"
rsync -a -- "$remote_host:$remote_output/phase1/fit-receipt.json" "$local_fit_receipt"
sha256sum "$local_result" "$local_artifact" "$local_fit_receipt" "$local_execution"
