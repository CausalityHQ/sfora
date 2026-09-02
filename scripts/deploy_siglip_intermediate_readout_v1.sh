#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-intermediate-readout-screens}
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-intermediate-readout-$source_revision.json}

git diff --quiet HEAD --
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard -- src scripts sitecustomize.py)"
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-intermediate-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-intermediate-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM

bundle=$scratch/source.bundle
git bundle create "$bundle" HEAD
remote_bundle=/tmp/sfora-intermediate-source-$source_revision.bundle

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_bundle" <<'PREFLIGHT'
set -euo pipefail
if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
if pgrep -f '[d]iagnose_siglip_intermediate_readout.py' >/dev/null; then
  echo 'intermediate process is already active' >&2
  exit 75
fi
test -f "$3/control.receipt.json"
test -f "$3/seed-017.receipt.json"
test -f "$3/seed-029.receipt.json"
test -f "$3/seed-043.receipt.json"
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt"
test -f "$3/seed-029/checkpoints/seed-029-epoch-060.pt"
test -f "$3/seed-043/checkpoints/seed-043-epoch-060.pt"
test ! -e "$1"
test ! -e "$2"
test ! -e "$2.partial"
test ! -e "$4"
PREFLIGHT

rsync -a -- "$bundle" "$remote_host:$remote_bundle"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$source_revision" "$remote_bundle" <<'REMOTE'
set -euo pipefail
source_dir=$1
output=$2
control=$3
revision=$4
bundle=$5
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output.partial
authority=$staging/authority
images=$staging/images
child=
cleanup_remote() {
  if [[ -n ${child:-} ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  test ! -e "$bundle" || unlink "$bundle"
  test "$staging" = "$output.partial" || exit 99
  test ! -e "$staging" || rm -rf -- "$staging"
}
trap cleanup_remote EXIT
trap 'cleanup_remote; exit 130' INT
trap 'cleanup_remote; exit 143' TERM

git clone --quiet --no-checkout "$bundle" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$revision"
unlink "$bundle"
cd "$source_dir"
test "$(git rev-parse HEAD)" = "$revision"
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

if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
if pgrep -f '[d]iagnose_siglip_intermediate_readout.py' >/dev/null; then
  echo 'intermediate process is already active' >&2
  exit 75
fi
test -x "$python"
test ! -e "$output"
test ! -e "$staging"
mkdir -p "$authority"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$python" -B - "$staging/control-manifest.json" "$images" <<'PY'
import pathlib
import sys
from run_siglip_proxy_control import load_control_examples, write_control_manifest_artifacts

write_control_manifest_artifacts(
    output=pathlib.Path(sys.argv[1]),
    optimization_image_root=pathlib.Path(sys.argv[2]),
    bands=load_control_examples(),
)
PY

"$python" -B - "$control" "$staging/control-manifest.json" "$authority" <<'PY'
import pathlib
import sys
from run_siglip_rsta_stage_a import project_stage_a_authority

control = pathlib.Path(sys.argv[1])
projected = project_stage_a_authority(
    seed_receipts=tuple(control / f"seed-{seed:03d}.receipt.json" for seed in (17, 29, 43)),
    aggregate_receipt=control / "control.receipt.json",
    checkpoints=tuple(
        control / f"seed-{seed:03d}/checkpoints/seed-{seed:03d}-epoch-060.pt"
        for seed in (17, 29, 43)
    ),
    control_manifest=pathlib.Path(sys.argv[2]),
)
authority = pathlib.Path(sys.argv[3])
(authority / "control-binding.json").write_bytes(projected.control_binding_bytes)
(authority / "optimization-manifest.json").write_bytes(projected.optimization_manifest_bytes)
PY

binding=$authority/control-binding.json
manifest=$authority/optimization-manifest.json
binding_sha=$(sha256sum "$binding" | awk '{print $1}')
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)

setsid timeout --foreground --signal=TERM --kill-after=30s 3600s \
  "$python" -B scripts/diagnose_siglip_intermediate_readout.py \
  --control-binding "$binding" \
  --control-binding-sha256 "$binding_sha" \
  --checkpoint-seed17 "$control/seed-017/checkpoints/seed-017-epoch-060.pt" \
  --optimization-manifest "$manifest" \
  --optimization-manifest-sha256 "$manifest_sha" \
  --image-root "$images" \
  --result "$staging/result.json" \
  --execute-intermediate-readout &
child=$!
stop_reason=
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  ((rss <= 17179869184)) || stop_reason=rss-cap
  awk -v x="$psi" 'BEGIN{exit !(x>=0.50)}' && stop_reason=psi || true
  ((swap-swap0 <= 131072)) || stop_reason=swap-delta
  if [[ -n $stop_reason ]]; then
    kill -TERM -- "-$child" 2>/dev/null || true
    break
  fi
  sleep 5
done
set +e
wait "$child"
status=$?
set -e
child=
if [[ -n $stop_reason ]]; then
  echo "STOP:$stop_reason" >&2
  exit 125
fi
((status == 0)) || exit "$status"
test -s "$staging/result.json"

"$python" -B - "$staging/result.json" "$manifest_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

from sfora.siglip_intermediate_readout import validate_intermediate_readout_result_bytes

path = pathlib.Path(sys.argv[1])
raw = path.read_bytes()
value = json.loads(raw)
validated = validate_intermediate_readout_result_bytes(raw)
assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
assert value["schema"] == "sfora-siglip-intermediate-readout-v1"
assert value["claim_eligible"] is False
assert value["official_test_access"] is False
assert value["feature_manifest_sha256"] == sys.argv[2]
assert value["expected_depth_count"] == 27
assert value["output_dimensions"] == 512
assert value["fold_count"] == 4
assert isinstance(value["passed"], bool)
assert validated.passed is value["passed"]
print(hashlib.sha256(raw).hexdigest())
PY

rm -rf -- "$images"
unlink "$staging/control-manifest.json"
mv "$staging" "$output"
trap - EXIT INT TERM
REMOTE

rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"
sha256sum "$local_output"
