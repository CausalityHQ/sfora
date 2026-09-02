#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-control-audit-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-control-checkpoint-audits}
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-control-checkpoint-audit-$source_revision}
case "$local_output" in
  /tmp/sfora-control-checkpoint-audit-*) ;;
  *) echo "local output must remain under /tmp/sfora-control-checkpoint-audit-*" >&2; exit 64 ;;
esac

git diff --quiet HEAD --
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard -- src scripts sitecustomize.py)"
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-control-audit-deploy.XXXXXX)
local_created=false
local_completed=false
cleanup() {
  if [[ $local_created == true && $local_completed != true ]]; then
    rm -rf -- "$local_output"
  fi
  case "$scratch" in
    /tmp/sfora-control-audit-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM

bundle=$scratch/source.bundle
git bundle create "$bundle" HEAD
remote_bundle=/tmp/sfora-control-audit-source-$source_revision.bundle

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_bundle" \
  "$remote_root" <<'PREFLIGHT'
set -euo pipefail
mkdir -p "$5"
if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py' >/dev/null; then
  echo 'control or native probe is still active' >&2
  exit 75
fi
if pgrep -f '[a]udit_siglip_control_checkpoint.py' >/dev/null; then
  echo 'checkpoint audit is already active' >&2
  exit 75
fi
test -f "$3/control.receipt.json"
for seed in 017 029 043; do test -f "$3/seed-$seed.receipt.json"; done
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt" || {
  echo 'missing seed-017 terminal checkpoint' >&2; exit 76;
}
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.checkpoint.json" || {
  echo 'missing seed-017 terminal checkpoint receipt' >&2; exit 76;
}
test ! -e "$1" || { echo "stale source clone: $1" >&2; exit 76; }
test ! -e "$2" || { echo "existing audit output: $2" >&2; exit 76; }
test ! -e "$2.partial" || { echo "stale audit staging: $2.partial" >&2; exit 76; }
test ! -e "$4" || { echo "stale source bundle: $4" >&2; exit 76; }
PREFLIGHT

rsync -a -- "$bundle" "$remote_host:$remote_bundle"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$source_revision" \
  "$remote_bundle" "$remote_root" <<'REMOTE'
set -euo pipefail
source_dir=$1
output=$2
control=$3
revision=$4
bundle=$5
root=$6
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output.partial
launcher=$output.launcher.partial
child=
completed=false
cleanup_remote() {
  if [[ -n ${child:-} ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  test ! -e "$bundle" || unlink "$bundle"
  test ! -e "$launcher" || unlink "$launcher"
  test "$staging" = "$output.partial" || exit 99
  test ! -e "$staging" || rm -rf -- "$staging"
  if [[ $completed != true ]]; then
    case "$source_dir" in
      "$root/$revision")
        test ! -e "$source_dir" || rm -rf -- "$source_dir"
        ;;
      *) exit 99 ;;
    esac
  fi
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

if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py' >/dev/null; then
  echo 'control or native probe is still active' >&2
  exit 75
fi
if pgrep -f '[a]udit_siglip_control_checkpoint.py' >/dev/null; then
  echo 'checkpoint audit is already active' >&2
  exit 75
fi
test -x "$python"
test ! -e "$output"
test ! -e "$staging"
mkdir -p "$staging"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
setsid timeout --foreground --signal=TERM --kill-after=30s 14400s \
  "$python" -B scripts/audit_siglip_control_checkpoint.py \
  --aggregate "$control/control.receipt.json" \
  --seed-receipt "$control/seed-017.receipt.json" \
  --seed-receipt "$control/seed-029.receipt.json" \
  --seed-receipt "$control/seed-043.receipt.json" \
  --checkpoint-directory "$control/seed-017/checkpoints" \
  --selected-seed 17 \
  --initial-output "$staging/initial.json" \
  --output "$staging/trained.json" \
  --raw-twin-output "$staging/raw-twin.json" \
  --projected-twin-output "$staging/projected-twin.json" \
  --raw-twin-inference-output "$staging/raw-twin-inference.json" \
  --projected-twin-inference-output "$staging/projected-twin-inference.json" \
  --execute-checkpoint-audit >"$launcher" &
child=$!
stop_reason=
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  # The authenticated GB10 PyTorch stack peaks near 31.3 GB while restoring
  # SigLIP-so400m; retain measured headroom without approaching host capacity.
  ((rss <= 42949672960)) || stop_reason=rss-cap
  awk -v x="$psi" 'BEGIN{exit !(x>=0.50)}' && stop_reason=psi || true
  ((swap-swap0 <= 262144)) || stop_reason=swap-delta
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

"$python" -B - "$staging" "$launcher" <<'PY'
import hashlib
import json
import pathlib
import sys

from run_siglip_proxy_control import load_control_examples
from sfora.siglip_checkpoint_audit import (
    SiglipCheckpointAuditAuthority,
    validate_siglip_checkpoint_audit_bytes,
)
from sfora.siglip_initial_control_audit import (
    SiglipInitialControlAuditAuthority,
    validate_siglip_initial_control_audit_bytes,
)
from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    validate_twin_reachability_artifact_bytes,
    validate_twin_reachability_inference_artifact_bytes,
)

root = pathlib.Path(sys.argv[1])
launcher = pathlib.Path(sys.argv[2])
expected = {
    "initial.json": ("initial_sha256", "initial_output", "initial", None),
    "trained.json": ("sha256", "output", "trained", None),
    "raw-twin.json": ("raw_twin_sha256", "raw_twin_output", "twin", "trained-raw"),
    "projected-twin.json": (
        "projected_twin_sha256",
        "projected_twin_output",
        "twin",
        "trained-projected",
    ),
    "raw-twin-inference.json": (
        "raw_twin_inference_sha256",
        "raw_twin_inference_output",
        "inference",
        "trained-raw",
    ),
    "projected-twin-inference.json": (
        "projected_twin_inference_sha256",
        "projected_twin_inference_output",
        "inference",
        "trained-projected",
    ),
}
if len(expected) != 6:
    raise AssertionError("exactly six audit artifacts are required")
bands = load_control_examples()
examples = bands.burned_diagnostic
example_ids = tuple(row.example_id for row in examples)
labels = tuple(row.label for row in examples)
receipt_raw = launcher.read_bytes()
receipt = json.loads(receipt_raw)
if receipt_raw != (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode():
    raise ValueError("launcher receipt differs")
digests = []
for name, (digest_key, path_key, kind, plane) in expected.items():
    path = root / name
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != expected:
        raise ValueError(f"{name} canonical output differs")
    assert value["claim_eligible"] is False
    digest = hashlib.sha256(raw).hexdigest()
    digests.append(digest)
    if receipt.get(digest_key) != digest or receipt.get(path_key) != str(path):
        raise ValueError("launcher receipt differs")
    if kind == "initial":
        assert value["official_test_access"] is False
        authority = SiglipInitialControlAuditAuthority(**value["authority"])
        validate_siglip_initial_control_audit_bytes(
            raw,
            expected_authority=authority,
            expected_example_ids=example_ids,
            expected_labels=labels,
        )
    elif kind == "trained":
        assert value["official_test_access"] is False
        authority = SiglipCheckpointAuditAuthority(**value["authority"])
        validate_siglip_checkpoint_audit_bytes(
            raw,
            expected_authority=authority,
            expected_example_ids=example_ids,
            expected_labels=labels,
        )
    else:
        assert value["authority"]["plane"] == plane
        authority = TwinReachabilityAuthority(**value["authority"])
        if kind == "twin":
            validate_twin_reachability_artifact_bytes(raw, expected=authority)
        else:
            validate_twin_reachability_inference_artifact_bytes(raw, expected=authority)
    print(f"{digest}  {name}")
if len(set(digests)) != 6:
    raise ValueError("audit artifact digests are not distinct")
PY

unlink "$launcher"
test "$(find "$staging" -maxdepth 1 -type f | wc -l)" -eq 6
mv "$staging" "$output"
completed=true
trap - EXIT INT TERM
REMOTE

mkdir "$local_output"
local_created=true
rsync -a -- "$remote_host:$remote_output/" "$local_output/"
test "$(find "$local_output" -maxdepth 1 -type f | wc -l)" -eq 6
sha256sum "$local_output"/*.json
local_completed=true
