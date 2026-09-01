#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-rsta-screens}
control_source_revision=034e66407c5de6e2ff1acf3d18455b10760d3509
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-rsta-$source_revision.json}

git diff --quiet HEAD --
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard -- src scripts sitecustomize.py)"
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-rsta-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-rsta-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

bundle=$scratch/source.bundle
git bundle create "$bundle" HEAD

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "/tmp/sfora-rsta-source-$source_revision.bundle" <<'PREFLIGHT'
set -euo pipefail
if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
if pgrep -f '[d]iagnose_siglip_rsta_stage_a.py' >/dev/null; then
  echo 'RSTA scientific process is already active' >&2
  exit 75
fi
test -f "$3/control.receipt.json"
for seed in 017 029 043; do
  test -f "$3/seed-$seed.receipt.json"
  test -f "$3/seed-$seed/checkpoints/seed-$seed-epoch-060.pt"
done
test ! -e "$1"
test ! -e "$2"
test ! -e "$2.partial"
test ! -e "$4"
PREFLIGHT

remote_bundle=/tmp/sfora-rsta-source-$source_revision.bundle
rsync -a -- "$bundle" "$remote_host:$remote_bundle"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$source_revision" "$remote_bundle" \
  "$control_source_revision" <<'REMOTE'
set -euo pipefail
source_dir=$1
output=$2
control=$3
revision=$4
bundle=$5
expected_control_source=$6
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output.partial
images=$staging/images
cleanup_remote() {
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
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)"
printf '%s\n' "$revision" >SOURCE_REVISION
cp /dev/null SOURCE_MANIFEST.sha256
git ls-files -z | while IFS= read -r -d '' relative; do
  digest=$(sha256sum "$relative" | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$relative" >>SOURCE_MANIFEST.sha256
done
printf '%s  SOURCE_REVISION\n' "$(sha256sum SOURCE_REVISION | awk '{print $1}')" >>SOURCE_MANIFEST.sha256
LC_ALL=C sort -k2 -o SOURCE_MANIFEST.sha256 SOURCE_MANIFEST.sha256
test "$(<SOURCE_REVISION)" = "$revision"
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
if pgrep -f '[d]iagnose_siglip_rsta_stage_a.py' >/dev/null; then
  echo 'RSTA scientific process is already active' >&2
  exit 75
fi
test -x "$python"
test ! -e "$output"
test ! -e "$staging"
mkdir -p "$staging/controller-scratch"

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

control_source=$("$python" -B - "$control/seed-017.receipt.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], "rb"))
revision = value["source"]["revision"]
assert isinstance(revision, str) and len(revision) == 40
print(revision)
PY
)
test "$control_source" = "$expected_control_source"
hostname=$(hostname)

set +e
timeout --foreground --signal=TERM --kill-after=30s 7200s bash -s -- \
  "$python" scripts/run_siglip_rsta_stage_a.py \
  --seed-receipt "$control/seed-017.receipt.json" \
  --seed-receipt "$control/seed-029.receipt.json" \
  --seed-receipt "$control/seed-043.receipt.json" \
  --aggregate-receipt "$control/control.receipt.json" \
  --checkpoint "$control/seed-017/checkpoints/seed-017-epoch-060.pt" \
  --checkpoint "$control/seed-029/checkpoints/seed-029-epoch-060.pt" \
  --checkpoint "$control/seed-043/checkpoints/seed-043-epoch-060.pt" \
  --control-manifest "$staging/control-manifest.json" \
  --optimization-image-root "$images" \
  --scientific-cli "$source_dir/scripts/diagnose_siglip_rsta_stage_a.py" \
  --scratch-root "$staging/controller-scratch" \
  --result-output "$staging/result.json" \
  --terminal-output "$staging/terminal.json" \
  --expected-hostname "$hostname" \
  --expected-source-commit "$control_source" \
  --expected-controller-source-commit "$revision" \
  --execute-controller 2>"$staging/controller.stderr" <<'SUPERVISOR'
set -euo pipefail
python=$1
shift
controller_pid=
stop_tree() {
  if [[ -n ${controller_pid:-} ]] && kill -0 "$controller_pid" 2>/dev/null; then
    while read -r child; do
      test -z "$child" || kill -TERM -- "-$child" 2>/dev/null || true
    done < <(pgrep -P "$controller_pid" || true)
    kill -TERM "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" 2>/dev/null || true
  fi
}
trap stop_tree EXIT INT TERM
"$python" -B "$@" &
controller_pid=$!
set +e
wait "$controller_pid"
status=$?
set -e
controller_pid=
exit "$status"
SUPERVISOR
controller_status=$?
set -e
printf '%s\n' "$controller_status" >"$staging/controller.exit"
printf '%s\n' "$control_source" >"$staging/control.source"

if test ! -f "$staging/result.json" && test ! -f "$staging/terminal.json"; then
  "$python" -B - "$controller_status" "$staging/controller.stderr" \
    >"$staging/terminal.json" <<'PY'
import hashlib
import json
import pathlib
import sys

status = int(sys.argv[1])
stderr = pathlib.Path(sys.argv[2]).read_bytes()
value = {
    "schema": "rsta-terminal-v1",
    "claim_eligible": False,
    "reason": "controller-exit",
    "exit_code": status,
    "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
}
sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY
fi
unlink "$staging/controller.stderr"
test "$staging" = "$output.partial" || exit 99
rm -rf -- "$images" "$staging/controller-scratch"

result_count=0
test ! -f "$staging/result.json" || result_count=$((result_count + 1))
test ! -f "$staging/terminal.json" || result_count=$((result_count + 1))
test "$result_count" -eq 1
if test -f "$staging/result.json"; then
  printf 'result.json\n' >"$staging/artifact.kind"
else
  printf 'terminal.json\n' >"$staging/artifact.kind"
fi
mv "$staging" "$output"
trap - EXIT INT TERM
REMOTE

rsync -a -- "$remote_host:$remote_output/controller.exit" "$scratch/controller.exit"
rsync -a -- "$remote_host:$remote_output/artifact.kind" "$scratch/artifact.kind"
rsync -a -- "$remote_host:$remote_output/control.source" "$scratch/control.source"
artifact=$(<"$scratch/artifact.kind")
case "$artifact" in
  result.json|terminal.json) ;;
  *) exit 98 ;;
esac
rsync -a -- "$remote_host:$remote_output/$artifact" "$local_output"
controller_status=$(<"$scratch/controller.exit")
control_source=$(<"$scratch/control.source")
[[ $controller_status =~ ^[0-9]+$ ]]
[[ $control_source =~ ^[0-9a-f]{40}$ ]]

python3 - "$local_output" "$controller_status" "$control_source" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
control_source = sys.argv[3]
raw = path.read_bytes()
value = json.loads(raw)
assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
assert value["claim_eligible"] is False
if value["schema"] == "siglip-rsta-stage-a-scientific-result-v1":
    assert value["result"]["verdict"] in {"PASS_ONWARD", "FAIL", "UNRESOLVED"}
    assert value["authority"]["control_binding"]["source_commit"] == control_source
    assert (
        value["authority"]["optimization_manifest_sha256"]
        == value["authority"]["control_binding"]["optimization_manifest_sha256"]
    )
    assert status == 0
elif value["schema"] == "siglip-rsta-stage-a-result-v1":
    assert value["verdict"] == "INVALID"
    assert status != 0
else:
    assert value["schema"] == "rsta-terminal-v1"
    assert status != 0
print(hashlib.sha256(raw).hexdigest())
raise SystemExit(0 if status == 0 else 1)
PY
