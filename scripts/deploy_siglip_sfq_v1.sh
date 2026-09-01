#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-sfq-screens}
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-sfq-$source_revision.json}

git diff --quiet HEAD --
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard -- src scripts sitecustomize.py)"
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-sfq-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-sfq-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM

paths=$scratch/paths.z
manifest=$scratch/SOURCE_MANIFEST.sha256
revision_file=$scratch/SOURCE_REVISION
git ls-files -z >"$paths"
: >"$manifest"
while IFS= read -r -d '' relative; do
  digest=$(sha256sum "$relative" | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$relative" >>"$manifest"
done <"$paths"
printf '%s\n' "$source_revision" >"$revision_file"
printf '%s  SOURCE_REVISION\n' "$(sha256sum "$revision_file" | awk '{print $1}')" >>"$manifest"
LC_ALL=C sort -k2 -o "$manifest" "$manifest"

ssh -o BatchMode=yes "$remote_host" bash -s -- "$remote_source" "$remote_output" <<'PREFLIGHT'
set -euo pipefail
if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
test ! -e "$1"
test ! -e "$2"
mkdir -p "$1"
PREFLIGHT
rsync -a --from0 --files-from="$paths" ./ "$remote_host:$remote_source/"
rsync -a -- "$revision_file" "$remote_host:$remote_source/SOURCE_REVISION"
rsync -a -- "$manifest" "$remote_host:$remote_source/SOURCE_MANIFEST.sha256"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$source_revision" <<'REMOTE'
set -euo pipefail
source_dir=$1
output=$2
revision=$3
python=/home/riomus/group-learning/.venv/bin/python3
features=$output/features

cd "$source_dir"
test "$(<SOURCE_REVISION)" = "$revision"
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
if pgrep -f '[r]un_siglip_proxy_control.py' >/dev/null; then
  echo 'control process is still active' >&2
  exit 75
fi
test -x "$python"
test ! -e "$output"
mkdir -p "$output"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$python" -B - <<'PY' >"$output/control-manifest.json"
import sys
from run_siglip_proxy_control import control_manifest_artifact_bytes, load_control_examples

sys.stdout.buffer.write(control_manifest_artifact_bytes(load_control_examples()))
PY
control_sha=$(sha256sum "$output/control-manifest.json" | awk '{print $1}')

timeout --signal=TERM --kill-after=30s 7200s \
  "$python" -B scripts/prepare_siglip_head_features.py \
  --control-manifest "$output/control-manifest.json" \
  --control-manifest-sha256 "$control_sha" \
  --output "$features" \
  --execute-feature-cache >"$output/prepare-receipt.json"

feature_sha=$(sha256sum "$features/features.json" | awk '{print $1}')
test -f "$features/clean-validation.npy"
test -f "$features/burned-diagnostic.npy"
unlink "$features/clean-validation.npy"
unlink "$features/burned-diagnostic.npy"

timeout --signal=TERM --kill-after=30s 7200s \
  "$python" -B scripts/diagnose_siglip_sfq.py \
  --feature-manifest "$features/features.json" \
  --feature-manifest-sha256 "$feature_sha" \
  --feature-source-commit "$revision" \
  --result "$output/result.json" \
  --execute-sfq-folds

"$python" -B - "$features/features.json" "$output/result.json" <<'PY'
import hashlib
import json
import pathlib
import sys

feature_path = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])
feature_raw = feature_path.read_bytes()
result_raw = result_path.read_bytes()
feature = json.loads(feature_raw)
result = json.loads(result_raw)
assert feature_raw == (json.dumps(feature, sort_keys=True, separators=(",", ":")) + "\n").encode()
assert result_raw == (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
assert result["schema"] == "sfora-siglip-sfq-fold-diagnostic-v1"
assert result["claim_eligible"] is False
assert result["official_test_access"] is False
assert result["source_manifest_sha256"] == feature["source_manifest_sha256"]
assert result["feature_cache_manifest_sha256"] == hashlib.sha256(feature_raw).hexdigest()
assert result["input_dimensions"] == 1152
assert result["output_dimensions"] == 512
assert result["fold_count"] == 4
assert isinstance(result["passed"], bool)
print(hashlib.sha256(result_raw).hexdigest())
PY
REMOTE

rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"
sha256sum "$local_output"
