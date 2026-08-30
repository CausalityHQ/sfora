#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$repo"
host=${REMOTE_HOST:-riomus@100.104.199.68}; root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
venv=${REMOTE_UV_ENVIRONMENT:-/home/riomus/group-learning/.venv}
ref=${SOURCE_REF:-origin/devbox/emafactorial}; revision=$(git rev-parse HEAD)
cell=${CELL:-siglip2-so400m}
[[ $cell == siglip2-so400m || $cell == siglip-so400m ]]
local_output=${LOCAL_OUTPUT:-reports/generated/cars-frozen-substrate-$cell-f0-2026-08-30.json}
remote_output=reports/generated/cars-frozen-substrate-$cell-f0-2026-08-30.json
local_error_manifest=${LOCAL_ERROR_MANIFEST:-}
remote_error_manifest=${REMOTE_ERROR_MANIFEST:-}
[[ $(git rev-parse "$ref") == "$revision" && ! -e $local_output ]]; git diff --quiet; git diff --cached --quiet
if [[ -n $local_error_manifest || -n $remote_error_manifest ]]; then
  [[ -n $local_error_manifest && -n $remote_error_manifest ]]
  [[ $cell == siglip-so400m ]]
  [[ ! -e $local_error_manifest ]]
  [[ $(realpath -m -- "$local_error_manifest") != "$(realpath -m -- "$local_output")" ]]
  [[ $remote_error_manifest =~ ^reports/generated/[A-Za-z0-9._-]+\.json$ ]]
fi
paths=$(mktemp /tmp/sfora-substrate2-paths.XXXXXX); manifest=$(mktemp /tmp/sfora-substrate2-manifest.XXXXXX); rev=$(mktemp /tmp/sfora-substrate2-rev.XXXXXX)
cleanup(){ for p in "$paths" "$manifest" "$rev"; do [[ ! -e $p ]] || unlink "$p"; done; }; trap cleanup EXIT INT TERM
printf '%s\n' "$revision" >"$rev"; LC_ALL=C git ls-files -z | LC_ALL=C sort -z >"$paths"
while IFS= read -r -d '' p; do sha256sum -- "$p" >>"$manifest"; done <"$paths"
printf '%s  SOURCE_REVISION\n' "$(sha256sum "$rev"|awk '{print $1}')" >>"$manifest"; LC_ALL=C sort -k2 -o "$manifest" "$manifest"
dir=$root/$revision; ssh "$host" bash -s -- "$dir" <<'REMOTE'
set -euo pipefail; [[ ! -e $1 ]]; mkdir -p "$1"
REMOTE
rsync -a --from0 --files-from="$paths" ./ "$host:$dir/"; rsync -a "$rev" "$host:$dir/SOURCE_REVISION"; rsync -a "$manifest" "$host:$dir/SOURCE_MANIFEST.sha256"
ssh "$host" bash -s -- "$dir" "$revision" "$remote_output" "$venv" "$cell" "${remote_error_manifest:-none}" <<'REMOTE'
set -euo pipefail; cd "$1"; sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
export PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=$4
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_substrate_screen.py tests/test_probe_frozen_substrate.py tests/test_validate_pass209_m2_artifacts.py tests/test_deploy_frozen_substrate_f0.py
if [[ $6 != none ]]; then
  SOURCE_REVISION=$2 SOURCE_MANIFEST=$PWD/SOURCE_MANIFEST.sha256 OUTPUT=$3 CELL=$5 ERROR_MANIFEST="$6" EXPECTED_CORRECT=1242 bash scripts/run_frozen_substrate_f0_siglip2_v1.sh
else
  SOURCE_REVISION=$2 SOURCE_MANIFEST=$PWD/SOURCE_MANIFEST.sha256 OUTPUT=$3 CELL=$5 bash scripts/run_frozen_substrate_f0_siglip2_v1.sh
fi
REMOTE
mkdir -p "$(dirname -- "$local_output")"; rsync -a -- "$host:$dir/$remote_output" "$local_output"
if [[ -n $local_error_manifest ]]; then
  mkdir -p "$(dirname -- "$local_error_manifest")"
  rsync -a -- "$host:$dir/$remote_error_manifest" "$local_error_manifest"
  uv run --offline --locked python scripts/validate_pass209_m2_artifacts.py --receipt "$local_output" --error-manifest "$local_error_manifest"
fi
