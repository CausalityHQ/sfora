#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_uv_environment=${REMOTE_UV_ENVIRONMENT:-/home/riomus/group-learning/.venv}
source_ref=${SOURCE_REF:-origin/devbox/emafactorial}
local_output=${LOCAL_OUTPUT:-reports/generated/cars-frozen-substrate-f0-2026-08-30.json}
remote_output=reports/generated/cars-frozen-substrate-f0-2026-08-30.json
source_revision=$(git rev-parse HEAD)

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ $(git rev-parse "$source_ref") == "$source_revision" ]]
git diff --quiet
git diff --cached --quiet
[[ ! -e $local_output ]]

paths_file=$(mktemp /tmp/sfora-substrate-paths.XXXXXX)
manifest_file=$(mktemp /tmp/sfora-substrate-manifest.XXXXXX)
revision_file=$(mktemp /tmp/sfora-substrate-revision.XXXXXX)
cleanup(){ for path in "$paths_file" "$manifest_file" "$revision_file"; do [[ ! -e $path ]] || unlink "$path"; done; }
trap cleanup EXIT INT TERM
printf '%s\n' "$source_revision" >"$revision_file"
LC_ALL=C git ls-files -z | LC_ALL=C sort -z >"$paths_file"
while IFS= read -r -d '' path; do sha256sum -- "$path" >>"$manifest_file"; done <"$paths_file"
printf '%s  SOURCE_REVISION\n' "$(sha256sum "$revision_file" | awk '{print $1}')" >>"$manifest_file"
LC_ALL=C sort -k2 -o "$manifest_file" "$manifest_file"

remote_dir=$remote_root/$source_revision
ssh "$remote_host" bash -s -- "$remote_dir" <<'REMOTE'
set -euo pipefail
[[ ! -e $1 ]]
mkdir -p "$1"
REMOTE
rsync -a --from0 --files-from="$paths_file" ./ "$remote_host:$remote_dir/"
rsync -a "$revision_file" "$remote_host:$remote_dir/SOURCE_REVISION"
rsync -a "$manifest_file" "$remote_host:$remote_dir/SOURCE_MANIFEST.sha256"

ssh "$remote_host" bash -s -- "$remote_dir" "$source_revision" "$remote_output" "$remote_uv_environment" <<'REMOTE'
set -euo pipefail
cd "$1"
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
export PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=$4
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_substrate_screen.py \
  tests/test_probe_frozen_substrate.py \
  tests/test_deploy_frozen_substrate_f0.py
SOURCE_REVISION=$2 SOURCE_MANIFEST=$PWD/SOURCE_MANIFEST.sha256 OUTPUT=$3 bash scripts/run_frozen_substrate_f0_v1.sh
REMOTE

mkdir -p "$(dirname -- "$local_output")"
rsync -a "$remote_host:$remote_dir/$remote_output" "$local_output"
printf 'Frozen substrate F0 result copied to %s from source %s\n' "$local_output" "$source_revision"
