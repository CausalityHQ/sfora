#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_uv_environment=${REMOTE_UV_ENVIRONMENT:-/home/riomus/group-learning/.venv}
source_ref=${SOURCE_REF:-origin/devbox/emafactorial}
local_output=${LOCAL_OUTPUT:-reports/generated/cars-token-set-f0-2026-08-30.json}
remote_output=reports/generated/cars-token-set-f0-2026-08-30.json
source_revision=$(git rev-parse HEAD)

[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ $remote_host =~ ^[A-Za-z0-9_.@:-]+$ ]]
[[ $remote_root =~ ^/[A-Za-z0-9_./-]+$ ]]
[[ $remote_uv_environment =~ ^/[A-Za-z0-9_./-]+$ ]]
[[ $(git rev-parse "$source_ref") == "$source_revision" ]]
git diff --quiet
git diff --cached --quiet
[[ ! -e $local_output ]]

paths_file=$(mktemp /tmp/sfora-token-set-paths.XXXXXX)
manifest_file=$(mktemp /tmp/sfora-token-set-manifest.XXXXXX)
revision_file=$(mktemp /tmp/sfora-token-set-revision.XXXXXX)
cleanup() {
  [[ ! -e $paths_file ]] || unlink "$paths_file"
  [[ ! -e $manifest_file ]] || unlink "$manifest_file"
  [[ ! -e $revision_file ]] || unlink "$revision_file"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' "$source_revision" >"$revision_file"
LC_ALL=C git ls-files -z | LC_ALL=C sort -z >"$paths_file"
while IFS= read -r -d '' path; do
  [[ $path != *$'\n'* && $path != *$'\r'* && $path != *$'\\'* ]]
  sha256sum -- "$path" >>"$manifest_file"
done <"$paths_file"
printf '%s  SOURCE_REVISION\n' \
  "$(sha256sum -- "$revision_file" | awk '{print $1}')" >>"$manifest_file"
LC_ALL=C sort -k2 -o "$manifest_file" "$manifest_file"

LC_ALL=C sort -c -k2 "$manifest_file"
grep -v '  SOURCE_REVISION$' "$manifest_file" \
  | sha256sum --check --strict >/dev/null
for required in \
  scripts/probe_siglip_token_set.py \
  scripts/run_siglip_token_set_f0_v49.sh \
  src/sfora/data.py \
  src/sfora/kernels/set_maxsim.py \
  src/sfora/token_set_proxy_anchor.py \
  tests/test_set_maxsim_kernel.py \
  uv.lock; do
  grep -Fqx "$(sha256sum -- "$required")" "$manifest_file"
done

remote_dir=$remote_root/$source_revision
ssh "$remote_host" bash -s -- "$remote_dir" <<'REMOTE_PREPARE'
set -euo pipefail
remote_dir=$1
[[ ! -e $remote_dir ]]
mkdir -p "$remote_dir"
REMOTE_PREPARE

rsync -a --from0 --files-from="$paths_file" ./ "$remote_host:$remote_dir/"
rsync -a "$revision_file" "$remote_host:$remote_dir/SOURCE_REVISION"
rsync -a "$manifest_file" "$remote_host:$remote_dir/SOURCE_MANIFEST.sha256"

ssh "$remote_host" bash -s -- \
  "$remote_dir" "$source_revision" "$remote_output" \
  "$remote_uv_environment" <<'REMOTE_RUN'
set -euo pipefail
remote_dir=$1
source_revision=$2
output=$3
uv_environment=$4
cd "$remote_dir"

sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT="$uv_environment"
uv run --offline --locked python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA parity gate cannot skip"'
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_set_maxsim_kernel.py -k triton
SOURCE_REVISION="$source_revision" \
SOURCE_MANIFEST="$remote_dir/SOURCE_MANIFEST.sha256" \
OUTPUT="$output" \
  bash scripts/run_siglip_token_set_f0_v49.sh
REMOTE_RUN

mkdir -p "$(dirname -- "$local_output")"
rsync -a "$remote_host:$remote_dir/$remote_output" "$local_output"
printf 'F0 result copied to %s from source %s\n' "$local_output" "$source_revision"
