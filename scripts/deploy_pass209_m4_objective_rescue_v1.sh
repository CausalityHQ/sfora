#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

remote=${DGX_REMOTE:-riomus@100.104.199.68}
remote_root=${DGX_SOURCE_ROOT:-/home/riomus/sfora-revisions}
remote_output=${DGX_OUTPUT_DIR:?DGX_OUTPUT_DIR is required}
remote_uv=${DGX_UV_PROJECT_ENVIRONMENT:?DGX_UV_PROJECT_ENVIRONMENT is required}
source_revision=$(git rev-parse HEAD)
[[ $source_revision =~ ^[0-9a-f]{40}$ ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
for required in \
  scripts/run_pass209_m4_cell.py \
  scripts/analyze_pass209_m4.py \
  scripts/run_pass209_m4_objective_rescue_v1.sh \
  scripts/deploy_pass209_m4_objective_rescue_v1.sh \
  src/sfora/pass209_m4.py; do
  git ls-files --error-unmatch -- "$required" >/dev/null
done

scratch=$(mktemp -d /tmp/sfora-m4-deploy.XXXXXX)
archive=$scratch/source.tar
manifest=$scratch/SOURCE_MANIFEST.sha256
revision_file=$scratch/SOURCE_REVISION
cleanup() {
  [[ ! -e $archive ]] || unlink "$archive"
  [[ ! -e $manifest ]] || unlink "$manifest"
  [[ ! -e $revision_file ]] || unlink "$revision_file"
  rmdir "$scratch"
}
trap cleanup EXIT INT TERM

printf '%s\n' "$source_revision" >"$revision_file"
git ls-files -z | sort -z | xargs -0 sha256sum >"$manifest"
sha256sum "$revision_file" | sed "s#  $revision_file#  SOURCE_REVISION#" >>"$manifest"
LC_ALL=C sort -o "$manifest" "$manifest"
git archive --format=tar --output="$archive" HEAD
tar --append --file="$archive" -C "$scratch" SOURCE_REVISION SOURCE_MANIFEST.sha256
archive_sha256=$(sha256sum "$archive" | awk '{print $1}')
remote_archive="$remote_root/$archive_sha256.tar"
remote_source="$remote_root/$archive_sha256"

ssh "$remote" "set -euo pipefail
mkdir -p '$remote_root'
[[ ! -e '$remote_source' && ! -e '$remote_archive' ]] || {
  echo 'refusing existing content-addressed M4 source' >&2
  exit 73
}"
scp "$archive" "$remote:$remote_archive"

ssh "$remote" "set -euo pipefail
[[ \$(sha256sum '$remote_archive' | awk '{print \$1}') == '$archive_sha256' ]]
mkdir '$remote_source'
tar --extract --file='$remote_archive' --directory='$remote_source'
cd '$remote_source'
[[ \$(<SOURCE_REVISION) == '$source_revision' ]]
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
if pgrep -f 'run_siglip_proxy_control.py (train|aggregate)' >/dev/null; then
  echo 'active M1/M3 control has not released the DGX' >&2
  exit 75
fi
if pgrep -f 'run_pass209_m4_(cell|objective_rescue)' >/dev/null || [[ -e '$remote_output' ]]; then
  echo 'refusing to duplicate a live M4 campaign' >&2
  exit 73
fi
"

# The four evidence locations are explicit operator-provided remote authorities.
: "${DGX_ERROR_MANIFEST:?DGX_ERROR_MANIFEST is required}"
: "${DGX_DINOV2_PREREQUISITE:?DGX_DINOV2_PREREQUISITE is required}"
: "${DGX_SIGLIP2_PREREQUISITE:?DGX_SIGLIP2_PREREQUISITE is required}"
: "${DGX_SELECTING_PREREQUISITE:?DGX_SELECTING_PREREQUISITE is required}"
: "${DGX_M3_SEED_017:?DGX_M3_SEED_017 is required}"
: "${DGX_M3_SEED_029:?DGX_M3_SEED_029 is required}"
: "${DGX_M3_SEED_043:?DGX_M3_SEED_043 is required}"
: "${DGX_M3_AGGREGATE:?DGX_M3_AGGREGATE is required}"

ssh "$remote" "cd '$remote_source' &&
nohup setsid env \
  SOURCE_REVISION='$source_revision' \
  SOURCE_MANIFEST='$remote_source/SOURCE_MANIFEST.sha256' \
  UV_PROJECT_ENVIRONMENT='$remote_uv' \
  OUTPUT_DIR='$remote_output' \
  ERROR_MANIFEST='$DGX_ERROR_MANIFEST' \
  DINOV2_PREREQUISITE='$DGX_DINOV2_PREREQUISITE' \
  SIGLIP2_PREREQUISITE='$DGX_SIGLIP2_PREREQUISITE' \
  SELECTING_PREREQUISITE='$DGX_SELECTING_PREREQUISITE' \
  M3_SEED_017='$DGX_M3_SEED_017' \
  M3_SEED_029='$DGX_M3_SEED_029' \
  M3_SEED_043='$DGX_M3_SEED_043' \
  M3_AGGREGATE='$DGX_M3_AGGREGATE' \
  bash scripts/run_pass209_m4_objective_rescue_v1.sh \
  >'$remote_output.launch.log' 2>&1 </dev/null &
printf '%s\n' \$!"
