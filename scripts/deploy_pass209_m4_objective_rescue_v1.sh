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

# Every evidence location is an explicit remote authority and must exist before upload.
: "${DGX_ERROR_MANIFEST:?DGX_ERROR_MANIFEST is required}"
: "${DGX_DINOV2_PREREQUISITE:?DGX_DINOV2_PREREQUISITE is required}"
: "${DGX_SIGLIP2_PREREQUISITE:?DGX_SIGLIP2_PREREQUISITE is required}"
: "${DGX_SELECTING_PREREQUISITE:?DGX_SELECTING_PREREQUISITE is required}"
: "${DGX_M3_SEED_017:?DGX_M3_SEED_017 is required}"
: "${DGX_M3_SEED_029:?DGX_M3_SEED_029 is required}"
: "${DGX_M3_SEED_043:?DGX_M3_SEED_043 is required}"
: "${DGX_M3_AGGREGATE:?DGX_M3_AGGREGATE is required}"

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
LC_ALL=C sort -k2,2 -o "$manifest" "$manifest"
git archive --format=tar --output="$archive" HEAD
tar --append --file="$archive" -C "$scratch" SOURCE_REVISION SOURCE_MANIFEST.sha256
archive_sha256=$(sha256sum "$archive" | awk '{print $1}')
remote_archive="$remote_root/$archive_sha256.tar"
remote_source="$remote_root/$archive_sha256"

{
  printf 'root=%q\n' "$remote_root"
  printf 'source_dir=%q\n' "$remote_source"
  printf 'archive=%q\n' "$remote_archive"
  printf 'output=%q\n' "$remote_output"
  printf 'uv_environment=%q\n' "$remote_uv"
  printf 'prerequisites=('
  for prerequisite in \
    "$DGX_ERROR_MANIFEST" "$DGX_DINOV2_PREREQUISITE" \
    "$DGX_SIGLIP2_PREREQUISITE" "$DGX_SELECTING_PREREQUISITE" \
    "$DGX_M3_SEED_017" "$DGX_M3_SEED_029" "$DGX_M3_SEED_043" \
    "$DGX_M3_AGGREGATE"; do
    printf ' %q' "$prerequisite"
  done
  printf ' )\n'
  cat <<'PREFLIGHT'
set -euo pipefail
set -- "${prerequisites[@]}"
mkdir -p "$root"
if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py|[a]udit_siglip_control_checkpoint.py|[p]robe_frozen_substrate.py|[r]un_siglip_rsta_stage_a.py|[d]iagnose_siglip_(rsta_stage_a|intermediate_readout|head_screen|sfq).py' >/dev/null; then
  echo 'active GPU campaign has not released the DGX' >&2
  exit 75
fi
if pgrep -f '[r]un_pass209_m4_(cell|objective_rescue)' >/dev/null; then
  echo 'refusing to duplicate a live M4 campaign' >&2
  exit 73
fi
[[ ! -e $source_dir && ! -e $archive ]] || {
  echo 'refusing existing content-addressed M4 source' >&2
  exit 73
}
[[ ! -e $output && ! -e $output.launch.log ]] || {
  echo 'refusing existing M4 output' >&2
  exit 73
}
output_parent=${output%/*}
[[ -d $output_parent && -w $output_parent ]] || {
  echo "unusable M4 output parent: $output_parent" >&2
  exit 76
}
[[ -d $uv_environment ]] || {
  echo "missing uv environment: $uv_environment" >&2
  exit 76
}
for prerequisite in "$@"; do
  [[ -f $prerequisite ]] || {
    echo "missing M4 prerequisite: $prerequisite" >&2
    exit 76
  }
done
PREFLIGHT
} | ssh -o BatchMode=yes "$remote" bash -s
scp -o BatchMode=yes "$archive" "$remote:$remote_archive"

{
  printf 'root=%q\n' "$remote_root"
  printf 'source_dir=%q\n' "$remote_source"
  printf 'archive=%q\n' "$remote_archive"
  printf 'archive_sha256=%q\n' "$archive_sha256"
  printf 'source_revision=%q\n' "$source_revision"
  printf 'output=%q\n' "$remote_output"
  cat <<'REMOTE'
set -euo pipefail
committed=false
cleanup_remote() {
  if [[ $committed != true ]]; then
    test ! -e "$archive" || unlink "$archive"
    case "$source_dir" in
      "$root"/*) test ! -e "$source_dir" || rm -rf -- "$source_dir" ;;
      *) exit 99 ;;
    esac
  fi
}
trap cleanup_remote EXIT INT TERM
[[ $(sha256sum "$archive" | awk '{print $1}') == "$archive_sha256" ]]
mkdir "$source_dir"
tar --extract --file="$archive" --directory="$source_dir"
cd "$source_dir"
[[ $(<SOURCE_REVISION) == "$source_revision" ]]
sha256sum --check --strict SOURCE_MANIFEST.sha256 >/dev/null
if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py|[a]udit_siglip_control_checkpoint.py|[p]robe_frozen_substrate.py|[r]un_siglip_rsta_stage_a.py|[d]iagnose_siglip_(rsta_stage_a|intermediate_readout|head_screen|sfq).py' >/dev/null; then
  echo 'active GPU campaign has not released the DGX' >&2
  exit 75
fi
if pgrep -f '[r]un_pass209_m4_(cell|objective_rescue)' >/dev/null || [[ -e $output ]]; then
  echo 'refusing to duplicate a live M4 campaign' >&2
  exit 73
fi
test ! -e "$archive" || unlink "$archive"
committed=true
trap - EXIT INT TERM
REMOTE
} | ssh -o BatchMode=yes "$remote" bash -s

{
  printf 'root=%q\n' "$remote_root"
  printf 'source_dir=%q\n' "$remote_source"
  printf 'source_revision=%q\n' "$source_revision"
  printf 'uv_environment=%q\n' "$remote_uv"
  printf 'output=%q\n' "$remote_output"
  printf 'error_manifest=%q\n' "$DGX_ERROR_MANIFEST"
  printf 'dinov2_prerequisite=%q\n' "$DGX_DINOV2_PREREQUISITE"
  printf 'siglip2_prerequisite=%q\n' "$DGX_SIGLIP2_PREREQUISITE"
  printf 'selecting_prerequisite=%q\n' "$DGX_SELECTING_PREREQUISITE"
  printf 'm3_seed_017=%q\n' "$DGX_M3_SEED_017"
  printf 'm3_seed_029=%q\n' "$DGX_M3_SEED_029"
  printf 'm3_seed_043=%q\n' "$DGX_M3_SEED_043"
  printf 'm3_aggregate=%q\n' "$DGX_M3_AGGREGATE"
  cat <<'LAUNCH'
set -euo pipefail
launch_committed=false
cleanup_failed_launch() {
  if [[ $launch_committed != true && ! -e $output ]]; then
    test ! -e "$output.launch.log" || unlink "$output.launch.log"
    case "$source_dir" in
      "$root"/*) test ! -e "$source_dir" || rm -rf -- "$source_dir" ;;
      *) exit 99 ;;
    esac
  fi
}
trap cleanup_failed_launch EXIT INT TERM
cd "$source_dir"
if pgrep -f '[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py|[a]udit_siglip_control_checkpoint.py|[p]robe_frozen_substrate.py|[r]un_siglip_rsta_stage_a.py|[d]iagnose_siglip_(rsta_stage_a|intermediate_readout|head_screen|sfq).py' >/dev/null; then
  echo 'active GPU campaign has not released the DGX' >&2
  exit 75
fi
if pgrep -f '[r]un_pass209_m4_(cell|objective_rescue)' >/dev/null || [[ -e $output || -e $output.launch.log ]]; then
  echo 'refusing to duplicate a live M4 campaign' >&2
  exit 73
fi
nohup setsid env \
  SOURCE_REVISION="$source_revision" \
  SOURCE_MANIFEST="$source_dir/SOURCE_MANIFEST.sha256" \
  UV_PROJECT_ENVIRONMENT="$uv_environment" \
  OUTPUT_DIR="$output" \
  ERROR_MANIFEST="$error_manifest" \
  DINOV2_PREREQUISITE="$dinov2_prerequisite" \
  SIGLIP2_PREREQUISITE="$siglip2_prerequisite" \
  SELECTING_PREREQUISITE="$selecting_prerequisite" \
  M3_SEED_017="$m3_seed_017" \
  M3_SEED_029="$m3_seed_029" \
  M3_SEED_043="$m3_seed_043" \
  M3_AGGREGATE="$m3_aggregate" \
  bash scripts/run_pass209_m4_objective_rescue_v1.sh \
  >"$output.launch.log" 2>&1 </dev/null &
pid=$!
sleep 2
kill -0 "$pid" 2>/dev/null || {
  cat "$output.launch.log" >&2
  echo 'M4 launch did not survive' >&2
  exit 70
}
launch_committed=true
trap - EXIT INT TERM
printf '%s\n' "$pid"
LAUNCH
} | ssh -o BatchMode=yes "$remote" bash -s
