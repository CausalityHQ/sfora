#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_artifact=${REMOTE_ARTIFACT:-/home/riomus/sfora-spatial-tail-recovery/498a10d34b9095fa92a7f45f8e0af6031b33344d/spatial-tail.safetensors}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-spatial-tail-latency}
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-spatial-tail-latency-$source_revision.json}
artifact_sha=cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9
artifact_bytes=67973944

source_files=(
  scripts/probe_siglip_spatial_tail_latency.py
  scripts/deploy_siglip_spatial_tail_latency_v1.sh
  scripts/probe_siglip_depth_recovery.py
  scripts/probe_siglip_spatial_tail_recovery.py
  scripts/run_siglip_proxy_control.py
  src/sfora/siglip_depth_recovery.py
  src/sfora/siglip_proxy_control.py
)
git diff --quiet HEAD -- "${source_files[@]}"
git diff --cached --quiet -- "${source_files[@]}"
for path in "${source_files[@]}"; do
  git ls-files --error-unmatch "$path" >/dev/null
done
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-spatial-tail-latency-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-spatial-tail-latency-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM
bundle=$scratch/source.bundle
git bundle create "$bundle" HEAD
remote_bundle=/tmp/sfora-spatial-tail-latency-source-$source_revision.bundle

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_artifact" \
  "$remote_bundle" "$artifact_sha" "$artifact_bytes" <<'PREFLIGHT'
set -euo pipefail
pgrep -f '[p]robe_siglip_spatial_tail_latency.py' >/dev/null && {
  echo 'spatial tail latency process is already active' >&2
  exit 75
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -f "$3/control.receipt.json"
test -f "$3/seed-017.receipt.json"
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt"
test -f "$4"
test "$(stat -c%s "$4")" = "$7"
test "$(sha256sum "$4" | awk '{print $1}')" = "$6"
test ! -e "$1"
test ! -e "$2"
test ! -e "$2.partial"
test ! -e "$5"
PREFLIGHT
rsync -a -- "$bundle" "$remote_host:$remote_bundle"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_artifact" \
  "$source_revision" "$remote_bundle" "$artifact_sha" "$artifact_bytes" <<'REMOTE'
set -euo pipefail
source_dir=$1
output=$2
control_root=$3
artifact=$4
revision=$5
bundle=$6
artifact_sha=$7
artifact_bytes=$8
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output.partial
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

pgrep -f '[p]robe_siglip_spatial_tail_latency.py' >/dev/null && {
  echo 'spatial tail latency process is already active' >&2
  exit 75
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -x "$python"
test -f "$artifact"
test "$(stat -c%s "$artifact")" = "$artifact_bytes"
test "$(sha256sum "$artifact" | awk '{print $1}')" = "$artifact_sha"
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
setsid timeout --foreground --signal=TERM --kill-after=30s 3600s \
  "$python" -B scripts/probe_siglip_spatial_tail_latency.py \
  --control-root "$control_root" \
  --spatial-artifact "$artifact" \
  --spatial-artifact-sha256 "$artifact_sha" \
  --output "$staging/result.json" \
  --execute-spatial-tail-latency &
child=$!
stop_reason=
psi_hits=0
progress_gap=0
last_cpu=
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  cpu=$(ps -o time= -g "$child" 2>/dev/null || true)
  ((rss <= 51539607552)) || stop_reason=rss-cap
  if awk -v x="$psi" 'BEGIN{exit !(x>=0.50)}'; then
    ((psi_hits+=1))
  else
    psi_hits=0
  fi
  awk -v x="$psi" 'BEGIN{exit !(x>=0.79)}' && stop_reason=psi-immediate || true
  ((psi_hits < 3)) || stop_reason=psi-sustained
  ((swap-swap0 <= 262144)) || stop_reason=swap-delta
  if [[ $cpu = "$last_cpu" ]]; then ((progress_gap+=5)); else progress_gap=0; last_cpu=$cpu; fi
  ((progress_gap < 300)) || stop_reason=progress-gap
  if [[ -n $stop_reason ]]; then kill -TERM -- "-$child" 2>/dev/null || true; break; fi
  sleep 5
done
set +e
wait "$child"
status=$?
set -e
child=
if [[ -n $stop_reason ]]; then echo "STOP:$stop_reason" >&2; exit 125; fi
((status == 0)) || exit "$status"
test -s "$staging/result.json"
"$python" -B - "$staging/result.json" "$artifact_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

from sfora.siglip_depth_recovery import speed_gate

raw = pathlib.Path(sys.argv[1]).read_bytes()
assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
value = json.loads(raw)
assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
assert value["schema"] == "sfora-siglip-spatial-tail-latency-v1"
assert value["claim_eligible"] is False
assert value["quality_measured"] is False
assert value["external_evaluation_access"] is False
assert value["selected_arm"] == "tokenwise-control"
assert value["retained_leading_blocks"] == 18
assert value["spatial_artifact_sha256"] == sys.argv[2]
assert value["speed_passed"] is speed_gate(value["timing"]["windows"])
print(hashlib.sha256(raw).hexdigest())
PY
mv "$staging" "$output"
trap - EXIT INT TERM
REMOTE

rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"
sha256sum "$local_output"
