#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-spatial-tail-recovery}
source_revision=$(git rev-parse HEAD)
remote_source=$remote_root/$source_revision
remote_output=$remote_output_root/$source_revision
local_output=${LOCAL_OUTPUT:-/tmp/sfora-spatial-tail-$source_revision.json}

source_files=(
  src/sfora/siglip_spatial_tail_recovery.py
  scripts/probe_siglip_spatial_tail_recovery.py
  scripts/deploy_siglip_spatial_tail_recovery_v1.sh
)
git diff --quiet HEAD -- "${source_files[@]}"
git diff --cached --quiet -- "${source_files[@]}"
for path in "${source_files[@]}"; do
  git ls-files --error-unmatch "$path" >/dev/null
done
test ! -e "$local_output"

scratch=$(mktemp -d /tmp/sfora-spatial-tail-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-spatial-tail-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM
bundle=$scratch/source.bundle
git bundle create "$bundle" HEAD
remote_bundle=/tmp/sfora-spatial-tail-source-$source_revision.bundle

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_bundle" <<'PREFLIGHT'
set -euo pipefail
pgrep -f '[p]robe_siglip_spatial_tail_recovery.py' >/dev/null && {
  echo 'spatial tail process is already active' >&2
  exit 75
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -f "$3/control.receipt.json"
test -f "$3/seed-017.receipt.json"
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt"
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
optimization_images=$staging/optimization-images
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

pgrep -f '[p]robe_siglip_spatial_tail_recovery.py' >/dev/null && {
  echo 'spatial tail process is already active' >&2
  exit 75
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
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

"$python" -B - "$staging/control-manifest.json" "$optimization_images" <<'PY'
import pathlib
import sys

from run_siglip_proxy_control import load_control_examples, write_control_manifest_artifacts

write_control_manifest_artifacts(
    output=pathlib.Path(sys.argv[1]),
    optimization_image_root=pathlib.Path(sys.argv[2]),
    bands=load_control_examples(),
    png_compress_level=0,
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
optimization_manifest=$authority/optimization-manifest.json
binding_sha=$(sha256sum "$binding" | awk '{print $1}')
optimization_sha=$(sha256sum "$optimization_manifest" | awk '{print $1}')
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
start_ns=$(date +%s%N)
setsid timeout --foreground --signal=TERM --kill-after=30s 5400s \
  "$python" -B scripts/probe_siglip_spatial_tail_recovery.py \
  --control-binding "$binding" \
  --control-binding-sha256 "$binding_sha" \
  --checkpoint-seed17 "$control/seed-017/checkpoints/seed-017-epoch-060.pt" \
  --optimization-manifest "$optimization_manifest" \
  --optimization-manifest-sha256 "$optimization_sha" \
  --optimization-image-root "$optimization_images" \
  --artifact "$staging/spatial-tail.safetensors" \
  --result "$staging/result.json" \
  --execute-spatial-tail &
child=$!
stop_reason=
psi_hits=0
progress_gap=0
last_cpu=
peak_rss=0
peak_cuda_mib=0
peak_psi=0
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  cuda_rows=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits || true)
  if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"$cuda_rows"; then
    cuda_mib=$(awk '{s+=$1}END{printf "%.0f",s}' <<<"$cuda_rows")
  else
    cuda_mib=0
  fi
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  cpu=$(ps -o time= -g "$child" 2>/dev/null | sha256sum | awk '{print $1}')
  ((rss > peak_rss)) && peak_rss=$rss
  ((cuda_mib > peak_cuda_mib)) && peak_cuda_mib=$cuda_mib
  peak_psi=$(awk -v a="$peak_psi" -v p="$psi" 'BEGIN{print (a > p ? a : p)}')
  ((rss <= 118111600640)) || stop_reason=rss-cap
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
end_ns=$(date +%s%N)
swap1=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
artifact_sha=$(sha256sum "$staging/spatial-tail.safetensors" | awk '{print $1}')
"$python" -B - "$staging/result.json" "$optimization_sha" "$artifact_sha" <<'PY'
import pathlib
import sys
from sfora.siglip_spatial_tail_recovery import validate_spatial_tail_result_bytes

value = validate_spatial_tail_result_bytes(pathlib.Path(sys.argv[1]).read_bytes())
assert value["claim_eligible"] is False
assert value["external_evaluation_access"] is False
assert value["optimization_manifest_sha256"] == sys.argv[2]
assert value["artifact_sha256"] == sys.argv[3]
PY
"$python" -B - "$staging/monitor.json" "$revision" "$start_ns" "$end_ns" \
  "$peak_rss" "$peak_cuda_mib" "$peak_psi" "$swap0" "$swap1" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema": "sfora-spatial-tail-monitor-v1",
    "source_revision": sys.argv[2],
    "elapsed_ns": int(sys.argv[4]) - int(sys.argv[3]),
    "peak_rss_bytes": int(sys.argv[5]),
    "peak_cuda_mib": int(sys.argv[6]),
    "peak_memory_psi_full_avg10": float(sys.argv[7]),
    "swap_start_kib": int(sys.argv[8]),
    "swap_end_kib": int(sys.argv[9]),
    "cuda_memory_cap_enforced_in_process": True,
}, sort_keys=True, separators=(",", ":")) + "\n")
PY
unlink "$staging/control-manifest.json"
rm -rf -- "$optimization_images" "$authority"
mv "$staging" "$output"
trap - EXIT INT TERM
REMOTE

rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"
sha256sum "$local_output"
