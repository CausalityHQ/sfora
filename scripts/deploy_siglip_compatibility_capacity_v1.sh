#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-riomus@100.104.199.68}
remote_root=${REMOTE_ROOT:-/home/riomus/sfora-revisions}
remote_control=${REMOTE_CONTROL:-/home/riomus/sfora-pass209-control-034e66407c5de6e2ff1acf3d18455b10760d3509}
remote_authority=${REMOTE_AUTHORITY:-/home/riomus/sfora-attention-readout-recovery/801937e5aeaffc61b886c263be1eaa5766f495d9/authority}
remote_spatial=${REMOTE_SPATIAL:-/home/riomus/sfora-spatial-tail-recovery/498a10d34b9095fa92a7f45f8e0af6031b33344d/spatial-tail.safetensors}
remote_output_root=${REMOTE_OUTPUT_ROOT:-/home/riomus/sfora-compatibility-capacity}
revision=$(git rev-parse HEAD)
remote_source=$remote_root/$revision
remote_output=$remote_output_root/$revision
local_result=${LOCAL_RESULT:-/tmp/sfora-compatibility-capacity-$revision.json}
local_artifact=${LOCAL_ARTIFACT:-/tmp/sfora-compatibility-capacity-$revision.safetensors}
spatial_sha=cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9
spatial_bytes=67973944
binding_sha=39f5e0518ea509dede79e79a45757553429f9468465209f9ea4092e4d28314b7
manifest_sha=045ca751f97ae097eed5a1b850b970e245961b09a1bf39490d882cfff2a5358e

source_files=(
  scripts/landlock_exec.c
src/sfora/siglip_compatibility_capacity.py
scripts/probe_siglip_compatibility_capacity.py
scripts/deploy_siglip_compatibility_capacity_v1.sh
src/sfora/siglip_gallery_compatibility_alignment.py
scripts/probe_siglip_gallery_compatibility_alignment.py
  src/sfora/siglip_spatial_tail_recovery.py
  scripts/probe_siglip_spatial_tail_recovery.py
  scripts/diagnose_siglip_rsta_stage_a.py
  src/sfora/siglip_rsta_stage_a.py
  src/sfora/siglip_proxy_control.py
  src/sfora/pass209_m4.py
  src/sfora/substrate_screen.py
  src/sfora/token_set_proxy_anchor.py
  src/sfora/token_set_screen.py
  src/sfora/kernels/set_maxsim.py
)
git diff --quiet HEAD -- src/sfora scripts \
  ':(exclude)src/sfora/qwen_geometry_control.py' \
  ':(exclude)scripts/run_qwen_geometry_control.py'
git diff --cached --quiet -- src/sfora scripts \
  ':(exclude)src/sfora/qwen_geometry_control.py' \
  ':(exclude)scripts/run_qwen_geometry_control.py'
for path in "${source_files[@]}"; do git ls-files --error-unmatch "$path" >/dev/null; done
test ! -e "$local_result"
test ! -e "$local_artifact"

scratch=$(mktemp -d /tmp/sfora-compatibility-capacity-deploy.XXXXXX)
cleanup() {
  case "$scratch" in
    /tmp/sfora-compatibility-capacity-deploy.*) rm -rf -- "$scratch" ;;
    *) exit 99 ;;
  esac
}
trap cleanup EXIT INT TERM
bundle=$scratch/source.bundle
remote_bundle=/tmp/sfora-compatibility-capacity-source-$revision.bundle
git bundle create "$bundle" HEAD

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_spatial" \
  "$remote_authority" "$remote_bundle" "$spatial_sha" "$spatial_bytes" \
  "$binding_sha" "$manifest_sha" <<'PREFLIGHT'
set -euo pipefail
pgrep -f '[p]robe_siglip_compatibility_capacity.py' >/dev/null && {
  echo 'compatibility capacity process is already active' >&2; exit 75;
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -f "$3/control.receipt.json"
test -f "$3/seed-017/checkpoints/seed-017-epoch-060.pt"
test -f "$4"
test "$(stat -c%s "$4")" = "$8"
test "$(sha256sum "$4" | awk '{print $1}')" = "$7"
test "$(sha256sum "$5/control-binding.json" | awk '{print $1}')" = "$9"
test "$(sha256sum "$5/optimization-manifest.json" | awk '{print $1}')" = "${10}"
test ! -e "$1"; test ! -e "$2"; test ! -e "$2.partial"; test ! -e "$6"
PREFLIGHT
rsync -a -- "$bundle" "$remote_host:$remote_bundle"

ssh -o BatchMode=yes "$remote_host" bash -s -- \
  "$remote_source" "$remote_output" "$remote_control" "$remote_spatial" \
  "$remote_authority" "$revision" "$remote_bundle" "$spatial_sha" \
  "$binding_sha" "$manifest_sha" <<'REMOTE'
set -euo pipefail
source_dir=$1; output=$2; control=$3; spatial=$4; sealed_authority=$5
revision=$6; bundle=$7; spatial_sha=$8; binding_sha=$9; manifest_sha=${10}
python=/home/riomus/group-learning/.venv/bin/python3
staging=$output.partial
authority=$staging/authority
optimization_images=$staging/optimization-images
child= group= source_owned=0 source_checkout_complete=0
cleanup_remote() {
  if [[ -n ${group:-} ]]; then
    kill -TERM -- "-$group" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$group" 2>/dev/null || true
  fi
  if [[ -n ${child:-} ]]; then
    wait "$child" 2>/dev/null || true
  fi
  test ! -e "$bundle" || unlink "$bundle"
  test "$staging" = "$output.partial" || exit 99
  test ! -e "$staging" || rm -rf -- "$staging"
  if [[ "$source_owned" = 1 && ! -e "$output" && -e "$source_dir" ]]; then
    test ! -L "$source_dir"
    test "${source_dir##*/}" = "$revision"
    test "$(dirname "$source_dir")" != /
    if [[ "$source_checkout_complete" = 1 ]]; then
      test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision" || exit 99
    fi
    rm -rf -- "$source_dir"
  fi
}
trap cleanup_remote EXIT
trap 'cleanup_remote; exit 130' INT
trap 'cleanup_remote; exit 143' TERM

test ! -e "$source_dir"
source_owned=1
git clone --quiet --no-checkout "$bundle" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$revision"
unlink "$bundle"
cd "$source_dir"
test "$(git rev-parse HEAD)" = "$revision"
source_checkout_complete=1
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

pgrep -f '[p]robe_siglip_compatibility_capacity.py' >/dev/null && {
  echo 'compatibility capacity process is already active' >&2; exit 75;
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test -x "$python"; test -f "$spatial"; test ! -e "$output"; test ! -e "$staging"
mkdir -p "$authority"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cp -- "$sealed_authority/control-binding.json" "$authority/control-binding.json"
cp -- "$sealed_authority/optimization-manifest.json" "$authority/optimization-manifest.json"
binding=$authority/control-binding.json
manifest=$authority/optimization-manifest.json
test "$(sha256sum "$binding" | awk '{print $1}')" = "$binding_sha"
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha"
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
setsid timeout --signal=TERM --kill-after=30s 7200s \
  bash -s -- "$python" "$manifest" "$optimization_images" "$binding" "$binding_sha" \
  "$control/seed-017/checkpoints/seed-017-epoch-060.pt" "$manifest_sha" "$spatial" \
  "$spatial_sha" "$staging/descriptors.safetensors" "$staging/result.json" \
  "$source_dir" "$control" "$staging" <<'RUN' &
set -euo pipefail
python=$1; manifest=$2; optimization_images=$3; binding=$4; binding_sha=$5
checkpoint=$6; manifest_sha=$7; spatial=$8; spatial_sha=$9
descriptors=${10}; result=${11}
source_dir=${12}; control=${13}; staging=${14}
"$python" -B - "$manifest" "$optimization_images" <<'PY'
import pathlib, sys
from probe_siglip_gallery_compatibility_alignment import (
    materialize_registered_optimization_images,
)
materialize_registered_optimization_images(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
PY
test "$descriptors" = "$staging/descriptors.safetensors"
test "$result" = "$staging/result.json"
cc -std=c17 -O2 -Wall -Wextra -Werror scripts/landlock_exec.c -o "$staging/landlock-exec"
private_tmp="$staging/private-tmp"
mkdir -p "$private_tmp"
export HOME="$private_tmp" TMPDIR="$private_tmp"
export HF_HOME=/home/riomus/.cache/huggingface TORCH_HOME="$private_tmp/torch"
exec "$staging/landlock-exec" --ro /usr --ro /etc --rw /proc --ro /sys \
  --rw /dev/null --rw /dev/zero --rw /dev/random --rw /dev/urandom \
  --rw /dev/nvidiactl --rw /dev/nvidia0 --rw /dev/nvidia-uvm \
  --rw /dev/nvidia-uvm-tools \
  --ro /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu \
  --ro /home/riomus/group-learning/.venv --ro "$source_dir" \
  --traverse "$control" --ro "$checkpoint" \
  --ro "$spatial" \
  --traverse /home/riomus/.cache/huggingface/hub \
  --ro /home/riomus/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384 \
  --rw "$staging" -- \
  "$python" -B scripts/probe_siglip_compatibility_capacity.py \
  --control-binding "$binding" \
  --control-binding-sha256 "$binding_sha" \
  --checkpoint-seed17 "$checkpoint" \
  --optimization-manifest "$manifest" \
  --optimization-manifest-sha256 "$manifest_sha" \
  --optimization-image-root "$optimization_images" \
  --spatial-artifact "$spatial" --spatial-artifact-sha256 "$spatial_sha" \
  --descriptor-artifact "$descriptors" --result "$result" \
  --execute-capacity-diagnostic
RUN
child=$!
group=$child
stop_reason=; psi_hits=0; progress_gap=0; last_cpu=
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -g "$child" 2>/dev/null | awk '{s+=$1}END{printf "%.0f",s*1024}' || true)
  test -n "$rss" || rss=0
  psi=$(awk '/^full /{sub("avg10=","",$2);print $2}' /proc/pressure/memory)
  swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  cpu=$(ps -o time= -g "$child" 2>/dev/null || true)
  gpu_mib=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
    | awk '{s+=$1}END{printf "%.0f",s}' || true)
  test -n "$gpu_mib" || gpu_mib=0
  ((rss <= 51539607552)) || stop_reason=rss-cap
  ((gpu_mib <= 49152)) || stop_reason=gpu-memory-cap
  if awk -v x="$psi" 'BEGIN{exit !(x>=0.50)}'; then ((psi_hits+=1)); else psi_hits=0; fi
  awk -v x="$psi" 'BEGIN{exit !(x>=0.79)}' && stop_reason=psi-immediate || true
  ((psi_hits < 3)) || stop_reason=psi-sustained
  ((swap-swap0 <= 262144)) || stop_reason=swap-delta
  if [[ $cpu = "$last_cpu" ]]; then ((progress_gap+=5)); else progress_gap=0; last_cpu=$cpu; fi
  ((progress_gap < 300)) || stop_reason=progress-gap
  if [[ -n $stop_reason ]]; then kill -TERM -- "-$child" 2>/dev/null || true; break; fi
  sleep 5
done
set +e; wait "$child"; status=$?; set -e; child=
if pgrep -g "$group" >/dev/null 2>&1; then
  kill -TERM -- "-$group" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    pgrep -g "$group" >/dev/null 2>&1 || break
    sleep 1
  done
  pgrep -g "$group" >/dev/null 2>&1 && kill -KILL -- "-$group" 2>/dev/null || true
fi
test -z "$(pgrep -g "$group" 2>/dev/null || true)"
group=
if [[ -n $stop_reason ]]; then echo "STOP:$stop_reason" >&2; exit 125; fi
((status == 0)) || exit "$status"
test -s "$staging/result.json"; test -s "$staging/descriptors.safetensors"
artifact_sha=$(sha256sum "$staging/descriptors.safetensors" | awk '{print $1}')
"$python" -B - "$staging/result.json" "$artifact_sha" "$spatial_sha" <<'PY'
import pathlib, sys
from safetensors.torch import load_file
from sfora.siglip_compatibility_capacity import (
    validate_compatibility_capacity_result_bytes,
)
value = validate_compatibility_capacity_result_bytes(pathlib.Path(sys.argv[1]).read_bytes())
assert value["claim_eligible"] is False
assert value["external_evaluation_access"] is False
assert value["descriptor_artifact_sha256"] == sys.argv[2]
payload = load_file(str(pathlib.Path(sys.argv[1]).with_name("descriptors.safetensors")))
assert set(payload) == {"student", "teacher", "id_sha256", "labels"}
assert payload["student"].shape == payload["teacher"].shape
assert payload["student"].dtype == payload["teacher"].dtype
assert payload["id_sha256"].shape == (payload["student"].shape[0], 32)
assert payload["labels"].shape == (payload["student"].shape[0],)
PY
unlink "$staging/landlock-exec"
rm -rf -- "$staging/private-tmp"
rm -rf -- "$optimization_images" "$authority"
mv "$staging" "$output"
trap - EXIT INT TERM
REMOTE

rsync -a -- "$remote_host:$remote_output/result.json" "$local_result"
rsync -a -- "$remote_host:$remote_output/descriptors.safetensors" "$local_artifact"
sha256sum "$local_result" "$local_artifact"
