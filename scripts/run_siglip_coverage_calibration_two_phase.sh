#!/usr/bin/env bash
set -euo pipefail

test "$#" = 19
python=$1
landlock=$2
source_dir=$3
control=$4
spatial=$5
binding=$6
checkpoint=$7
optimization_manifest=$8
optimization_images=$9
evaluation_manifest=${10}
support_images=${11}
heldout_images=${12}
spatial_sha=${13}
revision=${14}
staging=${15}
phase1=${16}
phase2=${17}
tmp1=${18}
tmp2=${19}
test "$phase1" = "$staging/phase1"
test "$phase2" = "$staging/phase2"

authority=$(dirname "$binding")
test "$optimization_manifest" = "$authority/optimization-manifest.json"
test "$evaluation_manifest" = "$authority/evaluation-manifest.json"
test -d "$staging"
mkdir -p "$phase1" "$phase2" "$tmp1" "$tmp2"
"$python" -B scripts/prepare_siglip_coverage_calibration.py \
  --control "$control" --authority "$authority" \
  --optimization-image-root "$optimization_images" \
  --support-image-root "$support_images" --heldout-image-root "$heldout_images" \
  --execute-preparation
binding_sha=$(sha256sum "$binding" | awk '{print $1}')
optimization_sha=$(sha256sum "$optimization_manifest" | awk '{print $1}')
evaluation_sha=$(sha256sum "$evaluation_manifest" | awk '{print $1}')
cc -std=c17 -O2 -Wall -Wextra -Werror scripts/landlock_exec.c -o "$landlock"
: >"$staging/preparation.complete"

common_landlock=(
  --ro /usr --ro /etc --rw /proc --ro /sys
  --rw /dev/null --rw /dev/zero --rw /dev/random --rw /dev/urandom
  --rw /dev/nvidiactl --rw /dev/nvidia0 --rw /dev/nvidia-uvm
  --rw /dev/nvidia-uvm-tools
  --ro /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu
  --ro /home/riomus/group-learning/.venv --ro "$source_dir"
  --traverse "$control" --ro "$checkpoint"
  --ro "$spatial" --ro "$binding" --ro "$optimization_manifest"
  --ro "$evaluation_manifest"
  --traverse /home/riomus/.cache/huggingface/hub
  --ro /home/riomus/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384
)
common_probe=(
  --control-binding "$binding" --control-binding-sha256 "$binding_sha"
  --checkpoint-seed17 "$checkpoint"
  --optimization-manifest "$optimization_manifest"
  --optimization-manifest-sha256 "$optimization_sha"
  --optimization-image-root "$optimization_images"
  --evaluation-manifest "$evaluation_manifest"
  --evaluation-manifest-sha256 "$evaluation_sha"
  --support-image-root "$support_images"
  --heldout-image-root "$heldout_images"
  --spatial-artifact "$spatial" --spatial-artifact-sha256 "$spatial_sha"
  --execution-source-commit "$revision"
  --map-artifact "$phase1/maps.safetensors"
  --fit-receipt "$phase1/fit-receipt.json"
  --execute-coverage-calibration
)

HOME="$tmp1" TMPDIR="$tmp1" TORCH_HOME="$tmp1/torch" \
  "$landlock" "${common_landlock[@]}" \
  --ro "$optimization_images" --ro "$support_images" \
  --rw "$phase1" --rw "$tmp1" -- \
  "$python" -B scripts/probe_siglip_coverage_calibration.py \
  --phase fit "${common_probe[@]}"

test -s "$phase1/maps.safetensors"
test -s "$phase1/fit-receipt.json"
fit_receipt_sha=$(sha256sum "$phase1/fit-receipt.json" | awk '{print $1}')

HOME="$tmp2" TMPDIR="$tmp2" TORCH_HOME="$tmp2/torch" \
  "$landlock" "${common_landlock[@]}" \
  --ro "$optimization_images" --ro "$support_images" --ro "$heldout_images" \
  --ro "$phase1" --rw "$phase2" --rw "$tmp2" -- \
  "$python" -B scripts/probe_siglip_coverage_calibration.py \
  --phase evaluate "${common_probe[@]}" \
  --fit-receipt-sha256 "$fit_receipt_sha" --result "$phase2/result.json"

test -s "$phase2/result.json"
