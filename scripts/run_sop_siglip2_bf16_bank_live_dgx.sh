#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-bf16-rankmatched-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
output="$run_base/sfora-siglip2-bf16-bank-live-seed179024-v1.json"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
tileiras_dir="$(dirname "$CUTILE_TILEIRAS_PATH")"
export PATH="$tileiras_dir:$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
  echo "cannot verify an idle DGX GPU" >&2; exit 1;
}
[[ -z "$gpu_pids" ]] || { echo "DGX GPU is active" >&2; exit 1; }
[[ ! -e "$output" && ! -L "$output" ]] || { echo "live timing output exists" >&2; exit 1; }

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$root/scripts/benchmark_sop_siglip2_bank_live.py" 53885581715cb545fe3320d2913a76f341b2252a2a4ca3bc8c97c2d0f64be111
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$run_base/sfora-siglip2-bf16-member-bank-179024-arcface-v1/receipt.json" 7a4c950f135c9aa23530552b5ff7c808729405b154597aa229bf759abc01d824
check_sha "$run_base/sfora-siglip2-bf16-member-bank-179024-bank-v1/receipt.json" 3905bc80278c3fc10bf1eae44363ba9fdd0daafc8299ca82aef2405832605ece

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/benchmark_sop_siglip2_bank_live.py" \
  --source-archive "$archive" \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --model-snapshot "$model" \
  --control-dir "$run_base/sfora-siglip2-bf16-member-bank-179024-arcface-v1" \
  --bank-dir "$run_base/sfora-siglip2-bf16-member-bank-179024-bank-v1" \
  --native-library "$native" \
  --training-profile bf16_bank \
  --control-receipt-sha256 7a4c950f135c9aa23530552b5ff7c808729405b154597aa229bf759abc01d824 \
  --bank-receipt-sha256 3905bc80278c3fc10bf1eae44363ba9fdd0daafc8299ca82aef2405832605ece \
  --blocks 100 \
  --output "$output"
