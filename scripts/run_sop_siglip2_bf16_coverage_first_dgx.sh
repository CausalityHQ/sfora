#!/usr/bin/env bash
set -euo pipefail

expected_commit="${1:?pass frozen coverage-screen commit}"
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
root=/home/riomus/sfora-siglip2-bf16-coverage-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
preflight="$run_base/sfora-siglip2-arcface-1000-v10-55040a7/member_bank_preflight_v1.json"
cost="$run_base/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
python=/home/riomus/group-learning/.venv/bin/python
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

[[ "$(git -C "$root" rev-parse HEAD)" == "$expected_commit" ]] || exit 1
[[ -z "$(git -C "$root" status --porcelain)" ]] || exit 1
check_sha() {
  [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]] || { echo "hash differs: $1" >&2; exit 1; }
}
check_sha "$root/scripts/train_sop_siglip2_compact.py" 400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118
check_sha "$root/src/sfora/unicom_rank_finish.py" bb97a0e0e97c0452ba48e10caf003b95c19d42ec9ce84204190be4b727d966a3
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$preflight" 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
check_sha "$cost" 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || exit 1

for arm in fixed_float matched_float bank; do
  output="$run_base/sfora-siglip2-bf16-coverage-179023-${arm}-v1"
  [[ ! -e "$output" && ! -L "$output" ]] || { echo "output exists: $output" >&2; exit 1; }
done

for arm in fixed_float matched_float bank; do
  output="$run_base/sfora-siglip2-bf16-coverage-179023-${arm}-v1"
  extra=()
  coefficient=58.64
  if [[ "$arm" == fixed_float ]]; then
    coefficient=21.93
  elif [[ "$arm" == bank ]]; then
    coefficient=8.0
    extra=(--member-bank --member-bank-preflight "$preflight"
      --expected-member-bank-preflight-sha256 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
      --member-bank-cost-receipt "$cost"
      --expected-member-bank-cost-sha256 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9)
  fi
  echo "COVERAGE FIRST seed=179023 arm=$arm output=$output" >&2
  "$python" "$root/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library "$native" --output-dir "$output" --arm float_rank \
    --seed 179023 --updates 1000 --batch-size 64 --workers 2 \
    --rank-coefficient "$coefficient" --train-vision-dtype bf16 \
    --coverage-first-schedule --evaluate "${extra[@]}"
  test -s "$output/receipt.json"
done
