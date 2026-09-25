#!/usr/bin/env bash
set -euo pipefail

commit="${1:?pass the pinned evaluator commit}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
root=/home/riomus/sfora-siglip2-coverage-official-v1
training_root=/home/riomus/sfora-siglip2-bf16-coverage-v1
run_base=/home/riomus/runs
decision="$run_base/sfora-siglip2-bf16-coverage-replication-v1.json"
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
manifest="$run_base/sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin"
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
python=/home/riomus/group-learning/.venv/bin/python
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

[[ "$(git -C "$root" rev-parse HEAD)" == "$commit" && -z "$(git -C "$root" status --porcelain)" ]] || exit 1
[[ "$(git -C "$training_root" rev-parse HEAD)" == 3b279b56e7b19239002550d44a20e267f8d076d7 ]] || exit 1
check_sha() {
  [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$decision" ccfdb7055643f2a16124ecab62b30d31a2b025f380319e950cb4a481066f1852
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$manifest" 28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 1
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || exit 1

for seed in 179023 179024 179025; do
  output="$run_base/sfora-siglip2-bf16-coverage-official-$seed-bank-v1"
  [[ ! -e "$output" && ! -L "$output" ]] || { echo "output exists: $output" >&2; exit 1; }
done
for seed in 179023 179024 179025; do
  source_dir="$run_base/sfora-siglip2-bf16-coverage-$seed-bank-v1"
  output="$run_base/sfora-siglip2-bf16-coverage-official-$seed-bank-v1"
  echo "COVERAGE OFFICIAL SOP TEST seed=$seed output=$output" >&2
  "$python" "$root/scripts/evaluate_sop_siglip2_official.py" \
    --decision "$decision" --expected-decision-sha256 ccfdb7055643f2a16124ecab62b30d31a2b025f380319e950cb4a481066f1852 \
    --seed "$seed" --arm bank --training-receipt "$source_dir/receipt.json" \
    --training-checkpoint "$source_dir/checkpoint.pt" --training-source-root "$training_root" \
    --source-archive "$archive" --test-image-manifest "$manifest" \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --model-snapshot "$model" --native-library "$native" \
    --output-dir "$output" --workers 2
  test -s "$output/receipt.json"
done
