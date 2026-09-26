#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
case "$seed" in
  179024)
    decision_sha=31ebbe4ae71c5f1c7922ef9140721957ccc408eab165df1570678a8843988ed5
    control_sha=1bc737d77ea19b40b74ca6ca967923696403c656c42c17930ace732f8ca0965f
    freeze_sha=07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5 ;;
  179026)
    decision_sha=e12af7bc484e140e1bde577ff713c0eee711fa6854ec8babbf0607afa74cf364
    control_sha=c69de5e542d2d919de5875b4313904efafd8d03d5a0271a0bc0199b45dbd14e0
    freeze_sha=a3153e1d2429560ea5902bd68d3e43bee17bd0509330a581a48b40e6a824ea8a ;;
  179027)
    decision_sha=20d0e3aeaeb24093cf05cd87feb692c878f28bac85c9abd88f2bba5fde2ca894
    control_sha=35b7f82ccfa44adf9bfb58c02711d017dea88a49e7b8e859b9437458a9fa8c4a
    freeze_sha=9fe8e88615ebf653cb9f0ca890b7eac472d84297fbf78370ed3b1cc5bf77b633 ;;
  *) exit 2 ;;
esac

root=/home/riomus/sfora-sop-true-freeze-official-v1
training_root=/home/riomus/sfora-sop-true-freeze-v1
runs=/home/riomus/runs
python=/home/riomus/group-learning/.venv/bin/python
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
manifest="$runs/sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin"
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
decision="$runs/sfora-sop-true-freeze-decision-$seed-v1.json"
check_sha() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
check_sha "$training_root/scripts/train_sop_siglip2_compact.py" c5b8786c352ce6c8bedce9a5963ef43e2c18db61974e3c141c698227a23f1b3c
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$manifest" 28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$decision" "$decision_sha"
check_sha "$runs/sfora-sop-true-freeze-control-$seed-1000-v1/receipt.json" "$control_sha"
check_sha "$runs/sfora-sop-true-freeze-freeze-$seed-1000-v1/receipt.json" "$freeze_sha"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$runs/.sfora-siglip2-gpu.lock"
flock -n 9
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

for arm in control freeze; do
  test ! -e "$runs/sfora-sop-true-freeze-official-$seed-$arm-v1"
done
for arm in control freeze; do
  training="$runs/sfora-sop-true-freeze-$arm-$seed-1000-v1"
  output="$runs/sfora-sop-true-freeze-official-$seed-$arm-v1"
  echo "START official seed=$seed arm=$arm" >&2
  "$python" "$root/scripts/evaluate_sop_siglip2_official.py" \
    --decision "$decision" --expected-decision-sha256 "$decision_sha" \
    --seed "$seed" --arm "$arm" --training-receipt "$training/receipt.json" \
    --training-checkpoint "$training/checkpoint.pt" --training-source-root "$training_root" \
    --source-archive "$archive" --test-image-manifest "$manifest" \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --model-snapshot "$model" --native-library "$native" \
    --output-dir "$output" --workers 2
  echo "DONE official seed=$seed arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
