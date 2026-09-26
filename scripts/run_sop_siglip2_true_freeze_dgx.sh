#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/riomus/sfora-sop-true-freeze-v1
RUNS=/home/riomus/runs
PYTHON=/home/riomus/group-learning/.venv/bin/python
TRAINER_SHA=c5b8786c352ce6c8bedce9a5963ef43e2c18db61974e3c141c698227a23f1b3c
PREFLIGHT_SHA=54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
COST_SHA=8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9
ARCHIVE_SHA=1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
NATIVE_SHA=39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
preflight="$RUNS/sfora-siglip2-arcface-1000-v10-55040a7/member_bank_preflight_v1.json"
cost="$RUNS/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export HF_HUB_OFFLINE=1

mode="${1:?smoke or full required}"
case "$mode" in smoke) updates=17 ;; full) updates=1000 ;; *) exit 2 ;; esac
check_sha() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
check_sha "$ROOT/scripts/train_sop_siglip2_compact.py" "$TRAINER_SHA"
check_sha "$archive" "$ARCHIVE_SHA"
check_sha "$native" "$NATIVE_SHA"
check_sha "$preflight" "$PREFLIGHT_SHA"
check_sha "$cost" "$COST_SHA"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for arm in control freeze; do
  test ! -e "$RUNS/sfora-sop-true-freeze-$arm-179024-$updates-v1"
done
for arm in control freeze; do
  output="$RUNS/sfora-sop-true-freeze-$arm-179024-$updates-v1"
  extra=()
  if [[ "$arm" == freeze ]]; then extra=(--freeze-lower-stack); fi
  if [[ "$mode" == full ]]; then extra+=(--evaluate); fi
  echo "START mode=$mode arm=$arm updates=$updates" >&2
  "$PYTHON" "$ROOT/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot "$model" --candidate-dir "$RUNS/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library "$native" --output-dir "$output" --arm float_rank \
    --seed 179024 --updates "$updates" --batch-size 64 --workers 2 \
    --rank-coefficient 8.0 --train-vision-dtype bf16 --coverage-first-schedule \
    --member-bank --member-bank-preflight "$preflight" \
    --expected-member-bank-preflight-sha256 "$PREFLIGHT_SHA" \
    --member-bank-cost-receipt "$cost" --expected-member-bank-cost-sha256 "$COST_SHA" \
    "${extra[@]}"
  echo "DONE arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
