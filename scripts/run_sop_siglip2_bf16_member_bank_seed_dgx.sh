#!/usr/bin/env bash
set -euo pipefail

seed="${1:?pass preregistered BF16 seed}"
case "$seed" in
  179023) order=(float_rank bank arcface) ;;
  179024) order=(bank arcface float_rank) ;;
  179025) order=(arcface float_rank bank) ;;
  *) echo "unregistered BF16 seed: $seed" >&2; exit 2 ;;
esac

root=/home/riomus/sfora-siglip2-bf16-qual-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
preflight="$run_base/sfora-siglip2-arcface-1000-v10-55040a7/member_bank_preflight_v1.json"
cost="$run_base/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
qualification="$run_base/sfora-siglip2-arcface-bf16-qualification-179020-v1/receipt.json"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo "BF16 replication requires an idle DGX GPU" >&2
  exit 1
}
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$root/scripts/train_sop_siglip2_compact.py" ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$preflight" 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
check_sha "$cost" 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9

test -s "$qualification"
/home/riomus/group-learning/.venv/bin/python - "$qualification" "$root" "$model" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

qualification, root, model = map(Path, sys.argv[1:])
row = json.loads(qualification.read_text())
if (
    row.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
    or row.get("arm") != "arcface"
    or row.get("seed") != 179020
    or row.get("updates") != 1000
    or row.get("train_vision_dtype") != "bf16"
    or row.get("grad_scaler_initial_scale") != 1.0
    or row.get("quality") is not None
    or row.get("source_sha256")
    != "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23"
    or len(row.get("step_seconds", [])) != 1000
):
    raise SystemExit("BF16 qualification receipt does not establish 1000 stable ArcFace updates")
for relative, digest in row["source_files_sha256"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"BF16 source file differs: {relative}")
for name, digest in row["model_file_sha256"].items():
    if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"BF16 model file differs: {name}")
PY

for name in "${order[@]}"; do
  output="$run_base/sfora-siglip2-bf16-member-bank-${seed}-${name}-v1"
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "run output exists: $output" >&2
    exit 1
  }
done

for name in "${order[@]}"; do
  output="$run_base/sfora-siglip2-bf16-member-bank-${seed}-${name}-v1"
  arm="$name"
  extra=()
  if [[ "$name" == bank ]]; then
    arm=float_rank
    extra=(--member-bank --member-bank-preflight "$preflight"
      --expected-member-bank-preflight-sha256 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
      --member-bank-cost-receipt "$cost"
      --expected-member-bank-cost-sha256 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9)
  fi
  echo "RUN BF16 seed=$seed arm=$name output=$output" >&2
  /home/riomus/group-learning/.venv/bin/python "$root/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library "$native" --output-dir "$output" --arm "$arm" \
    --seed "$seed" --updates 1000 --batch-size 64 --workers 2 \
    --train-vision-dtype bf16 --evaluate "${extra[@]}"
  test -s "$output/receipt.json"
done
