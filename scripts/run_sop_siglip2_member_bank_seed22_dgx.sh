#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
preflight="$run_base/sfora-siglip2-arcface-1000-v10-55040a7/member_bank_preflight_v1.json"
cost="$run_base/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

if systemctl --user is-active --quiet sfora-siglip2-member-bank-multiseed-v2.service \
  || systemctl --user is-active --quiet sfora-siglip2-bank-live-paired-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "seed22 requires completed first unit and an idle DGX GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$root/scripts/train_sop_siglip2_compact.py" 32cce40aa7133bfc602c41b8d5fdb6e76609c71336b907a87c964cbb58348ade
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$preflight" 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
check_sha "$cost" 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9
check_sha "$run_base/sfora-siglip2-member-bank-control-1000-v1/receipt.json" d88167bfcbf8152ee912c8382061afaf248e45fe52ae24cf5f1a5da739477fc3

/home/riomus/group-learning/.venv/bin/python - "$root" "$run_base" "$model" "$CUTILE_TILEIRAS_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, run_base, model, tileiras = map(Path, sys.argv[1:])
reference = json.loads(
    (run_base / "sfora-siglip2-member-bank-control-1000-v1/receipt.json").read_text()
)
for relative, digest in reference["source_files_sha256"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"source file differs: {relative}")
for name, digest in reference["model_file_sha256"].items():
    if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"model file differs: {name}")
if hashlib.sha256(tileiras.read_bytes()).hexdigest() != reference["tileiras_sha256"]:
    raise SystemExit("toolchain differs")
PY

run_arm() {
  local seed="$1" name="$2" output
  output="$run_base/sfora-siglip2-member-bank-multiseed-${seed}-${name}-v1"
  local arm="$name" extra=()
  if [[ "$name" == bank ]]; then
    arm=float_rank
    extra=(--member-bank --member-bank-preflight "$preflight"
      --expected-member-bank-preflight-sha256 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
      --member-bank-cost-receipt "$cost"
      --expected-member-bank-cost-sha256 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9)
  fi
  [[ ! -e "$output" && ! -L "$output" ]] || { echo "run output exists: $output" >&2; exit 1; }
  echo "RUN seed=$seed arm=$name output=$output" >&2
  /home/riomus/group-learning/.venv/bin/python "$root/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library "$native" --output-dir "$output" --arm "$arm" \
    --seed "$seed" --updates 1000 --batch-size 64 --workers 2 --evaluate "${extra[@]}"
  test -s "$output/receipt.json"
}

run_arm 179022 arcface
run_arm 179022 float_rank
run_arm 179022 bank
