#!/usr/bin/env bash
set -euo pipefail

gate_sha="${1:?pass independently pinned three-seed gate receipt SHA-256}"
root=/home/riomus/sfora-siglip2-bf16-qual-v1
run_base=/home/riomus/runs
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
preflight="$run_base/sfora-siglip2-arcface-1000-v10-55040a7/member_bank_preflight_v1.json"
cost="$run_base/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
gate="$run_base/sfora-siglip2-bf16-member-bank-multiseed-v1.json"
python=/home/riomus/group-learning/.venv/bin/python
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
tileiras_dir="$(dirname "$CUTILE_TILEIRAS_PATH")"
export PATH="$tileiras_dir:$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

[[ "$gate_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid gate digest" >&2; exit 2; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo "rank diagnostic requires an idle DGX GPU" >&2; exit 1;
}
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }

check_sha() {
  local actual
  actual="$(sha256sum "$1")"
  [[ "${actual%% *}" == "$2" ]] || { echo "source hash differs: $1" >&2; exit 1; }
}
check_sha "$gate" "$gate_sha"
check_sha "$root/scripts/train_sop_siglip2_compact.py" ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23
check_sha "$archive" 1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a
check_sha "$native" 39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c
check_sha "$preflight" 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
check_sha "$cost" 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9

"$python" - "$gate" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text())
if (
    row.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1"
    or row.get("seeds") != [179023, 179024, 179025]
    or row.get("continuation_gate_pass") is not True
    or row.get("source_archive_sha256")
    != "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
):
    raise SystemExit("rank diagnostic needs a passing BF16 three-seed gate")
PY

for seed in 179023 179024 179025; do
  for arm in float_rank bank; do
    output="$run_base/sfora-siglip2-bf16-rank-contribution-${seed}-${arm}-v1"
    [[ ! -e "$output" && ! -L "$output" ]] || {
      echo "rank diagnostic output exists: $output" >&2; exit 1;
    }
  done
done

for seed in 179023 179024 179025; do
  for name in float_rank bank; do
    output="$run_base/sfora-siglip2-bf16-rank-contribution-${seed}-${name}-v1"
    extra=()
    if [[ "$name" == bank ]]; then
      extra=(--member-bank --member-bank-preflight "$preflight"
        --expected-member-bank-preflight-sha256 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
        --member-bank-cost-receipt "$cost"
        --expected-member-bank-cost-sha256 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9)
    fi
    echo "RUN BF16 rank contribution seed=$seed arm=$name" >&2
    "$python" "$root/scripts/train_sop_siglip2_compact.py" \
      --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
      --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
      --native-library "$native" --output-dir "$output" --arm float_rank \
      --seed "$seed" --updates 1 --batch-size 64 --workers 2 \
      --gradient-diagnostic-steps 1 --train-vision-dtype bf16 "${extra[@]}"
    test -s "$output/receipt.json"
  done
done
