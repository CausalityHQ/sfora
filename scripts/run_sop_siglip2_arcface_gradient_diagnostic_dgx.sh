#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-arcface-overflow-diag-v1
run_base=/home/riomus/runs
output="$run_base/sfora-siglip2-arcface-overflow-diag-179020-v1"
archive=/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
native=/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

if systemctl --user is-active --quiet sfora-siglip2-member-bank-multiseed-v2.service \
  || systemctl --user is-active --quiet sfora-siglip2-member-bank-seed22-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "gradient diagnostic requires an idle DGX GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "gradient diagnostic output already exists" >&2
  exit 1
}
actual="$(sha256sum "$root/scripts/train_sop_siglip2_compact.py")"
[[ "${actual%% *}" == a4135aad21c11721695502588e427a868a5a14dfaa70a987d914c493a076f8b3 ]] || {
  echo "gradient diagnostic source hash differs" >&2
  exit 1
}
actual="$(sha256sum "$root/instrumentation.diff")"
[[ "${actual%% *}" == ddfe3c96c40d267d5d56c112035657d3c0c26e2fd28104eb70cb55396b0330bf ]] || {
  echo "gradient diagnostic patch hash differs" >&2
  exit 1
}

exec /home/riomus/group-learning/.venv/bin/python "$root/scripts/train_sop_siglip2_compact.py" \
  --model-snapshot "$model" --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
  --unicom-l14-archive "$archive" --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --native-library "$native" --output-dir "$output" --arm arcface \
  --seed 179020 --updates 1000 --batch-size 64 --workers 2
