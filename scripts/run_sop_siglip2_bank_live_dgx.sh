#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

if systemctl --user is-active --quiet sfora-siglip2-member-bank-multiseed-v2.service \
  || systemctl --user is-active --quiet sfora-siglip2-member-bank-seed22-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "paired live timing requires an idle DGX GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }

exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/benchmark_sop_siglip2_bank_live.py" \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --control-dir "$run_base/sfora-siglip2-member-bank-control-1000-v1" \
  --bank-dir "$run_base/sfora-siglip2-member-bank-treatment-1000-v1" \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --blocks 100 \
  --output "$run_base/sfora-siglip2-bank-live-paired-v1.json"
