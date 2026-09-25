#!/usr/bin/env bash
set -euo pipefail

seed="${1:?post-selection seed required}"
arm="${2:?arcface, float_rank, or bank required}"
decision_sha="${3:?independently pinned decision SHA-256 required}"
case "$seed" in 179020|179021|179022) ;; *) exit 2;; esac
case "$arm" in arcface|float_rank|bank) ;; *) exit 2;; esac
[[ "$decision_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2

root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

if systemctl --user is-active --quiet sfora-siglip2-member-bank-multiseed-v2.service \
  || systemctl --user is-active --quiet sfora-siglip2-member-bank-seed22-v1.service \
  || systemctl --user is-active --quiet sfora-siglip2-bank-live-paired-v1.service \
  || [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "official SOP evaluation requires finished training and an idle GPU" >&2
  exit 1
fi
exec 9>"$run_base/.sfora-siglip2-gpu.lock"
flock -n 9 || { echo "SOP SigLIP2 GPU lock is held" >&2; exit 1; }

run_dir="$run_base/sfora-siglip2-member-bank-multiseed-${seed}-${arm}-v1"
exec /home/riomus/group-learning/.venv/bin/python \
  "$root/scripts/evaluate_sop_siglip2_official.py" \
  --decision "$run_base/sfora-siglip2-member-bank-multiseed-decision-v1.json" \
  --expected-decision-sha256 "$decision_sha" \
  --seed "$seed" --arm "$arm" \
  --training-receipt "$run_dir/receipt.json" \
  --training-checkpoint "$run_dir/checkpoint.pt" \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --test-image-manifest "$run_base/sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin" \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --output-dir "$run_base/sfora-siglip2-official-${seed}-${arm}-v1" \
  --workers 2
