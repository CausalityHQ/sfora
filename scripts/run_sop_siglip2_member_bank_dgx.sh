#!/usr/bin/env bash
set -euo pipefail

mode="${1:?choose smoke or pair}"
root=/home/riomus/sfora-siglip2-member-bank-v1
run_base=/home/riomus/runs
control_run="$run_base/sfora-siglip2-arcface-1000-v10-55040a7"
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:"$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

train() {
  local arm="$1" updates="$2" output="$3" bank="$4"
  local extra=()
  if [[ "$bank" == yes ]]; then
    extra=(
      --member-bank
      --member-bank-preflight "$control_run/member_bank_preflight_v1.json"
      --expected-member-bank-preflight-sha256 54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c
      --member-bank-cost-receipt "$run_base/sfora-siglip2-member-bank-cost-v1/receipt_v2.json"
      --expected-member-bank-cost-sha256 8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9
    )
  fi
  if [[ "$updates" == 1000 ]]; then
    extra+=(--evaluate)
  elif [[ "$bank" == yes ]]; then
    extra+=(--gradient-diagnostic-steps "$updates")
  fi
  /home/riomus/group-learning/.venv/bin/python "$root/scripts/train_sop_siglip2_compact.py" \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --candidate-dir "$run_base/sfora-siglip2-train-28e19203" \
    --unicom-l14-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
    --dataset-root /home/riomus/datasets/Stanford_Online_Products \
    --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
    --output-dir "$output" \
    --arm "$arm" --seed 179019 --updates "$updates" --batch-size 64 --workers 2 \
    "${extra[@]}"
}

case "$mode" in
  smoke)
    train float_rank 2 "$run_base/sfora-siglip2-member-bank-smoke-v1" yes
    ;;
  pair)
    train arcface 1000 "$run_base/sfora-siglip2-member-bank-control-1000-v1" no
    train float_rank 1000 "$run_base/sfora-siglip2-member-bank-treatment-1000-v1" yes
    ;;
  *)
    echo "choose smoke or pair" >&2
    exit 2
    ;;
esac
