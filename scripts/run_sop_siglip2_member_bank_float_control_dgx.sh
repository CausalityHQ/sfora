#!/usr/bin/env bash
set -euo pipefail

root=/home/riomus/sfora-siglip2-member-bank-v1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:"$PATH"
export PYTHONPATH="$root/src:$root/scripts"
export HF_HUB_OFFLINE=1

exec /home/riomus/group-learning/.venv/bin/python "$root/scripts/train_sop_siglip2_compact.py" \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --candidate-dir /home/riomus/runs/sfora-siglip2-train-28e19203 \
  --unicom-l14-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --output-dir /home/riomus/runs/sfora-siglip2-member-bank-float-1000-v1 \
  --arm float_rank --seed 179019 --updates 1000 --batch-size 64 --workers 2 --evaluate
