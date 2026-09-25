#!/usr/bin/env bash
set -euo pipefail

precision="${1:?expected fp32_autocast or fp16_native}"
case "$precision" in fp32_autocast|fp16_native) ;; *) exit 2;; esac
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:"$PATH"
export PYTHONPATH=/home/riomus/sfora-siglip2-train-matched-v10-55040a7/src
export HF_HUB_OFFLINE=1

run=/home/riomus/runs/sfora-siglip2-arcface-1000-v10-55040a7
exec /home/riomus/group-learning/.venv/bin/python \
  /home/riomus/sfora-siglip2-live-trained-v1/verify_siglip2_compact_serving.py \
  --source-archive /home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz \
  --dataset-root /home/riomus/datasets/Stanford_Online_Products \
  --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
  --training-receipt "$run/receipt.json" \
  --training-checkpoint "$run/checkpoint.pt" \
  --train-embeddings "$run/train_embeddings.npy" \
  --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
  --paired-receipt "$run/live_timing_precision_pair.json" \
  --precision "$precision" \
  --output "$run/serving_api_${precision}_parity_v2.json"
