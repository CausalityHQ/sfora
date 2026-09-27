#!/usr/bin/env bash
set -euo pipefail
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
cd /home/riomus/runs/sfora-inshop-sop-warmstart-smoke-v1
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
python=/home/riomus/group-learning/.venv/bin/python
dataset='/home/riomus/datasets/In-shop Clothes Retrieval Benchmark'
snapshot='/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c'
source=/home/riomus/runs/sfora-inshop-siglip2-train-features-v1
checkpoint=/home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1/checkpoint.pt
checkpoint_sha=2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172
base=/home/riomus/runs
case "${1:?seed required}" in
  179024) preflight="$base/inshop-unseen-gallery-preflight-179024-v2.json"; preflight_sha=f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 ;;
  179025) preflight="$base/inshop-unseen-gallery-preflight-179025-v2.json"; preflight_sha=9dc46a73d7000996028551600ae84a6e6fe1d8d3e113d6503547a2fc8ca927f7 ;;
  179026) preflight="$base/sfora-valid-anchor-confirmation-179026-27/preflight-179026.json"; preflight_sha=4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254 ;;
  *) echo 'unfrozen seed' >&2; exit 2 ;;
esac
seed=$1

"$python" train_inshop_siglip2_unseen_gallery.py \
  --dataset-root "$dataset" --model-snapshot "$snapshot" --features-dir "$source" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --arm freeze_emb --updates 1000 --seed "$seed" \
  --output-dir "$base/sfora-inshop-sop-warmstart-control-$seed-1000-v1"
"$python" train_inshop_siglip2_unseen_gallery.py \
  --dataset-root "$dataset" --model-snapshot "$snapshot" \
  --features-dir "$base/sfora-inshop-sop-warmstart-cache-v1" \
  --vision-init-checkpoint "$checkpoint" --vision-init-sha256 "$checkpoint_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --arm freeze_emb --updates 1000 --seed "$seed" \
  --output-dir "$base/sfora-inshop-sop-warmstart-treatment-$seed-1000-v1"
"$python" compare_inshop_sop_warmstart_full_pair.py --seed "$seed"
