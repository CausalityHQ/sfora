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
preflight=/home/riomus/runs/inshop-unseen-gallery-preflight-179024-v2.json
preflight_sha=f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
source=/home/riomus/runs/sfora-inshop-siglip2-train-features-v1
checkpoint=/home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1/checkpoint.pt
checkpoint_sha=2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172
base=/home/riomus/runs

"$python" train_inshop_siglip2_unseen_gallery.py \
  --dataset-root "$dataset" --model-snapshot "$snapshot" --features-dir "$source" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --arm freeze_emb --updates 17 --seed 179024 \
  --output-dir "$base/sfora-inshop-sop-warmstart-control-179024-17-v1"
"$python" - <<'PY'
import json
from pathlib import Path
a = json.loads(Path('/home/riomus/runs/sfora-inshop-true-freeze-freeze_emb-179024-17-v1/receipt.json').read_text())
b = json.loads(Path('/home/riomus/runs/sfora-inshop-sop-warmstart-control-179024-17-v1/receipt.json').read_text())
assert a['first_input_batch_sha256'] == b['first_input_batch_sha256']
assert a['executed_schedule_sha256'] == b['executed_schedule_sha256']
assert a['features_sha256'] == b['features_sha256']
print('archived control batch/schedule/cache replay verified', flush=True)
PY

"$python" export_inshop_siglip2_train_features.py \
  --dataset-root "$dataset" --model-snapshot "$snapshot" \
  --vision-init-checkpoint "$checkpoint" --vision-init-sha256 "$checkpoint_sha" \
  --output-dir "$base/sfora-inshop-sop-warmstart-cache-v1"
"$python" train_inshop_siglip2_unseen_gallery.py \
  --dataset-root "$dataset" --model-snapshot "$snapshot" \
  --features-dir "$base/sfora-inshop-sop-warmstart-cache-v1" \
  --vision-init-checkpoint "$checkpoint" --vision-init-sha256 "$checkpoint_sha" \
  --preflight "$preflight" --preflight-sha256 "$preflight_sha" \
  --arm freeze_emb --updates 17 --seed 179024 \
  --output-dir "$base/sfora-inshop-sop-warmstart-treatment-179024-17-v1"
