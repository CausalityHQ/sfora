#!/usr/bin/env bash
# Frozen three-seed official query/gallery read after TRAIN-only bank selection.
set -euo pipefail

RUNS=/home/riomus/runs
CHECKOUT=/home/riomus/sfora-siglip2-deployed-batch-v1
PYTHON=/home/riomus/group-learning/.venv/bin/python
DECISION_SHA=43036079d17aa6a7010fc70c15716c5b939dd8e82629eb1187763395a386526f
EVALUATOR_SHA=6fee591bffdb2af8d77355bd4f0c282ac52979fa2125ff46bfb31df0ea985411
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
export PATH="$(dirname "$CUTILE_TILEIRAS_PATH"):$PATH"
export PYTHONPATH="$RUNS:$CHECKOUT/src:$CHECKOUT/scripts"
export HF_HUB_OFFLINE=1

test "$(sha256sum "$RUNS/sfora-inshop-siglip2-paired-holdout-v1.json" | cut -d' ' -f1)" = "$DECISION_SHA"
test "$(sha256sum "$RUNS/evaluate_inshop_siglip2_official.py" | cut -d' ' -f1)" = "$EVALUATOR_SHA"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec 9>"$RUNS/.sfora-siglip2-gpu.lock"
flock -n 9
for seed in 179023 179024 179025; do
  test ! -e "$RUNS/sfora-inshop-siglip2-official-bank-$seed-v1"
done
for seed in 179023 179024 179025; do
  training="$RUNS/sfora-inshop-siglip2-bank-$seed-v1"
  output="$RUNS/sfora-inshop-siglip2-official-bank-$seed-v1"
  echo "START official bank seed=$seed" >&2
  "$PYTHON" "$RUNS/evaluate_inshop_siglip2_official.py" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --training-dir "$training" \
    --decision "$RUNS/sfora-inshop-siglip2-paired-holdout-v1.json" \
    --native-library /home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so \
    --output-dir "$output" --seed "$seed" --arm bank --workers 4
  test -s "$output/receipt.json"
  echo "DONE official bank seed=$seed receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
