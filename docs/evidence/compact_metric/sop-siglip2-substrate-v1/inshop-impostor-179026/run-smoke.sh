#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-inshop-impostor-179026-v1
updates="${1:?17 or 100 updates required}"
[[ "$updates" == 17 || "$updates" == 100 ]]
export PYTHONPATH="$RUN/src:/home/riomus/runs:/home/riomus/sfora-siglip2-deployed-batch-v1/scripts"
export HF_HUB_OFFLINE=1
test "$(sha256sum "$RUN/trainer.py" | cut -d' ' -f1)" = 9d64781961155fb57441f77f4fd2b76bf04652b1543bcdd62a4168c8e5f02401
test "$(sha256sum "$RUN/src/sfora/deployed_code_rank.py" | cut -d' ' -f1)" = d61e0e4b7b4a6ba88e7ee84dcce67c2172e0ca907d02af6d57938481abfbb354
test "$(sha256sum "$RUN/train_sop_siglip2_compact.py" | cut -d' ' -f1)" = ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
for arm in freeze_emb freeze_emb_impostor; do
  output="$RUN/$arm-$updates"
  test ! -e "$output"
  /home/riomus/group-learning/.venv/bin/python "$RUN/trainer.py" \
    --dataset-root /home/riomus/datasets/inshop_official_standard \
    --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
    --features-dir /home/riomus/runs/sfora-inshop-siglip2-train-features-v1 \
    --preflight /home/riomus/runs/sfora-inshop-expected-gallery-179026-v1/seed-179026.json \
    --preflight-sha256 4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254 \
    --output-dir "$output" --seed 179026 --arm "$arm" --updates "$updates"
  sha256sum "$output/receipt.json"
done
