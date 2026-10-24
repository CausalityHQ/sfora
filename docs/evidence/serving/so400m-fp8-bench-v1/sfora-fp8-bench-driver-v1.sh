#!/usr/bin/env bash
set -euo pipefail
R=/home/riomus/runs; S=$R/sfora-inshop-serving-src-v1/scripts
echo "FP8 $(date -u +%FT%TZ) BEGIN cgroup=$(cat /proc/$$/cgroup)"
systemd-run --user --wait --collect --unit=sfora-fp8-bench-v1 -p StandardOutput=append:$R/sfora-fp8-bench-v1.log -p StandardError=append:$R/sfora-fp8-bench-v1.log -p WorkingDirectory=$S -p MemoryMax=32G -p MemorySwapMax=0 -p RuntimeMaxSec=3600 --setenv=PYTHONPATH=$S/../src:$S --setenv=HF_HUB_OFFLINE=1 --setenv=CUDA_VISIBLE_DEVICES=0 /usr/bin/flock $R/.sfora-siglip2-gpu.lock /usr/bin/flock /home/riomus/.sfora-siglip2-gpu.lock /home/riomus/group-learning/.venv/bin/python $S/bench_so400m_fp8_linear_v1.py --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f --output $R/sfora-fp8-bench-v1.json && echo "FP8 $(date -u +%FT%TZ) DONE" || echo "FP8 $(date -u +%FT%TZ) HOLD"
