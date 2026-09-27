#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-sop-custom-gallery-scale-179024-v7
export PYTHONPATH="$RUN/src:$RUN:/home/riomus/runs"
export HF_HUB_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test "$(sha256sum "$RUN/src/sfora/siglip2_compact_serving.py" | cut -d' ' -f1)" = 73ad606c78ad22e6b377627e4e7c8fe07d7dfaafd07a029c40df2982221a7fd1
test "$(sha256sum "$RUN/verify_scale.py" | cut -d' ' -f1)" = 4992e64cf6cfaa6de6494e53e4f854d0a8ac941b65aed26da2501cc181a5df2f
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$RUN/verify_scale.py"
