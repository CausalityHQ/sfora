#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-sop-custom-gallery-scale-179024-v2
export PYTHONPATH="$RUN/src:$RUN:/home/riomus/runs"
export HF_HUB_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test "$(sha256sum "$RUN/src/sfora/siglip2_compact_serving.py" | cut -d' ' -f1)" = a960c521116a55249c31993cc494275bedc03b29adb8cc4d1eb8b5f29a0e30be
test "$(sha256sum "$RUN/verify_scale.py" | cut -d' ' -f1)" = 68fed904011604790d2a290e46316ff86c85e527c7abb9354cfa3945b210058d
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$RUN/verify_scale.py"
