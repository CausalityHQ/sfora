#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-sop-custom-gallery-179024-v1
export PYTHONPATH="$RUN/src:/home/riomus/runs"
export HF_HUB_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test "$(sha256sum "$RUN/src/sfora/siglip2_compact_serving.py" | cut -d' ' -f1)" = eba33aa34607c45b7b72b80e645dddb502f2904cfc123f32998460a266d80192
test "$(sha256sum "$RUN/verify.py" | cut -d' ' -f1)" = 02ff4129b8ff859f2d75524d49b21aab930fad99965949daa116a5183cdb040d
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$RUN/verify.py"
