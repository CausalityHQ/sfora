#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-sop-custom-gallery-smoke-179024-v5
export PYTHONPATH="$RUN/src:$RUN:/home/riomus/runs"
export HF_HUB_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test "$(sha256sum "$RUN/src/sfora/siglip2_compact_serving.py" | cut -d' ' -f1)" = c0bef23f3fae3f99cf33638d71a5966427a31bfc60549efd407138a593b578b3
test "$(sha256sum "$RUN/verify.py" | cut -d' ' -f1)" = 51fddbbd3c9520e4a214bcb3aedda9137247af324348e8194ec0b7104a20ec3e
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$RUN/verify.py"
