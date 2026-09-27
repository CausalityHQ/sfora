#!/usr/bin/env bash
set -euo pipefail
RUN=/home/riomus/runs/sfora-sop-custom-gallery-smoke-179024-v6
export PYTHONPATH="$RUN/src:$RUN:/home/riomus/runs"
export HF_HUB_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test "$(sha256sum "$RUN/src/sfora/siglip2_compact_serving.py" | cut -d' ' -f1)" = 99fdc2b7f20a01d62faf4f02f63af4e2c65b2391d8c4bf96df81346e8f1135ec
test "$(sha256sum "$RUN/verify.py" | cut -d' ' -f1)" = ac5cb5a4f69e584cc81a6bb007a1ad77e3508b41cfee133e96328572681de2bb
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$RUN/verify.py"
