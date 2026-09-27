#!/usr/bin/env bash
set -euo pipefail
run=/home/riomus/runs/sfora-sop-cars-transfer-public-v1
export PYTHONPATH="$run/src:$run/scripts"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
exec 9>/home/riomus/runs/.sfora-siglip2-gpu.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
exec /home/riomus/group-learning/.venv/bin/python "$run/scripts/verify_sop_siglip2_cars_transfer.py"
