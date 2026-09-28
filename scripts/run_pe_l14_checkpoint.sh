#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-pe-l14-checkpoint-v1
case "${1:-}" in
  17) phase=mechanics; runtime=119;;
  100) phase=pilot; runtime=299;;
  *) exit 2;;
esac
cd "$root"
output=/home/riomus/runs/sfora-pe-l14-checkpoint-$phase-v1
test ! -e "checkpoint-$phase-v1.log"
test ! -e "checkpoint-$phase-v1-time.txt"
test ! -e "$output"
if [[ $phase == pilot ]]; then
  /home/riomus/group-learning/.venv/bin/python -c 'import json; r=json.load(open("/home/riomus/runs/sfora-pe-l14-checkpoint-mechanics-v1/receipt.json")); assert r["advance"] and r["updates"]==17 and r["updated_gpu_strict_reload_exact"]'
fi
systemd-run --user --unit="sfora-pe-l14-checkpoint-$phase-v1" --wait --pipe \
  -p RuntimeMaxSec="$runtime" -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/checkpoint-$phase-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/train_pe_l14_checkpoint.py" \
  --execution-authority "$root/checkpoint-execution.json" --execution-authority-sha256 "${2:?frozen execution authority SHA required}" --preflight-sha256 be5a8bdcfd1749340f220d5669cebfb1c0d8b30d3c08667a0b72a4fb38701277 --mechanics-receipt-sha256 "${3:-}" --updates "$1" --output "$output" \
  > "$root/checkpoint-$phase-v1.log" 2>&1
