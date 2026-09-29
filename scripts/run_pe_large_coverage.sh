#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-coverage-training-v2
case "${1:-}" in
  mechanics) phase=mechanics; arm=full; updates=17; extra=();;
  half|full) phase=$1-100; arm=$1; updates=100; extra=(--mechanics /home/riomus/runs/sfora-large-coverage-mechanics-v1 --mechanics-sha256 "${3:?mechanics SHA required}");;
  *) exit 2;;
esac
output=/home/riomus/runs/sfora-large-coverage-$phase-v1
log=$root/coverage-$phase.log
test ! -e "$log"
test ! -e "$output"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
systemd-run --user --unit="sfora-large-coverage-$phase-v1" --wait --pipe \
  -p RuntimeMaxSec=299 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/coverage-$phase-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/train_pe_large_coverage.py" \
  --execution-sha256 "${2:?execution SHA required}" --output "$output" --arm "$arm" --updates "$updates" "${extra[@]}" \
  > "$log" 2>&1
