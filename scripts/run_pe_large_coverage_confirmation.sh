#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-coverage-checkpoint-v3
case "${1:-}" in half|full) arm=$1;; *) exit 2;; esac
output=/home/riomus/runs/sfora-large-coverage-$arm-official-b32-v1
log=$root/$arm-official-b32.log
test ! -e "$output"
test ! -e "$log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
compiler=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test -x "$compiler"
test "$(sha256sum "$compiler" | cut -d ' ' -f 1)" = df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae
systemd-run --user --unit="sfora-large-coverage-$arm-official-b32-v1" --wait --pipe \
  -p RuntimeMaxSec=299 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=CUTILE_TILEIRAS_PATH="$compiler" --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/$arm-official-b32-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/qualify_pe_large_coverage_checkpoint.py" \
  --execution-sha256 "${2:?execution SHA required}" --arm "$arm" --training-sha256 "${3:?training SHA required}" \
  --cpu-proof "$root/$arm-cpu-proof-v3.json" --cpu-sha256 "${4:?source CPU SHA required}" --output "$output" \
  > "$log" 2>&1
