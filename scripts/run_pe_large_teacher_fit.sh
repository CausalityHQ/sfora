#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-teacher-fit-v1
case "${1:-}" in
  cpu) phase=cpu; limit=119; visibility=; extra=(--qualify-cpu); locks=();;
  fit) phase=targets; limit=299; visibility=0; extra=(--cpu-sha256 "${3:?CPU SHA required}"); locks=(/usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock);;
  *) exit 2;;
esac
output=/home/riomus/runs/sfora-large-teacher-fit-$phase-v1
test ! -e "$root/teacher-fit-$phase-v1.log"
test ! -e "$output"
systemd-run --user --unit="sfora-large-teacher-fit-$phase-v1" --wait --pipe \
  -p RuntimeMaxSec="$limit" -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES="$visibility" --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  "${locks[@]}" /usr/bin/time -v -o "$root/teacher-fit-$phase-v1-time.txt" \
  /home/riomus/group-learning/.venv/bin/python "$root/export_pe_large_teacher_fit.py" \
  --execution-sha256 "${2:?execution SHA required}" --output "$output" "${extra[@]}" \
  > "$root/teacher-fit-$phase-v1.log" 2>&1
