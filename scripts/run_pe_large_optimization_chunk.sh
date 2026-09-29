#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-large-optimization-chunks-v1
arm=${1:?arm required}
end=${2:?chunk end required}
case "$arm" in half|full) ;; *) exit 2;; esac
[[ $end =~ ^[0-9]+$ ]] && ((end >= 100 && end <= 2000 && end % 100 == 0))
output=/home/riomus/runs/sfora-large-optimization-$arm-$end-v1
log=$root/$arm-$end.log
extra=()
if ((end > 100)); then
  previous=/home/riomus/runs/sfora-large-optimization-$arm-$((end-100))-v1
  extra=(--previous "$previous" --previous-sha256 "${5:?previous receipt SHA required}")
else
  test "$#" -eq 4
fi
test ! -e "$log"
test ! -e "$output"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
systemd-run --user --unit="sfora-large-optimization-$arm-$end-v1" --wait --pipe \
  -p RuntimeMaxSec=299 -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock \
  /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/$arm-$end-time.txt" \
  /home/riomus/group-learning/.venv/bin/python -u "$root/train_pe_large_optimization_chunk.py" \
  --execution-sha256 "${3:?execution SHA required}" \
  --mechanics /home/riomus/runs/sfora-large-optimization-mechanics-v2 \
  --mechanics-sha256 "${4:?mechanics SHA required}" \
  --output "$output" --arm "$arm" --chunk-end "$end" "${extra[@]}" \
  > "$log" 2>&1
