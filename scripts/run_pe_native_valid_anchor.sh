#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-native-valid-anchor-gpu-v1
mode=${1:?mode required}
execution=${2:?execution SHA required}
extra=()
cuda=
cap=119
case "$mode" in
  startup) name=native-valid-anchor-startup-v1; output=$root/startup-proof.json; extra=(--updates 17 --startup-only);;
  mechanics) name=native-valid-anchor-mechanics-v1; output=/home/riomus/runs/sfora-$name; cuda=0; extra=(--updates 17);;
  train)
    mechanics_sha=${3:?mechanics SHA required}
    seed=${4:?seed required}; arm=${5:?arm required}
    case "$seed" in 179041|179042) ;; *) exit 2;; esac
    case "$arm" in control|treatment) ;; *) exit 2;; esac
    name=native-valid-anchor-$seed-$arm-100-v1; output=/home/riomus/runs/sfora-$name; cuda=0; cap=299
    extra=(--updates 100 --seed "$seed" --arm "$arm" --mechanics /home/riomus/runs/sfora-native-valid-anchor-mechanics-v1 --mechanics-sha256 "$mechanics_sha");;
  *) exit 2;;
esac
test ! -e "$output"
test ! -e "$root/$name.log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
systemd-run --user --unit="sfora-$name" --wait --pipe \
  -p RuntimeMaxSec="$cap" -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES="$cuda" \
  --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/$name-time.txt" \
  /home/riomus/group-learning/.venv/bin/python -u "$root/train_pe_native_valid_anchor.py" \
  --execution-sha256 "$execution" --output "$output" "${extra[@]}" > "$root/$name.log" 2>&1
