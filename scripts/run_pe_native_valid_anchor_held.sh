#!/usr/bin/env bash
set -euo pipefail
root=/home/riomus/runs/sfora-native-valid-anchor-held-v3
mode=${1:?mode required}; seed=${2:?seed required}; arm=${3:?arm required}
execution=${4:?execution SHA required}; training=${5:?training SHA required}
case "$seed" in 179041|179042) ;; *) exit 2;; esac
case "$arm" in control|treatment) ;; *) exit 2;; esac
extra=(); cuda=; cap=119
proof=$root/source-$seed-$arm-proof.json
held=/home/riomus/runs/sfora-native-valid-anchor-held-$seed-$arm-v1
case "$mode" in
  cpu) name=source-$seed-$arm; output=$proof; extra=(--qualify-cpu);;
  held) name=held-$seed-$arm; output=$held; cuda=0; cap=299; extra=(--cpu-proof "$proof" --cpu-sha256 "${6:?source CPU SHA required}");;
  audit) name=audit-$seed-$arm; output=$held; extra=(--cpu-proof "$proof" --cpu-sha256 "${6:?source CPU SHA required}" --audit-cpu --receipt-sha256 "${7:?held receipt SHA required}");;
  *) exit 2;;
esac
test ! -e "$root/$name.log"
if [[ $mode == audit ]]; then test ! -e "$held/cpu-audit.json"; else test ! -e "$output"; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units --state=running --no-pager --no-legend 'sfora-*')"
compiler=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test -x "$compiler"
test "$(sha256sum "$compiler" | cut -d ' ' -f 1)" = df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae
systemd-run --user --unit="sfora-native-valid-anchor-$name-v3" --wait --pipe \
  -p RuntimeMaxSec="$cap" -p TimeoutStopSec=1 -p MemoryMax=8G -p MemorySwapMax=0 -p KillMode=control-group \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=CUDA_VISIBLE_DEVICES="$cuda" \
  --setenv=CUTILE_TILEIRAS_PATH="$compiler" --setenv=PYTHONPATH="$root:$root/src:$root/isolated-deps" \
  /usr/bin/flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
  /usr/bin/time -v -o "$root/$name-time.txt" \
  /home/riomus/group-learning/.venv/bin/python -u "$root/qualify_pe_native_valid_anchor_held.py" \
  --execution-sha256 "$execution" --seed "$seed" --arm "$arm" --training-sha256 "$training" --output "$output" "${extra[@]}" > "$root/$name.log" 2>&1
