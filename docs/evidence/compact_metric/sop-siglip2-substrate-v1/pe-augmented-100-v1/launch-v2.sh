#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-pe-augmented-pair-v1
out_root=/home/riomus/runs/sfora-pe-augmented-100-v2
cd "$run_root"
for name in attempt.json receipt.json large.pt pe.pt large.held.npy pe.held.npy large.json pe.json; do
  test ! -e "$out_root/$name"
done
test ! -e "$run_root/augmented-100-v2.log"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$(systemctl --user list-units 'sfora*' --state=running --no-legend --no-pager)"
flock -n /home/riomus/.sfora-siglip2-gpu.lock true
systemd-run --user --unit=sfora-pe-augmented-100-v2 \
 --property=RuntimeMaxSec=600 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$run_root" \
 --property="StandardOutput=append:$run_root/augmented-100-v2.log" \
 --property="StandardError=append:$run_root/augmented-100-v2.log" \
 --setenv=CUDA_VISIBLE_DEVICES=0 \
 --setenv="PYTHONPATH=$run_root:$run_root/src:$run_root/isolated-deps" \
 /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock \
 /usr/bin/time -v -o "$run_root/augmented-100-v2-time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$run_root/train_inshop_pe_pair.py" \
 --root "$run_root" --cache /home/riomus/runs/sfora-pe-fullfit-cache-v3 \
 --output "$out_root" --dataset-root /home/riomus/datasets/inshop_official_standard \
 --large-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c \
 --mechanics-dir /home/riomus/runs/sfora-pe-fp16-smoke-v1 \
 --preflight-sha256 41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293
systemctl --user show sfora-pe-augmented-100-v2 --property=InvocationID,ActiveState,SubState,RuntimeMaxUSec,MemoryMax,KillMode
