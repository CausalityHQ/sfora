#!/bin/bash
set -euo pipefail
root=/home/riomus/runs/sfora-pe-fp16-mechanics-v1
test ! -e "$root/fp16-smoke-v1.log"
test ! -e /home/riomus/runs/sfora-pe-fp16-smoke-v1/receipt.json
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
systemd-run --user --unit=sfora-pe-fp16-smoke-v1 \
  --property=RuntimeMaxSec=180 --property=MemoryMax=8G \
  --property=KillMode=control-group --working-directory="$root" \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=OMP_NUM_THREADS=8 \
  --setenv="PYTHONPATH=$root:$root/src:$root/isolated-deps" \
  /bin/bash -c 'set -euo pipefail; set -C; exec /usr/bin/flock -n /home/riomus/.sfora-siglip2-gpu.lock /usr/bin/time -v -o fp16-smoke-v1-time.txt /home/riomus/group-learning/.venv/bin/python probe_inshop_pe_training_smoke.py --root /home/riomus/runs/sfora-pe-fp16-mechanics-v1 --dataset-root /home/riomus/datasets/inshop_official_standard --large-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c --output /home/riomus/runs/sfora-pe-fp16-smoke-v1 --vision-precision fp16 > fp16-smoke-v1.log 2>&1'
