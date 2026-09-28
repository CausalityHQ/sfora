#!/usr/bin/env bash
set -euo pipefail
run_root=/home/riomus/runs/sfora-preprocessing-worker-v1
test ! -e "$run_root/worker-cpu-v1.json"
test ! -e "$run_root/worker-cpu-v1.log"
systemd-run --user --unit=sfora-preprocessing-worker-cpu-v1 --wait \
 --property=RuntimeMaxSec=180 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$run_root" \
 --property="StandardOutput=append:$run_root/worker-cpu-v1.log" \
 --property="StandardError=append:$run_root/worker-cpu-v1.log" \
 --setenv=CUDA_VISIBLE_DEVICES= \
 --setenv="PYTHONPATH=$run_root:$run_root/src" \
 /usr/bin/time -v -o "$run_root/worker-cpu-v1-time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$run_root/probe_sop_preprocessing_worker.py" \
 "$run_root/worker-cpu-v1.json"
