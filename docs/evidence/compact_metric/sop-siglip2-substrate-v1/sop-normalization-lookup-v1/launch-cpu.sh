#!/usr/bin/env bash
set -euo pipefail
code_root=/home/riomus/runs/sfora-preprocessing-worker-v1
run_root=/home/riomus/runs/sfora-normalization-lookup-v1
mkdir -p "$run_root"
test ! -e "$run_root/receipt.json"
test ! -e "$run_root/run.log"
systemd-run --user --unit=sfora-normalization-lookup-cpu-v1 --wait \
 --property=RuntimeMaxSec=120 --property=MemoryMax=8G --property=KillMode=control-group \
 --property="WorkingDirectory=$code_root" \
 --property="StandardOutput=append:$run_root/run.log" \
 --property="StandardError=append:$run_root/run.log" \
 --setenv=CUDA_VISIBLE_DEVICES= \
 --setenv="PYTHONPATH=$code_root:$code_root/src" \
 /usr/bin/time -v -o "$run_root/time.txt" \
 /home/riomus/group-learning/.venv/bin/python "$code_root/probe_sop_normalization_lookup.py" \
 "$run_root/receipt.json"
