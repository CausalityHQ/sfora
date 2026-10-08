#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/full-cpu-v4-launch.sh > /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/full-cpu-v4-original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/full-cpu-v4-terminal-status.txt
exit "$sfora_launch_status"
