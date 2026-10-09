#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-rank-routed-mlp-train-source-v3/cpu-v3-launch.sh > /home/riomus/runs/sfora-rank-routed-mlp-train-source-v3/cpu-v3-original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-rank-routed-mlp-train-source-v3/cpu-v3-terminal-status.txt
exit "$sfora_launch_status"
