#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-connected-mlp-train-source-v6/mechanics-control-179061-v2-launch.sh > /home/riomus/runs/sfora-connected-mlp-train-source-v6/mechanics-control-179061-v2-original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-connected-mlp-train-source-v6/mechanics-control-179061-v2-terminal-status.txt
exit "$sfora_launch_status"
