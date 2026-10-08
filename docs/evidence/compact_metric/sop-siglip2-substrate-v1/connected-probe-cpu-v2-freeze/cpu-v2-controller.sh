#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-connected-probe-train-source-v2/cpu-v2-launch.sh > /home/riomus/runs/sfora-connected-probe-train-source-v2/cpu-v2-original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-connected-probe-train-source-v2/cpu-v2-terminal-status.txt
exit "$sfora_launch_status"
