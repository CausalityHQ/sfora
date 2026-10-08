#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/full-export-candidate-179069-v2-launch.sh > /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/full-export-candidate-179069-v2-original.log 2>&1
sfora_launch_status=$?
printf '%s\n' "$sfora_launch_status" > /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/full-export-candidate-179069-v2-terminal-status.txt
exit "$sfora_launch_status"
