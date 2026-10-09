#!/bin/bash
set -u
set -o noclobber
/bin/bash /home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v3/launch.sh > /home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v3/original.log 2>&1
sfora_status=$?
printf '%s\n' "$sfora_status" > /home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v3/terminal-status.txt
exit "$sfora_status"
