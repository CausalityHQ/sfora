#!/bin/bash
set -eu
ulimit -v 1048576
/usr/bin/time -v -o /tmp/sfora-connected-archived-precision-run-v1/resources.log timeout 120s /usr/bin/python3 -B -I -S /home/rb/worktrees/sfora-positive-causality/scripts/diagnose_connected_archived_precision.py --launch /tmp/sfora-connected-archived-precision-run-v1/launch.json --launch-sha256 65dbf9e33ee339478c8a74245b1180577f431ff58749f2afec6f3b98a1d5df7e
