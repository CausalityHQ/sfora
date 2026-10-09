#!/bin/bash
# ONE outer timeout 120 / AS 1GiB around both full tests, sequential.
cd /home/rb/worktrees/sfora-restore-original-serving-sha-20261009
ulimit -v 1048576
exec /usr/bin/time -v -o /tmp/sfora-restore-original-serving-sha-20261009/final_run.time \
  timeout 120 bash -c 'python3 scripts/test_connected_inference_extraction.py && python3 scripts/test_connected_probe_serving.py'
