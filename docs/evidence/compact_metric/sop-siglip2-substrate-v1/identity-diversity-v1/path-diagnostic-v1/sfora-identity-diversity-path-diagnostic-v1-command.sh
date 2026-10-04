#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
printf '%s  %s\n' 7c07ae4448655219da677ad61febdb6431cf9d8ea4fb1cb5f3b243f82d768378 /home/riomus/runs/sfora-identity-diversity-path-diagnostic-v1.py | sha256sum -c
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-identity-diversity-path-diagnostic-v1.py
