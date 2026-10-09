#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'PINS'
ad9efe3876aba7fa35f418a5e224ea8692d99a2876e02e60722edfa254249831  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/inputs.sha256
f99170e5c728cb07b8825ec325b56d1c3baf8a455c3143edebbcc939a61b01fb  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/verify_fresh_sha_table.py
53409094fbbffd51eb8d7f38b47e6248d60b9e6346ed1ec184cf5a1574639922  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/residency_sampler.py
PINS
printf "COMPACT_PHASE launcher_scan_start\n"
/usr/bin/python3 -I -B -S /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/verify_fresh_sha_table.py < /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/inputs.sha256
printf "COMPACT_PHASE launcher_scan_end\n"
sha256sum -c <<'PINS'
ad9efe3876aba7fa35f418a5e224ea8692d99a2876e02e60722edfa254249831  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/inputs.sha256
f99170e5c728cb07b8825ec325b56d1c3baf8a455c3143edebbcc939a61b01fb  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/verify_fresh_sha_table.py
53409094fbbffd51eb8d7f38b47e6248d60b9e6346ed1ec184cf5a1574639922  /home/riomus/runs/sfora-connected-probe-launcher-sha-ab-source-v1/residency_sampler.py
PINS
