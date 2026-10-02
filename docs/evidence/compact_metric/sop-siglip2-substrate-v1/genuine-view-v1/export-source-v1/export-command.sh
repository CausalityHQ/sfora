#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-export-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
61db5c05eb55094b22b9a3f59254063fd0e49cbd4c02340fd7fa345035c25b0b  startup-terminal.json
374d2cf0562b462276ace3c5967b7f35c64d18bb69ebbc89cd3639ae0e6d805a  startup.log
34ac1412b0287c1ce891d49d49debecb6ff5e892f5c6e81a649fd925c7c316ce  export_siglip2_genuine_views.py
d7b5c1a607e7cbd0ff3a01d22e9d5889a250d29b7e406cf9ce3f934290f0fde0  test_siglip2_genuine_views.py
2e4bcececc3b32c01544cdd3b075bf3c6079a8e3680bcb4e668b1e080f030031  execution.json
862cdec3df3b12af88d00c3fa9412fa01804222fc6defbaff39c84d532a51577  authority.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  train_sop_siglip2_compact.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-export-source-v1/export_siglip2_genuine_views.py --execution-sha256 2e4bcececc3b32c01544cdd3b075bf3c6079a8e3680bcb4e668b1e080f030031 --authority /home/riomus/runs/sfora-so400-genuine-view-export-source-v1/authority.json --authority-sha256 862cdec3df3b12af88d00c3fa9412fa01804222fc6defbaff39c84d532a51577 --phase export --output /home/riomus/runs/sfora-so400-genuine-view-export-v1 --startup-terminal /home/riomus/runs/sfora-so400-genuine-view-export-source-v1/startup-terminal.json --startup-terminal-sha256 61db5c05eb55094b22b9a3f59254063fd0e49cbd4c02340fd7fa345035c25b0b
sha256sum -c <<'HASHES'
61db5c05eb55094b22b9a3f59254063fd0e49cbd4c02340fd7fa345035c25b0b  startup-terminal.json
374d2cf0562b462276ace3c5967b7f35c64d18bb69ebbc89cd3639ae0e6d805a  startup.log
34ac1412b0287c1ce891d49d49debecb6ff5e892f5c6e81a649fd925c7c316ce  export_siglip2_genuine_views.py
d7b5c1a607e7cbd0ff3a01d22e9d5889a250d29b7e406cf9ce3f934290f0fde0  test_siglip2_genuine_views.py
2e4bcececc3b32c01544cdd3b075bf3c6079a8e3680bcb4e668b1e080f030031  execution.json
862cdec3df3b12af88d00c3fa9412fa01804222fc6defbaff39c84d532a51577  authority.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  train_sop_siglip2_compact.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
