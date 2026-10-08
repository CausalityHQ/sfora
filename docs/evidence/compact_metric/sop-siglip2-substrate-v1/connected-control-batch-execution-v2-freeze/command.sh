#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
d09d9c438d160154c9ebe77bffefb8883b89653ccdbd60b004363556e99dc127  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/observe_connected_control_batch_execution.py
b017d9c42963dddd75fc14a2fe747cb7b633f1bf7477f55e8e0587ebb36f9d03  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/test_connected_control_batch_execution.py
4340657f54e8b73a73098af3a1002fc936a7169a736cfa7c13fed8544836ef17  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/execution.json
24e7312e91fea6ec04204bf0b4a83abcaa0423d1ced1613e2c3f2b25ce38989d  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/observe_connected_control_batch_execution.py --execution-sha256 4340657f54e8b73a73098af3a1002fc936a7169a736cfa7c13fed8544836ef17 --authority /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/authority.json --authority-sha256 24e7312e91fea6ec04204bf0b4a83abcaa0423d1ced1613e2c3f2b25ce38989d --output /home/riomus/runs/sfora-connected-control-batch-execution-observation-v2
sha256sum -c <<'HASHES'
d09d9c438d160154c9ebe77bffefb8883b89653ccdbd60b004363556e99dc127  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/observe_connected_control_batch_execution.py
b017d9c42963dddd75fc14a2fe747cb7b633f1bf7477f55e8e0587ebb36f9d03  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/test_connected_control_batch_execution.py
4340657f54e8b73a73098af3a1002fc936a7169a736cfa7c13fed8544836ef17  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/execution.json
24e7312e91fea6ec04204bf0b4a83abcaa0423d1ced1613e2c3f2b25ce38989d  /home/riomus/runs/sfora-connected-control-batch-execution-source-v2/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
