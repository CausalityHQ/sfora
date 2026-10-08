#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
dc274cc3ecc8a2ff08aa69fc9954b367f47768c52a6caee77fb508ececb2b524  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/observe_connected_control_batch_execution.py
34ad39653df9802de79ea2365b2acba17443313c18ea90646c45cc051d7d597d  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/test_connected_control_batch_execution.py
0472570c604b3bfbac0e6d8e7026c804321bc7d5525d6eaf4ea76122555893fc  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/execution.json
3ed85073ab003e09509983ccabbdffddbb11419e3c2e4fad9fe0dc201504d133  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/observe_connected_control_batch_execution.py --execution-sha256 0472570c604b3bfbac0e6d8e7026c804321bc7d5525d6eaf4ea76122555893fc --authority /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/authority.json --authority-sha256 3ed85073ab003e09509983ccabbdffddbb11419e3c2e4fad9fe0dc201504d133 --output /home/riomus/runs/sfora-connected-control-batch-execution-observation-v1
sha256sum -c <<'HASHES'
dc274cc3ecc8a2ff08aa69fc9954b367f47768c52a6caee77fb508ececb2b524  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/observe_connected_control_batch_execution.py
34ad39653df9802de79ea2365b2acba17443313c18ea90646c45cc051d7d597d  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/test_connected_control_batch_execution.py
0472570c604b3bfbac0e6d8e7026c804321bc7d5525d6eaf4ea76122555893fc  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/execution.json
3ed85073ab003e09509983ccabbdffddbb11419e3c2e4fad9fe0dc201504d133  /home/riomus/runs/sfora-connected-control-batch-execution-source-v1/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
