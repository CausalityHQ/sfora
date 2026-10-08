#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
977ae48e43feaded1301c4c4f3d2f2c46c9e0638ec1ad1e26b813cf2193a0c5a  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/observe_connected_control_batch_execution.py
abf32ffd41879e41cafc55a277dfcbf6974f2c7bc0727a1296babc9d4e1a133b  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/test_connected_control_batch_execution.py
03b5b4c4906957b2ad94dcf431b2f10f376c1b5766cc89f49c4260dc7588adbe  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/execution.json
2bed8237afb9eb4fd1b05a16505aeb224fab81320b5068524aabe0217c209b46  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/observe_connected_control_batch_execution.py --execution-sha256 03b5b4c4906957b2ad94dcf431b2f10f376c1b5766cc89f49c4260dc7588adbe --authority /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/authority.json --authority-sha256 2bed8237afb9eb4fd1b05a16505aeb224fab81320b5068524aabe0217c209b46 --output /home/riomus/runs/sfora-connected-control-batch-execution-observation-v4
sha256sum -c <<'HASHES'
977ae48e43feaded1301c4c4f3d2f2c46c9e0638ec1ad1e26b813cf2193a0c5a  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/observe_connected_control_batch_execution.py
abf32ffd41879e41cafc55a277dfcbf6974f2c7bc0727a1296babc9d4e1a133b  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/test_connected_control_batch_execution.py
03b5b4c4906957b2ad94dcf431b2f10f376c1b5766cc89f49c4260dc7588adbe  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/execution.json
2bed8237afb9eb4fd1b05a16505aeb224fab81320b5068524aabe0217c209b46  /home/riomus/runs/sfora-connected-control-batch-execution-source-v4/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
