#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
afe1b5de4135309b4099840aad77d644c0af90182cfe6c77b9ca265ad0742878  evaluate_siglip2_genuine_views.py
3e5c9de6efb016aade362ac5900e41cac7603f4719bfdabfc5be6a0398d89d58  test_siglip2_genuine_view_evaluation.py
0ef4f638e5a44ea287b2dd7b61dd265ee41f3bdd550f76ddb02b596ba1eb04db  execution.json
8e0fd3465cff9c6d24619d57b59ff2d30ab5b763cef5b3d149844afc5a5f3153  authority-first-selection-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v1/evaluate_siglip2_genuine_views.py --execution-sha256 0ef4f638e5a44ea287b2dd7b61dd265ee41f3bdd550f76ddb02b596ba1eb04db --authority /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v1/authority-first-selection-v1.json --authority-sha256 8e0fd3465cff9c6d24619d57b59ff2d30ab5b763cef5b3d149844afc5a5f3153 --phase cpu --output /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-cpu-v1
sha256sum -c <<'HASHES'
afe1b5de4135309b4099840aad77d644c0af90182cfe6c77b9ca265ad0742878  evaluate_siglip2_genuine_views.py
3e5c9de6efb016aade362ac5900e41cac7603f4719bfdabfc5be6a0398d89d58  test_siglip2_genuine_view_evaluation.py
0ef4f638e5a44ea287b2dd7b61dd265ee41f3bdd550f76ddb02b596ba1eb04db  execution.json
8e0fd3465cff9c6d24619d57b59ff2d30ab5b763cef5b3d149844afc5a5f3153  authority-first-selection-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
