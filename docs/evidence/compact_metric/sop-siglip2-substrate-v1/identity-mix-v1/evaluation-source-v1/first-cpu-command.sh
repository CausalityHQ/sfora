#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-identity-mix-evaluation-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
78caa9eae26bb229236b9a0a85e8db4a0ed40ab13e9bad8496adc444041ad682  authority-first-selection-v1.json
42ed70cb1e973acab0ef4f68f4a7e02943c28f29ac8d7f8405c8c93cad98c3b6  evaluate_siglip2_identity_mix.py
f5711ecd17b439d8807871a6e8fa8164f9e2a435d9030e8510f0e76cebd2e086  execution.json
37f94eab96fd5d9170444ea64ffd9b7f2aa7b04bc5120b3873ba30a2a47aaab1  test_siglip2_identity_mix_evaluation.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-identity-mix-evaluation-source-v1/evaluate_siglip2_identity_mix.py --execution-sha256 f5711ecd17b439d8807871a6e8fa8164f9e2a435d9030e8510f0e76cebd2e086 --authority /home/riomus/runs/sfora-so400-identity-mix-evaluation-source-v1/authority-first-selection-v1.json --authority-sha256 78caa9eae26bb229236b9a0a85e8db4a0ed40ab13e9bad8496adc444041ad682 --phase cpu --output /home/riomus/runs/sfora-so400-identity-mix-evaluation-first-cpu-v1
sha256sum -c <<'HASHES'
78caa9eae26bb229236b9a0a85e8db4a0ed40ab13e9bad8496adc444041ad682  authority-first-selection-v1.json
42ed70cb1e973acab0ef4f68f4a7e02943c28f29ac8d7f8405c8c93cad98c3b6  evaluate_siglip2_identity_mix.py
f5711ecd17b439d8807871a6e8fa8164f9e2a435d9030e8510f0e76cebd2e086  execution.json
37f94eab96fd5d9170444ea64ffd9b7f2aa7b04bc5120b3873ba30a2a47aaab1  test_siglip2_identity_mix_evaluation.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
