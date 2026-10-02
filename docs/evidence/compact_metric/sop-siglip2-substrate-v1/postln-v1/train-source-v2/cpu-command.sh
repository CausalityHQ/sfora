#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-postln-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
8cb549d62f7ea43196292196f366434322b6fd38098b5905905fa1b0fbd2fe2c  train_siglip2_postln_adaptation.py
17251204578ee5e469884d8d5400b8f9a06ed92bc4ef53cf303e1e25d39685a3  test_siglip2_postln_adaptation.py
5cddac9f95553747beb18ecf89e61a1cdf72a56fe63f477c99470b92acc0d31f  execution.json
1c85b23565ba98f915b4ce3ac6dd545073f8a73a97e0b5be5e26c3d3ff563b00  authority-cpu-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-postln-source-v2/train_siglip2_postln_adaptation.py --execution-sha256 5cddac9f95553747beb18ecf89e61a1cdf72a56fe63f477c99470b92acc0d31f --authority /home/riomus/runs/sfora-so400-postln-source-v2/authority-cpu-v2.json --authority-sha256 1c85b23565ba98f915b4ce3ac6dd545073f8a73a97e0b5be5e26c3d3ff563b00 --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-postln-cpu-v2
sha256sum -c <<'HASHES'
8cb549d62f7ea43196292196f366434322b6fd38098b5905905fa1b0fbd2fe2c  train_siglip2_postln_adaptation.py
17251204578ee5e469884d8d5400b8f9a06ed92bc4ef53cf303e1e25d39685a3  test_siglip2_postln_adaptation.py
5cddac9f95553747beb18ecf89e61a1cdf72a56fe63f477c99470b92acc0d31f  execution.json
1c85b23565ba98f915b4ce3ac6dd545073f8a73a97e0b5be5e26c3d3ff563b00  authority-cpu-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
