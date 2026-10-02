#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-train-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
1effe6e87b9d667d29bc7b8904023634e7913f53c4b0a17e360728f83e1cd77b  test_siglip2_genuine_view_training.py
788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96  train_siglip2_genuine_views.py
2003e9a6adba30c8f90fa17f6437cbba1f4b6cd2d12e75906195d16229846e08  execution.json
8a1b1f278d80f1447b901778a4613f78636fa8d53cb7f70651545a1c4115accf  authority-mechanics-candidate-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-train-source-v1/train_siglip2_genuine_views.py --execution-sha256 2003e9a6adba30c8f90fa17f6437cbba1f4b6cd2d12e75906195d16229846e08 --authority /home/riomus/runs/sfora-so400-genuine-view-train-source-v1/authority-mechanics-candidate-v2.json --authority-sha256 8a1b1f278d80f1447b901778a4613f78636fa8d53cb7f70651545a1c4115accf --phase mechanics --arm candidate --seed 179061 --output /home/riomus/runs/sfora-so400-genuine-view-mechanics-candidate-v2
sha256sum -c <<'HASHES'
1effe6e87b9d667d29bc7b8904023634e7913f53c4b0a17e360728f83e1cd77b  test_siglip2_genuine_view_training.py
788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96  train_siglip2_genuine_views.py
2003e9a6adba30c8f90fa17f6437cbba1f4b6cd2d12e75906195d16229846e08  execution.json
8a1b1f278d80f1447b901778a4613f78636fa8d53cb7f70651545a1c4115accf  authority-mechanics-candidate-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
