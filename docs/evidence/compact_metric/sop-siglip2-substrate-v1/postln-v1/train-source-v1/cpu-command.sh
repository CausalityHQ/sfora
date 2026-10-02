#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-postln-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
0265e9461da9c065d4f1fdd697a59cbe7acd822bc83a4563c4786ed9ce83de4f  train_siglip2_postln_adaptation.py
22ab44aeec564537b9dfb8ed4f0fffd517ca7ccdc6fc3cba6f1c7e165f5e2bc1  test_siglip2_postln_adaptation.py
8954a1ab95169297899da760ea4d4d9a168870ff1e7ace34b47c20bf295de2af  execution.json
6b7d7227c9d64ab53f5a351bb2d45548d77b235a3217756d8d5cb386f89dff20  authority-cpu-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-postln-source-v1/train_siglip2_postln_adaptation.py --execution-sha256 8954a1ab95169297899da760ea4d4d9a168870ff1e7ace34b47c20bf295de2af --authority /home/riomus/runs/sfora-so400-postln-source-v1/authority-cpu-v1.json --authority-sha256 6b7d7227c9d64ab53f5a351bb2d45548d77b235a3217756d8d5cb386f89dff20 --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-postln-cpu-v1
sha256sum -c <<'HASHES'
0265e9461da9c065d4f1fdd697a59cbe7acd822bc83a4563c4786ed9ce83de4f  train_siglip2_postln_adaptation.py
22ab44aeec564537b9dfb8ed4f0fffd517ca7ccdc6fc3cba6f1c7e165f5e2bc1  test_siglip2_postln_adaptation.py
8954a1ab95169297899da760ea4d4d9a168870ff1e7ace34b47c20bf295de2af  execution.json
6b7d7227c9d64ab53f5a351bb2d45548d77b235a3217756d8d5cb386f89dff20  authority-cpu-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
