#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-quadratic-readout-source-v4
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
83e835683d95e3feff653c63ca57b85b4be45f2dd0512a3057420a4ef8d1539a  quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
62a4b94b81dd8dd10b4f76d8f105b18cfee1e6c72ee29ff6045bfd644ea0ff63  test_siglip2_quadratic_readout.py
3493ac6027b00d52106cb9814a77d6efe5616b957253da99cce375ba8b722091  train_siglip2_quadratic_readout.py
86833af1d96bf7f8ad17e62860cb25a8a4755f901ec013027c014ef2db25bd67  execution.json
f200368da29992b5aef75327c384e14836ad281365bc72d6bfb03b5abff1a50e  authority-mechanics-control-179061-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-source-v4/train_siglip2_quadratic_readout.py --execution-sha256 86833af1d96bf7f8ad17e62860cb25a8a4755f901ec013027c014ef2db25bd67 --authority /home/riomus/runs/sfora-so400-quadratic-readout-source-v4/authority-mechanics-control-179061-v3.json --authority-sha256 f200368da29992b5aef75327c384e14836ad281365bc72d6bfb03b5abff1a50e --phase mechanics --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-quadratic-readout-mechanics-control-179061-v3
sha256sum -c <<'HASHES'
83e835683d95e3feff653c63ca57b85b4be45f2dd0512a3057420a4ef8d1539a  quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
62a4b94b81dd8dd10b4f76d8f105b18cfee1e6c72ee29ff6045bfd644ea0ff63  test_siglip2_quadratic_readout.py
3493ac6027b00d52106cb9814a77d6efe5616b957253da99cce375ba8b722091  train_siglip2_quadratic_readout.py
86833af1d96bf7f8ad17e62860cb25a8a4755f901ec013027c014ef2db25bd67  execution.json
f200368da29992b5aef75327c384e14836ad281365bc72d6bfb03b5abff1a50e  authority-mechanics-control-179061-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
