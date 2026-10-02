#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-quadratic-readout-source-v3
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
5172c92b47ac18ee2de62d09aaec9413d2fa42b8e519526d9d0da02d7cfab24c  test_siglip2_quadratic_readout.py
3639f0d129a508406fdb0f0047962d6d2bdb6349bb069a754675ea4bef003124  train_siglip2_quadratic_readout.py
8a25f31c4ba2fc0063b6ef61b9604d040bbef39ff6182a1a82d5cb97b71ccba4  execution.json
9e086f3d202112c82ec2324f4fd06584f37e14ed984b00d49db5d93b0bcc3fdb  authority-mechanics-candidate-179061-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-source-v3/train_siglip2_quadratic_readout.py --execution-sha256 8a25f31c4ba2fc0063b6ef61b9604d040bbef39ff6182a1a82d5cb97b71ccba4 --authority /home/riomus/runs/sfora-so400-quadratic-readout-source-v3/authority-mechanics-candidate-179061-v2.json --authority-sha256 9e086f3d202112c82ec2324f4fd06584f37e14ed984b00d49db5d93b0bcc3fdb --phase mechanics --arm candidate --seed 179061 --output /home/riomus/runs/sfora-so400-quadratic-readout-mechanics-candidate-179061-v2
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
5172c92b47ac18ee2de62d09aaec9413d2fa42b8e519526d9d0da02d7cfab24c  test_siglip2_quadratic_readout.py
3639f0d129a508406fdb0f0047962d6d2bdb6349bb069a754675ea4bef003124  train_siglip2_quadratic_readout.py
8a25f31c4ba2fc0063b6ef61b9604d040bbef39ff6182a1a82d5cb97b71ccba4  execution.json
9e086f3d202112c82ec2324f4fd06584f37e14ed984b00d49db5d93b0bcc3fdb  authority-mechanics-candidate-179061-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
