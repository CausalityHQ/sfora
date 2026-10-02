#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-quadratic-readout-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
aa1393baa254464cbd63ff1c20064ceb3fb1c9212fa627c02ccd0f7998e289a4  test_siglip2_quadratic_readout.py
c6b1886d3bbcc8d7fa8c10ba391f44284708a70f326b9dc6432e54d11768cf4b  train_siglip2_quadratic_readout.py
a5204aea1a6de718a7e7af557131002c2a64c27b2053af6bbac6a39c5ac1bf96  execution.json
9ad207f24391d255d30ee5dabe6273f54555e9eb0574d2b737dcd830870364df  authority-cpu-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-source-v2/train_siglip2_quadratic_readout.py --execution-sha256 a5204aea1a6de718a7e7af557131002c2a64c27b2053af6bbac6a39c5ac1bf96 --authority /home/riomus/runs/sfora-so400-quadratic-readout-source-v2/authority-cpu-v2.json --authority-sha256 9ad207f24391d255d30ee5dabe6273f54555e9eb0574d2b737dcd830870364df --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-quadratic-readout-cpu-v2
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
aa1393baa254464cbd63ff1c20064ceb3fb1c9212fa627c02ccd0f7998e289a4  test_siglip2_quadratic_readout.py
c6b1886d3bbcc8d7fa8c10ba391f44284708a70f326b9dc6432e54d11768cf4b  train_siglip2_quadratic_readout.py
a5204aea1a6de718a7e7af557131002c2a64c27b2053af6bbac6a39c5ac1bf96  execution.json
9ad207f24391d255d30ee5dabe6273f54555e9eb0574d2b737dcd830870364df  authority-cpu-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
