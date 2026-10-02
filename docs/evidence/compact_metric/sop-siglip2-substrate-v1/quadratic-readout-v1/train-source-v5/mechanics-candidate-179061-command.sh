#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-quadratic-readout-source-v5
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
d1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242  test_siglip2_quadratic_readout.py
f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b  train_siglip2_quadratic_readout.py
a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3  execution.json
2633aabe251d0eb86bca1fac928764d297c5cb0b25f17761f6d189f2d7de30e6  authority-mechanics-candidate-179061-v4.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py --execution-sha256 a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3 --authority /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/authority-mechanics-candidate-179061-v4.json --authority-sha256 2633aabe251d0eb86bca1fac928764d297c5cb0b25f17761f6d189f2d7de30e6 --phase mechanics --arm candidate --seed 179061 --output /home/riomus/runs/sfora-so400-quadratic-readout-mechanics-candidate-179061-v4
sha256sum -c <<'HASHES'
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
d1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242  test_siglip2_quadratic_readout.py
f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b  train_siglip2_quadratic_readout.py
a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3  execution.json
2633aabe251d0eb86bca1fac928764d297c5cb0b25f17761f6d189f2d7de30e6  authority-mechanics-candidate-179061-v4.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
