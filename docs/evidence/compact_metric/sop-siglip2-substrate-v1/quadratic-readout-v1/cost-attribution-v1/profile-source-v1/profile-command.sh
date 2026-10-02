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
eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/authority-mechanics-profile-control-179061-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
fb01a9f9ee1facd26ca4033ce228562a88995eedaaebc3ca803422df23aa0868  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/profile_once.py
28e04566328e27c522443455fae1e5ea041dfa52c91b326f099c3b7394e563fb  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/test_profile_once.py
1d6d400886e9a7cc1d59bcdfd325524e9889a8e7c07108b0197263acb4690c9a  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/runtime-manifest.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
7834ef85dff0d2563a968da9d1a61d9c36687820b47a557fcb8ad577b71aef52  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/cProfile.py
20de245bd1ac2702ee67d064584ed9153b2ea6df8597fb696a02952e34173877  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/profile.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/profile_once.py --manifest /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/runtime-manifest.json --manifest-sha256 1d6d400886e9a7cc1d59bcdfd325524e9889a8e7c07108b0197263acb4690c9a
sha256sum -c <<'HASHES'
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
d1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242  test_siglip2_quadratic_readout.py
f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b  train_siglip2_quadratic_readout.py
a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3  execution.json
eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/authority-mechanics-profile-control-179061-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
fb01a9f9ee1facd26ca4033ce228562a88995eedaaebc3ca803422df23aa0868  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/profile_once.py
28e04566328e27c522443455fae1e5ea041dfa52c91b326f099c3b7394e563fb  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/test_profile_once.py
1d6d400886e9a7cc1d59bcdfd325524e9889a8e7c07108b0197263acb4690c9a  /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/runtime-manifest.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
7834ef85dff0d2563a968da9d1a61d9c36687820b47a557fcb8ad577b71aef52  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/cProfile.py
20de245bd1ac2702ee67d064584ed9153b2ea6df8597fb696a02952e34173877  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/profile.py
HASHES
