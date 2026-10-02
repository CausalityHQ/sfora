#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
d0b4e3228ec33d332e3d61cb31f616f4bc003b416a21245f3e3a951631a8b885  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/compare_quadratic_fresh_copy.py
9d30b37530019fdb6966eec2004af2ef8adc4979944918a1cbde9260226e407c  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/test_compare_quadratic_fresh_copy.py
cc84ddd5e59afe23d6c319e857295103d5b22a11a32a3b59e68e8a53fab065a8  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/execution.json
eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/original-mechanics-authority.json
291b9afd13449f07ca2f6929dc49be999555fa17d9b5c081ebc8094d60b2fb65  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
3ab43bea68a86cd30e5c41bebd35623e4796c9fd46e9800bc23d9ec8d244e132  /home/riomus/runs/sfora-so400-quadratic-readout-source-v6/quadratic_encoder_frames.py
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_readout.py
d1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/test_siglip2_quadratic_readout.py
f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py
a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/execution.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/compare_quadratic_fresh_copy.py --execution-sha256 cc84ddd5e59afe23d6c319e857295103d5b22a11a32a3b59e68e8a53fab065a8 --authority /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/authority.json --authority-sha256 291b9afd13449f07ca2f6929dc49be999555fa17d9b5c081ebc8094d60b2fb65 --output /home/riomus/runs/sfora-so400-quadratic-fresh-copy-control-179061-v1
sha256sum -c <<'HASHES'
d0b4e3228ec33d332e3d61cb31f616f4bc003b416a21245f3e3a951631a8b885  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/compare_quadratic_fresh_copy.py
9d30b37530019fdb6966eec2004af2ef8adc4979944918a1cbde9260226e407c  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/test_compare_quadratic_fresh_copy.py
cc84ddd5e59afe23d6c319e857295103d5b22a11a32a3b59e68e8a53fab065a8  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/execution.json
eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/original-mechanics-authority.json
291b9afd13449f07ca2f6929dc49be999555fa17d9b5c081ebc8094d60b2fb65  /home/riomus/runs/sfora-so400-quadratic-fresh-copy-source-v1/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
3ab43bea68a86cd30e5c41bebd35623e4796c9fd46e9800bc23d9ec8d244e132  /home/riomus/runs/sfora-so400-quadratic-readout-source-v6/quadratic_encoder_frames.py
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_readout.py
d1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/test_siglip2_quadratic_readout.py
f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py
a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3  /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/execution.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
