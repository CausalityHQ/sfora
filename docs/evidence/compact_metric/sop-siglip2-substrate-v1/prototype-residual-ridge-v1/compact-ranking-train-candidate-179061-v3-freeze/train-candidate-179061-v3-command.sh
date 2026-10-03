#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/test_siglip2_compact_ranking.py
ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/execution.json
116a60fc2f545df9e11af609a6a192534d0104e995a2f7226ceaca1ca04ee4b0  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-cpu-v7.json
8e24c0534d2765190285c9da68b234c3bb2de4d50f9dfc66697d42b3abb929d2  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-candidate-179061-v3.json
f01ea8340e70c59196cb279ee86a3148d82932bf87a5c90871ddaa73b710b9ba  /home/riomus/runs/sfora-so400-compact-ranking-cpu-v8/receipt.json
9340a884d3707704eb9072e62222aabd7727abbfaa28f701d8172dea1095bef8  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/cpu-v8.log
021e7f3afe7fcffa1a23ea3508c506d7e3cccc4eefcafa69aeda0d2d94bb1cac  /home/riomus/runs/sfora-so400-compact-ranking-mechanics-control-v6/receipt.json
fd7a6d373d73174382e1b2bf700e86d48649faa5f233099f98c1528428e3ca24  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/mechanics-control-v6.log
09701090bf25f7d6277f48103c2d4c0362f4649da65785a900c8db1f3adb3e6c  /home/riomus/runs/sfora-so400-compact-ranking-mechanics-candidate-v6/receipt.json
a8aacfdfd6cad2a82e25bc4db28890b56cbc12cf87b8536752cb4fa520b3ae51  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/mechanics-candidate-v6.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py --execution-sha256 ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26 --authority /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-candidate-179061-v3.json --authority-sha256 8e24c0534d2765190285c9da68b234c3bb2de4d50f9dfc66697d42b3abb929d2 --phase train --arm candidate --seed 179061 --output /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3
sha256sum -c <<'HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/test_siglip2_compact_ranking.py
ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/execution.json
116a60fc2f545df9e11af609a6a192534d0104e995a2f7226ceaca1ca04ee4b0  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-cpu-v7.json
8e24c0534d2765190285c9da68b234c3bb2de4d50f9dfc66697d42b3abb929d2  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-candidate-179061-v3.json
f01ea8340e70c59196cb279ee86a3148d82932bf87a5c90871ddaa73b710b9ba  /home/riomus/runs/sfora-so400-compact-ranking-cpu-v8/receipt.json
9340a884d3707704eb9072e62222aabd7727abbfaa28f701d8172dea1095bef8  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/cpu-v8.log
021e7f3afe7fcffa1a23ea3508c506d7e3cccc4eefcafa69aeda0d2d94bb1cac  /home/riomus/runs/sfora-so400-compact-ranking-mechanics-control-v6/receipt.json
fd7a6d373d73174382e1b2bf700e86d48649faa5f233099f98c1528428e3ca24  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/mechanics-control-v6.log
09701090bf25f7d6277f48103c2d4c0362f4649da65785a900c8db1f3adb3e6c  /home/riomus/runs/sfora-so400-compact-ranking-mechanics-candidate-v6/receipt.json
a8aacfdfd6cad2a82e25bc4db28890b56cbc12cf87b8536752cb4fa520b3ae51  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/mechanics-candidate-v6.log
HASHES
