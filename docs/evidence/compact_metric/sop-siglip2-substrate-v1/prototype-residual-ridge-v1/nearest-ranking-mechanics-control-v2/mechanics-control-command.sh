#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
857ea3d7c7b39a1850d3073ccfcb58bb36470d0ef6bf6bf012725cdef62038b7  /home/riomus/runs/sfora-so400-nearest-ranking-cpu-v5/receipt.json
58e6b8a0eb2057a93734840277673a61625f11b68e91c91fca4a94e80edded5f  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/cpu-v5.log
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
5ee562302a3d5392291115e03890e1b1d49709a14b386d166088bdfb62ec2124  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-mechanics-control-v2.json
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
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py --execution-sha256 0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311 --authority /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-mechanics-control-v2.json --authority-sha256 5ee562302a3d5392291115e03890e1b1d49709a14b386d166088bdfb62ec2124 --phase mechanics --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-nearest-ranking-mechanics-control-v2
sha256sum -c <<'HASHES'
857ea3d7c7b39a1850d3073ccfcb58bb36470d0ef6bf6bf012725cdef62038b7  /home/riomus/runs/sfora-so400-nearest-ranking-cpu-v5/receipt.json
58e6b8a0eb2057a93734840277673a61625f11b68e91c91fca4a94e80edded5f  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/cpu-v5.log
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
5ee562302a3d5392291115e03890e1b1d49709a14b386d166088bdfb62ec2124  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-mechanics-control-v2.json
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
HASHES
