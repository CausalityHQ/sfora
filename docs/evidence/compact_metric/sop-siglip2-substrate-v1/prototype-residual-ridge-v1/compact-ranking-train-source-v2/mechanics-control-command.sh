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
48676ed912cbf349c77a44bdae5f05268865790b040a2af5b95288e81b35df3b  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/train_siglip2_compact_ranking.py
af71d3160df5b663eb283290d0d1433ead0d105d607fd38d72c53dfa3d9d92e3  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/test_siglip2_compact_ranking.py
c3efe5ec8b8f387cebd551dc5fd265a4acb851c1b6f230a3d3e5da366c39fcba  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/execution.json
2b6c676b1725196da7beab2bfc55596c0fbb8e6268fda8f5eca4421f15e28f05  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/authority-cpu-v2.json
fa6b99f7f00ef7cdd3be9abd5261fd444280166b81b13585ab862bd9c0c9e802  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/authority-mechanics-control-v1.json
6082653bba4f503a8b0d6ff413d59ff5c0ca5388a0cf8e1d728933c8a1a27ac7  /home/riomus/runs/sfora-so400-compact-ranking-cpu-v2/receipt.json
674bb15b1ef49ca4506bb9a02a1212743dcce27af7d7f69d9c8cddd6df4ede52  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/cpu-v2.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/train_siglip2_compact_ranking.py --execution-sha256 c3efe5ec8b8f387cebd551dc5fd265a4acb851c1b6f230a3d3e5da366c39fcba --authority /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/authority-mechanics-control-v1.json --authority-sha256 fa6b99f7f00ef7cdd3be9abd5261fd444280166b81b13585ab862bd9c0c9e802 --phase mechanics --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-compact-ranking-mechanics-control-v1
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
48676ed912cbf349c77a44bdae5f05268865790b040a2af5b95288e81b35df3b  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/train_siglip2_compact_ranking.py
af71d3160df5b663eb283290d0d1433ead0d105d607fd38d72c53dfa3d9d92e3  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/test_siglip2_compact_ranking.py
c3efe5ec8b8f387cebd551dc5fd265a4acb851c1b6f230a3d3e5da366c39fcba  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/execution.json
2b6c676b1725196da7beab2bfc55596c0fbb8e6268fda8f5eca4421f15e28f05  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/authority-cpu-v2.json
fa6b99f7f00ef7cdd3be9abd5261fd444280166b81b13585ab862bd9c0c9e802  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/authority-mechanics-control-v1.json
6082653bba4f503a8b0d6ff413d59ff5c0ca5388a0cf8e1d728933c8a1a27ac7  /home/riomus/runs/sfora-so400-compact-ranking-cpu-v2/receipt.json
674bb15b1ef49ca4506bb9a02a1212743dcce27af7d7f69d9c8cddd6df4ede52  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v2/cpu-v2.log
HASHES
