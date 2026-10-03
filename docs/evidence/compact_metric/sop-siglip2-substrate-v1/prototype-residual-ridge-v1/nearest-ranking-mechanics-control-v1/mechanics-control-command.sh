#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
13b4c6bf2f6ffc837a2832756a4236c4d1fcd23dd11cf43ba6af1df78150a9a0  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/authority-mechanics-control-v1.json
8678ae6f62df6d554728096c3442e06494f22db146ca502d3530876fee43026b  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/nearest_ranking_readout.py
60af70908cdbc60ed7a7dfc1aadb3395f27cda2395a638eeab10aafdb009edde  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/test_siglip2_nearest_ranking.py
bc86ee5a32ab899c881220348aa50503361e52a8574d9bfa55244b344e10f171  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
2e888839e3d38c343258db7deadd0f3048620dbddfb771234ec625d7fbdd0949  /home/riomus/runs/sfora-so400-nearest-ranking-cpu-v1/receipt.json
fbf9ab84122774fe9a08ea2da4c463c59e6013683f4c90bce0ac1eec8a69c59e  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/cpu-v1.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/train_siglip2_nearest_ranking.py --execution-sha256 8678ae6f62df6d554728096c3442e06494f22db146ca502d3530876fee43026b --authority /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/authority-mechanics-control-v1.json --authority-sha256 13b4c6bf2f6ffc837a2832756a4236c4d1fcd23dd11cf43ba6af1df78150a9a0 --phase mechanics --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-nearest-ranking-mechanics-control-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
13b4c6bf2f6ffc837a2832756a4236c4d1fcd23dd11cf43ba6af1df78150a9a0  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/authority-mechanics-control-v1.json
8678ae6f62df6d554728096c3442e06494f22db146ca502d3530876fee43026b  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/nearest_ranking_readout.py
60af70908cdbc60ed7a7dfc1aadb3395f27cda2395a638eeab10aafdb009edde  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/test_siglip2_nearest_ranking.py
bc86ee5a32ab899c881220348aa50503361e52a8574d9bfa55244b344e10f171  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
2e888839e3d38c343258db7deadd0f3048620dbddfb771234ec625d7fbdd0949  /home/riomus/runs/sfora-so400-nearest-ranking-cpu-v1/receipt.json
fbf9ab84122774fe9a08ea2da4c463c59e6013683f4c90bce0ac1eec8a69c59e  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v1/cpu-v1.log
HASHES
