#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
86adf642e3f8d1070bdb80d21e8760a39b5c37f8733086f3eb37bb35caac6565  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/observe_connected_full_authority.py
77fb110aff0f185a4384bf50fb9f778b5038326f657a067a5e1ed99dd152d518  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
0a807abc27cbf7150824b2dd202e7be75aaa548f0e27918a1d5b1c862908850b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/evaluate_siglip2_connected_mlp.py
046fbeb6db0e162e905492b7ae10828bf018c1ffd35359ac9c9e723172bed33f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/execution.json
156140e6ef654a620322c20d1ec4ef40c1beaa78c2256a5dcbf8d9a24ac06ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/authority-full-cpu-v4.json
635dae6ab5ba856654c9b000bc359b93173e63c98f2d5802735b27911d145763  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/full-cpu-v4-command.sh
09818ee1aab610160db21d95f16689c679000e26b1062296e81bb1f988fcbf63  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/test_connected_mlp_evaluation.py
953d4e7900b1bba38a35d55662836d522cf0aa4df3a7ffb46e4b938e09f5df41  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/observe_connected_full_authority.py --manifest /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/manifest.json --manifest-sha256 953d4e7900b1bba38a35d55662836d522cf0aa4df3a7ffb46e4b938e09f5df41
sha256sum -c <<'HASHES'
86adf642e3f8d1070bdb80d21e8760a39b5c37f8733086f3eb37bb35caac6565  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/observe_connected_full_authority.py
77fb110aff0f185a4384bf50fb9f778b5038326f657a067a5e1ed99dd152d518  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
0a807abc27cbf7150824b2dd202e7be75aaa548f0e27918a1d5b1c862908850b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/evaluate_siglip2_connected_mlp.py
046fbeb6db0e162e905492b7ae10828bf018c1ffd35359ac9c9e723172bed33f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/execution.json
156140e6ef654a620322c20d1ec4ef40c1beaa78c2256a5dcbf8d9a24ac06ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/authority-full-cpu-v4.json
635dae6ab5ba856654c9b000bc359b93173e63c98f2d5802735b27911d145763  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/full-cpu-v4-command.sh
09818ee1aab610160db21d95f16689c679000e26b1062296e81bb1f988fcbf63  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v8/test_connected_mlp_evaluation.py
953d4e7900b1bba38a35d55662836d522cf0aa4df3a7ffb46e4b938e09f5df41  /home/riomus/runs/sfora-connected-full-authority-observation-source-v5/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
