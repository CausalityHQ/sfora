#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
4b6836c15bd680c56551b5ba43ba201aa35327fe48b6bd1f06e2c54c50e24229  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/observe_connected_full_authority.py
1baee5e37efbc45891ad7a302e20b61a5002e40467087a217b6d2d10477610a9  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
9cf3ba0e005fcf1b9c1433359bae09a2540fb6143d6a5c83b8c2b1cfc60802a3  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/evaluate_siglip2_connected_mlp.py
56dd3f10a2eefdcb9ce63b63b134d9b934d14f50ecaf58ca67668b363002b6e7  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/execution.json
a7f4ea4e438df350b87c7a70d4591c1ba1dbcf3c1ea12c084bb71279b089b994  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/authority-full-cpu-v3.json
66d11f73c0a49ef0a66878b8517d13c31865c8fb4f88d6930bf197511b1bd3fd  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/full-cpu-v3-command.sh
2e869791c5667558a6c4d5d2459c4fba973a6705c08d5921a16ffac002ddef8d  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/test_connected_mlp_evaluation.py
8b83d33b5111311b5956872adac4d85df60d7be7da242a37a1c98cd8690b1f2f  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/observe_connected_full_authority.py --manifest /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/manifest.json --manifest-sha256 8b83d33b5111311b5956872adac4d85df60d7be7da242a37a1c98cd8690b1f2f
sha256sum -c <<'HASHES'
4b6836c15bd680c56551b5ba43ba201aa35327fe48b6bd1f06e2c54c50e24229  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/observe_connected_full_authority.py
1baee5e37efbc45891ad7a302e20b61a5002e40467087a217b6d2d10477610a9  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
9cf3ba0e005fcf1b9c1433359bae09a2540fb6143d6a5c83b8c2b1cfc60802a3  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/evaluate_siglip2_connected_mlp.py
56dd3f10a2eefdcb9ce63b63b134d9b934d14f50ecaf58ca67668b363002b6e7  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/execution.json
a7f4ea4e438df350b87c7a70d4591c1ba1dbcf3c1ea12c084bb71279b089b994  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/authority-full-cpu-v3.json
66d11f73c0a49ef0a66878b8517d13c31865c8fb4f88d6930bf197511b1bd3fd  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/full-cpu-v3-command.sh
2e869791c5667558a6c4d5d2459c4fba973a6705c08d5921a16ffac002ddef8d  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v7/test_connected_mlp_evaluation.py
8b83d33b5111311b5956872adac4d85df60d7be7da242a37a1c98cd8690b1f2f  /home/riomus/runs/sfora-connected-full-authority-observation-source-v4/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
