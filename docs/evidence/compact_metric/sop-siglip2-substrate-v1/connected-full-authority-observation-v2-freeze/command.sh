#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
97c653e45519549fd023909a41a88c61d7f80684ca2cd4ad73c0dd4b9813b9a0  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/observe_connected_full_authority.py
7bbe6b53d4f1855683d6ef7847e5bf2681747e8d2a16604d980390cff3629e6b  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/evaluate_siglip2_connected_mlp.py
a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/execution.json
d8f3b0f92a02937f63539782f2653a20c4d7fbc320022c1554ee261f718fe71f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/authority-full-cpu-v1.json
911428b835ed6323b2c58c927db996121e10b4845a0c306f3c815b2a012998f0  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/full-cpu-v1-command.sh
df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/test_connected_mlp_evaluation.py
30ddf964348d0c9cc9e1b2fe3f1691ab28656f231e3adb5d3e83e470186c8632  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/observe_connected_full_authority.py --manifest /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/manifest.json --manifest-sha256 30ddf964348d0c9cc9e1b2fe3f1691ab28656f231e3adb5d3e83e470186c8632
sha256sum -c <<'HASHES'
97c653e45519549fd023909a41a88c61d7f80684ca2cd4ad73c0dd4b9813b9a0  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/observe_connected_full_authority.py
7bbe6b53d4f1855683d6ef7847e5bf2681747e8d2a16604d980390cff3629e6b  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/evaluate_siglip2_connected_mlp.py
a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/execution.json
d8f3b0f92a02937f63539782f2653a20c4d7fbc320022c1554ee261f718fe71f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/authority-full-cpu-v1.json
911428b835ed6323b2c58c927db996121e10b4845a0c306f3c815b2a012998f0  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/full-cpu-v1-command.sh
df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/test_connected_mlp_evaluation.py
30ddf964348d0c9cc9e1b2fe3f1691ab28656f231e3adb5d3e83e470186c8632  /home/riomus/runs/sfora-connected-full-authority-observation-source-v2/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
