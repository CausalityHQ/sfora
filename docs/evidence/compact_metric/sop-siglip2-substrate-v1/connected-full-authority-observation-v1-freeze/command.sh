#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9dd557cb358db2a74603f52f73bba6bd2c060baffb139ab14a71eca512eb63aa  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/observe_connected_full_authority.py
c309b85f7100f0ad44c408f6105d37c2adfadfd4f30951666d373122951a2631  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/evaluate_siglip2_connected_mlp.py
a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/execution.json
d8f3b0f92a02937f63539782f2653a20c4d7fbc320022c1554ee261f718fe71f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/authority-full-cpu-v1.json
911428b835ed6323b2c58c927db996121e10b4845a0c306f3c815b2a012998f0  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/full-cpu-v1-command.sh
df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/test_connected_mlp_evaluation.py
0fe19765b5d9cdb4cded7fde48394d6d8bdf3b8a1dd6fa4a1d472cbf8e719f54  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/observe_connected_full_authority.py --manifest /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/manifest.json --manifest-sha256 0fe19765b5d9cdb4cded7fde48394d6d8bdf3b8a1dd6fa4a1d472cbf8e719f54
sha256sum -c <<'HASHES'
9dd557cb358db2a74603f52f73bba6bd2c060baffb139ab14a71eca512eb63aa  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/observe_connected_full_authority.py
c309b85f7100f0ad44c408f6105d37c2adfadfd4f30951666d373122951a2631  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/evaluate_siglip2_connected_mlp.py
a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/execution.json
d8f3b0f92a02937f63539782f2653a20c4d7fbc320022c1554ee261f718fe71f  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/authority-full-cpu-v1.json
911428b835ed6323b2c58c927db996121e10b4845a0c306f3c815b2a012998f0  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/full-cpu-v1-command.sh
df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/test_connected_mlp_evaluation.py
0fe19765b5d9cdb4cded7fde48394d6d8bdf3b8a1dd6fa4a1d472cbf8e719f54  /home/riomus/runs/sfora-connected-full-authority-observation-source-v1/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
