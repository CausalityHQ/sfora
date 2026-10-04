#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
4adf432ac2c32cdcabc0c4571db8e470620747cfe3bc5b6a2e51468b069b9add  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/evaluate_siglip2_compact_ranking.py
afaf8c47af2b4c9ff51cd6dc79f109c9461142508d29a11e45bd26ae6cd12de2  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/test_compact_ranking_evaluation.py
959d4e7908a48c6b979b3a29b6bfb3889f5f9e9591fe69e22180f89fdbd2113e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
8d2cc1e7a182d4c475fffd3a73920061a7bb9e66ad3e378939746181e5810967  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/import-only-v1.py
0c19f13c4e6f4fe187860345eefc2f3eebeb792d97bb543af421d357ba53f0da  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/import-only-authority-v1.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v6/import-only-v1.py
