#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
ad2e0872c5b81703763c3e34c247d55d0891cd54a19c2a5e27866bd53b418a9e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/evaluate_siglip2_compact_ranking.py
7057cc1b5d825ae0fbceb5d27d440a7a2c3bb255f5b6020897142fed1003d72e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/test_compact_ranking_evaluation.py
8ff136f0c902db9b60f408036fed283aacfedf653a56ea7014ed6c8af8e42c6d  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
d1d0824a2a7b01d84101216b48641998c7e2a1395f16de95c09ae11abee2a4a6  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/import-only-v5.py
d556fc64ff34d4035f64fa2f896b958c40af36d72229f9556076a701a7b4e5b2  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/import-only-authority-v5.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v10/import-only-v5.py
