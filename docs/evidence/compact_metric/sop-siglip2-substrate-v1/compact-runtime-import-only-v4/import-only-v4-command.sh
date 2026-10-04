#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
41a9405dc584a268268db4c9638427acd2912e782647ff9514b51ffd1d9baee8  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/evaluate_siglip2_compact_ranking.py
3dd42969d4acbcbb5550c732735e3e7f0e074edd7794672586ae5e66e09a8365  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/test_compact_ranking_evaluation.py
3cdf491a117a5ae6c5cfa9c2b5949f2da88bc368dbbac7d0e9fb30f7791f0e03  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
f031a8c7ee48e3b31fdeba7d3339f6b1156921f8b551ca851ffc38474042a636  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/import-only-v4.py
ece47b313f54d6b0930719ed6cfa423e2137364a3d8a3b02bc72ec1169887898  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/import-only-authority-v4.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v9/import-only-v4.py
