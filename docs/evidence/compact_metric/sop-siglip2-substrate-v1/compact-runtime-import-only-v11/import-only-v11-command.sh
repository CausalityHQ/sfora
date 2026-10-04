#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
bb4bd265d6907c5b4ecece87c48ac709a84e48795d2395cb32d1d30f69e69842  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/evaluate_siglip2_compact_ranking.py
ece11879484970ea25a54bb5ec8a014a713cdab0bab81cdac3556533fd03b8c0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/test_compact_ranking_evaluation.py
1a07a0ea0ab1f30678a38594cf579a6ad4f1b7b9e7187a7d370273affdca2377  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
ef93af822600a0cebf68adc75933a784ee3b413a1216568e80a99e48afe25c01  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/import-only-v11.py
fc8614b1cb08c162839ea7a6ec8854dd3a0d9b4e009894a80b7158556cdfb027  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/import-only-authority-v11.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v16/import-only-v11.py
