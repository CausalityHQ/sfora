#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
edbd5e928e242b870e60b6ea5ca92c7a1710ca74e59b5a9fb174d3b45b927965  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/evaluate_siglip2_compact_ranking.py
6d1cbc277839c968215388e68c34a4817f96fdd2a0a1268689c64b3af26a5e6a  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/test_compact_ranking_evaluation.py
3ad48e29e16797820488a5929c7248bfeb065c155cd49a728c8fd1ceef8d8737  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
2776c770f8f17b21990df7124a5ca75de1c3b8521edc0abfe5d87ca41a8f3bf5  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/import-only-v10.py
c4909892b56427c3ef4e0ecbaa10eaea916a058925e990a032ec4a617bf15af0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/import-only-authority-v10.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v15/import-only-v10.py
