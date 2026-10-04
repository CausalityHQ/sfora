#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3bee6de7f0b61900185487fc6281e15cd4094f39277ed45e8e6beed369305428  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/evaluate_siglip2_compact_ranking.py
fedfe810ff99cfada73fb4e59c52d6ac9bd34ef3c71338a30b07979c74288e98  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/test_compact_ranking_evaluation.py
df46725301b28a711a93974c7339e1aff6f80335fee6126f0ce8a9616a486b36  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
c117323d21bbe3e70eda6169d48829d627e8000e1f8e882e5c3d5f11147d04d0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/import-only-v9.py
0ce726441bede4903784b6e82261bcc18e2f4c22651a712612791f8335f13d2a  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/import-only-authority-v9.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v14/import-only-v9.py
