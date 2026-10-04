#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
32e8830c57c023ec3162a1003d9e3a6fd166a7fe953fc7832701df61f3cbf830  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/evaluate_siglip2_compact_ranking.py
267960ec73f630dca87ac6edbb62bcb2209beab047388f6d2f2404a87b5c5338  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/test_compact_ranking_evaluation.py
4f644dfaf1d1bdbd3f574f4bb6326ed4978d030b3247ba91c096ffeac8dd6042  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
c6271795ba45278e15e317debea3037e250e3cca82ae1141ed96fe44d53e3764  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/import-only-v8.py
bacd37c519ececd19f1d6634fd4bd9474cd4ca4345dd2035b549a8da4ba2831d  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/import-only-authority-v8.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v13/import-only-v8.py
