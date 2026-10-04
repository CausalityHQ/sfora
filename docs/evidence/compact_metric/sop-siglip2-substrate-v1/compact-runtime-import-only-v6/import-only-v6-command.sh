#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
8c7dd65ae30c2e87139c68e75fac0a4f7064a22ab500841dfa610e76da03bf71  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/evaluate_siglip2_compact_ranking.py
49ad2921c6ad2fcb47beaacbdd5e668468b9af919e602fe6d1465c45420fbd0f  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/test_compact_ranking_evaluation.py
c4a3fcff52a790a0a05a15a719896beb72a461c0711c236a7ee7e0475c9f4a84  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
30f48a9e8c6f1f0140e19659c5d65ddd9f1e2975ad808be5dbf7dc3eb6c25616  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/import-only-v6.py
227bcfeb94393fb862e3a12bf19a09698a9eb18e1372ae96030f90fb420eedcd  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/import-only-authority-v6.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v11/import-only-v6.py
