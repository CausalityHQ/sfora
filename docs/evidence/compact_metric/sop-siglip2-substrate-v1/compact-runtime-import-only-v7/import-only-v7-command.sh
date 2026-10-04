#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
045c27359ff8e670a0596b177f5b119b1304fe56874e27809af2b83c67ea39fd  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/evaluate_siglip2_compact_ranking.py
2deec95d3ab51a4557afc8750f745889176df37f82c5812174163fc783c1d82c  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/test_compact_ranking_evaluation.py
6ce59e332a04acaa824638f7e38dc142e003e141254ac7bcdaeac95d3fb0ff89  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/execution.json
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
55159154c0ca26766c6ad56685a88d426aa190642aa894ff359632ce47d5f8ec  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-cpu-v5/receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-native256-source-cpu-so400-v4/proof.json
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
dd28b43d7ccf2db5b14f02fc1357865e7747df3677b37e01abb0e3e99bf4d681  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/import-only-v7.py
d7affe260aa827edac91c8ba90421fdbe8280019daab10f8b9fba99836482df0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/import-only-authority-v7.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v12/import-only-v7.py
