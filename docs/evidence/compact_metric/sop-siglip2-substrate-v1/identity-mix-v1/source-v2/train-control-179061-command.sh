#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-identity-mix-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
2451f3f8b2bd117c09dbb2322dafe9c83e937acc2801ba08916345d74581679b  authority-train-control-179061-v2.json
8abb08aa570e0f436805cb4f7cf574eea3a5dae4db3fa7b73e43e45747e397ae  execution.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
fa474ad759b52b924b829f6ba6b17ea5d04d2ac86342b07845d89d0fa96faa9d  test_siglip2_identity_mix.py
7567309ca9f7e91f76a58ae96784d5fcd9c1efdc9924daf6f5386f827d60f85f  train_siglip2_identity_mix.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-identity-mix-source-v2/train_siglip2_identity_mix.py --execution-sha256 8abb08aa570e0f436805cb4f7cf574eea3a5dae4db3fa7b73e43e45747e397ae --authority /home/riomus/runs/sfora-so400-identity-mix-source-v2/authority-train-control-179061-v2.json --authority-sha256 2451f3f8b2bd117c09dbb2322dafe9c83e937acc2801ba08916345d74581679b --phase train --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-identity-mix-train-control-179061-v2
sha256sum -c <<'HASHES'
2451f3f8b2bd117c09dbb2322dafe9c83e937acc2801ba08916345d74581679b  authority-train-control-179061-v2.json
8abb08aa570e0f436805cb4f7cf574eea3a5dae4db3fa7b73e43e45747e397ae  execution.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
fa474ad759b52b924b829f6ba6b17ea5d04d2ac86342b07845d89d0fa96faa9d  test_siglip2_identity_mix.py
7567309ca9f7e91f76a58ae96784d5fcd9c1efdc9924daf6f5386f827d60f85f  train_siglip2_identity_mix.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
