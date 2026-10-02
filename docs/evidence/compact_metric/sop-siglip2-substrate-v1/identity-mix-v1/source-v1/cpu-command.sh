#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-identity-mix-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
d6530492c2a8c6269b8f04506bf8a7cf0f64d1ddd790553fdff1921c701ac810  authority-cpu-v1.json
c2e55e7a43b16679058a32fa0644964231867359cdac405f1c8b7bafb8fd779a  execution.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
f3124a1927ba6df62d4e7eb691b1be99b0c6d592d3b3d9f6768d997feeb80080  test_siglip2_identity_mix.py
3997facdf8748148c5afa9cc13eed4e8a6a0b9fc3dd250bd7bf6a79082e68cd3  train_siglip2_identity_mix.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-identity-mix-source-v1/train_siglip2_identity_mix.py --execution-sha256 c2e55e7a43b16679058a32fa0644964231867359cdac405f1c8b7bafb8fd779a --authority /home/riomus/runs/sfora-so400-identity-mix-source-v1/authority-cpu-v1.json --authority-sha256 d6530492c2a8c6269b8f04506bf8a7cf0f64d1ddd790553fdff1921c701ac810 --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-identity-mix-cpu-v1
sha256sum -c <<'HASHES'
d6530492c2a8c6269b8f04506bf8a7cf0f64d1ddd790553fdff1921c701ac810  authority-cpu-v1.json
c2e55e7a43b16679058a32fa0644964231867359cdac405f1c8b7bafb8fd779a  execution.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
f3124a1927ba6df62d4e7eb691b1be99b0c6d592d3b3d9f6768d997feeb80080  test_siglip2_identity_mix.py
3997facdf8748148c5afa9cc13eed4e8a6a0b9fc3dd250bd7bf6a79082e68cd3  train_siglip2_identity_mix.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
