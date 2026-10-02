#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-export-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
1e1ca5607e941a2e93c21df93d681e4aa237fbd830b5133d452dd9e97507ae4b  startup-terminal.json
43faad6250b92e147a874523f190b2bbff04b55bdd6a0f1184372cad75708d72  startup.log
e5e98f9bc85680cab013d53752e7fa140da9d5e7f7ecd4537413b29b1047f65e  export_siglip2_genuine_views.py
980fe18f9fd2baaf7e1bb5c19738bc7971c3010f6b4e66c83d284dd7c79afb2b  test_siglip2_genuine_views.py
8ec7f2687f7d1e7962de4f9753cefae989d0b6310d325b5e19da33a9961af3dc  execution.json
a0072203febf1ef3d16ea329deb0a52dccef34a63223d7dc4b5a430d43d8b6a4  authority.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  train_sop_siglip2_compact.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/export_siglip2_genuine_views.py --execution-sha256 8ec7f2687f7d1e7962de4f9753cefae989d0b6310d325b5e19da33a9961af3dc --authority /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/authority.json --authority-sha256 a0072203febf1ef3d16ea329deb0a52dccef34a63223d7dc4b5a430d43d8b6a4 --phase export --output /home/riomus/runs/sfora-so400-genuine-view-export-v2 --startup-terminal /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/startup-terminal.json --startup-terminal-sha256 1e1ca5607e941a2e93c21df93d681e4aa237fbd830b5133d452dd9e97507ae4b
sha256sum -c <<'HASHES'
1e1ca5607e941a2e93c21df93d681e4aa237fbd830b5133d452dd9e97507ae4b  startup-terminal.json
43faad6250b92e147a874523f190b2bbff04b55bdd6a0f1184372cad75708d72  startup.log
e5e98f9bc85680cab013d53752e7fa140da9d5e7f7ecd4537413b29b1047f65e  export_siglip2_genuine_views.py
980fe18f9fd2baaf7e1bb5c19738bc7971c3010f6b4e66c83d284dd7c79afb2b  test_siglip2_genuine_views.py
8ec7f2687f7d1e7962de4f9753cefae989d0b6310d325b5e19da33a9961af3dc  execution.json
a0072203febf1ef3d16ea329deb0a52dccef34a63223d7dc4b5a430d43d8b6a4  authority.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  partition.json
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  train_sop_siglip2_compact.py
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
