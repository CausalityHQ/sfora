#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v3
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
e6d5eb58d8df92c98b3d1063fcb0fa6763df5207d6914c4b567e58783bdf18dd  evaluate_siglip2_genuine_views.py
633a6f048c09dba943c311494350bcd4b3df679532e5f700fe128e11b66685e7  test_siglip2_genuine_view_evaluation.py
111b3c721418e4e6c5c51a47702e4b0a15375fca4690a311e61a73c71370d888  execution.json
bcf0d346a2a2f647206b21431762015fe0e71b711317796d3cd75e849748cead  authority-first-selection-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v3/evaluate_siglip2_genuine_views.py --execution-sha256 111b3c721418e4e6c5c51a47702e4b0a15375fca4690a311e61a73c71370d888 --authority /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v3/authority-first-selection-v3.json --authority-sha256 bcf0d346a2a2f647206b21431762015fe0e71b711317796d3cd75e849748cead --phase cpu --output /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-cpu-v3
sha256sum -c <<'HASHES'
e6d5eb58d8df92c98b3d1063fcb0fa6763df5207d6914c4b567e58783bdf18dd  evaluate_siglip2_genuine_views.py
633a6f048c09dba943c311494350bcd4b3df679532e5f700fe128e11b66685e7  test_siglip2_genuine_view_evaluation.py
111b3c721418e4e6c5c51a47702e4b0a15375fca4690a311e61a73c71370d888  execution.json
bcf0d346a2a2f647206b21431762015fe0e71b711317796d3cd75e849748cead  authority-first-selection-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
