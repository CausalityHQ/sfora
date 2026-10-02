#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  evaluate_siglip2_genuine_views.py
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  test_siglip2_genuine_view_evaluation.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  execution.json
a528f7da1c6839f219bcbc4f3c60539c96b1196e8731c0db464113afc0a8413d  authority-first-selection-v4.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py --execution-sha256 82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f --authority /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/authority-first-selection-v4.json --authority-sha256 a528f7da1c6839f219bcbc4f3c60539c96b1196e8731c0db464113afc0a8413d --phase cpu --output /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-cpu-v4
sha256sum -c <<'HASHES'
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  evaluate_siglip2_genuine_views.py
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  test_siglip2_genuine_view_evaluation.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  execution.json
a528f7da1c6839f219bcbc4f3c60539c96b1196e8731c0db464113afc0a8413d  authority-first-selection-v4.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
