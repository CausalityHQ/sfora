#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
82c78d8c6cea0ccd1d81250d4bb44659e1ef8bbbb972b6ba93192be557a379a0  evaluate_siglip2_genuine_views.py
67c7e957d3b7b87cf3cf6304468afbe3cddab473074e4d86a1abf4ddd09cfe56  test_siglip2_genuine_view_evaluation.py
3f406808619f6f27f3630155d37b4b8b1f9e5a54118ffa8620db53cb73509ee9  execution.json
b8fc0a0c0c1ef5a74cefa7fa733f2014d3e2f260ecd06b2f3bb5a98ec550c7c7  authority-first-selection-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v2/evaluate_siglip2_genuine_views.py --execution-sha256 3f406808619f6f27f3630155d37b4b8b1f9e5a54118ffa8620db53cb73509ee9 --authority /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v2/authority-first-selection-v2.json --authority-sha256 b8fc0a0c0c1ef5a74cefa7fa733f2014d3e2f260ecd06b2f3bb5a98ec550c7c7 --phase cpu --output /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-cpu-v2
sha256sum -c <<'HASHES'
82c78d8c6cea0ccd1d81250d4bb44659e1ef8bbbb972b6ba93192be557a379a0  evaluate_siglip2_genuine_views.py
67c7e957d3b7b87cf3cf6304468afbe3cddab473074e4d86a1abf4ddd09cfe56  test_siglip2_genuine_view_evaluation.py
3f406808619f6f27f3630155d37b4b8b1f9e5a54118ffa8620db53cb73509ee9  execution.json
b8fc0a0c0c1ef5a74cefa7fa733f2014d3e2f260ecd06b2f3bb5a98ec550c7c7  authority-first-selection-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
