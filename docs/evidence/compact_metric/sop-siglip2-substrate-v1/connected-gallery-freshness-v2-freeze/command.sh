#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
617564f93b78799f4d274bdd295088a53ebe220f8ba4c6ef157f5d69cd82f3d5  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/diagnose_connected_gallery_freshness.py
a93fbefb8523860a729987a86de7373ca14414fda870849b7933fd4dc1ebb075  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/test_connected_gallery_freshness.py
aa435769db7ba23975ee227f22488139820324245ca1c4d457d51b79981c9f27  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/execution.json
5422a9938258f3fce8c79f3f2a965aa1c682df827a55f15e650ef3e00737d333  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/authority.json
abcb080eecc205417f520e9d5b414f99fbdc2fb9ea6ac4793f47a573f87151be  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/score-verification.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/diagnose_connected_gallery_freshness.py --execution-sha256 aa435769db7ba23975ee227f22488139820324245ca1c4d457d51b79981c9f27 --authority /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/authority.json --authority-sha256 5422a9938258f3fce8c79f3f2a965aa1c682df827a55f15e650ef3e00737d333 --output /home/riomus/runs/sfora-connected-gallery-freshness-diagnostic-v2
sha256sum -c <<'HASHES'
617564f93b78799f4d274bdd295088a53ebe220f8ba4c6ef157f5d69cd82f3d5  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/diagnose_connected_gallery_freshness.py
a93fbefb8523860a729987a86de7373ca14414fda870849b7933fd4dc1ebb075  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/test_connected_gallery_freshness.py
aa435769db7ba23975ee227f22488139820324245ca1c4d457d51b79981c9f27  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/execution.json
5422a9938258f3fce8c79f3f2a965aa1c682df827a55f15e650ef3e00737d333  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/authority.json
abcb080eecc205417f520e9d5b414f99fbdc2fb9ea6ac4793f47a573f87151be  /home/riomus/runs/sfora-connected-gallery-freshness-source-v2/score-verification.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
