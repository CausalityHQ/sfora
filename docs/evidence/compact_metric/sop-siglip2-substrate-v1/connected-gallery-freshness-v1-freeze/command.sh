#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
4a52c22bdf53731361585787e5647bfea0a8c4f75a134eeac726c44edd2c4962  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/diagnose_connected_gallery_freshness.py
954e657c0a391847c8e8a9dd7dffa2e0691c95151d3b3f8964ef4a960b9f3900  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/test_connected_gallery_freshness.py
75caf981fd868417dcca15f512948dfad43d6f1eba6ff4cf2cc595aef01ae5c9  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/execution.json
9bd88118446ae34f9124570b2a0bc00842eb52135991e8ec2ec7f6229a4b6c6e  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/authority.json
abcb080eecc205417f520e9d5b414f99fbdc2fb9ea6ac4793f47a573f87151be  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/score-verification.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/diagnose_connected_gallery_freshness.py --execution-sha256 75caf981fd868417dcca15f512948dfad43d6f1eba6ff4cf2cc595aef01ae5c9 --authority /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/authority.json --authority-sha256 9bd88118446ae34f9124570b2a0bc00842eb52135991e8ec2ec7f6229a4b6c6e --output /home/riomus/runs/sfora-connected-gallery-freshness-diagnostic-v1
sha256sum -c <<'HASHES'
4a52c22bdf53731361585787e5647bfea0a8c4f75a134eeac726c44edd2c4962  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/diagnose_connected_gallery_freshness.py
954e657c0a391847c8e8a9dd7dffa2e0691c95151d3b3f8964ef4a960b9f3900  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/test_connected_gallery_freshness.py
75caf981fd868417dcca15f512948dfad43d6f1eba6ff4cf2cc595aef01ae5c9  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/execution.json
9bd88118446ae34f9124570b2a0bc00842eb52135991e8ec2ec7f6229a4b6c6e  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/authority.json
abcb080eecc205417f520e9d5b414f99fbdc2fb9ea6ac4793f47a573f87151be  /home/riomus/runs/sfora-connected-gallery-freshness-source-v1/score-verification.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
