#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
18f1124aa7ab4aa86be162dffae1db52c75f6a02f11668eb0d1241c689667554  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/observe_connected_full_authority.py
af16b339de90fa8701b47a3327ca57f9a7db9cc6f9da6ab5c09888d3d45551f5  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
2390c60fe5e87e82ab122c5c0101476b378792541bc454470c37d6c7f0410d40  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/evaluate_siglip2_connected_mlp.py
a0c1f27db4da404e7777d89518dfc83e2e20b607fcc1031776978e9ca8ec9f3c  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/execution.json
3444df504430a2ee92f33fac04cfed609c512de542d8993f08069dd4bd24e19d  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/authority-full-cpu-v2.json
8998b7abb9ad2cbbb119aff15b722578845b39d42ae179c8a183c3000fa0e1f7  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/full-cpu-v2-command.sh
28d0297cebd6a7bab395a44bc60a4aa1bcaacb9160a8093044a54fa270b104d9  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/test_connected_mlp_evaluation.py
e1d50ca169e7159b7d871eb17d0356b864e313ecbf407a7aadb9ee98e397f974  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/observe_connected_full_authority.py --manifest /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/manifest.json --manifest-sha256 e1d50ca169e7159b7d871eb17d0356b864e313ecbf407a7aadb9ee98e397f974
sha256sum -c <<'HASHES'
18f1124aa7ab4aa86be162dffae1db52c75f6a02f11668eb0d1241c689667554  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/observe_connected_full_authority.py
af16b339de90fa8701b47a3327ca57f9a7db9cc6f9da6ab5c09888d3d45551f5  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/test_connected_full_authority_observation.py
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
2390c60fe5e87e82ab122c5c0101476b378792541bc454470c37d6c7f0410d40  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/evaluate_siglip2_connected_mlp.py
a0c1f27db4da404e7777d89518dfc83e2e20b607fcc1031776978e9ca8ec9f3c  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/execution.json
3444df504430a2ee92f33fac04cfed609c512de542d8993f08069dd4bd24e19d  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/authority-full-cpu-v2.json
8998b7abb9ad2cbbb119aff15b722578845b39d42ae179c8a183c3000fa0e1f7  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/full-cpu-v2-command.sh
28d0297cebd6a7bab395a44bc60a4aa1bcaacb9160a8093044a54fa270b104d9  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/test_connected_mlp_evaluation.py
e1d50ca169e7159b7d871eb17d0356b864e313ecbf407a7aadb9ee98e397f974  /home/riomus/runs/sfora-connected-full-authority-observation-source-v3/manifest.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
