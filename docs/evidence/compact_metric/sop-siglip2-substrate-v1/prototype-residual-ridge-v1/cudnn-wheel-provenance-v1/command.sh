#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
eb546ed3d6e88f5058f8cbec7e6ed160b5da54cd6883f710dd64b21aca32aeed  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/execution.json
f7fb3c39d90151c0fd171262f3d813bbeff17cba97a8d2e4a70e84c2a13260e4  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/authority.json
78d25af1531f2e209d712055c7882fcbd2f7d4f0379ab2fc1e35096ec9ffb12e  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/installed-evidence.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
2c08f68be84e42a68bb3fbaa499121cf1ab11d0d9612a46647d406ec86ea0470  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/qualify_cudnn_wheel_provenance.py
3397b0d2f8e5f1ad7024acbde3c585446125ec3d1686ff051802903bbf7f74d5  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/test_cudnn_wheel_provenance.py
HASHES
/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -I -S -B /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/qualify_cudnn_wheel_provenance.py --execution-sha256 eb546ed3d6e88f5058f8cbec7e6ed160b5da54cd6883f710dd64b21aca32aeed --authority /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/authority.json --authority-sha256 f7fb3c39d90151c0fd171262f3d813bbeff17cba97a8d2e4a70e84c2a13260e4 --output /home/riomus/runs/sfora-cudnn-wheel-provenance-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
eb546ed3d6e88f5058f8cbec7e6ed160b5da54cd6883f710dd64b21aca32aeed  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/execution.json
f7fb3c39d90151c0fd171262f3d813bbeff17cba97a8d2e4a70e84c2a13260e4  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/authority.json
78d25af1531f2e209d712055c7882fcbd2f7d4f0379ab2fc1e35096ec9ffb12e  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/installed-evidence.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
2c08f68be84e42a68bb3fbaa499121cf1ab11d0d9612a46647d406ec86ea0470  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/qualify_cudnn_wheel_provenance.py
3397b0d2f8e5f1ad7024acbde3c585446125ec3d1686ff051802903bbf7f74d5  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/test_cudnn_wheel_provenance.py
HASHES
