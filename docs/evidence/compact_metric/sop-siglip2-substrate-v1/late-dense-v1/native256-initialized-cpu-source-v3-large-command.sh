#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-initialized-cpu-source-v3
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67  joint_relational_compaction.py
2bea89a983e366a5db7a08d2d9112393029a47326e34029d7e8d7f3b8b0367db  qualify_siglip2_initialized_cpu.py
0635805daf79c6595d6e06f208e6ea42c8bbcc84add0dc1bbce63e8cc4ec57b8  test_siglip2_initialized_cpu.py
483366ffe853d4bbcc2e2166533f692b80c114a34d6ad5ca81cdbbdce08d3ba9  execution.json
ac6d3ed54791d925cfbcfe80d01eeb20ea84801d1fa33c5e8c874740616832f6  authority-large-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-initialized-cpu-source-v3/qualify_siglip2_initialized_cpu.py --execution-sha256 483366ffe853d4bbcc2e2166533f692b80c114a34d6ad5ca81cdbbdce08d3ba9 --authority /home/riomus/runs/sfora-native256-initialized-cpu-source-v3/authority-large-v3.json --authority-sha256 ac6d3ed54791d925cfbcfe80d01eeb20ea84801d1fa33c5e8c874740616832f6 --arm large --output /home/riomus/runs/sfora-native256-initialized-cpu-large-v3
sha256sum -c <<'HASHES'
4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67  joint_relational_compaction.py
2bea89a983e366a5db7a08d2d9112393029a47326e34029d7e8d7f3b8b0367db  qualify_siglip2_initialized_cpu.py
0635805daf79c6595d6e06f208e6ea42c8bbcc84add0dc1bbce63e8cc4ec57b8  test_siglip2_initialized_cpu.py
483366ffe853d4bbcc2e2166533f692b80c114a34d6ad5ca81cdbbdce08d3ba9  execution.json
ac6d3ed54791d925cfbcfe80d01eeb20ea84801d1fa33c5e8c874740616832f6  authority-large-v3.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
