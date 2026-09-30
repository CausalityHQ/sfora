#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-initialized-cpu-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67  joint_relational_compaction.py
4000bf97b864d92890ef47a2ec2b2dbd6cb3db4bed1c3977727276220d18e807  qualify_siglip2_initialized_cpu.py
0635805daf79c6595d6e06f208e6ea42c8bbcc84add0dc1bbce63e8cc4ec57b8  test_siglip2_initialized_cpu.py
ce88515d639b8039b57a169770990763b3eb09f3d1c64d7b5d37a903bf50ef6c  execution.json
a53cecd64ae967f8cae89966fe0649fb3db0087a71d2a69c27b782db2578bdd2  authority-so400-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-initialized-cpu-source-v1/qualify_siglip2_initialized_cpu.py --execution-sha256 ce88515d639b8039b57a169770990763b3eb09f3d1c64d7b5d37a903bf50ef6c --authority /home/riomus/runs/sfora-native256-initialized-cpu-source-v1/authority-so400-v1.json --authority-sha256 a53cecd64ae967f8cae89966fe0649fb3db0087a71d2a69c27b782db2578bdd2 --arm so400 --output /home/riomus/runs/sfora-native256-initialized-cpu-so400-v1
sha256sum -c <<'HASHES'
4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67  joint_relational_compaction.py
4000bf97b864d92890ef47a2ec2b2dbd6cb3db4bed1c3977727276220d18e807  qualify_siglip2_initialized_cpu.py
0635805daf79c6595d6e06f208e6ea42c8bbcc84add0dc1bbce63e8cc4ec57b8  test_siglip2_initialized_cpu.py
ce88515d639b8039b57a169770990763b3eb09f3d1c64d7b5d37a903bf50ef6c  execution.json
a53cecd64ae967f8cae89966fe0649fb3db0087a71d2a69c27b782db2578bdd2  authority-so400-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
