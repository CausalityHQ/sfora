#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-cached-readout-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9a95ac46b4d9a15997837c5904ff5c40c3ba9e3af28fab2c5d0b02837e25aae4  train_siglip2_cached_readout.py
05b15d129e78fc5d914ef6706088328b3b7b37dfb43af39ac43ec4fd1482bd22  test_siglip2_cached_readout.py
16b29bc94b3ef74f72cc8047f3304ec719e66742a1f72d3b2f8ae72e57cb3c64  execution.json
e0ff2e1ffa286ec4fb85ac899eabc6160eb28027a4a557fb0548c9a7ebd8ede7  authority-cpu-v1.json
4694490f347949a7d814f5001655d1c7a9eb41c5ea369f4e838b231ad2ee7662  sfora-native256-fit-export-collected-inputs-v1.json
da925901082ff22994a90dcdb0ccfea0cd5e2144cb3f87c3950904e9928a503e  sfora-native256-full-initialized-cpu-collected-inputs-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-cached-readout-source-v1/train_siglip2_cached_readout.py --execution-sha256 16b29bc94b3ef74f72cc8047f3304ec719e66742a1f72d3b2f8ae72e57cb3c64 --authority /home/riomus/runs/sfora-so400-cached-readout-source-v1/authority-cpu-v1.json --authority-sha256 e0ff2e1ffa286ec4fb85ac899eabc6160eb28027a4a557fb0548c9a7ebd8ede7 --phase cpu --arm control --seed 179032 --output /home/riomus/runs/sfora-so400-cached-readout-cpu-v1
sha256sum -c <<'HASHES'
9a95ac46b4d9a15997837c5904ff5c40c3ba9e3af28fab2c5d0b02837e25aae4  train_siglip2_cached_readout.py
05b15d129e78fc5d914ef6706088328b3b7b37dfb43af39ac43ec4fd1482bd22  test_siglip2_cached_readout.py
16b29bc94b3ef74f72cc8047f3304ec719e66742a1f72d3b2f8ae72e57cb3c64  execution.json
e0ff2e1ffa286ec4fb85ac899eabc6160eb28027a4a557fb0548c9a7ebd8ede7  authority-cpu-v1.json
4694490f347949a7d814f5001655d1c7a9eb41c5ea369f4e838b231ad2ee7662  sfora-native256-fit-export-collected-inputs-v1.json
da925901082ff22994a90dcdb0ccfea0cd5e2144cb3f87c3950904e9928a503e  sfora-native256-full-initialized-cpu-collected-inputs-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
