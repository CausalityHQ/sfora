#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-cached-readout-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c  train_siglip2_cached_readout.py
9fd8a780c412f9423e9e75192a1e8d91ead520d46f6269e77482965e85a28500  test_siglip2_cached_readout.py
907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598  execution.json
51d9537c8da6946be2bb19165bc5a9bd5096670812c54ab0fc04a6ffc9c4e15e  authority-mechanics-candidate-v2.json
4694490f347949a7d814f5001655d1c7a9eb41c5ea369f4e838b231ad2ee7662  sfora-native256-fit-export-collected-inputs-v1.json
da925901082ff22994a90dcdb0ccfea0cd5e2144cb3f87c3950904e9928a503e  sfora-native256-full-initialized-cpu-collected-inputs-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-cached-readout-source-v2/train_siglip2_cached_readout.py --execution-sha256 907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598 --authority /home/riomus/runs/sfora-so400-cached-readout-source-v2/authority-mechanics-candidate-v2.json --authority-sha256 51d9537c8da6946be2bb19165bc5a9bd5096670812c54ab0fc04a6ffc9c4e15e --phase mechanics --arm candidate --seed 179032 --output /home/riomus/runs/sfora-so400-cached-readout-mechanics-candidate-v2
sha256sum -c <<'HASHES'
a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c  train_siglip2_cached_readout.py
9fd8a780c412f9423e9e75192a1e8d91ead520d46f6269e77482965e85a28500  test_siglip2_cached_readout.py
907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598  execution.json
51d9537c8da6946be2bb19165bc5a9bd5096670812c54ab0fc04a6ffc9c4e15e  authority-mechanics-candidate-v2.json
4694490f347949a7d814f5001655d1c7a9eb41c5ea369f4e838b231ad2ee7662  sfora-native256-fit-export-collected-inputs-v1.json
da925901082ff22994a90dcdb0ccfea0cd5e2144cb3f87c3950904e9928a503e  sfora-native256-full-initialized-cpu-collected-inputs-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
