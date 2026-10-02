#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-so400-quadratic-readout-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
de2787e8b6e34c9280a1267134a7c9eb8184a0023d1f515dcc40969b692716c6  test_siglip2_quadratic_readout.py
ea1417630658151a476b57fcdf227c54cebd7d434f61d460e130719309e5fc28  train_siglip2_quadratic_readout.py
3924c2ea4e029f728bbabab80afb8c45b1eb3995b14dcc1a6b802ae549ffc70d  execution.json
abc85c880260e2617f4ccf20cca450869c0c7e1893207d25462e7a6a75757951  authority-cpu-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-quadratic-readout-source-v1/train_siglip2_quadratic_readout.py --execution-sha256 3924c2ea4e029f728bbabab80afb8c45b1eb3995b14dcc1a6b802ae549ffc70d --authority /home/riomus/runs/sfora-so400-quadratic-readout-source-v1/authority-cpu-v1.json --authority-sha256 abc85c880260e2617f4ccf20cca450869c0c7e1893207d25462e7a6a75757951 --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-so400-quadratic-readout-cpu-v1
sha256sum -c <<'HASHES'
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  quadratic_readout.py
de2787e8b6e34c9280a1267134a7c9eb8184a0023d1f515dcc40969b692716c6  test_siglip2_quadratic_readout.py
ea1417630658151a476b57fcdf227c54cebd7d434f61d460e130719309e5fc28  train_siglip2_quadratic_readout.py
3924c2ea4e029f728bbabab80afb8c45b1eb3995b14dcc1a6b802ae549ffc70d  execution.json
abc85c880260e2617f4ccf20cca450869c0c7e1893207d25462e7a6a75757951  authority-cpu-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
