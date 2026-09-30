#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-adaptation-source-v1
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
f17008c37532d037f9963aebe77ef7038177a1377f3197c9e4a76af25a124d37  train_siglip2_substrate_adaptation.py
f14efe93e80f67a362d01e825c9521db8027bbf340736c7e2637a0f26692908e  test_siglip2_substrate_adaptation.py
435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870  deployed_code_rank.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  reference_train_sop_siglip2_compact.py
a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272  reference_unicom_training.py
2b418eeb1ea3c0e0beb8f1d6b53259de304a45c2c8e713a716122dbe9c5cd358  execution.json
e954ce377e353f810fe8e5b151a254969dc14031ef88b930b594e49903107f67  authority-so400-mechanics-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-adaptation-source-v1/train_siglip2_substrate_adaptation.py --execution-sha256 2b418eeb1ea3c0e0beb8f1d6b53259de304a45c2c8e713a716122dbe9c5cd358 --authority /home/riomus/runs/sfora-native256-adaptation-source-v1/authority-so400-mechanics-v1.json --authority-sha256 e954ce377e353f810fe8e5b151a254969dc14031ef88b930b594e49903107f67 --phase mechanics --arm so400 --seed 179032 --output /home/riomus/runs/sfora-native256-mechanics-so400-v1
sha256sum -c <<'HASHES'
f17008c37532d037f9963aebe77ef7038177a1377f3197c9e4a76af25a124d37  train_siglip2_substrate_adaptation.py
f14efe93e80f67a362d01e825c9521db8027bbf340736c7e2637a0f26692908e  test_siglip2_substrate_adaptation.py
435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870  deployed_code_rank.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  reference_train_sop_siglip2_compact.py
a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272  reference_unicom_training.py
2b418eeb1ea3c0e0beb8f1d6b53259de304a45c2c8e713a716122dbe9c5cd358  execution.json
e954ce377e353f810fe8e5b151a254969dc14031ef88b930b594e49903107f67  authority-so400-mechanics-v1.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
