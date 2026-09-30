#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-adaptation-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870  deployed_code_rank.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  reference_train_sop_siglip2_compact.py
a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272  reference_unicom_training.py
2e92f2ccc46ead2fb95af5d3a91ab8d528ce8d08904666f7d615d950b4bb2ba6  test_siglip2_substrate_adaptation.py
ef9f10efde1873f6df43dfd67d1fdbaed13d5b8160ae9d243219528f0f5dd97e  train_siglip2_substrate_adaptation.py
a2ecfcb73cc339a4d1ecb25c099367bab3d87a1a696e52408da73a506e83c1a5  execution.json
2ccf7c79363df808b48710256af582d4b3cc401826cc3036bbfece9909ac502e  authority-large-mechanics-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-adaptation-source-v2/train_siglip2_substrate_adaptation.py --execution-sha256 a2ecfcb73cc339a4d1ecb25c099367bab3d87a1a696e52408da73a506e83c1a5 --authority /home/riomus/runs/sfora-native256-adaptation-source-v2/authority-large-mechanics-v2.json --authority-sha256 2ccf7c79363df808b48710256af582d4b3cc401826cc3036bbfece9909ac502e --phase mechanics --arm large --seed 179032 --output /home/riomus/runs/sfora-native256-mechanics-large-v2
sha256sum -c <<'HASHES'
435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870  deployed_code_rank.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  reference_train_sop_siglip2_compact.py
a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272  reference_unicom_training.py
2e92f2ccc46ead2fb95af5d3a91ab8d528ce8d08904666f7d615d950b4bb2ba6  test_siglip2_substrate_adaptation.py
ef9f10efde1873f6df43dfd67d1fdbaed13d5b8160ae9d243219528f0f5dd97e  train_siglip2_substrate_adaptation.py
a2ecfcb73cc339a4d1ecb25c099367bab3d87a1a696e52408da73a506e83c1a5  execution.json
2ccf7c79363df808b48710256af582d4b3cc401826cc3036bbfece9909ac502e  authority-large-mechanics-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
HASHES
