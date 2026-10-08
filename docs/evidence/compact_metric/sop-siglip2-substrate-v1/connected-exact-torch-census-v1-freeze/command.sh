#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
exec 8<>/home/riomus/runs/.sfora-siglip2-gpu.lock
/usr/bin/flock -n 8
exec 9<>/home/riomus/.sfora-siglip2-gpu.lock
/usr/bin/flock -n 9
sha256sum -c <<'HASHES'
0fdc170019de1137ac9d82b55868ec8c8f8b21867d2a71c8c31a29f9a5b333d2  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/census_connected_core_errors.py
ccbae2316db1637803982ebf7de14214b4f9239a740c5235b9c259d05cc62db8  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/census_connected_core_errors_torch.py
8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/compare_inshop_sop_warmstart_100.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/evaluate_siglip2_prototype_residual.py
163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/export_siglip2_substrate_fit.py
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/extract_siglip2_vision_source.py
6a4d310d4eb883b7bf3d2ea74a1e96223cee096d0bdb9bc6c2a651eecf9ff163  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/qualify_connected_serving_requests.py
eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/qualify_siglip2_substrate_cpu.py
5dff16f60da6529777f965ba70ea10d49d6f09a66d7785892d67a1c6fb89080d  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/test_connected_core_errors_torch.py
17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/train_siglip2_quadratic_readout.py
a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/train_siglip2_substrate_adaptation.py
dc273876d6e62528d1b852b85d4a1ee68e85d9666a54617a729e410e4bb263d6  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/inputs.json
01ae023cb89b828c029817582cdb48204ff8177e76047ccec0773e5d90aafbfa  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/accepted-selection-receipt.json
e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/source-cpu-proof.json
731abe64e6c0d0a1ba75a1cf7cb3abcfcc4d36770719c14539c821651b6f6328  /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/source-cpu-original.log
8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b  /home/riomus/runs/sfora-native256-source-cpu-v4/sources.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/census_connected_core_errors_torch.py --authority /home/riomus/runs/sfora-connected-exact-torch-census-source-v2/authority.json --authority-sha256 "$1" --output /home/riomus/runs/sfora-connected-exact-torch-census-v1.json
