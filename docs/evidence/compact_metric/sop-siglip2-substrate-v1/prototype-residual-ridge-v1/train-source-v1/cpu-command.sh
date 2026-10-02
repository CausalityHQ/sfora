#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
6929334fa05a915fb6e7bad85af35fc50059f6f1d020d1394d1535bf5a077238  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/fit_siglip2_prototype_residual.py
52e8df6c406f1af94a0f5596659038275f8e14c91944ab2ecca7885d0971f01c  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/test_siglip2_prototype_residual.py
c9931cdc193580c8ccd85a058e2001bdb344f0a9c07f82c560c7690634d72801  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/execution.json
9dd2f30def2d8bea836e3687a2d661cbb4b88a3c253ad77ee6c0eeb2a04fc555  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/authority-cpu-v1.json
854e7d92b5cea2ef78dd49cf5deaded13f0af651a6dfae73368ec78acb939e27  /home/riomus/runs/sfora-so400-prototype-residual-solver-v1/foundation_adapter.py
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/quadratic_readout.py
1410060ac38065f39aa8e3f0a43fe47fb42321331bb00c9d43264381c4a23ba8  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/test_siglip2_quadratic_readout.py
17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/train_siglip2_quadratic_readout.py
84414302a6749842cae20e2608e932066334916218df68f91f10fd3c589ca382  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/execution.json
d703e4c4b6a7be0fd287bb721cc2769dd4b958f196938a8182c1762cadcd4eb2  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/authority-cpu-v7.json
7e8566c24cae41e71b67f42bb68eda0717b8a7351a7ae9e5c145a35ab0bfa78f  /home/riomus/runs/sfora-so400-quadratic-readout-cpu-v7/receipt.json
3bf689314d9d7ad4b111223417ae9ca41785fa75b2cef1cf0642a072d452c373  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/cpu-v7.log
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-prototype-residual-source-v1/fit_siglip2_prototype_residual.py --execution-sha256 c9931cdc193580c8ccd85a058e2001bdb344f0a9c07f82c560c7690634d72801 --authority /home/riomus/runs/sfora-so400-prototype-residual-source-v1/authority-cpu-v1.json --authority-sha256 9dd2f30def2d8bea836e3687a2d661cbb4b88a3c253ad77ee6c0eeb2a04fc555 --phase cpu --arm linear --output /home/riomus/runs/sfora-so400-prototype-residual-cpu-v1
sha256sum -c <<'HASHES'
6929334fa05a915fb6e7bad85af35fc50059f6f1d020d1394d1535bf5a077238  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/fit_siglip2_prototype_residual.py
52e8df6c406f1af94a0f5596659038275f8e14c91944ab2ecca7885d0971f01c  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/test_siglip2_prototype_residual.py
c9931cdc193580c8ccd85a058e2001bdb344f0a9c07f82c560c7690634d72801  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/execution.json
9dd2f30def2d8bea836e3687a2d661cbb4b88a3c253ad77ee6c0eeb2a04fc555  /home/riomus/runs/sfora-so400-prototype-residual-source-v1/authority-cpu-v1.json
854e7d92b5cea2ef78dd49cf5deaded13f0af651a6dfae73368ec78acb939e27  /home/riomus/runs/sfora-so400-prototype-residual-solver-v1/foundation_adapter.py
1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/quadratic_encoder_frames.py
12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/quadratic_readout.py
1410060ac38065f39aa8e3f0a43fe47fb42321331bb00c9d43264381c4a23ba8  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/test_siglip2_quadratic_readout.py
17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/train_siglip2_quadratic_readout.py
84414302a6749842cae20e2608e932066334916218df68f91f10fd3c589ca382  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/execution.json
d703e4c4b6a7be0fd287bb721cc2769dd4b958f196938a8182c1762cadcd4eb2  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/authority-cpu-v7.json
7e8566c24cae41e71b67f42bb68eda0717b8a7351a7ae9e5c145a35ab0bfa78f  /home/riomus/runs/sfora-so400-quadratic-readout-cpu-v7/receipt.json
3bf689314d9d7ad4b111223417ae9ca41785fa75b2cef1cf0642a072d452c373  /home/riomus/runs/sfora-so400-quadratic-readout-source-v7/cpu-v7.log
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
HASHES
