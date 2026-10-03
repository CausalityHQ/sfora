#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
83e47dc0534d83e7e2848f453077109ec9f713a2efcba01a7f6e2e47f6a91ece  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/prototype_residual_readout.py
78b32ba8633bde2e853c4633680fabfeb9138d745347a8ae648c376dc88d5b2b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/test_siglip2_prototype_residual.py
ced15941c69f674e79740449cc12a3b0a1716221d6fb7cc4bc794a5726430fd8  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/execution.json
d0dee86d2da0767ff87ad366fcf7e859a4ca5556761ee5fef78a3386f6c8df5a  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/authority-cpu-v1.json
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
cc3ffb6d63a78448e3bff0535304b50b30eb451b0617c15c4b8cf1727bed7d2d  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit_siglip2_prototype_residual.py
4f5b5ba1947e57c9adb93da512a5a7aed87528104f0c9acd873be9398d5e3c27  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/test_siglip2_prototype_residual.py
9c726ec7e9c8e339a82d9284fd41495bf9feb2b39ba76df10eda75a05684fef2  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/execution.json
3405b72d92ab289ac36373ab192fd74872a0083f1e2a0d91247a8c9e159cf597  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/authority-fit-linear-v1.json
2a861afc762d92018b93eaf4f3b5a566a7be095d71b6da3fabfba6d8c4cada09  /home/riomus/runs/sfora-so400-prototype-residual-fit-linear-v1/receipt.json
0a02d3eb758e7de6a930bc2908937890ff1b26c388cf8d26ed3d72895788dc44  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit-linear-v1.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/fit_siglip2_prototype_residual.py --execution-sha256 ced15941c69f674e79740449cc12a3b0a1716221d6fb7cc4bc794a5726430fd8 --authority /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/authority-cpu-v1.json --authority-sha256 d0dee86d2da0767ff87ad366fcf7e859a4ca5556761ee5fef78a3386f6c8df5a --phase cpu --arm linear --output /home/riomus/runs/sfora-so400-signed-concat-cpu-v1
sha256sum -c <<'HASHES'
83e47dc0534d83e7e2848f453077109ec9f713a2efcba01a7f6e2e47f6a91ece  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/prototype_residual_readout.py
78b32ba8633bde2e853c4633680fabfeb9138d745347a8ae648c376dc88d5b2b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/test_siglip2_prototype_residual.py
ced15941c69f674e79740449cc12a3b0a1716221d6fb7cc4bc794a5726430fd8  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/execution.json
d0dee86d2da0767ff87ad366fcf7e859a4ca5556761ee5fef78a3386f6c8df5a  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v1/authority-cpu-v1.json
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
cc3ffb6d63a78448e3bff0535304b50b30eb451b0617c15c4b8cf1727bed7d2d  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit_siglip2_prototype_residual.py
4f5b5ba1947e57c9adb93da512a5a7aed87528104f0c9acd873be9398d5e3c27  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/test_siglip2_prototype_residual.py
9c726ec7e9c8e339a82d9284fd41495bf9feb2b39ba76df10eda75a05684fef2  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/execution.json
3405b72d92ab289ac36373ab192fd74872a0083f1e2a0d91247a8c9e159cf597  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/authority-fit-linear-v1.json
2a861afc762d92018b93eaf4f3b5a566a7be095d71b6da3fabfba6d8c4cada09  /home/riomus/runs/sfora-so400-prototype-residual-fit-linear-v1/receipt.json
0a02d3eb758e7de6a930bc2908937890ff1b26c388cf8d26ed3d72895788dc44  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit-linear-v1.log
HASHES
