#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
cc3ffb6d63a78448e3bff0535304b50b30eb451b0617c15c4b8cf1727bed7d2d  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit_siglip2_prototype_residual.py
4f5b5ba1947e57c9adb93da512a5a7aed87528104f0c9acd873be9398d5e3c27  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/test_siglip2_prototype_residual.py
9c726ec7e9c8e339a82d9284fd41495bf9feb2b39ba76df10eda75a05684fef2  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/execution.json
64b2492079cdd9adb0f3bf8660f5a775949050d32cbbe0a101682e6cfbf6f06b  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/authority-cpu-v3.json
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
dab6656c23bc4efcb54092773163a20ecda5b4ef4ad96388344107b088c56f5c  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/authority-selection-v1.json
da90ff2c71d374f6d1912f0500fe02797e12b0ade5a16324814b899d4b7eaf3c  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/evaluate_siglip2_prototype_residual.py
5dacb69b400565a3b3d01953bcca8b3fe40339b4eaa04ec05ed435925a1da89f  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/execution.json
5d8ff48c3691b0f49d077b67bc5d09f2998705430c33751d9090c102db1b0126  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/root-source-verification.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/source-selection-inventory.json
fed62cb3b179d47150e7bdedb2f6d2b8d9134c13cf85271a06a9fa1daff6bc51  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/test_siglip2_prototype_residual_evaluation.py
7bfaabeb855abecbfa87664c4dfc9381c1213196ffc5a40fc1bf60b2caacc1f8  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/evaluate_siglip2_quadratic_readout.py
5bedd27b10020846ef2374835808d6654a559852ce79629d8abeff2029856e9e  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/execution.json
1ea5312edddfc162bedf94859839e3b02d50433b67a7a250fa054e442b9155d9  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/test_siglip2_quadratic_readout_evaluation.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/evaluate_siglip2_prototype_residual.py --execution-sha256 5dacb69b400565a3b3d01953bcca8b3fe40339b4eaa04ec05ed435925a1da89f --authority /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/authority-selection-v1.json --authority-sha256 dab6656c23bc4efcb54092773163a20ecda5b4ef4ad96388344107b088c56f5c --phase cpu --output /home/riomus/runs/sfora-so400-prototype-residual-evaluation-cpu-v5
sha256sum -c <<'HASHES'
cc3ffb6d63a78448e3bff0535304b50b30eb451b0617c15c4b8cf1727bed7d2d  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/fit_siglip2_prototype_residual.py
4f5b5ba1947e57c9adb93da512a5a7aed87528104f0c9acd873be9398d5e3c27  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/test_siglip2_prototype_residual.py
9c726ec7e9c8e339a82d9284fd41495bf9feb2b39ba76df10eda75a05684fef2  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/execution.json
64b2492079cdd9adb0f3bf8660f5a775949050d32cbbe0a101682e6cfbf6f06b  /home/riomus/runs/sfora-so400-prototype-residual-source-v3/authority-cpu-v3.json
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
dab6656c23bc4efcb54092773163a20ecda5b4ef4ad96388344107b088c56f5c  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/authority-selection-v1.json
da90ff2c71d374f6d1912f0500fe02797e12b0ade5a16324814b899d4b7eaf3c  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/evaluate_siglip2_prototype_residual.py
5dacb69b400565a3b3d01953bcca8b3fe40339b4eaa04ec05ed435925a1da89f  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/execution.json
5d8ff48c3691b0f49d077b67bc5d09f2998705430c33751d9090c102db1b0126  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/root-source-verification.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/source-selection-inventory.json
fed62cb3b179d47150e7bdedb2f6d2b8d9134c13cf85271a06a9fa1daff6bc51  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v5/test_siglip2_prototype_residual_evaluation.py
7bfaabeb855abecbfa87664c4dfc9381c1213196ffc5a40fc1bf60b2caacc1f8  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/evaluate_siglip2_quadratic_readout.py
5bedd27b10020846ef2374835808d6654a559852ce79629d8abeff2029856e9e  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/execution.json
1ea5312edddfc162bedf94859839e3b02d50433b67a7a250fa054e442b9155d9  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/test_siglip2_quadratic_readout_evaluation.py
HASHES
