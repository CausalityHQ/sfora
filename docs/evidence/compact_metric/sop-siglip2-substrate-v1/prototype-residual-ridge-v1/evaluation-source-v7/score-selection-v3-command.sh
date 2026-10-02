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
edb510d2a2f9883c8c99f835d3a54349fc0807baf4765f75f66b7be823b92a03  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/authority-selection-v1.json
9f5cd1e70619a0946c75f31dec6cbba4ce7fb33498e9a505ea0a1ff3630f82e9  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/evaluate_siglip2_prototype_residual.py
265f1c8e9827f829fb45fe0bcadd3a41624e57d13994dd3fdfcd1c8bed9673f2  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/execution.json
c74cb1851b5d0eb545a8cb7b85e42ae37be43fb5543e4018c79ce56846013339  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/root-source-verification.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/source-selection-inventory.json
0b9e5bea3502798a40bb5ae9f37c462579b098c0bc6f4015c46c44c291d17abf  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/test_siglip2_prototype_residual_evaluation.py
7bfaabeb855abecbfa87664c4dfc9381c1213196ffc5a40fc1bf60b2caacc1f8  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/evaluate_siglip2_quadratic_readout.py
5bedd27b10020846ef2374835808d6654a559852ce79629d8abeff2029856e9e  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/execution.json
1ea5312edddfc162bedf94859839e3b02d50433b67a7a250fa054e442b9155d9  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/test_siglip2_quadratic_readout_evaluation.py
1e8db2ef028612e92f2e44997ce34ecdd46e68e6fb8b3998f02c67a90ef465df  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/cpu-v7-terminal.json
eab813fda9597acc612f97bc21015700794727649ed07baebe4e78295950a7e3  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-cpu-v7/receipt.json
2162efc8dcacce22d020d118457fa76aff57b5a25e0b00fecb34f32278a5ca67  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/cpu-v7.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/evaluate_siglip2_prototype_residual.py --execution-sha256 265f1c8e9827f829fb45fe0bcadd3a41624e57d13994dd3fdfcd1c8bed9673f2 --authority /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/authority-selection-v1.json --authority-sha256 edb510d2a2f9883c8c99f835d3a54349fc0807baf4765f75f66b7be823b92a03 --phase score --output /home/riomus/runs/sfora-so400-prototype-residual-evaluation-selection-score-v3 --prerequisite /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/cpu-v7-terminal.json --prerequisite-sha256 1e8db2ef028612e92f2e44997ce34ecdd46e68e6fb8b3998f02c67a90ef465df
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
edb510d2a2f9883c8c99f835d3a54349fc0807baf4765f75f66b7be823b92a03  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/authority-selection-v1.json
9f5cd1e70619a0946c75f31dec6cbba4ce7fb33498e9a505ea0a1ff3630f82e9  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/evaluate_siglip2_prototype_residual.py
265f1c8e9827f829fb45fe0bcadd3a41624e57d13994dd3fdfcd1c8bed9673f2  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/execution.json
c74cb1851b5d0eb545a8cb7b85e42ae37be43fb5543e4018c79ce56846013339  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/root-source-verification.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/source-selection-inventory.json
0b9e5bea3502798a40bb5ae9f37c462579b098c0bc6f4015c46c44c291d17abf  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/test_siglip2_prototype_residual_evaluation.py
7bfaabeb855abecbfa87664c4dfc9381c1213196ffc5a40fc1bf60b2caacc1f8  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/evaluate_siglip2_quadratic_readout.py
5bedd27b10020846ef2374835808d6654a559852ce79629d8abeff2029856e9e  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/execution.json
1ea5312edddfc162bedf94859839e3b02d50433b67a7a250fa054e442b9155d9  /home/riomus/runs/sfora-so400-prototype-original-evaluator-source-v1/test_siglip2_quadratic_readout_evaluation.py
1e8db2ef028612e92f2e44997ce34ecdd46e68e6fb8b3998f02c67a90ef465df  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/cpu-v7-terminal.json
eab813fda9597acc612f97bc21015700794727649ed07baebe4e78295950a7e3  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-cpu-v7/receipt.json
2162efc8dcacce22d020d118457fa76aff57b5a25e0b00fecb34f32278a5ca67  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/cpu-v7.log
HASHES
