#!/bin/bash
set -euo pipefail
sha256sum -c <<'SHA_PRE'
66bcc2918230e097aa492a476310c97a5e7b2fd4a90c4fbee582b29867ebf461  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/qualify_connected_gradient_decomposition.py
f9334f72e38b477478de313f9ade5893a76266db60c422e377f9c7044ba237e8  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/test_connected_gradient_decomposition.py
d5d698fbbe16ac78e665947097778c89f88152e6c1c6f845800b4ac93a425258  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/execution.json
b4dc742f99ff95151a412919a84af1a9999d96397d582ab3a9dc0fbf5b094c2d  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
62e69a04bf4df55df5185366231d0237b5991c912f2a0e23e6af31f41c5dd24f  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/observe_connected_control_batch_execution.py
9459187171def85aaa436bbef3032b8bbf97d76222601b883c6e6c044d609c52  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-control-179061-v1.json
8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25  /home/riomus/runs/sfora-connected-mlp-train-source-v6/test_siglip2_connected_mlp.py
79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train_siglip2_connected_mlp.py
b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/evaluate_siglip2_connected_mlp.py
48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/test_connected_mlp_evaluation.py
578b92a6ae1b7831820dde7ab1d50fb0ad0fa5fa4b681e1e310c24fa8cc96ec3  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-candidate-179061-v1.json
cc20ff1b808fabd158d5545e2f270fd0e7d30e12eefc1d2e5ced1b1859a0a8b3  /home/riomus/runs/sfora-connected-mlp-train-candidate-179061-v1/candidate-179061-terminal.pt
bc6c7357e4cb667e86c15e8f10453f1d27a14f79498a760e81f417b0e881a22b  /home/riomus/runs/sfora-connected-mlp-train-candidate-179061-v1/receipt.json
ec0a5e7603bc846a9a783692031ac0ed3e8a6714e487fbd3fce770324cd1b6c0  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train-candidate-179061-v1-original.log
9b4c5291e6e2c6dd97898eb5d7007f23f74ba860d13389f69eab3278fa47c793  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-candidate-179069-v1.json
85c2503e5399451197c589f5b154eba914200f029de8a846812bbb9fe89aaf71  /home/riomus/runs/sfora-connected-mlp-train-candidate-179069-v1/candidate-179069-terminal.pt
955cd7389a8d9f55b600225b8ce6b576b222b927627cd4f2410786e5ea804c20  /home/riomus/runs/sfora-connected-mlp-train-candidate-179069-v1/receipt.json
18d2718455d90368759f27e1016ea1605abe4520729f2889d9875c4ec064447d  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train-candidate-179069-v1-original.log
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
SHA_PRE
set +e
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/qualify_connected_gradient_decomposition.py --execution-sha256 d5d698fbbe16ac78e665947097778c89f88152e6c1c6f845800b4ac93a425258 --authority /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/authority.json --authority-sha256 b4dc742f99ff95151a412919a84af1a9999d96397d582ab3a9dc0fbf5b094c2d --output /home/riomus/runs/sfora-connected-gradient-decomposition-v1
result=$?
set -e
sha256sum -c <<'SHA_POST'
66bcc2918230e097aa492a476310c97a5e7b2fd4a90c4fbee582b29867ebf461  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/qualify_connected_gradient_decomposition.py
f9334f72e38b477478de313f9ade5893a76266db60c422e377f9c7044ba237e8  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/test_connected_gradient_decomposition.py
d5d698fbbe16ac78e665947097778c89f88152e6c1c6f845800b4ac93a425258  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/execution.json
b4dc742f99ff95151a412919a84af1a9999d96397d582ab3a9dc0fbf5b094c2d  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v1/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
62e69a04bf4df55df5185366231d0237b5991c912f2a0e23e6af31f41c5dd24f  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/observe_connected_control_batch_execution.py
9459187171def85aaa436bbef3032b8bbf97d76222601b883c6e6c044d609c52  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-control-179061-v1.json
8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25  /home/riomus/runs/sfora-connected-mlp-train-source-v6/test_siglip2_connected_mlp.py
79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train_siglip2_connected_mlp.py
b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/evaluate_siglip2_connected_mlp.py
48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/test_connected_mlp_evaluation.py
578b92a6ae1b7831820dde7ab1d50fb0ad0fa5fa4b681e1e310c24fa8cc96ec3  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-candidate-179061-v1.json
cc20ff1b808fabd158d5545e2f270fd0e7d30e12eefc1d2e5ced1b1859a0a8b3  /home/riomus/runs/sfora-connected-mlp-train-candidate-179061-v1/candidate-179061-terminal.pt
bc6c7357e4cb667e86c15e8f10453f1d27a14f79498a760e81f417b0e881a22b  /home/riomus/runs/sfora-connected-mlp-train-candidate-179061-v1/receipt.json
ec0a5e7603bc846a9a783692031ac0ed3e8a6714e487fbd3fce770324cd1b6c0  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train-candidate-179061-v1-original.log
9b4c5291e6e2c6dd97898eb5d7007f23f74ba860d13389f69eab3278fa47c793  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-candidate-179069-v1.json
85c2503e5399451197c589f5b154eba914200f029de8a846812bbb9fe89aaf71  /home/riomus/runs/sfora-connected-mlp-train-candidate-179069-v1/candidate-179069-terminal.pt
955cd7389a8d9f55b600225b8ce6b576b222b927627cd4f2410786e5ea804c20  /home/riomus/runs/sfora-connected-mlp-train-candidate-179069-v1/receipt.json
18d2718455d90368759f27e1016ea1605abe4520729f2889d9875c4ec064447d  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train-candidate-179069-v1-original.log
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
SHA_POST
/usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py "$result"
exit "$result"
