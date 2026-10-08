#!/bin/bash
set -euo pipefail
sha256sum -c <<'SHA_PRE'
2bc273f3db332c0bf2b3b3e3b45722e581035c6509aef9d7c2dc5456494de63c  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/qualify_connected_gradient_decomposition.py
3c9c67a7ba3d52b1a73f4aeb3aaebf8e53793c026ed4c2725cb44448cf666f9e  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/test_connected_gradient_decomposition.py
7bbdd3a7910a155859dca1c1014966fb2089b380ae1ce909a432e17a479ab307  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/execution.json
d807851d43a5413545854fecd576ce5484ad6575b7b80e4fba010310677f6f4a  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
6dc7c0e78447837a517fbec83d6bcc0228cd3824a6b09e441da4d9a1f3508fbf  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/observe_connected_control_batch_execution.py
9459187171def85aaa436bbef3032b8bbf97d76222601b883c6e6c044d609c52  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-control-179061-v1.json
a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c  /home/riomus/runs/sfora-connected-mlp-train-source-v6/execution.json
8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25  /home/riomus/runs/sfora-connected-mlp-train-source-v6/test_siglip2_connected_mlp.py
79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train_siglip2_connected_mlp.py
3b9ade347801291b2cd4f4a3849be9eb2811b4237ddd562303d46cc3af505029  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/execution.json
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
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/qualify_connected_gradient_decomposition.py --execution-sha256 7bbdd3a7910a155859dca1c1014966fb2089b380ae1ce909a432e17a479ab307 --authority /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/authority.json --authority-sha256 d807851d43a5413545854fecd576ce5484ad6575b7b80e4fba010310677f6f4a --output /home/riomus/runs/sfora-connected-gradient-decomposition-v3
result=$?
set -e
sha256sum -c <<'SHA_POST'
2bc273f3db332c0bf2b3b3e3b45722e581035c6509aef9d7c2dc5456494de63c  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/qualify_connected_gradient_decomposition.py
3c9c67a7ba3d52b1a73f4aeb3aaebf8e53793c026ed4c2725cb44448cf666f9e  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/test_connected_gradient_decomposition.py
7bbdd3a7910a155859dca1c1014966fb2089b380ae1ce909a432e17a479ab307  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/execution.json
d807851d43a5413545854fecd576ce5484ad6575b7b80e4fba010310677f6f4a  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/authority.json
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
6dc7c0e78447837a517fbec83d6bcc0228cd3824a6b09e441da4d9a1f3508fbf  /home/riomus/runs/sfora-connected-gradient-decomposition-source-v3/observe_connected_control_batch_execution.py
9459187171def85aaa436bbef3032b8bbf97d76222601b883c6e6c044d609c52  /home/riomus/runs/sfora-connected-mlp-train-source-v6/authority-train-control-179061-v1.json
a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c  /home/riomus/runs/sfora-connected-mlp-train-source-v6/execution.json
8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25  /home/riomus/runs/sfora-connected-mlp-train-source-v6/test_siglip2_connected_mlp.py
79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b  /home/riomus/runs/sfora-connected-mlp-train-source-v6/train_siglip2_connected_mlp.py
3b9ade347801291b2cd4f4a3849be9eb2811b4237ddd562303d46cc3af505029  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/execution.json
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
SFORA_COMMAND_STATUS="$result" /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
exit "$result"
