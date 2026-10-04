#!/bin/bash
set -euo pipefail
sha256sum -c <<'CONTINUE_HASHES'
02f79083cecd1ddf4d9f8c28996df305c517be8af65abbfd47e1e8d866e44392  /home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-first-selection-score-v1/receipt.json
a14e340b59c6cdb704ee86e26c2d607db84746a4dafe25fb0b7f883263941c87  /home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1/first-selection-score-v1.log
CONTINUE_HASHES
sha256sum -c <<'ANCHOR_HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179069-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
ANCHOR_HASHES
sha256sum -c <<'ANCHOR_HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179069-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
ANCHOR_HASHES
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
dbc18bd35ff6d61a0e748f31301239433bf6ab79722b7d5b361c6d533e7947fb  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-train-candidate-179069-v1.json
232e9a821b61b8420ae7328395a33b9878a0fcd834473c348e8a062448bcf7ec  /home/riomus/runs/sfora-so400-fullfeature-residual-cpu-v1/receipt.json
4ff36c8e587eea20ef3db4c2cc51e495f18ae72bf9f52f784ff5a9bc61e458ea  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/cpu-v1.log
c4b8ef60934897b92f6971b60e5f2005267f906e0de920e2ba0b4c072edce5cf  /home/riomus/runs/sfora-so400-fullfeature-residual-mechanics-control-v1/receipt.json
d97b914531db90705baa960ff1a1c6a7b17ac276bb4344691ee44428b592b599  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/mechanics-control-v1.log
2c5d6e7b3b8055ede673655b59dec76e72b5f8c01383907cd4e3ae2fc0b5611f  /home/riomus/runs/sfora-so400-fullfeature-residual-mechanics-candidate-v1/receipt.json
a5e792ddd40188c4ff896552aae7047d423b58439b3be214e703d4e014752ddb  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/mechanics-candidate-v1.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py --execution-sha256 996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15 --authority /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-train-candidate-179069-v1.json --authority-sha256 dbc18bd35ff6d61a0e748f31301239433bf6ab79722b7d5b361c6d533e7947fb --phase train --arm candidate --seed 179069 --output /home/riomus/runs/sfora-so400-fullfeature-residual-train-candidate-179069-v1
sha256sum -c <<'HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
dbc18bd35ff6d61a0e748f31301239433bf6ab79722b7d5b361c6d533e7947fb  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-train-candidate-179069-v1.json
232e9a821b61b8420ae7328395a33b9878a0fcd834473c348e8a062448bcf7ec  /home/riomus/runs/sfora-so400-fullfeature-residual-cpu-v1/receipt.json
4ff36c8e587eea20ef3db4c2cc51e495f18ae72bf9f52f784ff5a9bc61e458ea  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/cpu-v1.log
c4b8ef60934897b92f6971b60e5f2005267f906e0de920e2ba0b4c072edce5cf  /home/riomus/runs/sfora-so400-fullfeature-residual-mechanics-control-v1/receipt.json
d97b914531db90705baa960ff1a1c6a7b17ac276bb4344691ee44428b592b599  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/mechanics-control-v1.log
2c5d6e7b3b8055ede673655b59dec76e72b5f8c01383907cd4e3ae2fc0b5611f  /home/riomus/runs/sfora-so400-fullfeature-residual-mechanics-candidate-v1/receipt.json
a5e792ddd40188c4ff896552aae7047d423b58439b3be214e703d4e014752ddb  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/mechanics-candidate-v1.log
HASHES

sha256sum -c <<'ANCHOR_HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179069-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
ANCHOR_HASHES

sha256sum -c <<'ANCHOR_HASHES'
7c1c78ef3f2c784f217822690e93e52e3ef55b27d6a27d8f9daf9b9b19295f51  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/native-authority.json
5910eaca2a004aacd68bf15975222c0dfa5ddf270aafe5f383608ca1c4c4f027  /home/riomus/runs/sfora-cudnn-wheel-provenance-v1/proof.json
15a6e7d687d9eb1a1598a0961c87a1cf0374b76547b3cda1662b23c33b86f287  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/decision.json
223cdb698747af05ccbd7184a8501651ec7cee1aa50a4be80a0d722833447558  /home/riomus/runs/sfora-cudnn-wheel-provenance-source-v1/original.log
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
3a50af8c634a2166ac010e8fc1df6a0dd1a575b63e1ad9c1d405f82f209ca522  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/authority-cpu-v5.json
0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/execution.json
862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/nearest_ranking_readout.py
a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/test_siglip2_nearest_ranking.py
4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c  /home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5/train_siglip2_nearest_ranking.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/train_siglip2_compact_ranking.py
6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/test_siglip2_compact_ranking.py
996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/execution.json
1c963e44b14740d9eaca073cb3e381e58a31440c24dc405189a4d2c04e12c62d  /home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1/authority-cpu-v1.json
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179069-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
ANCHOR_HASHES

sha256sum -c <<'CONTINUE_HASHES'
02f79083cecd1ddf4d9f8c28996df305c517be8af65abbfd47e1e8d866e44392  /home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-first-selection-score-v1/receipt.json
a14e340b59c6cdb704ee86e26c2d607db84746a4dafe25fb0b7f883263941c87  /home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1/first-selection-score-v1.log
CONTINUE_HASHES
