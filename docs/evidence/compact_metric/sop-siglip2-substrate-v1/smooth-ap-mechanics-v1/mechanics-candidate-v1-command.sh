#!/bin/bash
set -euo pipefail
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
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
eea9468a0067470c2fb8706bc8d86c423e107aeb194a78ea4c1df9830f9e15df  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-cpu-v1.json
701ec74d14fd7fe62f968a0124e0ee4f85ef0c017128fda53de849979ac616ca  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-mechanics-candidate-v1.json
04aea98937cad01660088a8578a71035c654727ad2888fc22f5830deef133ee2  /home/riomus/runs/sfora-so400-smooth-ap-cpu-v1/receipt.json
6fd56ee538d7ebed9ec0e0e9cdcfea96cf4e62348b9ae174cd998c59760a1f5d  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/cpu-v1.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py --execution-sha256 bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272 --authority /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-mechanics-candidate-v1.json --authority-sha256 701ec74d14fd7fe62f968a0124e0ee4f85ef0c017128fda53de849979ac616ca --phase mechanics --arm candidate --seed 179061 --output /home/riomus/runs/sfora-so400-smooth-ap-mechanics-candidate-v1
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
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
eea9468a0067470c2fb8706bc8d86c423e107aeb194a78ea4c1df9830f9e15df  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-cpu-v1.json
701ec74d14fd7fe62f968a0124e0ee4f85ef0c017128fda53de849979ac616ca  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-mechanics-candidate-v1.json
04aea98937cad01660088a8578a71035c654727ad2888fc22f5830deef133ee2  /home/riomus/runs/sfora-so400-smooth-ap-cpu-v1/receipt.json
6fd56ee538d7ebed9ec0e0e9cdcfea96cf4e62348b9ae174cd998c59760a1f5d  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/cpu-v1.log
HASHES
