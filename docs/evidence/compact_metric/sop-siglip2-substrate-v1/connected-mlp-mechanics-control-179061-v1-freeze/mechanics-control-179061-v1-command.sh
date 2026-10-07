#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
92f7c6fb22d79a3399c018de023ad89761dfc8bbbd982511c81c9dbd93519d5e  /home/riomus/runs/sfora-connected-mlp-train-source-v3/authority-mechanics-control-179061-v1.json
5947257ef8e2656fe9b30e94b13c873a55f571f3ed5990f2988d29a085603a3f  /home/riomus/runs/sfora-connected-mlp-train-source-v3/execution.json
35daffe103bc52fe3fbbc4af33a0348494b9477ce49052db7f5d0cf5be083cc1  /home/riomus/runs/sfora-connected-mlp-train-source-v3/test_siglip2_connected_mlp.py
55935d5a7e7299d1a11f14617cd5ef07f4232afd32148abf942c74edbc636e99  /home/riomus/runs/sfora-connected-mlp-train-source-v3/train_siglip2_connected_mlp.py
393384096ffa869e1be20d2f91fdf08dfadb9f3e531dfe724085d8501d3f85d9  /usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1
fe5966a43e068ad7cb389c3affa069f4ee6f296e07d7ccc0398a23cfde4f0b7e  /usr/lib/aarch64-linux-gnu/libc.so.6
f6fb476bbefda386491dc5151a593a98bedeff6a4eb0ec4ef766e3b2fe3b9032  /usr/lib/aarch64-linux-gnu/libcuda.so.580.159.03
97387fb55ccfc0cc109dfe99b81de2b7bbaac723d1c215107dc69a068c4b1fae  /usr/lib/aarch64-linux-gnu/libdl.so.2
f25617883b0c10df0f143174c49454ca61c571a1051b5480070341223f95fbdb  /usr/lib/aarch64-linux-gnu/libfribidi.so.0.4.0
f2d3ad2bf0b61f6bc944cc37d7b6ab7f88d2582b41ff989ca164803f56cc5f20  /usr/lib/aarch64-linux-gnu/libgcc_s.so.1
d5b262e559d38769e4959f036a1cc2c4009fa60b4ec836a7a19ca8554aebf5a5  /usr/lib/aarch64-linux-gnu/libm.so.6
66763ad0d2cf14c12eca195de0f8c1065390d0b0bbf66fb495f65762893cbbd3  /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.580.159.03
97b288827c0ab29659dfcdcfab6c2e9e1225c19fc5abfe106d8a53c2c7b34402  /usr/lib/aarch64-linux-gnu/libpthread.so.0
4b38331a53bc2bd10c1ecf4f02a71c913c40bad13f5fbf93bee5a0e1f25dd493  /usr/lib/aarch64-linux-gnu/librt.so.1
6e3112d35cfc86db7ee85b27e1746f67408e2837ff8628b46386c3eabd5682a4  /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33
fe15c414e7e585bf749be683eba39d2f8db556b678ed8253f6884be660caac31  /usr/lib/aarch64-linux-gnu/libutil.so.1
170380b4e7ab28ec86eb090b48df90f84089392cb72fecd5067e5b7a4dc5239f  /usr/lib/aarch64-linux-gnu/libz.so.1.3
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-mlp-train-source-v3/train_siglip2_connected_mlp.py --execution-sha256 5947257ef8e2656fe9b30e94b13c873a55f571f3ed5990f2988d29a085603a3f --authority /home/riomus/runs/sfora-connected-mlp-train-source-v3/authority-mechanics-control-179061-v1.json --authority-sha256 92f7c6fb22d79a3399c018de023ad89761dfc8bbbd982511c81c9dbd93519d5e --phase mechanics --arm control --seed 179061 --output /home/riomus/runs/sfora-connected-mlp-mechanics-control-179061-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
92f7c6fb22d79a3399c018de023ad89761dfc8bbbd982511c81c9dbd93519d5e  /home/riomus/runs/sfora-connected-mlp-train-source-v3/authority-mechanics-control-179061-v1.json
5947257ef8e2656fe9b30e94b13c873a55f571f3ed5990f2988d29a085603a3f  /home/riomus/runs/sfora-connected-mlp-train-source-v3/execution.json
35daffe103bc52fe3fbbc4af33a0348494b9477ce49052db7f5d0cf5be083cc1  /home/riomus/runs/sfora-connected-mlp-train-source-v3/test_siglip2_connected_mlp.py
55935d5a7e7299d1a11f14617cd5ef07f4232afd32148abf942c74edbc636e99  /home/riomus/runs/sfora-connected-mlp-train-source-v3/train_siglip2_connected_mlp.py
393384096ffa869e1be20d2f91fdf08dfadb9f3e531dfe724085d8501d3f85d9  /usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1
fe5966a43e068ad7cb389c3affa069f4ee6f296e07d7ccc0398a23cfde4f0b7e  /usr/lib/aarch64-linux-gnu/libc.so.6
f6fb476bbefda386491dc5151a593a98bedeff6a4eb0ec4ef766e3b2fe3b9032  /usr/lib/aarch64-linux-gnu/libcuda.so.580.159.03
97387fb55ccfc0cc109dfe99b81de2b7bbaac723d1c215107dc69a068c4b1fae  /usr/lib/aarch64-linux-gnu/libdl.so.2
f25617883b0c10df0f143174c49454ca61c571a1051b5480070341223f95fbdb  /usr/lib/aarch64-linux-gnu/libfribidi.so.0.4.0
f2d3ad2bf0b61f6bc944cc37d7b6ab7f88d2582b41ff989ca164803f56cc5f20  /usr/lib/aarch64-linux-gnu/libgcc_s.so.1
d5b262e559d38769e4959f036a1cc2c4009fa60b4ec836a7a19ca8554aebf5a5  /usr/lib/aarch64-linux-gnu/libm.so.6
66763ad0d2cf14c12eca195de0f8c1065390d0b0bbf66fb495f65762893cbbd3  /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.580.159.03
97b288827c0ab29659dfcdcfab6c2e9e1225c19fc5abfe106d8a53c2c7b34402  /usr/lib/aarch64-linux-gnu/libpthread.so.0
4b38331a53bc2bd10c1ecf4f02a71c913c40bad13f5fbf93bee5a0e1f25dd493  /usr/lib/aarch64-linux-gnu/librt.so.1
6e3112d35cfc86db7ee85b27e1746f67408e2837ff8628b46386c3eabd5682a4  /usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33
fe15c414e7e585bf749be683eba39d2f8db556b678ed8253f6884be660caac31  /usr/lib/aarch64-linux-gnu/libutil.so.1
170380b4e7ab28ec86eb090b48df90f84089392cb72fecd5067e5b7a4dc5239f  /usr/lib/aarch64-linux-gnu/libz.so.1.3
HASHES
