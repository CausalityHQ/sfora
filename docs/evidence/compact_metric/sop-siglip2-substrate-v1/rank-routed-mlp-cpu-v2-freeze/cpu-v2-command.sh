#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
f769e49b86a0e9ebe9d3475b54361a6de8a53459a08f416422c2915480d1549b  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/authority-cpu-v2.json
a7d7f45f3366ee45f8027f372eaa8dde9e752af62626671007a7192f3ef2c389  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/execution.json
8f61921da94679231da41734aa73375b9dad311b74adc04fa9c86a2376a9e15a  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/test_siglip2_rank_routed_mlp.py
2fb4f237345f346080e499cc99d97fbc40c7087f255027ed810ef02b56537b1f  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/train_siglip2_rank_routed_mlp.py
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
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/train_siglip2_rank_routed_mlp.py --execution-sha256 a7d7f45f3366ee45f8027f372eaa8dde9e752af62626671007a7192f3ef2c389 --authority /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/authority-cpu-v2.json --authority-sha256 f769e49b86a0e9ebe9d3475b54361a6de8a53459a08f416422c2915480d1549b --phase cpu --arm control --seed 179061 --output /home/riomus/runs/sfora-rank-routed-mlp-cpu-v2
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
f769e49b86a0e9ebe9d3475b54361a6de8a53459a08f416422c2915480d1549b  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/authority-cpu-v2.json
a7d7f45f3366ee45f8027f372eaa8dde9e752af62626671007a7192f3ef2c389  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/execution.json
8f61921da94679231da41734aa73375b9dad311b74adc04fa9c86a2376a9e15a  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/test_siglip2_rank_routed_mlp.py
2fb4f237345f346080e499cc99d97fbc40c7087f255027ed810ef02b56537b1f  /home/riomus/runs/sfora-rank-routed-mlp-train-source-v2/train_siglip2_rank_routed_mlp.py
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
