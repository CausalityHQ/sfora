#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'OBSERVER_HASHES'
3833eb15c06525bd156bda3e7f9036f1e67bacd2d318dcee5f872998d24ea20a  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/observe_connected_mlp_phases.py
81b18c9ac34937d01a58e0a6a6777a8ac6db4b8ea080ac854a572d7537657167  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/test_connected_mlp_phase_observer.py
7852e23715354d590acabb4f9abf06153bef51a6de8b65c2f52a0ca240b16716  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/manifest.json
OBSERVER_HASHES
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
56f37121ce44dee2f305961d540ce592b32d0e938c229b65f62d8c627f47c779  /home/riomus/runs/sfora-connected-mlp-train-source-v2/authority-cpu-v2.json
cb18b71dad340f49964679bf47b14449e546ab654e67d4d823256e4f212234b0  /home/riomus/runs/sfora-connected-mlp-train-source-v2/execution.json
e0d407400265234c5585b350d619ce272e3008c45dc22d487e6dab95f60953f0  /home/riomus/runs/sfora-connected-mlp-train-source-v2/test_siglip2_connected_mlp.py
cd68f9b109ca48dfc488a248b66c6e2f4a1e58e884aa3c598f4325d4c912edb2  /home/riomus/runs/sfora-connected-mlp-train-source-v2/train_siglip2_connected_mlp.py
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
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/observe_connected_mlp_phases.py --manifest /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/manifest.json --manifest-sha256 7852e23715354d590acabb4f9abf06153bef51a6de8b65c2f52a0ca240b16716
sha256sum -c <<'OBSERVER_HASHES'
3833eb15c06525bd156bda3e7f9036f1e67bacd2d318dcee5f872998d24ea20a  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/observe_connected_mlp_phases.py
81b18c9ac34937d01a58e0a6a6777a8ac6db4b8ea080ac854a572d7537657167  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/test_connected_mlp_phase_observer.py
7852e23715354d590acabb4f9abf06153bef51a6de8b65c2f52a0ca240b16716  /home/riomus/runs/sfora-connected-mlp-phase-observer-source-v1/manifest.json
OBSERVER_HASHES
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
56f37121ce44dee2f305961d540ce592b32d0e938c229b65f62d8c627f47c779  /home/riomus/runs/sfora-connected-mlp-train-source-v2/authority-cpu-v2.json
cb18b71dad340f49964679bf47b14449e546ab654e67d4d823256e4f212234b0  /home/riomus/runs/sfora-connected-mlp-train-source-v2/execution.json
e0d407400265234c5585b350d619ce272e3008c45dc22d487e6dab95f60953f0  /home/riomus/runs/sfora-connected-mlp-train-source-v2/test_siglip2_connected_mlp.py
cd68f9b109ca48dfc488a248b66c6e2f4a1e58e884aa3c598f4325d4c912edb2  /home/riomus/runs/sfora-connected-mlp-train-source-v2/train_siglip2_connected_mlp.py
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
