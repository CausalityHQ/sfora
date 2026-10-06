#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
257c02985581be6b946e99649a7d6470bd21696fffa717758d3977cbdf214ccb  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/driver.py
840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/train_siglip2_identity_diversity.py
fa7c15251aec590ec613f5a42e14118d3a99b558f7c214e129fbb1f6a13972de  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/authority.json
032a5774afd3c3c2bf1817155dd60d4ed6623f7365c94c831528cf43cd8686ef  /home/riomus/runs/sfora-so400-identity-diversity-evaluation-first-export-control-179061-v1/receipt.json
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
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/driver.py /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/authority.json fa7c15251aec590ec613f5a42e14118d3a99b558f7c214e129fbb1f6a13972de
sha256sum -c <<'HASHES'
257c02985581be6b946e99649a7d6470bd21696fffa717758d3977cbdf214ccb  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/driver.py
840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/train_siglip2_identity_diversity.py
fa7c15251aec590ec613f5a42e14118d3a99b558f7c214e129fbb1f6a13972de  /home/riomus/runs/sfora-export-exit-scan-ab-source-v1/authority.json
032a5774afd3c3c2bf1817155dd60d4ed6623f7365c94c831528cf43cd8686ef  /home/riomus/runs/sfora-so400-identity-diversity-evaluation-first-export-control-179061-v1/receipt.json
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
