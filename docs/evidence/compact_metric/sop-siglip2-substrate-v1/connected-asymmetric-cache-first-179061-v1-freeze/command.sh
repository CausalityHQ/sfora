#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
exec 8<>/home/riomus/runs/.sfora-siglip2-gpu.lock
/usr/bin/flock -n 8
exec 9<>/home/riomus/.sfora-siglip2-gpu.lock
/usr/bin/flock -n 9
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
ce1f85d1744240b06a744496cc7cd47fcf89fb47d758e2b67557cb09e8aad307  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/authority.json
f68b4aa480a17f60b5f48522cbb9d952bbd03f9594c0d1214c8f6f624662b181  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/diagnose_connected_asymmetric_cache.py
e74b18f7f838f869737bc0b6debe14bfe10d6be239642db402e6faea947932f3  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/execution.json
7370dd94065d50756232a220d0eb7e3c2be2ce92df31652398142e2e4f895791  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/test_connected_asymmetric_cache.py
e6c57fecb9ddc0772e1396b771876864ff181d3de58804e3622dda74983232c2  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/execution.json
62e69a04bf4df55df5185366231d0237b5991c912f2a0e23e6af31f41c5dd24f  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/observe_connected_control_batch_execution.py
6d08228ab87cc1cf9aba5159a4b72ad671087462ef335c03c659d64b0a750c8c  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/test_connected_control_batch_execution.py
cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1  /home/riomus/runs/sfora-connected-control-serving-native-authority-v5/native-authority.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407  /home/riomus/runs/sfora-connected-mlp-evaluation-full-export-control-179061-v2/receipt.json
0875aec90fe11c1523b66584bbec8164be5dc886d3da81131a4c220a1f624c8b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/authority-full-export-control-179061-v2.json
b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/evaluate_siglip2_connected_mlp.py
3b9ade347801291b2cd4f4a3849be9eb2811b4237ddd562303d46cc3af505029  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/execution.json
48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/test_connected_mlp_evaluation.py
e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153  /home/riomus/runs/sfora-connected-mlp-train-control-179061-v1/control-179061-bundle/bundle.json
fc8795be3cca792aa328f087b1908c8fc12ff75362802042a35e874c02732442  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/connected_control_native_authority.py
b255c6e835ad3d66b1143f2ca2192e500958fe8ffd6f62ec135a0326a6ad48de  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/observe_connected_serving.py
6a4d310d4eb883b7bf3d2ea74a1e96223cee096d0bdb9bc6c2a651eecf9ff163  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/qualify_connected_serving_requests.py
173eb393ed82d01281557b2f76edb1d50ffcc29faf85b9f9deb07d0df3c90d87  /home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages/sfora/cutile_int8.py
ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4  /home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages/sfora/packed_int8.py
3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526  /home/riomus/runs/sfora-cutile-threads-v1/candidate.so
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae  /home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
HASHES
export CUTILE_TILEIRAS_PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
test -x "$CUTILE_TILEIRAS_PATH"
export PYTHONPATH=/home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages
set +e
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/diagnose_connected_asymmetric_cache.py --execution-sha256 e74b18f7f838f869737bc0b6debe14bfe10d6be239642db402e6faea947932f3 --authority /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/authority.json --authority-sha256 ce1f85d1744240b06a744496cc7cd47fcf89fb47d758e2b67557cb09e8aad307 --output /home/riomus/runs/sfora-connected-asymmetric-cache-first-179061-v1
sfora_native_status=$?
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
ce1f85d1744240b06a744496cc7cd47fcf89fb47d758e2b67557cb09e8aad307  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/authority.json
f68b4aa480a17f60b5f48522cbb9d952bbd03f9594c0d1214c8f6f624662b181  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/diagnose_connected_asymmetric_cache.py
e74b18f7f838f869737bc0b6debe14bfe10d6be239642db402e6faea947932f3  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/execution.json
7370dd94065d50756232a220d0eb7e3c2be2ce92df31652398142e2e4f895791  /home/riomus/runs/sfora-connected-asymmetric-cache-source-v1/test_connected_asymmetric_cache.py
e6c57fecb9ddc0772e1396b771876864ff181d3de58804e3622dda74983232c2  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/execution.json
62e69a04bf4df55df5185366231d0237b5991c912f2a0e23e6af31f41c5dd24f  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/observe_connected_control_batch_execution.py
6d08228ab87cc1cf9aba5159a4b72ad671087462ef335c03c659d64b0a750c8c  /home/riomus/runs/sfora-connected-control-batch-execution-source-v5/test_connected_control_batch_execution.py
cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1  /home/riomus/runs/sfora-connected-control-serving-native-authority-v5/native-authority.json
179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/authority.json
3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/diagnose_connected_gallery_freshness.py
6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4  /home/riomus/runs/sfora-connected-gallery-freshness-source-v3/execution.json
db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407  /home/riomus/runs/sfora-connected-mlp-evaluation-full-export-control-179061-v2/receipt.json
0875aec90fe11c1523b66584bbec8164be5dc886d3da81131a4c220a1f624c8b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/authority-full-export-control-179061-v2.json
b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/evaluate_siglip2_connected_mlp.py
3b9ade347801291b2cd4f4a3849be9eb2811b4237ddd562303d46cc3af505029  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/execution.json
48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b  /home/riomus/runs/sfora-connected-mlp-evaluation-source-v9/test_connected_mlp_evaluation.py
e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153  /home/riomus/runs/sfora-connected-mlp-train-control-179061-v1/control-179061-bundle/bundle.json
fc8795be3cca792aa328f087b1908c8fc12ff75362802042a35e874c02732442  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/connected_control_native_authority.py
b255c6e835ad3d66b1143f2ca2192e500958fe8ffd6f62ec135a0326a6ad48de  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/observe_connected_serving.py
6a4d310d4eb883b7bf3d2ea74a1e96223cee096d0bdb9bc6c2a651eecf9ff163  /home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v5/qualify_connected_serving_requests.py
173eb393ed82d01281557b2f76edb1d50ffcc29faf85b9f9deb07d0df3c90d87  /home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages/sfora/cutile_int8.py
ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4  /home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages/sfora/packed_int8.py
3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526  /home/riomus/runs/sfora-cutile-threads-v1/candidate.so
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae  /home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras
HASHES
sfora_postflight_status=$?
if (( sfora_native_status != 0 )); then exit "$sfora_native_status"; fi
exit "$sfora_postflight_status"
