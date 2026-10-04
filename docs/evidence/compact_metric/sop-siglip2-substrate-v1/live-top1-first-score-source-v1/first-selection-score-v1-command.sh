#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
0ef7bac8033e6fb08b26066489188d49e6a6c6c2209fd43c7fccda44b3325b4b  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/authority-first-selection-score-v1.json
8ecaa206d0343dde117cec97fa92a0179cb8a5991127d4c18dc7917a7c535563  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
804bde987374b1416e8cedaaf1a9a44d888e4072a26d1af86a788bdae17ef072  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/execution.json
bd49b89abdf6a29a1b18ebe058ffe0b265a41b2736a0e819d17f1b7f1d674e63  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/test_compact_ranking_evaluation.py
44838ad3b4a8c43cdfc73a9a1a2bc8d972c7ba21732902bb877e8ca8cd52b0be  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/bundle/bundle.json
cd759c80170d0257ef9cfeefe552aa58834429ed32a8320bb0a19c115d22ba6c  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/receipt.json
2d4d38019bbec85d7ab4943243f5f0c01686853af3649255681262ad05415507  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/resume.pt
1d9b343ec48b497cc9f5e628a1c41cf814a9da7a7752432f821c28f2fa08b747  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/bundle/bundle.json
5c12a68fe86f8c9f05100c767431a22a57468488fe19ad1ea16d5d59893bbc3c  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/receipt.json
d6dac8cdcea26d5ff2a4c55c3ed381eb55ed03f559d12d140eadd94ac62995a3  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/resume.pt
c34dfd339f808e53aa9e797ce06ea0196cc0c5e52797f994a5133ddd927ee684  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/authority-train-candidate-179061-v1.json
5b28eac8b0c1fb8a30a78da27c5e9ca7cc2d80d82824e0cf18b2dd1d5b3addfc  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/authority-train-control-179061-v1.json
3a7e7a42866e44ada935287fb15025fd0ab3469acea1a4579ca90e3ab8584f88  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/execution.json
70f86f22065888d569f834fc1b4b70f47e3a5d10331b013e158e220c2892a92e  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/test_siglip2_compact_ranking.py
4d7c83a06f3a6deeaab52632358ceb048e7033e7f50a660f03b1b123cb201a67  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train-candidate-179061-v1.log
d424a62d13e6074ff8ce6945769fbc5e3da570d14563b5489f507c2165138221  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train-control-179061-v1.log
80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
f89a8304bcb2999246e44480e19eabde029e44a6770e8c7f6ffa19ba17d375d7  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/receipt.json
391e98fd6edb1ec63fa69c51d18e2257002eea29d4a9bd9bfe6efbb64ff542b8  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/first-export-control-179061-v1.log
031d0cf2225156535cfe6ebb23b3ccf507e0d01778123e1b543683a7ece45709  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.packed.bin
6bd74a3c8ec5cdeafb7dff9b0cfb92bf263877822901e846d68e591156a5f7b0  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.raw.npy
40860a4f470875c04bbb83af51bd6f11902d9f3416662db9f6e96a384b670cee  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.unit.npy
5f0a6a0a83dd134c8bc99fcfbffd2b4e315917e299f61736bc4d4217af8eb895  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/receipt.json
d2bcdd6c8a736ea113b04a14a9237f76d9f8ec7e894fb7af18f806ccd62deb38  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/first-export-candidate-179061-v1.log
c9f1ee8fe32b9a6723005453a428de71ad8fe1aaa4ddbcf99dc06828dea266c3  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.packed.bin
238476a3b9962e108a866daa61b3f8b2cb4e1a4837f9c9db6cf3fd4011755ef2  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.raw.npy
e6e1ba60734ebd73ecf25288863e80bbb7567de6b6adc423c1cdfd8dcbfe1399  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.unit.npy
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/evaluate_siglip2_compact_ranking.py --execution-sha256 804bde987374b1416e8cedaaf1a9a44d888e4072a26d1af86a788bdae17ef072 --authority /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/authority-first-selection-score-v1.json --authority-sha256 0ef7bac8033e6fb08b26066489188d49e6a6c6c2209fd43c7fccda44b3325b4b --phase score --output /home/riomus/runs/sfora-so400-live-top1-evaluation-first-selection-score-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
0ef7bac8033e6fb08b26066489188d49e6a6c6c2209fd43c7fccda44b3325b4b  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/authority-first-selection-score-v1.json
8ecaa206d0343dde117cec97fa92a0179cb8a5991127d4c18dc7917a7c535563  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
804bde987374b1416e8cedaaf1a9a44d888e4072a26d1af86a788bdae17ef072  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/execution.json
bd49b89abdf6a29a1b18ebe058ffe0b265a41b2736a0e819d17f1b7f1d674e63  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/test_compact_ranking_evaluation.py
44838ad3b4a8c43cdfc73a9a1a2bc8d972c7ba21732902bb877e8ca8cd52b0be  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/bundle/bundle.json
cd759c80170d0257ef9cfeefe552aa58834429ed32a8320bb0a19c115d22ba6c  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/receipt.json
2d4d38019bbec85d7ab4943243f5f0c01686853af3649255681262ad05415507  /home/riomus/runs/sfora-so400-live-top1-train-candidate-179061-v1/resume.pt
1d9b343ec48b497cc9f5e628a1c41cf814a9da7a7752432f821c28f2fa08b747  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/bundle/bundle.json
5c12a68fe86f8c9f05100c767431a22a57468488fe19ad1ea16d5d59893bbc3c  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/receipt.json
d6dac8cdcea26d5ff2a4c55c3ed381eb55ed03f559d12d140eadd94ac62995a3  /home/riomus/runs/sfora-so400-live-top1-train-control-179061-v1/resume.pt
c34dfd339f808e53aa9e797ce06ea0196cc0c5e52797f994a5133ddd927ee684  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/authority-train-candidate-179061-v1.json
5b28eac8b0c1fb8a30a78da27c5e9ca7cc2d80d82824e0cf18b2dd1d5b3addfc  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/authority-train-control-179061-v1.json
3a7e7a42866e44ada935287fb15025fd0ab3469acea1a4579ca90e3ab8584f88  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/execution.json
70f86f22065888d569f834fc1b4b70f47e3a5d10331b013e158e220c2892a92e  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/test_siglip2_compact_ranking.py
4d7c83a06f3a6deeaab52632358ceb048e7033e7f50a660f03b1b123cb201a67  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train-candidate-179061-v1.log
d424a62d13e6074ff8ce6945769fbc5e3da570d14563b5489f507c2165138221  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train-control-179061-v1.log
80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218  /home/riomus/runs/sfora-so400-live-top1-train-source-v2/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
f89a8304bcb2999246e44480e19eabde029e44a6770e8c7f6ffa19ba17d375d7  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/receipt.json
391e98fd6edb1ec63fa69c51d18e2257002eea29d4a9bd9bfe6efbb64ff542b8  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/first-export-control-179061-v1.log
031d0cf2225156535cfe6ebb23b3ccf507e0d01778123e1b543683a7ece45709  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.packed.bin
6bd74a3c8ec5cdeafb7dff9b0cfb92bf263877822901e846d68e591156a5f7b0  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.raw.npy
40860a4f470875c04bbb83af51bd6f11902d9f3416662db9f6e96a384b670cee  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-control-179061-v1/control-179061.unit.npy
5f0a6a0a83dd134c8bc99fcfbffd2b4e315917e299f61736bc4d4217af8eb895  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/receipt.json
d2bcdd6c8a736ea113b04a14a9237f76d9f8ec7e894fb7af18f806ccd62deb38  /home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1/first-export-candidate-179061-v1.log
c9f1ee8fe32b9a6723005453a428de71ad8fe1aaa4ddbcf99dc06828dea266c3  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.packed.bin
238476a3b9962e108a866daa61b3f8b2cb4e1a4837f9c9db6cf3fd4011755ef2  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.raw.npy
e6e1ba60734ebd73ecf25288863e80bbb7567de6b6adc423c1cdfd8dcbfe1399  /home/riomus/runs/sfora-so400-live-top1-evaluation-first-export-candidate-179061-v1/candidate-179061.unit.npy
HASHES
