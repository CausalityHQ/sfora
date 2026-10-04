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
5618f9e4d3a9772c0ca6063c5374052399627b338ed09ca97bccee844c7a1eab  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/authority-first-export-control-179061-v2.json
8604043becd1d430bf90d0b5f99e02bf1c35fbd278fb755036130e811b4ddede  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/evaluate_siglip2_compact_ranking.py
09415ae900226b2a89e639ae890005116f2b38c911c091fde15b38b12522c1d2  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/execution.json
9d0f2c983051deec30b869f6255166eb4265783cd177c62556f1538909d774dc  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/test_compact_ranking_evaluation.py
435c2eb490ad020f61adfa9aa0d9667b1c366ee86688a24d3015735168a9c022  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
07a936c20b648135447235374d44b13de8bc092a3ff3386963694039a324a749  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/receipt.json
2d45bae9a7b66d65e3723ba1ddd929c532507898986ea7277991b4ecb1b6fba7  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/resume.pt
936b4c599142860d946a3a6841f39b53e7675e0ab86153ac9fe66826ebe66d90  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/bundle/bundle.json
f41553b7a5f0c6310c73a517a275dd9fe9d61fb9564bfb52a00049eb9b4168d2  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/receipt.json
f4eb3b1db5f972f3e9f5c25a08b5dd3e9adc0748450dd2814474c4b5a4619ceb  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/resume.pt
b48d00d895f30b9d0b91d0ff86910822605611fa461bf1e1fe2521320e0afe1d  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/authority-train-candidate-179061-v1.json
327dc3866088ec9bef38dfa9903d337817a374d2995504ef8d55c3a5f7a85b54  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/authority-train-control-179061-v1.json
1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/execution.json
b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/test_siglip2_compact_ranking.py
e2f0183814c2f51cc0349d47034ee26466dc53c960fcb6a4527a91d014597c68  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train-candidate-179061-v1.log
1f774398dec5224b6ea695a2695b93218e0f0a1762529dc434ddd60e7b7b4e3c  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train-control-179061-v1.log
40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
cf5e2fc574924ef89a73af2168f38da5e0e00f3f7ec616aebf0752cd84309752  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-first-cpu-v2/receipt.json
c96b5b802c9071d07d402497b95bc5002b62e4c959c6e2d82385cfd21d33adf5  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/first-cpu-v2.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/evaluate_siglip2_compact_ranking.py --execution-sha256 09415ae900226b2a89e639ae890005116f2b38c911c091fde15b38b12522c1d2 --authority /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/authority-first-export-control-179061-v2.json --authority-sha256 5618f9e4d3a9772c0ca6063c5374052399627b338ed09ca97bccee844c7a1eab --phase export --seed 179061 --arm control --output /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-first-export-control-179061-v2
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
5618f9e4d3a9772c0ca6063c5374052399627b338ed09ca97bccee844c7a1eab  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/authority-first-export-control-179061-v2.json
8604043becd1d430bf90d0b5f99e02bf1c35fbd278fb755036130e811b4ddede  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/evaluate_siglip2_compact_ranking.py
09415ae900226b2a89e639ae890005116f2b38c911c091fde15b38b12522c1d2  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/execution.json
9d0f2c983051deec30b869f6255166eb4265783cd177c62556f1538909d774dc  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/test_compact_ranking_evaluation.py
435c2eb490ad020f61adfa9aa0d9667b1c366ee86688a24d3015735168a9c022  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
07a936c20b648135447235374d44b13de8bc092a3ff3386963694039a324a749  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/receipt.json
2d45bae9a7b66d65e3723ba1ddd929c532507898986ea7277991b4ecb1b6fba7  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-candidate-179061-v1/resume.pt
936b4c599142860d946a3a6841f39b53e7675e0ab86153ac9fe66826ebe66d90  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/bundle/bundle.json
f41553b7a5f0c6310c73a517a275dd9fe9d61fb9564bfb52a00049eb9b4168d2  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/receipt.json
f4eb3b1db5f972f3e9f5c25a08b5dd3e9adc0748450dd2814474c4b5a4619ceb  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-control-179061-v1/resume.pt
b48d00d895f30b9d0b91d0ff86910822605611fa461bf1e1fe2521320e0afe1d  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/authority-train-candidate-179061-v1.json
327dc3866088ec9bef38dfa9903d337817a374d2995504ef8d55c3a5f7a85b54  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/authority-train-control-179061-v1.json
1ed4790f6716ded3972a0e6fd86cffdd45edb7d2060cf949d07da2d4a1adf141  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/execution.json
b56265f631a589cc40aa5e0b7cc779bb54d84451eabd87628cc850b704eeba9d  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/test_siglip2_compact_ranking.py
e2f0183814c2f51cc0349d47034ee26466dc53c960fcb6a4527a91d014597c68  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train-candidate-179061-v1.log
1f774398dec5224b6ea695a2695b93218e0f0a1762529dc434ddd60e7b7b4e3c  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train-control-179061-v1.log
40c271217c74b89d3b7c20d0dfe05571746303ed6f75f7dcdcffbef3f77a2004  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
cf5e2fc574924ef89a73af2168f38da5e0e00f3f7ec616aebf0752cd84309752  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-first-cpu-v2/receipt.json
c96b5b802c9071d07d402497b95bc5002b62e4c959c6e2d82385cfd21d33adf5  /home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v2/first-cpu-v2.log
HASHES
