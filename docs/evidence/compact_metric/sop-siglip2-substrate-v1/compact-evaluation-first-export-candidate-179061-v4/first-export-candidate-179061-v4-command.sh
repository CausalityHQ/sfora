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
afca0b389c5c6acadaac0733d6257465ac0590c906a5da42a3092ecc3edcc471  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/authority-first-export-candidate-179061-v4.json
9acc2332dc4c2471299f92a634bab65243a9b9addbd3823af3471716dc7eb7d0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/evaluate_siglip2_compact_ranking.py
b5940d3791344fba6fa65f7bd6051c7eb0a550bb4ac39218cdadb878b272750a  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/execution.json
eb8476be0a4fd02e98bbbc7f98757eb619f95c662027f045621b7edf4d388d6b  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/test_compact_ranking_evaluation.py
1f873fa0f4b48ea7f0fdc3d74390745c6c5751fa2f270925c5e4502a4a15d15c  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/bundle/bundle.json
cabfad45d73b563874cd2b3db0f1bff736c989906e8182b13f5ed54df65191d1  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/receipt.json
dda28fbd4542158e7d9f166d6e32630b84a39fa181496c176256f9605adb31dc  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/resume.pt
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
edb53a7159254ef2809307e6992a6c1d34e84bad562996348553ea9085272eb4  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/receipt.json
016dea3cb3d1d41c216ff93c99b4c8f4d691e112d6da240c04579ed753602473  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/resume.pt
8e24c0534d2765190285c9da68b234c3bb2de4d50f9dfc66697d42b3abb929d2  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-candidate-179061-v3.json
d46e72f796723414a7ad0237b0f6d8d548fb783f84b97df80e16e3386e05a983  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-control-179061-v3.json
ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/execution.json
544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/test_siglip2_compact_ranking.py
7aebd3c6e219647978a40f60d089214794eba4ea51e39749e198bfe79d96825c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train-candidate-179061-v3.log
cba5da3f7661281ce0231c528022b85603f76323fd23de31cbff15e06f0a4cec  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train-control-179061-v3.log
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/evaluate_siglip2_compact_ranking.py --execution-sha256 b5940d3791344fba6fa65f7bd6051c7eb0a550bb4ac39218cdadb878b272750a --authority /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/authority-first-export-candidate-179061-v4.json --authority-sha256 afca0b389c5c6acadaac0733d6257465ac0590c906a5da42a3092ecc3edcc471 --phase export --seed 179061 --arm candidate --output /home/riomus/runs/sfora-so400-compact-ranking-evaluation-first-export-candidate-179061-v4
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
afca0b389c5c6acadaac0733d6257465ac0590c906a5da42a3092ecc3edcc471  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/authority-first-export-candidate-179061-v4.json
9acc2332dc4c2471299f92a634bab65243a9b9addbd3823af3471716dc7eb7d0  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/evaluate_siglip2_compact_ranking.py
b5940d3791344fba6fa65f7bd6051c7eb0a550bb4ac39218cdadb878b272750a  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/execution.json
eb8476be0a4fd02e98bbbc7f98757eb619f95c662027f045621b7edf4d388d6b  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20/test_compact_ranking_evaluation.py
1f873fa0f4b48ea7f0fdc3d74390745c6c5751fa2f270925c5e4502a4a15d15c  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/bundle/bundle.json
cabfad45d73b563874cd2b3db0f1bff736c989906e8182b13f5ed54df65191d1  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/receipt.json
dda28fbd4542158e7d9f166d6e32630b84a39fa181496c176256f9605adb31dc  /home/riomus/runs/sfora-so400-compact-ranking-train-candidate-179061-v3/resume.pt
097169fbeec8b19fa6194d470e6613a83f84e967201fc72793e3b2f53a3609d2  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/bundle/bundle.json
edb53a7159254ef2809307e6992a6c1d34e84bad562996348553ea9085272eb4  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/receipt.json
016dea3cb3d1d41c216ff93c99b4c8f4d691e112d6da240c04579ed753602473  /home/riomus/runs/sfora-so400-compact-ranking-train-control-179061-v3/resume.pt
8e24c0534d2765190285c9da68b234c3bb2de4d50f9dfc66697d42b3abb929d2  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-candidate-179061-v3.json
d46e72f796723414a7ad0237b0f6d8d548fb783f84b97df80e16e3386e05a983  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/authority-train-control-179061-v3.json
ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/execution.json
544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/test_siglip2_compact_ranking.py
7aebd3c6e219647978a40f60d089214794eba4ea51e39749e198bfe79d96825c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train-candidate-179061-v3.log
cba5da3f7661281ce0231c528022b85603f76323fd23de31cbff15e06f0a4cec  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train-control-179061-v3.log
880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c  /home/riomus/runs/sfora-so400-compact-ranking-train-source-v7/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
HASHES
