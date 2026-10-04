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
4d2d02a1b200a36084f4f95dba93604555ec495d8f3dadced963507342e03f2f  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/authority-first-cpu-v1.json
0128c543768d8f9eb964231777f5b1fe30e1c39b34c41dc5065441fe2d92c294  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
1e69b40217e47098b1d088c1cdacd9a50de9471a7f59ff162faed89eafb8632d  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/execution.json
e46e55e5b9d6548ab03d467791b3fba0ff08c6af8c51cf875bae21aee7f5f46e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/test_compact_ranking_evaluation.py
264f73615568233abcfcdadc59d3c8084f20b0917b7deeed00719fb68d6faa1a  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
2c22ac1be19a228d0d96cef023da7471af9c13ebff30299e8dc711eaddb62e93  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/receipt.json
4a398c05263652896ac9b34ea618df645b92a0ab7fa071f8b0a6dbbb01958e56  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/resume.pt
2f00a8e881d9223074ef47bbe7db2ab0650c480931f2544c8bf3a7f28629dfd4  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/bundle/bundle.json
15c8133816847404527ff6564a44e7ab011b77f51075cf7e92dff26dcea15feb  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/receipt.json
e189eb1f544e9cbe24e086c2d30afceb044a13a19f84065c54db51c5e152c2ef  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/resume.pt
9d17967daa5a7fd3bcaeab547f53b1af668c04961bdb6f9a5db637e5adeef8f5  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/authority-train-candidate-179061-v1.json
4e7c1405e78c176d44e1d2df517dc5c5aa8f2ac1f2d800a94039e0cae262cd8e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/authority-train-control-179061-v1.json
73cbc39e857ad99e2962a6252c8c158b232636889b5023c28a97b1be7edda68e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/execution.json
32f9ade5717d98e9aa5f6644ad7ac0dc5992bb21e0861730e154d0d78c8a7744  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
e75cd6b0d58b6a221dd76ded17d807056cf4deba2e743775649061db091d9bbc  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train-candidate-179061-v1.log
a48a06bbfbe9e0daf6b9643f1c0233e171b31514d850d33a4d1092009184d08a  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train-control-179061-v1.log
affb911bd76a760b480e616d13dc78710cc56c74ce92ddd27e1cc8aaa3483480  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py --execution-sha256 1e69b40217e47098b1d088c1cdacd9a50de9471a7f59ff162faed89eafb8632d --authority /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/authority-first-cpu-v1.json --authority-sha256 4d2d02a1b200a36084f4f95dba93604555ec495d8f3dadced963507342e03f2f --phase cpu --output /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-first-cpu-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
4d2d02a1b200a36084f4f95dba93604555ec495d8f3dadced963507342e03f2f  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/authority-first-cpu-v1.json
0128c543768d8f9eb964231777f5b1fe30e1c39b34c41dc5065441fe2d92c294  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
1e69b40217e47098b1d088c1cdacd9a50de9471a7f59ff162faed89eafb8632d  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/execution.json
e46e55e5b9d6548ab03d467791b3fba0ff08c6af8c51cf875bae21aee7f5f46e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-evaluation-source-v1/test_compact_ranking_evaluation.py
264f73615568233abcfcdadc59d3c8084f20b0917b7deeed00719fb68d6faa1a  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
2c22ac1be19a228d0d96cef023da7471af9c13ebff30299e8dc711eaddb62e93  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/receipt.json
4a398c05263652896ac9b34ea618df645b92a0ab7fa071f8b0a6dbbb01958e56  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-candidate-179061-v1/resume.pt
2f00a8e881d9223074ef47bbe7db2ab0650c480931f2544c8bf3a7f28629dfd4  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/bundle/bundle.json
15c8133816847404527ff6564a44e7ab011b77f51075cf7e92dff26dcea15feb  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/receipt.json
e189eb1f544e9cbe24e086c2d30afceb044a13a19f84065c54db51c5e152c2ef  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-control-179061-v1/resume.pt
9d17967daa5a7fd3bcaeab547f53b1af668c04961bdb6f9a5db637e5adeef8f5  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/authority-train-candidate-179061-v1.json
4e7c1405e78c176d44e1d2df517dc5c5aa8f2ac1f2d800a94039e0cae262cd8e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/authority-train-control-179061-v1.json
73cbc39e857ad99e2962a6252c8c158b232636889b5023c28a97b1be7edda68e  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/execution.json
32f9ade5717d98e9aa5f6644ad7ac0dc5992bb21e0861730e154d0d78c8a7744  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
e75cd6b0d58b6a221dd76ded17d807056cf4deba2e743775649061db091d9bbc  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train-candidate-179061-v1.log
a48a06bbfbe9e0daf6b9643f1c0233e171b31514d850d33a4d1092009184d08a  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train-control-179061-v1.log
affb911bd76a760b480e616d13dc78710cc56c74ce92ddd27e1cc8aaa3483480  /home/riomus/runs/sfora-so400-current-gallery-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
HASHES
