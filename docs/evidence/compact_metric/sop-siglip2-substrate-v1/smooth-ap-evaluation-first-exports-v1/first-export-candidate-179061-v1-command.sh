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
3feb2831d6589e151b46ec1c355a23b155909731f3c5c397c0ccf60ed24b18e7  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/authority-first-export-candidate-179061-v1.json
2d9198924fe7004730f25ee8929900e0360d73a32442ad7444426cb0c447d317  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
458ba8a4767027bc7c6b1e0c76f7d94ada13b29ebd62c308475c3e91c6414194  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/execution.json
5bd23d26304e8c4c88d00c024aec124e8f8a55b987ff0accfcd0513e1ae9c604  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/test_compact_ranking_evaluation.py
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
f3813ecee4cefdc7579ecd502dfa5d8098f4357aaa141b0bcc3f5eda499d9891  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/bundle/bundle.json
0e5eb26ff7853f9ddf09f6ade2dc364f99ce10f007b56977ddf72926a8d5fb85  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/receipt.json
4a48437574a68fdc3c6852faed71e047c0c56a24f617342bd3cd2d0d886d0b6b  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179061-v1.json
877528b4c34359d14754d56beb847b78a8349d234e8eb5416933158bc0569129  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-control-179061-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
017e99d4fb2fffeac8cb1ba78abc2bd339007d7b7d9e189671154dbcb7a17fd6  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-control-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
d5d87a94e77bf8b0a8436aceff4c0e4cf7138ddaa59ee64a485ad2b8504c46b6  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-first-cpu-v1/receipt.json
037f8e89a28bbb59ad3c4d0726afabb106d5c3a1d7761e6ff30ddaaccc48eff2  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/first-cpu-v1.log
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py --execution-sha256 458ba8a4767027bc7c6b1e0c76f7d94ada13b29ebd62c308475c3e91c6414194 --authority /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/authority-first-export-candidate-179061-v1.json --authority-sha256 3feb2831d6589e151b46ec1c355a23b155909731f3c5c397c0ccf60ed24b18e7 --phase export --seed 179061 --arm candidate --output /home/riomus/runs/sfora-so400-smooth-ap-evaluation-first-export-candidate-179061-v1
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/evaluate_siglip2_nearest_ranking.py
5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/execution.json
a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e  /home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1/test_nearest_ranking_evaluation.py
3feb2831d6589e151b46ec1c355a23b155909731f3c5c397c0ccf60ed24b18e7  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/authority-first-export-candidate-179061-v1.json
2d9198924fe7004730f25ee8929900e0360d73a32442ad7444426cb0c447d317  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/evaluate_siglip2_compact_ranking.py
458ba8a4767027bc7c6b1e0c76f7d94ada13b29ebd62c308475c3e91c6414194  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/execution.json
5bd23d26304e8c4c88d00c024aec124e8f8a55b987ff0accfcd0513e1ae9c604  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/test_compact_ranking_evaluation.py
a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json
49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json
4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3  /home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt
f3813ecee4cefdc7579ecd502dfa5d8098f4357aaa141b0bcc3f5eda499d9891  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/bundle/bundle.json
0e5eb26ff7853f9ddf09f6ade2dc364f99ce10f007b56977ddf72926a8d5fb85  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/receipt.json
4a48437574a68fdc3c6852faed71e047c0c56a24f617342bd3cd2d0d886d0b6b  /home/riomus/runs/sfora-so400-smooth-ap-train-control-179061-v1/resume.pt
4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179061-v1.json
877528b4c34359d14754d56beb847b78a8349d234e8eb5416933158bc0569129  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-control-179061-v1.json
bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/execution.json
5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/test_siglip2_compact_ranking.py
c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log
017e99d4fb2fffeac8cb1ba78abc2bd339007d7b7d9e189671154dbcb7a17fd6  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-control-179061-v1.log
74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679  /home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py
85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/evaluate_siglip2_genuine_views.py
82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/execution.json
ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/test_siglip2_genuine_view_evaluation.py
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
d5d87a94e77bf8b0a8436aceff4c0e4cf7138ddaa59ee64a485ad2b8504c46b6  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-first-cpu-v1/receipt.json
037f8e89a28bbb59ad3c4d0726afabb106d5c3a1d7761e6ff30ddaaccc48eff2  /home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1/first-cpu-v1.log
HASHES
