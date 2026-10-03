#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
914f7a082570da853a404e8dd1960f41bbbdd634e92c01c9dd92aac8abd3f06d  /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/receipt.json
b74fd184eeafe42fe15b0aaf1af4d4226bf3c2847f7d61436e2e64348ee66650  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/first-score.log
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/partition.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/source-selection-inventory.json
5ba809ccf300779fbb45b04df3da1fea1254ba3e225753f7878e4ab3aa447566  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/authority-selection-v1.json
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
cc88228c956232dce08563b12189dd80f99d1e02c31c2da86925a74fa4bca6d2  /home/riomus/runs/sfora-so400-signed-concat-fit-linear-v1/receipt.json
8279cf16bd9ff4162e804ca38011ae0ad661daf7a18bf9d1076e8cbd684270fb  /home/riomus/runs/sfora-so400-signed-concat-fit-linear-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a1998762b729be99f818ab9631a233bf88cb539007489985bc9fbe6d1e570609  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-linear-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
0d3bfc16198cebc76db43460675fc098ade2418f3f9d4cc6cab5f3743641acee  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-linear-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
e497a1f6da45c9ab5eb5b6ecfc2f7c2ceb4ffd2c68bed642ea6264335e83a288  /home/riomus/runs/sfora-so400-signed-concat-evaluation-cpu-v2/receipt.json
eab6fed2d9bb97343d2f30d24d0b0c34f6db28d823e314ab80b52dc54d0a1506  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/cpu-v2.log
89ef2f26a511a2d6272570cd4b83c072ad870eeddf63b00c06e4c605db9fdb45  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/cpu-v2-terminal.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py --execution-sha256 c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970 --authority /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/authority-selection-v1.json --authority-sha256 5ba809ccf300779fbb45b04df3da1fea1254ba3e225753f7878e4ab3aa447566 --phase score --output /home/riomus/runs/sfora-so400-signed-concat-evaluation-selection-score-v1 --prerequisite /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/cpu-v2-terminal.json --prerequisite-sha256 89ef2f26a511a2d6272570cd4b83c072ad870eeddf63b00c06e4c605db9fdb45
sha256sum -c <<'HASHES'
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
914f7a082570da853a404e8dd1960f41bbbdd634e92c01c9dd92aac8abd3f06d  /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/receipt.json
b74fd184eeafe42fe15b0aaf1af4d4226bf3c2847f7d61436e2e64348ee66650  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/first-score.log
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/partition.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/source-selection-inventory.json
5ba809ccf300779fbb45b04df3da1fea1254ba3e225753f7878e4ab3aa447566  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/authority-selection-v1.json
e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py
c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/execution.json
5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/test_siglip2_prototype_residual_evaluation.py
b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json
b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf  /home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt
cc88228c956232dce08563b12189dd80f99d1e02c31c2da86925a74fa4bca6d2  /home/riomus/runs/sfora-so400-signed-concat-fit-linear-v1/receipt.json
8279cf16bd9ff4162e804ca38011ae0ad661daf7a18bf9d1076e8cbd684270fb  /home/riomus/runs/sfora-so400-signed-concat-fit-linear-v1/resume.pt
109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json
a1998762b729be99f818ab9631a233bf88cb539007489985bc9fbe6d1e570609  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-linear-v1.json
a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/execution.json
93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log
0d3bfc16198cebc76db43460675fc098ade2418f3f9d4cc6cab5f3743641acee  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-linear-v1.log
95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit_siglip2_prototype_residual.py
2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py
c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4  /home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/test_siglip2_prototype_residual.py
e497a1f6da45c9ab5eb5b6ecfc2f7c2ceb4ffd2c68bed642ea6264335e83a288  /home/riomus/runs/sfora-so400-signed-concat-evaluation-cpu-v2/receipt.json
eab6fed2d9bb97343d2f30d24d0b0c34f6db28d823e314ab80b52dc54d0a1506  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/cpu-v2.log
89ef2f26a511a2d6272570cd4b83c072ad870eeddf63b00c06e4c605db9fdb45  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/cpu-v2-terminal.json
HASHES
