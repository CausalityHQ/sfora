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
dba7a472694f3e4d218c58ca27d3fbe4686b45913604ca997e1066e62c887012  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/authority-selection-v1.json
64660923102e987cd4d16d9e063dc19260f1396fc86e9cb1a9ecb95e1a7d1c91  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/evaluate_siglip2_prototype_residual.py
42d60a615811cfb76a9def2115f6a1c725d2ebc442a204a586a846395ee0c677  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/execution.json
19dbd915c749f3d70a20556e17756897669dc1da5817135fdede99e7eaab6ee7  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/test_siglip2_prototype_residual_evaluation.py
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
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/evaluate_siglip2_prototype_residual.py --execution-sha256 42d60a615811cfb76a9def2115f6a1c725d2ebc442a204a586a846395ee0c677 --authority /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/authority-selection-v1.json --authority-sha256 dba7a472694f3e4d218c58ca27d3fbe4686b45913604ca997e1066e62c887012 --phase cpu --output /home/riomus/runs/sfora-so400-signed-concat-evaluation-cpu-v1
sha256sum -c <<'HASHES'
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
914f7a082570da853a404e8dd1960f41bbbdd634e92c01c9dd92aac8abd3f06d  /home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/receipt.json
b74fd184eeafe42fe15b0aaf1af4d4226bf3c2847f7d61436e2e64348ee66650  /home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/first-score.log
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  /home/riomus/runs/sfora-so400-genuine-view-export-source-v2/partition.json
ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985  /home/riomus/runs/sfora-so400-prototype-residual-evaluation-source-v7/source-selection-inventory.json
dba7a472694f3e4d218c58ca27d3fbe4686b45913604ca997e1066e62c887012  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/authority-selection-v1.json
64660923102e987cd4d16d9e063dc19260f1396fc86e9cb1a9ecb95e1a7d1c91  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/evaluate_siglip2_prototype_residual.py
42d60a615811cfb76a9def2115f6a1c725d2ebc442a204a586a846395ee0c677  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/execution.json
19dbd915c749f3d70a20556e17756897669dc1da5817135fdede99e7eaab6ee7  /home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v1/test_siglip2_prototype_residual_evaluation.py
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
HASHES
