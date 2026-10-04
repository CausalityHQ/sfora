#!/bin/bash
set -euo pipefail
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
771ff4d7b0c8bc5007a91756ea5e838e2ecf34cc935b18a4cc678142be77fa5d  /home/riomus/runs/sfora-identity-diversity-metadata-v1/freeze_identity_diversity_scope.py
fd0f6cb3376ecce3dc0b27402af2fb3671539a7aaf6246412f4a2009dc098ad6  /home/riomus/runs/sfora-identity-diversity-metadata-v1/mapping.json
cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c  /home/riomus/runs/sfora-identity-diversity-metadata-v1/official-partition.txt
21dabd04d9208695b6d3f03e0e6575ee79c53bdb54cc19ccf7f1210eef316fdf  /home/riomus/runs/sfora-identity-diversity-metadata-v1/plan.json
41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293  /home/riomus/runs/sfora-identity-diversity-metadata-v1/preflight.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  /home/riomus/runs/sfora-identity-diversity-metadata-v1/roles.json
80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218  /home/riomus/runs/sfora-identity-diversity-metadata-v1/schedule-consumer.py
788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96  /home/riomus/runs/sfora-identity-diversity-metadata-v1/schedule-source.py
55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726  /home/riomus/runs/sfora-identity-diversity-metadata-v1/scope.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  /home/riomus/runs/sfora-so400-genuine-view-export-source-v1/train_sop_siglip2_compact.py
d506aa29a3a6f1e0e5453670dc634dc202692eb70711e10f0b06f0065879a802  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/authority.json
d6ae3b99f721fd659035d4ee64a9fbdaead0380798e6d1fdf5047499f599637f  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/execution.json
67f228d24f3af71eba4d854148e2c351df04129347a761b8e7967ef144f1ef39  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/export_siglip2_identity_diversity_views.py
61e6fc421587e2acf29aafff325f9076e4b3847a559249202701e4e934b29809  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/test_siglip2_identity_diversity_views.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/export_siglip2_identity_diversity_views.py --execution-sha256 d6ae3b99f721fd659035d4ee64a9fbdaead0380798e6d1fdf5047499f599637f --authority /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/authority.json --authority-sha256 d506aa29a3a6f1e0e5453670dc634dc202692eb70711e10f0b06f0065879a802 --arm candidate --phase startup --output /home/riomus/runs/sfora-so400-identity-diversity-view-startup-candidate-v2
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
771ff4d7b0c8bc5007a91756ea5e838e2ecf34cc935b18a4cc678142be77fa5d  /home/riomus/runs/sfora-identity-diversity-metadata-v1/freeze_identity_diversity_scope.py
fd0f6cb3376ecce3dc0b27402af2fb3671539a7aaf6246412f4a2009dc098ad6  /home/riomus/runs/sfora-identity-diversity-metadata-v1/mapping.json
cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c  /home/riomus/runs/sfora-identity-diversity-metadata-v1/official-partition.txt
21dabd04d9208695b6d3f03e0e6575ee79c53bdb54cc19ccf7f1210eef316fdf  /home/riomus/runs/sfora-identity-diversity-metadata-v1/plan.json
41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293  /home/riomus/runs/sfora-identity-diversity-metadata-v1/preflight.json
702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c  /home/riomus/runs/sfora-identity-diversity-metadata-v1/roles.json
80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218  /home/riomus/runs/sfora-identity-diversity-metadata-v1/schedule-consumer.py
788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96  /home/riomus/runs/sfora-identity-diversity-metadata-v1/schedule-source.py
55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726  /home/riomus/runs/sfora-identity-diversity-metadata-v1/scope.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py
e5f08a75c38ee6579ccc9681aee4946fbc59ddfa36e376a3fb81e27d9e58ef45  /home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py
1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610  /home/riomus/runs/sfora-so400-genuine-view-export-source-v1/train_sop_siglip2_compact.py
d506aa29a3a6f1e0e5453670dc634dc202692eb70711e10f0b06f0065879a802  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/authority.json
d6ae3b99f721fd659035d4ee64a9fbdaead0380798e6d1fdf5047499f599637f  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/execution.json
67f228d24f3af71eba4d854148e2c351df04129347a761b8e7967ef144f1ef39  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/export_siglip2_identity_diversity_views.py
61e6fc421587e2acf29aafff325f9076e4b3847a559249202701e4e934b29809  /home/riomus/runs/sfora-so400-identity-diversity-view-source-v2/test_siglip2_identity_diversity_views.py
HASHES
