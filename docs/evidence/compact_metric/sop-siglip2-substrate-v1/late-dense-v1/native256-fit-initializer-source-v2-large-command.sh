#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-fit-initializer-source-v2
trap 'sfora_command_status=$?; trap - EXIT; set +e; export SFORA_COMMAND_STATUS=$sfora_command_status; /usr/bin/python3 /home/riomus/runs/sfora-native256-fit-initializer-so400-v2-final-footer.py; sfora_footer_status=$?; if (( sfora_command_status != 0 )); then exit "$sfora_command_status"; fi; exit "$sfora_footer_status"' EXIT
sha256sum -c <<'HASHES'
7ac147b85de1fefe23671c05adf9147c75e43f1ffcaf27c0395254f35d5e1122  initialize_siglip2_substrate_fit.py
1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c  representation_ceiling.py
bde0252dea6aa6680289fbae8b270d0d878ac0b8153ed94a5a075b5a203802d2  test_siglip2_substrate_initializer.py
52e14ceaa4d1a75a01c9187a769066dae379b0f20f9176434f273b46c7b50b8a  execution.json
e971e85878af2f2372d3b6287106d8ebc6b810754ab7c8f46a39d5ced0923765  authority-large-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-fit-initializer-so400-v2-final-footer.py
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-fit-initializer-source-v2/initialize_siglip2_substrate_fit.py --execution-sha256 52e14ceaa4d1a75a01c9187a769066dae379b0f20f9176434f273b46c7b50b8a --authority /home/riomus/runs/sfora-native256-fit-initializer-source-v2/authority-large-v2.json --authority-sha256 e971e85878af2f2372d3b6287106d8ebc6b810754ab7c8f46a39d5ced0923765 --arm large --output /home/riomus/runs/sfora-native256-fit-initializer-large-v2
sha256sum -c <<'HASHES'
7ac147b85de1fefe23671c05adf9147c75e43f1ffcaf27c0395254f35d5e1122  initialize_siglip2_substrate_fit.py
1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c  representation_ceiling.py
bde0252dea6aa6680289fbae8b270d0d878ac0b8153ed94a5a075b5a203802d2  test_siglip2_substrate_initializer.py
52e14ceaa4d1a75a01c9187a769066dae379b0f20f9176434f273b46c7b50b8a  execution.json
e971e85878af2f2372d3b6287106d8ebc6b810754ab7c8f46a39d5ced0923765  authority-large-v2.json
bfe7cd329ad359bdcbc077e732ce654a94ef248b8567f0f32fd4234652df1995  /home/riomus/runs/sfora-native256-fit-initializer-so400-v2-final-footer.py
HASHES
