#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-source-extraction-v1
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
7aa542685ea1e2a2a4d542588bf1505a00d3287fe60ff56b8e11c9c3c4a3675e  test_extract_siglip2_vision_source.py
a019105abb4423c2e38f213a1a196ccaaff689310aa00543933403dc4345ca5a  inputs.json
1cbf827797068443b2451304c41dcac5f3bf384a7594f4d13e4b09d2df667945  execution.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-source-extraction-v1/extract_siglip2_vision_source.py --source /home/riomus/runs/sfora-so400-native256-upstream-v1/model.safetensors --source-sha256 810ddc85ac019e6b2738b9130ae3602e41eb093a848a297a0ad296ca1f39dc67 --source-size 4542792928 --config /home/riomus/runs/sfora-so400-native256-upstream-v1/config.json --config-sha256 13ee943037e446415ff6e7e406ecf79fd955f482f4ba5b437d3bc10fba7f1362 --preprocessor /home/riomus/runs/sfora-so400-native256-upstream-v1/preprocessor_config.json --preprocessor-sha256 d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff --source-model google/siglip2-so400m-patch16-256 --revision e8708ab72d125807e45b36fb7d4e0aacbb59f379 --output /home/riomus/runs/sfora-native256-source-extraction-v1/so400/vision.safetensors --provenance /home/riomus/runs/sfora-native256-source-extraction-v1/so400/provenance.json
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
7aa542685ea1e2a2a4d542588bf1505a00d3287fe60ff56b8e11c9c3c4a3675e  test_extract_siglip2_vision_source.py
a019105abb4423c2e38f213a1a196ccaaff689310aa00543933403dc4345ca5a  inputs.json
1cbf827797068443b2451304c41dcac5f3bf384a7594f4d13e4b09d2df667945  execution.json
HASHES
