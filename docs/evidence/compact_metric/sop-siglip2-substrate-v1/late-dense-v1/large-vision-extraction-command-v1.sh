#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-source-extraction-v1
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
7aa542685ea1e2a2a4d542588bf1505a00d3287fe60ff56b8e11c9c3c4a3675e  test_extract_siglip2_vision_source.py
a019105abb4423c2e38f213a1a196ccaaff689310aa00543933403dc4345ca5a  inputs.json
1cbf827797068443b2451304c41dcac5f3bf384a7594f4d13e4b09d2df667945  execution.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-source-extraction-v1/extract_siglip2_vision_source.py --source /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/blobs/fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a --source-sha256 fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a --source-size 3526204360 --config /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/blobs/1cce28fde65d915dbd9c38d7865a91b842ce174d --config-sha256 172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104 --preprocessor /home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/blobs/130986149eee6a6fd9a2eb53da12fbc1a415c0e6 --preprocessor-sha256 d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff --source-model google/siglip2-large-patch16-256 --revision 787800c8990e6f058423089178e718139608408c --output /home/riomus/runs/sfora-native256-source-extraction-v1/large/vision.safetensors --provenance /home/riomus/runs/sfora-native256-source-extraction-v1/large/provenance.json
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
7aa542685ea1e2a2a4d542588bf1505a00d3287fe60ff56b8e11c9c3c4a3675e  test_extract_siglip2_vision_source.py
a019105abb4423c2e38f213a1a196ccaaff689310aa00543933403dc4345ca5a  inputs.json
1cbf827797068443b2451304c41dcac5f3bf384a7594f4d13e4b09d2df667945  execution.json
HASHES
