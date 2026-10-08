#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
e1b1056e573360aeba3135e29e395696cac8966c1d4c584b197185c520ae5f2f  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/freeze.json
4c610106bd596baf411db928e510d02dd9a331861cd27830452d3753ac423c00  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/control.py
0c0320787ee8d9e3a61bfd265859ec8785874b9727c686d6bd290fd4af19679d  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/joint_relational_compaction.py
ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/packed_int8.py
cd87559f81b14ce3ba8b1d5742b2410d8ee3df581400c4ac387176b1e376e7c8  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/request.py
312536f149c5e4e8411c4725697b5438d78f304339ae25432145f62928b8993c  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/runner.py
HASHES
/usr/bin/time -v /usr/bin/timeout 15 /usr/bin/prlimit --as=1073741824 /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13 -I -S -B /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/runner.py /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/request.py /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/control.py /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/packed_int8.py /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/joint_relational_compaction.py
sha256sum -c <<'HASHES'
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
e1b1056e573360aeba3135e29e395696cac8966c1d4c584b197185c520ae5f2f  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/freeze.json
4c610106bd596baf411db928e510d02dd9a331861cd27830452d3753ac423c00  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/control.py
0c0320787ee8d9e3a61bfd265859ec8785874b9727c686d6bd290fd4af19679d  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/joint_relational_compaction.py
ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/packed_int8.py
cd87559f81b14ce3ba8b1d5742b2410d8ee3df581400c4ac387176b1e376e7c8  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/request.py
312536f149c5e4e8411c4725697b5438d78f304339ae25432145f62928b8993c  /home/riomus/runs/sfora-installed-packing-alias-py313-source-v1/runner.py
HASHES
