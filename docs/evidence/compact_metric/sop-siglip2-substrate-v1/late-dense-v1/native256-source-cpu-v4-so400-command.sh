#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-source-cpu-v4
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38  qualify_siglip2_substrate_cpu.py
aa45d7db558194ce294a70cc465a8250c871374d99d95c6adcb016dc60e7fc04  test_siglip2_substrate_cpu.py
3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b  execution.json
8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b  sources.json
d32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251  fit.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-source-cpu-v4/qualify_siglip2_substrate_cpu.py --execution-sha256 3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b --sources /home/riomus/runs/sfora-native256-source-cpu-v4/sources.json --sources-sha256 8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b --fit-manifest /home/riomus/runs/sfora-native256-source-cpu-v4/fit.json --fit-manifest-sha256 d32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251 --arm so400 --output /home/riomus/runs/sfora-native256-source-cpu-so400-v4
sha256sum -c <<'HASHES'
a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d  extract_siglip2_vision_source.py
eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38  qualify_siglip2_substrate_cpu.py
aa45d7db558194ce294a70cc465a8250c871374d99d95c6adcb016dc60e7fc04  test_siglip2_substrate_cpu.py
3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b  execution.json
8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b  sources.json
d32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251  fit.json
HASHES
/usr/bin/python3 - <<'PY'
import json,os
from pathlib import Path
cg=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
root=Path('/sys/fs/cgroup')/cg.lstrip('/')
values={n:(root/n).read_text().strip() for n in ('memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.peak','memory.swap.max','memory.events')}
assert values['memory.max']=='8589934592' and 0<int(values['memory.peak'])<=8589934592
assert all(values[n]=='0' for n in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
events=dict(x.split() for x in values['memory.events'].splitlines()); assert all(int(events[k])==0 for k in ('oom','oom_kill','max'))
print('FINAL_CGROUP '+json.dumps({'path':str(root),'invocation_id':os.environ.get('INVOCATION_ID'),'values':values},sort_keys=True))
PY
