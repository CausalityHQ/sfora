#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-fit-initializer-source-v1
sha256sum -c <<'HASHES'
c2caa8c4985910dd5496dbf8fc5b1ccbd3a66ee28dfe883c78b5cf91d0cbcd06  initialize_siglip2_substrate_fit.py
7f9eb6a079ef55fc0bc86a60302533df2b7fd77119a2b6d1bfd19fef4607e8d0  test_siglip2_substrate_initializer.py
1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c  representation_ceiling.py
3a716471c65e0e2484ac746cf5bc3937970b16934e2d14aef8510580d8c004c2  execution.json
e0a749356c5e11b1e414370c65aadee671e4a363fcaaca7647c580bad98e9af9  authority-so400-v1.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-fit-initializer-source-v1/initialize_siglip2_substrate_fit.py --execution-sha256 3a716471c65e0e2484ac746cf5bc3937970b16934e2d14aef8510580d8c004c2 --authority /home/riomus/runs/sfora-native256-fit-initializer-source-v1/authority-so400-v1.json --authority-sha256 e0a749356c5e11b1e414370c65aadee671e4a363fcaaca7647c580bad98e9af9 --arm so400 --output /home/riomus/runs/sfora-native256-fit-initializer-so400-v1
sha256sum -c <<'HASHES'
c2caa8c4985910dd5496dbf8fc5b1ccbd3a66ee28dfe883c78b5cf91d0cbcd06  initialize_siglip2_substrate_fit.py
7f9eb6a079ef55fc0bc86a60302533df2b7fd77119a2b6d1bfd19fef4607e8d0  test_siglip2_substrate_initializer.py
1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c  representation_ceiling.py
3a716471c65e0e2484ac746cf5bc3937970b16934e2d14aef8510580d8c004c2  execution.json
e0a749356c5e11b1e414370c65aadee671e4a363fcaaca7647c580bad98e9af9  authority-so400-v1.json
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
