#!/bin/bash
set -euo pipefail
cd /home/riomus/runs/sfora-native256-fit-export-source-v1
sha256sum -c <<'HASHES'
163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8  export_siglip2_substrate_fit.py
d16ee66a830a1b12a8567686bad36dd65c2ef27e396f630fc14e608e53a3af44  test_siglip2_substrate_fit.py
ad32df859712e3d608341ed9eda24fd96db375ec85c61779265a9c16cb8dd3b5  execution.json
065b32e841e8a05bc71aa36e5c9b32db07b139ba7159f428dc21eba23a57361c  authority.json
HASHES
/home/riomus/group-learning/.venv/bin/python -B /home/riomus/runs/sfora-native256-fit-export-source-v1/export_siglip2_substrate_fit.py --execution-sha256 ad32df859712e3d608341ed9eda24fd96db375ec85c61779265a9c16cb8dd3b5 --authority /home/riomus/runs/sfora-native256-fit-export-source-v1/authority.json --authority-sha256 065b32e841e8a05bc71aa36e5c9b32db07b139ba7159f428dc21eba23a57361c --arm so400 --startup /home/riomus/runs/sfora-native256-fit-export-source-v1/startup-so400-v1.json --startup-sha256 f6809cca2d4219c412e07673fec44d4da4ca3f2210df93d9cdbbd623bd4b1551 --output /home/riomus/runs/sfora-native256-fit-export-so400-v1
sha256sum -c <<'HASHES'
163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8  export_siglip2_substrate_fit.py
d16ee66a830a1b12a8567686bad36dd65c2ef27e396f630fc14e608e53a3af44  test_siglip2_substrate_fit.py
ad32df859712e3d608341ed9eda24fd96db375ec85c61779265a9c16cb8dd3b5  execution.json
065b32e841e8a05bc71aa36e5c9b32db07b139ba7159f428dc21eba23a57361c  authority.json
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
