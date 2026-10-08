import hashlib,importlib.util,json,shlex,sys
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parent
f=p/'freeze.json'
assert len(sys.argv)==2 and hashlib.sha256(f.read_bytes()).hexdigest()==sys.argv[1]
j=json.loads(f.read_text())
for n,h in j['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
assert str(p)==j['source_root'] and not Path(j['output']).exists()
s=importlib.util.spec_from_file_location('frozen_preflight',p/'train_siglip2_connected_probe.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=json.loads((p/'authority-cpu-v2.json').read_text());m.check_launch(a,SimpleNamespace(execution_sha256=j['execution_sha256'],phase='cpu',arm='control',seed=179061))
lines=(p/'cpu-v2-command.sh').read_text().splitlines();tables=[];table=None
for line in lines:
 if line=="sha256sum -c <<'HASHES'":table=[]
 elif line=='HASHES':tables.append(table);table=None
 elif table is not None:table.append(line.split())
assert len(tables)==2 and tables[0]==tables[1]
for digest,path in tables[0]:
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while block:=stream.read(1024*1024):h.update(block)
 assert h.hexdigest()==digest,path
cli=[shlex.split(line) for line in lines if line.startswith('/home/riomus/group-learning/.venv/bin/python -B ')]
assert len(cli)==1 and cli[0][2:]==m.cli(p,p/'authority-cpu-v2.json',j['authority_sha256'],j['execution_sha256'],'cpu','control',179061,j['output'])
assert [m.policy(v)['seconds'] for v in ('cpu','mechanics','train')]==[600,1200,3000]
assert not {'torch','numpy','transformers','PIL'} & sys.modules.keys()
print('PASS full freeze + BOTH command hash tables/current bytes/exact actual CLI/policy/outputabsent/no_native_imports')
