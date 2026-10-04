"""Root integration only: freeze actual accepted trainer bytes; never launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

repo = Path('/home/rb/worktrees/sfora-positive-causality')
preparation = Path('/tmp/sfora-live-top1-cpu-freeze-preparation.json')
assert len(sys.argv) == 2, 'final verified contract path required'
contract = json.loads(Path(sys.argv[1]).read_text())
p = json.loads(preparation.read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(preparation) == 'dd96140a4d6b86b075d7c3eb558f1fc82f367ea9652b77d9ab3e34556d1c6b45'
assert all(sha(v['path']) == v['sha256'] for v in p['templates'].values())
new = Path('/tmp/sfora-live-top1-train-source-v1')
assert not new.exists(), 'exclusive new source directory required'
spec = importlib.util.spec_from_file_location('live_top1_freeze', repo/'scripts/train_siglip2_compact_ranking.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
assert t.SCHEMA == 'siglip2-compact-live-top1-v1'
assert t.RECIPE['trainable_names'] == {'control':['A','C'], 'candidate':['A','C']}
assert t.RECIPE['updates'] == 128 and t.RECIPE['batch'] == 64 and t.RECIPE['microbatch'] == 16
assert all(t.policy(phase)['seconds'] == seconds and t.policy(phase)['host_bytes'] == 8589934592
           and t.policy(phase)['swap_bytes'] == 0 for phase,seconds in [('cpu',500),('mechanics',600),('train',600)])
code = {n:sha(repo/'scripts'/n) for n in sorted(t.FILES)}
assert code == contract['code'], 'verified final contract/source mismatch'
new.mkdir()
for n in sorted(t.FILES): shutil.copyfile(repo/'scripts'/n,new/n)
(new/'execution.json').write_text(json.dumps(code,sort_keys=True,indent=2)+'\n')
execution = sha(new/'execution.json')
assert execution == contract['execution']['sha256'], 'final execution manifest bytes differ'
a = json.loads(Path(p['templates']['authority-cpu-v1.json']['path']).read_text())
a.update(schema=t.AUTHORITY_SCHEMA, execution_sha256=execution, recipe=t.RECIPE)
ap = new/'authority-cpu-v1.json'
ap.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n')
t.check_launch(a,SimpleNamespace(execution_sha256=execution,phase='cpu',arm='control',seed=179061))
remote = p['new_source_root']
output = '/home/riomus/runs/'+p['new_unit']
rows = '\n'.join(f'{sha(new/n)}  {remote}/{n}' for n in sorted(t.FILES|{'execution.json','authority-cpu-v1.json'}))
command = Path(p['templates']['cpu-v1-command.sh']['path']).read_text()
assert command.count('\nHASHES\n') == 2
command = command.replace('\nHASHES\n','\n'+rows+'\nHASHES\n')
lines = command.splitlines()
indices = [i for i,line in enumerate(lines) if '--phase cpu --arm control --seed 179061 --output ' in line]
assert len(indices) == 1
lines[indices[0]] = shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',
    *t.cli(remote,remote+'/authority-cpu-v1.json',sha(ap),execution,'cpu','control',179061,output)])
cp = new/'cpu-v1-command.sh'
cp.write_text('\n'.join(lines)+'\n')
blocks = re.findall(r"sha256sum -c <<'HASHES'\n(.*?)\nHASHES",cp.read_text(),re.S)
assert len(blocks) == 2 and blocks[0] == blocks[1]
guards = dict((path,digest) for digest,path in (line.split(None,1) for line in blocks[0].splitlines()))
assert all(guards[path] == digest for path,digest in p['original_prepost_guards'].items())
launch = Path(p['templates']['cpu-v1-launch.sh']['path']).read_text()
launch = launch.replace(p['templates']['cpu-v1-command.sh']['sha256'],sha(cp))
launch = launch.replace(p['old_oracle_root'],remote).replace('sfora-so400-fullfeature-residual-cpu-v1',p['new_unit'])
lp = new/'cpu-v1-launch.sh'
lp.write_text(launch)
assert 'RuntimeMaxSec=500' in launch and 'MemoryMax=8589934592' in launch and 'MemorySwapMax=0' in launch
for f in (cp,lp): subprocess.run(['bash','-n',str(f)],check=True)
print(json.dumps({'root':remote,'code':code,'execution_sha256':execution,
    'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),
    'original_guards_retained':len(p['original_prepost_guards']), 'native_launched':False},sort_keys=True,indent=2))
