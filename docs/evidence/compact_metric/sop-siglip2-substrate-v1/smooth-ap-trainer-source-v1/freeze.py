"""Root integration freezer: run only after verified trainer commit integration."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

repo=Path('/home/rb/worktrees/sfora-positive-causality')
old=Path('/tmp/sfora-compact-ranking-train-source-v7')
new=Path('/tmp/sfora-smooth-ap-train-source-v1')
remote='/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads(Path('/tmp/sfora-smooth-ap-compact-trainer-contract.json').read_text())
assert not new.exists()
spec=importlib.util.spec_from_file_location('smooth_ap_freeze',repo/'scripts/train_siglip2_compact_ranking.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
assert t.SCHEMA=='siglip2-compact-smooth-ap-v1' and t.AUTHORITY_SCHEMA==contract['schemas']['AUTHORITY_SCHEMA']
assert json.loads(json.dumps(t.RECIPE))==contract['RECIPE']
assert t.policy('cpu')['seconds']==500 and t.policy('mechanics')['seconds']==600 and t.policy('train')['seconds']==600
new.mkdir()
for n in sorted(t.FILES):shutil.copyfile(repo/'scripts'/n,new/n)
code={n:sha(new/n) for n in sorted(t.FILES)}
(new/'execution.json').write_text(json.dumps(code,indent=2,sort_keys=True)+'\n')
a=json.loads((old/'authority-cpu-v7.json').read_text())
a.update(schema=t.AUTHORITY_SCHEMA,execution_sha256=sha(new/'execution.json'),recipe=t.RECIPE)
ap=new/'authority-cpu-v1.json';ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
a=json.loads(ap.read_text())
t.check_launch(a,SimpleNamespace(execution_sha256=sha(new/'execution.json'),phase='cpu',arm='control',seed=179061))
replace={str('/home/riomus/runs/sfora-so400-compact-ranking-train-source-v7'):remote,
 'sfora-so400-compact-ranking-cpu-v8':'sfora-so400-smooth-ap-cpu-v1',
 'authority-cpu-v7.json':'authority-cpu-v1.json','cpu-v8-command.sh':'cpu-v1-command.sh',
 sha(old/'execution.json'):sha(new/'execution.json'),sha(old/'authority-cpu-v7.json'):sha(ap)}
for n in t.FILES:replace[sha(old/n)]=sha(new/n)
command=(old/'cpu-v8-command.sh').read_text()
for x,y in replace.items():command=command.replace(x,y)
cp=new/'cpu-v1-command.sh';cp.write_text(command)
launch=(old/'cpu-v8-launch.sh').read_text().replace(sha(old/'cpu-v8-command.sh'),sha(cp))
for x,y in replace.items():launch=launch.replace(x,y)
lp=new/'cpu-v1-launch.sh';lp.write_text(launch)
for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
assert 'RuntimeMaxSec=500' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
assert 'compact-ranking-train-source-v7' not in command+launch
print(json.dumps({'root':remote,'code':code,'execution_sha256':sha(new/'execution.json'),
 'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp)},indent=2))
