"""Root-owned prospective freeze; run only after verified child integration."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

old = Path('/tmp/sfora-compact-ranking-evaluation-source-v26')
new = Path('/tmp/sfora-compact-ranking-evaluation-source-v27')
old_remote = '/home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v26'
new_remote = old_remote.replace('v26', 'v27')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not new.exists(), 'prospective source already exists; inspect, never overwrite'
new.mkdir()
names = ('evaluate_siglip2_compact_ranking.py', 'test_compact_ranking_evaluation.py')
for name in names:
    shutil.copyfile(Path('scripts') / name, new / name)
code = {name: sha(new / name) for name in names}
(new / 'execution.json').write_text(json.dumps(code, indent=2, sort_keys=True) + '\n')
old_authority = old / 'authority-first-cpu-v13.json'
authority = json.loads(old_authority.read_text())
original = json.loads(json.dumps(authority))
authority['execution_sha256'] = sha(new / 'execution.json')
authority['resource_policies']['export']['seconds'] = 600
original['resource_policies']['export']['seconds'] = 600
assert {k:v for k,v in authority.items() if k != 'execution_sha256'} == {
    k:v for k,v in original.items() if k != 'execution_sha256'}
authority_path = new / 'authority-first-cpu-v14.json'
import importlib.util
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('new_authority_check',new/names[0]);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
e.check_launch(authority,SimpleNamespace(execution_sha256=authority['execution_sha256'],phase='cpu',arm=None,seed=None))
authority_path.write_text(json.dumps(authority, indent=2, sort_keys=True) + '\n')
changes = {
    old_remote:new_remote,
    'first-cpu-v13':'first-cpu-v14',
    sha(old / 'execution.json'):sha(new / 'execution.json'),
    sha(old_authority):sha(authority_path),
    **{sha(old / name):code[name] for name in names},
}
command = (old / 'first-cpu-v13-command.sh').read_text()
for before, after in changes.items():
    assert before in command, before
    command = command.replace(before, after)
command_path = new / 'first-cpu-v14-command.sh'
command_path.write_text(command)
launch = (old / 'first-cpu-v13-launch.sh').read_text()
for before, after in {old_remote:new_remote, 'first-cpu-v13':'first-cpu-v14',
                     sha(old / 'first-cpu-v13-command.sh'):sha(command_path)}.items():
    assert before in launch, before
    launch = launch.replace(before, after)
launch_path = new / 'first-cpu-v14-launch.sh'
launch_path.write_text(launch)
for path in (command_path, launch_path):
    subprocess.run(['bash', '-n', str(path)], check=True)
assert 'RuntimeMaxSec=500' in launch and 'MemoryMax=8589934592' in launch
assert 'MemorySwapMax=0' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
print(json.dumps({'root':str(new), 'execution_sha256':sha(new/'execution.json'),
                  'authority_sha256':sha(authority_path), 'command_sha256':sha(command_path),
                  'launch_sha256':sha(launch_path), 'code':code}, indent=2))
