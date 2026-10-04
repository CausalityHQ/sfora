"""Freeze retained-pair exports only after the original CPUv10 is verified."""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import importlib.util

old=Path('/tmp/sfora-compact-ranking-evaluation-source-v22')
new=Path('/tmp/sfora-compact-ranking-evaluation-source-v23')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
verified=json.loads(Path('/tmp/compact-evaluation-cpu-v10-verification.json').read_text())
assert verified['pass'] and verified['quality_read'] is False
assert verified['invocation_id']=='bf8317d990b74d278934bdc6a12f28e1'
spec=importlib.util.spec_from_file_location('export_freeze',new/'evaluate_siglip2_compact_ranking.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
old_remote='/home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v22'
new_remote=old_remote.replace('v22','v23')
for arm in ('control','candidate'):
    stem='first-export-'+arm+'-179061-'
    original_path=old/('authority-'+stem+'v6.json')
    authority=json.loads(original_path.read_text())
    prior=authority['selected_cpu']
    cpu={**prior,'invocation_id':verified['invocation_id'],
         'service_seconds':verified['service_seconds'],'native_peak_rss_kib':verified['rss_kib'],
         'unit':prior['unit'].replace('v9','v10'),
         'receipt':{'path':prior['receipt']['path'].replace('v9','v10'),'sha256':verified['receipt_sha256']},
         'log':{'path':prior['log']['path'].replace('source-v22','source-v23').replace('v9.log','v10.log'),
                'sha256':verified['log_sha256']}}
    authority['selected_cpu']=cpu;authority['execution_sha256']=sha(new/'execution.json')
    e.check_launch(authority,SimpleNamespace(execution_sha256=authority['execution_sha256'],phase='export',arm=arm,seed=179061))
    path=new/('authority-'+stem+'v7.json');assert not path.exists();path.write_text(json.dumps(authority,indent=2,sort_keys=True)+'\n')
    replacements={old_remote:new_remote,stem+'v6':stem+'v7','first-cpu-v9':'first-cpu-v10',
        sha(original_path):sha(path),sha(old/'execution.json'):sha(new/'execution.json'),
        prior['receipt']['sha256']:cpu['receipt']['sha256'],prior['log']['sha256']:cpu['log']['sha256']}
    for name in ('evaluate_siglip2_compact_ranking.py','test_compact_ranking_evaluation.py'):
        replacements[sha(old/name)]=sha(new/name)
    command=(old/(stem+'v6-command.sh')).read_text()
    for before,after in replacements.items():command=command.replace(before,after)
    command_path=new/(stem+'v7-command.sh');command_path.write_text(command)
    launch=(old/(stem+'v6-launch.sh')).read_text()
    for before,after in {old_remote:new_remote,stem+'v6':stem+'v7',sha(old/(stem+'v6-command.sh')):sha(command_path)}.items():
        assert before in launch;launch=launch.replace(before,after)
    launch_path=new/(stem+'v7-launch.sh');launch_path.write_text(launch)
    for p in (command_path,launch_path):subprocess.run(['bash','-n',str(p)],check=True)
    assert 'RuntimeMaxSec=300' in launch and 'MemoryMax=8589934592' in launch and 'MemorySwapMax=0' in launch
    print(json.dumps({'arm':arm,'authority_sha256':sha(path),'command_sha256':sha(command_path),'launch_sha256':sha(launch_path)}))
