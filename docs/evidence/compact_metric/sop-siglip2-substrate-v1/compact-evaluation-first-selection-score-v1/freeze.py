"""Root-owned score freeze; execute only after both original exports pass."""
import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

root=Path('/tmp/sfora-compact-ranking-evaluation-source-v27')
remote='/home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v27'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
authority=json.loads((root/'authority-first-export-control-179061-v10.json').read_text())
authority.update(phase='score',arm=None,seed=None)
exports={};extra=[]
for arm in ('control','candidate'):
    verified=json.loads(Path('/tmp/compact-evaluation-export-'+arm+'-v10-verification.json').read_text())
    assert verified['pass'] and verified['arm']==arm
    receipt=Path('/tmp/compact-first-export-'+arm+'-v10-receipt.json')
    log=Path('/tmp/compact-first-export-'+arm+'-v10-original.log')
    assert sha(receipt)==verified['receipt_sha256'] and sha(log)==verified['log_sha256']
    record=json.loads(receipt.read_text());assert record['quality_read'] is False
    unit='sfora-so400-compact-ranking-evaluation-first-export-'+arm+'-179061-v10'
    descriptor=dict(authority['selected_cpu'])
    descriptor.update(unit=unit,invocation_id=verified['invocation_id'],service_seconds=verified['service_seconds'],
        native_peak_rss_kib=verified['rss_kib'],
        receipt={'path':record['output']+'/receipt.json','sha256':verified['receipt_sha256']},
        log={'path':remote+'/first-export-'+arm+'-179061-v10.log','sha256':verified['log_sha256']})
    exports[arm+'-179061']=descriptor
    extra.extend((d['sha256']+'  '+d['path']) for d in (descriptor['receipt'],descriptor['log']))
    extra.extend(digest+'  '+record['output']+'/'+name for name,digest in verified['files'].items())
authority['exports']=exports
spec=importlib.util.spec_from_file_location('score_freeze',root/'evaluate_siglip2_compact_ranking.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
e.check_launch(authority,SimpleNamespace(execution_sha256=sha(root/'execution.json'),phase='score',arm=None,seed=None))
stem='first-selection-score-v1';path=root/('authority-'+stem+'.json');assert not path.exists()
path.write_text(json.dumps(authority,indent=2,sort_keys=True)+'\n')
old=root/'authority-first-cpu-v14.json'
command=(root/'first-cpu-v14-command.sh').read_text().replace('first-cpu-v14',stem).replace(sha(old),sha(path)).replace('--phase cpu','--phase score')
command=command.replace('\nHASHES\n','\n'+'\n'.join(extra)+'\nHASHES\n')
command_path=root/(stem+'-command.sh');command_path.write_text(command)
launch=(root/'first-cpu-v14-launch.sh').read_text().replace('first-cpu-v14',stem).replace(sha(root/'first-cpu-v14-command.sh'),sha(command_path))
launch_path=root/(stem+'-launch.sh');launch_path.write_text(launch)
for p in (command_path,launch_path):subprocess.run(['bash','-n',str(p)],check=True)
assert 'RuntimeMaxSec=500' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
print(json.dumps({'authority_sha256':sha(path),'command_sha256':sha(command_path),'launch_sha256':sha(launch_path)},indent=2))
