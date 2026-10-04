"""Root integration only; refuses to freeze before both original TRAIN passes."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
old=Path('/tmp/sfora-smooth-ap-evaluation-source-v1')
new=Path('/tmp/sfora-image-anchor-smooth-ap-evaluation-source-v1')
train=Path('/tmp/sfora-image-anchor-smooth-ap-train-source-v2')
remote='/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=module('frozen_trainer',train/'train_siglip2_compact_ranking.py')
e=module('frozen_evaluator',new/'evaluate_siglip2_compact_ranking.py')
original=json.loads((old/'authority-first-cpu-v1.json').read_text())
a=json.loads(json.dumps(original));a.update(schema=e.AUTHORITY_SCHEMA,execution_sha256=sha(new/'execution.json'),training=e.TRAINING,endpoints=[])
records={};repl={'/home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1':remote,
'sfora-so400-smooth-ap-evaluation-first-cpu-v1':'sfora-so400-image-anchor-smooth-ap-evaluation-first-cpu-v1',sha(old/'execution.json'):sha(new/'execution.json')}

for n in e.FILES:repl[sha(old/n)]=sha(new/n)
for n in t.FILES:repl[original['training']['code'][n]]=e.TRAINING['code'][n]
repl[original['training']['execution_sha256']]=e.TRAINING['execution_sha256']
repl[original['training']['root']]=e.TRAINING['root']
for arm,endpoint in zip(t.ARMS,original['endpoints'],strict=True):
 stem=f'/tmp/sfora-image-anchor-smooth-ap-train-{arm}-179061-v1'
 r=json.loads(Path(stem+'-receipt.json').read_text());v=json.loads(Path(stem+'-verification.json').read_text())
 ap=train/f'authority-train-{arm}-179061-v1.json';launch=json.loads(ap.read_text())
 assert v['pass'] and r['launch']==launch and v['receipt_sha256']==sha(stem+'-receipt.json') and v['log_sha256']==sha(stem+'-original.log')
 t.check_terminal_record(r,launch,'train',arm,179061)
 unit={'unit':f'sfora-so400-image-anchor-smooth-ap-train-{arm}-179061-v1','invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':r['process_peak_rss_kib'],'both_locks_held':True,'receipt':{'path':r['output']+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':e.TRAINING['root']+f'/train-{arm}-179061-v1.log','sha256':v['log_sha256']}}
 fresh={'seed':179061,'arm':arm,'checkpoint':r['checkpoint'],'bundle':r['bundle'],'terminal_state_sha256':r['terminal_state_sha256'],'inference_state_sha256':r['inference_state_sha256'],'launch':{'path':e.TRAINING['root']+'/'+ap.name,'sha256':sha(ap)},'terminal':unit}
 a['endpoints'].append(fresh);records[179061,arm]={**r,'service_seconds':v['service_seconds']}
 for key in ('checkpoint','bundle','launch'):
  for field in ('path','sha256'):repl[endpoint[key][field]]=fresh[key][field]
 for key in ('receipt','log'):
  for field in ('path','sha256'):repl[endpoint['terminal'][key][field]]=unit[key][field]
costs=e.paired_cost(records,'first');assert all(row['pass'] for row in costs.values()),costs
ap=new/'authority-first-cpu-v1.json';assert not ap.exists()
e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='cpu',arm=None,seed=None))
ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');repl[sha(old/'authority-first-cpu-v1.json')]=sha(ap)
command=(old/'first-cpu-v1-command.sh').read_text()
# Match complete old output namespace before generic first-cpu token.
for before,after in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):command=command.replace(before,after)
cp=new/'first-cpu-v1-command.sh';cp.write_text(command)
launch=(old/'first-cpu-v1-launch.sh').read_text().replace(sha(old/'first-cpu-v1-command.sh'),sha(cp))
for before,after in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):launch=launch.replace(before,after)
lp=new/'first-cpu-v1-launch.sh';lp.write_text(launch)
for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
assert 'RuntimeMaxSec=500' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
assert 'compact-ranking-train-source-v7' not in command and 'compact-ranking-train-control-179061-v3' not in command
print(json.dumps({'root':remote,'execution_sha256':sha(new/'execution.json'),'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'costs':costs},indent=2))
