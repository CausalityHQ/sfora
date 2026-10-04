import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
old=Path('/tmp/sfora-smooth-ap-train-source-v1')
new=Path('/tmp/sfora-image-anchor-smooth-ap-train-source-v2')
remote='/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('frozen',new/'train_siglip2_compact_ranking.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
base=json.loads((new/'authority-mechanics-control-v1.json').read_text())
units={}
for arm in t.ARMS:
 stem=f'/tmp/sfora-image-anchor-smooth-ap-mechanics-{arm}-v1'
 r=json.loads(Path(stem+'-receipt.json').read_text());v=json.loads(Path(stem+'-verification.json').read_text())
 a=json.loads((new/f'authority-mechanics-{arm}-v1.json').read_text())
 assert v['pass'] and r['launch']==a and v['receipt_sha256']==sha(stem+'-receipt.json') and v['log_sha256']==sha(stem+'-original.log')
 t.check_terminal_record(r,a,'mechanics',arm,179061)
 units[arm]={'both_locks_held':True,'invocation_id':v['invocation_id'],'log':{'path':remote+f'/mechanics-{arm}-v1.log','sha256':v['log_sha256']},'native_peak_rss_kib':r['process_peak_rss_kib'],'receipt':{'path':f'/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-mechanics-{arm}-v1/receipt.json','sha256':v['receipt_sha256']},'service_seconds':v['service_seconds'],'unit':f'sfora-so400-image-anchor-smooth-ap-mechanics-{arm}-v1'}
for arm in t.ARMS:
 a=json.loads(json.dumps(base));a.update(phase='train',arm=arm,resource_policy=t.policy('train'),selected_mechanics=units)
 ap=new/f'authority-train-{arm}-179061-v1.json';assert not ap.exists();ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
 t.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='train',arm=arm,seed=179061))
 oa=old/f'authority-train-{arm}-179061-v1.json';original=json.loads(oa.read_text())
 repl={'/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1':remote,
 f'sfora-so400-smooth-ap-train-{arm}-179061-v1':f'sfora-so400-image-anchor-smooth-ap-train-{arm}-179061-v1',
 'authority-cpu-v1.json':'authority-cpu-v2.json',sha(oa):sha(ap),sha(old/'execution.json'):sha(new/'execution.json'),sha(old/'authority-cpu-v1.json'):sha(new/'authority-cpu-v2.json')}

 for n in t.FILES:repl[sha(old/n)]=sha(new/n)
 for before,after in [(original['selected_cpu'],a['selected_cpu'])]+[(original['selected_mechanics'][k],units[k]) for k in t.ARMS]:
  for key in ('receipt','log'):
   repl[before[key]['path']]=after[key]['path'];repl[before[key]['sha256']]=after[key]['sha256']
 for k in t.ARMS:
  repl[f'authority-mechanics-{k}-v1.json']=f'authority-mechanics-{k}-v1.json'
  repl[sha(old/f'authority-mechanics-{k}-v1.json')]=sha(new/f'authority-mechanics-{k}-v1.json')
 command=(old/f'train-{arm}-179061-v1-command.sh').read_text()
 # Replace absolute original prerequisite paths before their common root.
 for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):command=command.replace(x,y)
 anchor_rows=(new/'cpu-v2-command.sh').read_text().split("sha256sum -c <<'HASHES'\n")[1].split('\nHASHES')[0]
 command=command.replace("set -euo pipefail\n","set -euo pipefail\nsha256sum -c <<'ANCHOR_HASHES'\n"+anchor_rows+"\nANCHOR_HASHES\n",1)+"\nsha256sum -c <<'ANCHOR_HASHES'\n"+anchor_rows+"\nANCHOR_HASHES\n"
 cp=new/f'train-{arm}-179061-v1-command.sh';cp.write_text(command)
 launch=(old/f'train-{arm}-179061-v1-launch.sh').read_text().replace(sha(old/f'train-{arm}-179061-v1-command.sh'),sha(cp))
 for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):launch=launch.replace(x,y)
 lp=new/f'train-{arm}-179061-v1-launch.sh';lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 assert 'RuntimeMaxSec=600' in launch and '--setenv=CUDA_VISIBLE_DEVICES=0' in launch
 assert 'compact-ranking-train-source-v7' not in command+launch and 'mechanics-control-v6' not in command and 'cpu-v8' not in command
 print(json.dumps({'arm':arm,'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp)}))
