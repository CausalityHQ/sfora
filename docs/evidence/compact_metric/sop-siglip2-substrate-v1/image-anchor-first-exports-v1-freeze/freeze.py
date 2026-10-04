"""Root-owned integration; no execution before original evaluator CPU passes."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
old=Path('/tmp/sfora-smooth-ap-evaluation-source-v1')
new=Path('/tmp/sfora-image-anchor-smooth-ap-evaluation-source-v1')
remote='/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads(Path('/tmp/sfora-image-anchor-smooth-ap-evaluation-first-cpu-v1-verification.json').read_text())
assert v['pass'] and v['quality_read'] is False
assert v['receipt_sha256']==sha('/tmp/sfora-image-anchor-smooth-ap-evaluation-first-cpu-v1-receipt.json')
assert v['log_sha256']==sha('/tmp/sfora-image-anchor-smooth-ap-evaluation-first-cpu-v1-original.log')
s=importlib.util.spec_from_file_location('frozen_evaluator',new/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
base=json.loads((new/'authority-first-cpu-v1.json').read_text())
cpu={'both_locks_held':True,'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'unit':'sfora-so400-image-anchor-smooth-ap-evaluation-first-cpu-v1','receipt':{'path':'/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-evaluation-first-cpu-v1/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+'/first-cpu-v1.log','sha256':v['log_sha256']}}
def strings(before,after,out):
 if isinstance(before,dict) and isinstance(after,dict):
  for key in before.keys() & after.keys():strings(before[key],after[key],out)
 elif isinstance(before,list) and isinstance(after,list):
  for x,y in zip(before,after,strict=True):strings(x,y,out)
 elif isinstance(before,str) and isinstance(after,str) and before!=after:
  if before in out:assert out[before]==after
  out[before]=after
for arm in e.ARMS:
 stem=f'first-export-{arm}-179061-'
 oa=old/f'authority-{stem}v1.json';original=json.loads(oa.read_text())
 a=json.loads(json.dumps(base));a.update(phase='export',arm=arm,seed=179061,selected_cpu=cpu)
 e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='export',arm=arm,seed=179061))
 ap=new/f'authority-{stem}v1.json';encoded=json.dumps(a,indent=2,sort_keys=True)+'\n'
 if ap.exists():assert ap.read_text()==encoded, 'pending source authority changed'
 else:ap.write_text(encoded)
 repl={};strings(original,a,repl)
 repl.update({str('/home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1'):remote,stem+'v1':stem+'v1',f'sfora-so400-smooth-ap-evaluation-{stem}v1':f'sfora-so400-image-anchor-smooth-ap-evaluation-{stem}v1',sha(oa):sha(ap),sha(old/'execution.json'):sha(new/'execution.json')})
 for n in e.FILES:repl[sha(old/n)]=sha(new/n)
 command=(old/f'{stem}v1-command.sh').read_text()
 for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):command=command.replace(x,y)
 # The historical shell delegated CPU receipt/log admission to the native driver.
 # Pin both explicitly before and after this new prospective invocation too.
 cpu_lines=''.join(f"{d['sha256']}  {d['path']}\n" for d in (cpu['receipt'],cpu['log']))
 assert command.count('\nHASHES\n')==2
 assert all(d['sha256']+'  '+d['path'] in command for d in (cpu['receipt'],cpu['log']))
 cp=new/f'{stem}v1-command.sh';cp.write_text(command)
 launch=(old/f'{stem}v1-launch.sh').read_text().replace(sha(old/f'{stem}v1-command.sh'),sha(cp))
 for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):launch=launch.replace(x,y)
 lp=new/f'{stem}v1-launch.sh';lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 assert 'RuntimeMaxSec=600' in launch and '--setenv=CUDA_VISIBLE_DEVICES=0' in launch
 assert 'compact-ranking-train-source-v7' not in command and 'first-cpu-v14' not in command+launch
 for desc in [cpu['receipt'],cpu['log']]+[f for ep in a['endpoints'] for f in (ep['checkpoint'],ep['bundle'],ep['launch'],ep['terminal']['receipt'],ep['terminal']['log'])]:assert desc['sha256']+'  '+desc['path'] in command
 print(json.dumps({'arm':arm,'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp)}))
