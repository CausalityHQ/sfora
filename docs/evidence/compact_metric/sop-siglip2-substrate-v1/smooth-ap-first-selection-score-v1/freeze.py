"""Root integration of frozen first-stage scorer; no new quality reads here."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
old=Path('/tmp/sfora-compact-ranking-evaluation-source-v27')
new=Path('/tmp/sfora-smooth-ap-evaluation-source-v1')
remote='/home/riomus/runs/sfora-so400-smooth-ap-evaluation-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('frozen_eval',new/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
original=json.loads((old/'authority-first-selection-score-v1.json').read_text())
a=json.loads((new/'authority-first-export-control-179061-v1.json').read_text());a.update(phase='score',arm=None,seed=None,exports={})
repl={};
def strings(x,y):
 if isinstance(x,dict) and isinstance(y,dict):
  for k in x.keys() & y.keys():strings(x[k],y[k])
 elif isinstance(x,list) and isinstance(y,list):
  for xx,yy in zip(x,y,strict=True):strings(xx,yy)
 elif isinstance(x,str) and isinstance(y,str) and x!=y:
  if x in repl:assert repl[x]==y
  repl[x]=y
for arm in e.ARMS:
 stem=f'/tmp/sfora-smooth-ap-first-export-{arm}-179061-v1'
 v=json.loads(Path(stem+'-verification.json').read_text());r=json.loads(Path(stem+'-receipt.json').read_text())
 assert v['pass'] and v['quality_read'] is False and v['arm']==arm
 assert v['receipt_sha256']==sha(stem+'-receipt.json') and v['log_sha256']==sha(stem+'-original.log')
 assert r['launch']==json.loads((new/f'authority-first-export-{arm}-179061-v1.json').read_text())
 unit={'both_locks_held':True,'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'unit':f'sfora-so400-smooth-ap-evaluation-first-export-{arm}-179061-v1','receipt':{'path':r['output']+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+f'/first-export-{arm}-179061-v1.log','sha256':v['log_sha256']}}
 a['exports'][arm+'-179061']=unit
 old_receipt=Path(f'/tmp/compact-first-export-{arm}-v10-receipt.json')
 assert sha(old_receipt)==original['exports'][arm+'-179061']['receipt']['sha256']
 old_r=json.loads(old_receipt.read_text())
 assert old_r['files'].keys()==r['files'].keys() and v['files']==r['files']
 repl[old_r['output']]=r['output']
 for name in r['files']:repl[old_r['files'][name]]=r['files'][name]
strings(original,a)
ap=new/'authority-first-selection-score-v1.json';assert not ap.exists()
e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='score',arm=None,seed=None))
ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
repl.update({str('/home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v27'):remote,'sfora-so400-compact-ranking-evaluation-first-selection-score-v1':'sfora-so400-smooth-ap-evaluation-first-selection-score-v1',sha(old/'authority-first-selection-score-v1.json'):sha(ap),sha(old/'execution.json'):sha(new/'execution.json')})
for n in e.FILES:repl[sha(old/n)]=sha(new/n)
command=(old/'first-selection-score-v1-command.sh').read_text()
for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):command=command.replace(x,y)
cp=new/'first-selection-score-v1-command.sh';cp.write_text(command)
launch=(old/'first-selection-score-v1-launch.sh').read_text().replace(sha(old/'first-selection-score-v1-command.sh'),sha(cp))
for x,y in sorted(repl.items(),key=lambda pair:len(pair[0]),reverse=True):launch=launch.replace(x,y)
lp=new/'first-selection-score-v1-launch.sh';lp.write_text(launch)
for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
assert 'RuntimeMaxSec=500' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
assert 'compact-ranking-train-source-v7' not in command and 'export-control-179061-v10' not in command
for u in a['exports'].values():
 for key in ('receipt','log'):assert u[key]['sha256']+'  '+u[key]['path'] in command
print(json.dumps({'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'execution_sha256':sha(new/'execution.json')},indent=2))
