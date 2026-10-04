"""Root template integration; run only after four original full exports pass."""
import hashlib,importlib.util,json,re,subprocess
from pathlib import Path
from types import SimpleNamespace
root=Path('/tmp/sfora-fullfeature-residual-evaluation-source-v1')
remote='/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('full_score_evaluator',root/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
a=json.loads((root/'authority-full-export-control-179061-v1.json').read_text());a.update(phase='score',arm=None,seed=None,exports={})
original=json.loads((root/'authority-first-selection-score-v1.json').read_text())
wire_by_path={};units=[]
for seed,arm in [(179061,'control'),(179061,'candidate'),(179069,'candidate'),(179069,'control')]:
 stem=f'full-export-{arm}-{seed}-v1';local=Path('/tmp/sfora-fullfeature-residual-'+stem)
 v=json.loads(Path(str(local)+'-verification.json').read_text());r=json.loads(Path(str(local)+'-receipt.json').read_text())
 assert v['pass'] and v['quality_read'] is False and v['arm']==arm and v['seed']==seed
 assert v['receipt_sha256']==sha(str(local)+'-receipt.json') and v['log_sha256']==sha(str(local)+'-original.log')
 assert r['launch']==json.loads((root/f'authority-{stem}.json').read_text()) and v['files']==r['files']
 unit={'both_locks_held':True,'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'unit':'sfora-so400-fullfeature-residual-evaluation-'+stem,'receipt':{'path':r['output']+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+'/'+stem+'.log','sha256':v['log_sha256']}}
 a['exports'][arm+'-'+str(seed)]=unit;units.append(unit)
 for name,h in r['files'].items():wire_by_path[r['output']+'/'+name]=h
assert len(a['exports'])==4 and a['stage']=='full'
e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='score',arm=None,seed=None))
ap=root/'authority-full-selection-score-v1.json';assert not ap.exists();ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
repl={'first-selection-score-v1':'full-selection-score-v1',sha(root/'authority-first-selection-score-v1.json'):sha(ap)}
for key in ('receipt','log'):
 old=original['selected_cpu'][key];new=a['selected_cpu'][key];repl[old['path']]=new['path'];repl[old['sha256']]=new['sha256']
for arm in ('control','candidate'):
 old=original['exports'][arm+'-179061'];new=a['exports'][arm+'-179061']
 for key in ('receipt','log'):repl[old[key]['path']]=new[key]['path'];repl[old[key]['sha256']]=new[key]['sha256']
 repl['/home/riomus/runs/'+old['unit']]='/home/riomus/runs/'+new['unit']
command=(root/'first-selection-score-v1-command.sh').read_text()
for x,y in sorted(repl.items(),key=lambda kv:len(kv[0]),reverse=True):command=command.replace(x,y)
lines=command.splitlines(keepends=True)
for i,line in enumerate(lines):
 if len(line)>=67 and line[64:66]=='  ':
  path=line[66:].strip()
  if path in wire_by_path:lines[i]=wire_by_path[path]+'  '+path+'\n'
command=''.join(lines)
descriptors=[a['selected_cpu']['receipt'],a['selected_cpu']['log'],a['first_selection']['receipt'],a['first_selection']['log']]+[d for ep in a['endpoints'] for d in (ep['checkpoint'],ep['bundle'],ep['launch'],ep['terminal']['receipt'],ep['terminal']['log'])]+[u[k] for u in units for k in ('receipt','log')]+[{'path':p,'sha256':h} for p,h in wire_by_path.items()]
rows=[]
for d in descriptors:
 row=d['sha256']+'  '+d['path'];count=command.count(row+'\n');assert count in (0,2),(d,count)
 if count==0 and row not in rows:rows.append(row)
assert command.count('\nHASHES\n')==2
if rows:command=command.replace('\nHASHES\n','\n'+'\n'.join(rows)+'\nHASHES\n')
for d in descriptors:assert command.count(d['sha256']+'  '+d['path']+'\n')==2
cp=root/'full-selection-score-v1-command.sh';assert not cp.exists();cp.write_text(command)
launch=(root/'first-selection-score-v1-launch.sh').read_text().replace(sha(root/'first-selection-score-v1-command.sh'),sha(cp)).replace('first-selection-score-v1','full-selection-score-v1')
lp=root/'full-selection-score-v1-launch.sh';assert not lp.exists();lp.write_text(launch)
for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
assert 'RuntimeMaxSec=500' in launch and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
print(json.dumps({'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'execution_sha256':sha(root/'execution.json')},indent=2))
