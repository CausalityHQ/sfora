"""Root-only template integration after accepted original full CPU."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
root=Path('/tmp/sfora-fullfeature-residual-evaluation-source-v1')
remote='/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads(Path('/tmp/sfora-fullfeature-residual-evaluation-full-cpu-v1-verification.json').read_text())
assert v['pass'] and v['quality_read'] is False
assert sha('/tmp/sfora-fullfeature-residual-evaluation-full-cpu-v1-receipt.json')==v['receipt_sha256']
assert sha('/tmp/sfora-fullfeature-residual-evaluation-full-cpu-v1-original.log')==v['log_sha256']
s=importlib.util.spec_from_file_location('frozen_full_evaluator',root/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
base=json.loads(Path('/tmp/sfora-fullfeature-confirmation-cpu-freeze-v1/authority-full-cpu-v1.json').read_text())
cpu={'both_locks_held':True,'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'unit':v['unit'],'receipt':{'path':'/home/riomus/runs/'+v['unit']+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+'/full-cpu-v1.log','sha256':v['log_sha256']}}
for seed,arm in [(179061,'control'),(179061,'candidate'),(179069,'candidate'),(179069,'control')]:
 oldstem=f'first-export-{arm}-179061-v1';stem=f'full-export-{arm}-{seed}-v1'
 original=json.loads((root/f'authority-{oldstem}.json').read_text())
 a=json.loads(json.dumps(base));a.update(phase='export',arm=arm,seed=seed,selected_cpu=cpu)
 e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='export',arm=arm,seed=seed))
 ap=root/f'authority-{stem}.json';assert not ap.exists();ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
 repl={oldstem:stem,sha(root/f'authority-{oldstem}.json'):sha(ap)}
 for key in ('receipt','log'):
  old=original['selected_cpu'][key];new=cpu[key];repl[old['path']]=new['path'];repl[old['sha256']]=new['sha256']
 command=(root/f'{oldstem}-command.sh').read_text()
 for x,y in sorted(repl.items(),key=lambda kv:len(kv[0]),reverse=True):command=command.replace(x,y)
 # CLI seed is an explicit value; historical source paths stay untouched.
 assert command.count('--seed 179061')==1
 command=command.replace('--seed 179061',f'--seed {seed}')
 assert command.count('\nHASHES\n')==2
 descriptors=[cpu['receipt'],cpu['log']]+[d for ep in a['endpoints'] for d in (ep['checkpoint'],ep['bundle'],ep['launch'],ep['terminal']['receipt'],ep['terminal']['log'])]
 rows=[]
 for d in descriptors:
  row=d['sha256']+'  '+d['path']
  count=command.count(row+'\n');assert count in (0,2),(d,count)
  if not count and row not in rows:rows.append(row)
 if rows:command=command.replace('\nHASHES\n','\n'+'\n'.join(rows)+'\nHASHES\n')
 for d in descriptors:assert command.count(d['sha256']+'  '+d['path']+'\n')==2
 cp=root/f'{stem}-command.sh';assert not cp.exists();cp.write_text(command)
 launch=(root/f'{oldstem}-launch.sh').read_text().replace(sha(root/f'{oldstem}-command.sh'),sha(cp)).replace(oldstem,stem)
 lp=root/f'{stem}-launch.sh';assert not lp.exists();lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 assert 'RuntimeMaxSec=900' in launch and '--setenv=CUDA_VISIBLE_DEVICES=0' in launch
 print(json.dumps({'seed':seed,'arm':arm,'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp)}))
