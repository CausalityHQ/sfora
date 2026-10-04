"""Root matched evaluator integration; actual accepted pair required; no launch."""
import json,hashlib,re,shlex,importlib.util,subprocess
from pathlib import Path
from types import SimpleNamespace
root=Path('/tmp/sfora-live-top1-evaluation-source-v1');remote='/home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1';tr='/home/riomus/runs/sfora-so400-live-top1-train-source-v2';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_eval',root/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
templates=Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/fullfeature-residual-evaluation-cpu-source-v1')
a=json.loads((templates/'authority-first-cpu-v1.json').read_text());a.update(schema=e.AUTHORITY_SCHEMA,execution_sha256=sha(root/'execution.json'),training=e.TRAINING);endpoints=[];rows=[]
for arm in e.ARMS:
 v=json.loads(Path(f'/tmp/sfora-live-top1-train-{arm}-179061-v1-verification.json').read_text());r=json.loads(Path(f'/tmp/sfora-live-top1-train-{arm}-179061-v1-receipt.json').read_text());assert v['pass'] and not v['quality_read'] and r['completed_step']==128
 run='/home/riomus/runs/'+v['unit'];launch=tr+f'/authority-train-{arm}-179061-v1.json'
 u={'receipt':{'path':run+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':tr+f'/train-{arm}-179061-v1.log','sha256':v['log_sha256']},'unit':v['unit'],'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'both_locks_held':True}
 endpoint={'seed':179061,'arm':arm,'terminal':u,'launch':{'path':launch,'sha256':sha(Path('/tmp/sfora-live-top1-train-source-v2')/Path(launch).name)},'checkpoint':r['checkpoint'],'bundle':{'path':run+'/bundle/bundle.json','sha256':r['bundle']['sha256']},'terminal_state_sha256':r['terminal_state_sha256'],'inference_state_sha256':r['inference_state_sha256']};endpoints.append(endpoint)
 for fact in [u['receipt'],u['log'],endpoint['launch'],endpoint['checkpoint'],endpoint['bundle']]:rows.append(fact['sha256']+'  '+fact['path'])
controls=json.loads(Path('/tmp/sfora-live-top1-train-control-179061-v1-receipt.json').read_text());candidate=json.loads(Path('/tmp/sfora-live-top1-train-candidate-179061-v1-receipt.json').read_text())
cv=json.loads(Path('/tmp/sfora-live-top1-train-control-179061-v1-verification.json').read_text());av=json.loads(Path('/tmp/sfora-live-top1-train-candidate-179061-v1-verification.json').read_text())
assert av['service_seconds']/cv['service_seconds']<=1.50 and candidate['total_training_core_seconds']/controls['total_training_core_seconds']<=1.50
a['endpoints']=endpoints;ap=root/'authority-first-cpu-v1.json';assert not ap.exists();ap.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n');e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='cpu',arm=None,seed=None))
for name in sorted(e.FILES|{'execution.json',ap.name}):rows.append(sha(root/name)+'  '+remote+'/'+name)
for name in ('train_siglip2_compact_ranking.py','test_siglip2_compact_ranking.py','execution.json'):rows.append(sha(Path('/tmp/sfora-live-top1-train-source-v2')/name)+'  '+tr+'/'+name)
command=(templates/'first-cpu-v1-command.sh').read_text().replace('\nHASHES\n','\n'+'\n'.join(rows)+'\nHASHES\n');lines=command.splitlines();indices=[i for i,l in enumerate(lines) if '--phase cpu --output ' in l];assert len(indices)==1
lines[indices[0]]=shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',remote+'/evaluate_siglip2_compact_ranking.py','--execution-sha256',a['execution_sha256'],'--authority',remote+'/'+ap.name,'--authority-sha256',sha(ap),'--phase','cpu','--output','/home/riomus/runs/sfora-so400-live-top1-evaluation-first-cpu-v1'])
cp=root/'first-cpu-v1-command.sh';cp.write_text('\n'.join(lines)+'\n')
blocks=re.findall(r"sha256sum -c <<'HASHES'\n(.*?)\nHASHES",cp.read_text(),re.S);assert len(blocks)==2 and blocks[0]==blocks[1]
launch=(templates/'first-cpu-v1-launch.sh').read_text().replace(sha(templates/cp.name),sha(cp)).replace('/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1',remote).replace('sfora-so400-fullfeature-residual-evaluation-first-cpu-v1','sfora-so400-live-top1-evaluation-first-cpu-v1');lp=root/'first-cpu-v1-launch.sh';lp.write_text(launch)
for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
print(json.dumps({'execution_sha256':a['execution_sha256'],'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'native_launched':False}))
