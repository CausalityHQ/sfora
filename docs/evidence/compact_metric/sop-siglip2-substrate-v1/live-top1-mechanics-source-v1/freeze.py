import json,hashlib,re,shlex,importlib.util,subprocess
from pathlib import Path
from types import SimpleNamespace
root=Path('/tmp/sfora-live-top1-train-source-v2');remote='/home/riomus/runs/sfora-so400-live-top1-train-source-v2';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads(Path('/tmp/sfora-live-top1-cpu-v2-verification.json').read_text());assert v['pass'] and not v['quality_read']
s=importlib.util.spec_from_file_location('frozen',root/'train_siglip2_compact_ranking.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
a=json.loads((root/'authority-cpu-v1.json').read_text());a.update(phase='mechanics',resource_policy=t.policy('mechanics'),selected_cpu={'receipt':{'path':'/home/riomus/runs/sfora-so400-live-top1-cpu-v2/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+'/cpu-v2.log','sha256':v['log_sha256']},'unit':v['unit'],'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'both_locks_held':True})
templates=Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/fullfeature-residual-mechanics-source-v1')
for arm in t.ARMS:
 a['arm']=arm;ap=root/f'authority-mechanics-{arm}-v1.json';assert not ap.exists();ap.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n');t.check_launch(a,SimpleNamespace(phase='mechanics',arm=arm,seed=179061,execution_sha256=a['execution_sha256']))
 unit=f'sfora-so400-live-top1-mechanics-{arm}-v1';output='/home/riomus/runs/'+unit
 command=(templates/f'mechanics-{arm}-v1-command.sh').read_text();rows='\n'.join(f'{sha(root/n)}  {remote}/{n}' for n in sorted(t.FILES|{'execution.json',ap.name}));rows+='\n'+v['receipt_sha256']+'  '+a['selected_cpu']['receipt']['path']+'\n'+v['log_sha256']+'  '+a['selected_cpu']['log']['path']
 command=command.replace('\nHASHES\n','\n'+rows+'\nHASHES\n');lines=command.splitlines();indices=[i for i,x in enumerate(lines) if '--phase mechanics --arm '+arm in x];assert len(indices)==1;lines[indices[0]]=shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',*t.cli(remote,remote+'/'+ap.name,sha(ap),a['execution_sha256'],'mechanics',arm,179061,output)])
 cp=root/f'mechanics-{arm}-v1-command.sh';cp.write_text('\n'.join(lines)+'\n');blocks=re.findall(r"sha256sum -c <<'HASHES'\n(.*?)\nHASHES",cp.read_text(),re.S);assert len(blocks)==2 and blocks[0]==blocks[1]
 oldcp=templates/cp.name;launch=(templates/f'mechanics-{arm}-v1-launch.sh').read_text().replace(sha(oldcp),sha(cp)).replace('/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1',remote).replace('sfora-so400-fullfeature-residual-mechanics-'+arm+'-v1',unit);lp=root/f'mechanics-{arm}-v1-launch.sh';lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 print(json.dumps({'arm':arm,'unit':unit,'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'native_launched':False}))
