import json,hashlib,re,shlex,subprocess,importlib.util
from pathlib import Path
from types import SimpleNamespace
root=Path('/tmp/sfora-live-top1-evaluation-source-v1');remote='/home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1';old='/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1'
templates=Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/fullfeature-residual-export-source-v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_export',root/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
v=json.loads(Path('/tmp/sfora-live-top1-evaluation-first-cpu-v1-verification.json').read_text());assert v['pass'] and not v['quality_read']
cpu={'receipt':{'path':'/home/riomus/runs/'+v['unit']+'/receipt.json','sha256':v['receipt_sha256']},'log':{'path':remote+'/first-cpu-v1.log','sha256':v['log_sha256']},'unit':v['unit'],'invocation_id':v['invocation_id'],'service_seconds':v['service_seconds'],'native_peak_rss_kib':v['rss_kib'],'both_locks_held':True}
base=json.loads((root/'authority-first-cpu-v1.json').read_text());first=(root/'first-cpu-v1-command.sh').read_text();blocks=re.findall(r"sha256sum -c <<'HASHES'\n(.*?)\nHASHES",first,re.S);assert len(blocks)==2 and blocks[0]==blocks[1];base_rows=blocks[0].splitlines();facts={}
for arm in e.ARMS:
 stem='first-export-'+arm+'-179061-v1';a=dict(base,phase='export',seed=179061,arm=arm,selected_cpu=cpu);ap=root/('authority-'+stem+'.json');assert not ap.exists();ap.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n');e.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='export',seed=179061,arm=arm))
 rows=base_rows+[sha(ap)+'  '+remote+'/'+ap.name]+[f['sha256']+'  '+f['path'] for f in (cpu['receipt'],cpu['log'])]
 command=(templates/(stem+'-command.sh')).read_text();command=re.sub(r"(sha256sum -c <<'HASHES'\n).*?(\nHASHES)",lambda m:m[1]+'\n'.join(rows)+m[2],command,flags=re.S)
 lines=command.splitlines();indices=[i for i,l in enumerate(lines) if '--phase export --seed' in l];assert len(indices)==1
 unit='sfora-so400-live-top1-evaluation-'+stem
 lines[indices[0]]=shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',remote+'/evaluate_siglip2_compact_ranking.py','--execution-sha256',a['execution_sha256'],'--authority',remote+'/'+ap.name,'--authority-sha256',sha(ap),'--phase','export','--seed','179061','--arm',arm,'--output','/home/riomus/runs/'+unit])
 cp=root/(stem+'-command.sh');cp.write_text('\n'.join(lines)+'\n');lp=root/(stem+'-launch.sh');launch=(templates/lp.name).read_text().replace(sha(templates/cp.name),sha(cp)).replace(old,remote).replace('sfora-so400-fullfeature-residual-evaluation-'+stem,unit);lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 assert 'RuntimeMaxSec=900' in launch and 'MemoryMax=8589934592' in launch and 'MemorySwapMax=0' in launch
 (root/(stem+'-preflight.sha256')).write_text('\n'.join(dict.fromkeys(rows))+'\n');facts[arm]={'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp),'unit':unit,'native_launched':False}
print(json.dumps(facts,indent=2))
