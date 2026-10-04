import hashlib,importlib.util,json,subprocess
from pathlib import Path
from types import SimpleNamespace
old=Path('/tmp/sfora-compact-ranking-train-source-v7')
new=Path('/tmp/sfora-smooth-ap-train-source-v1')
remote='/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('frozen',new/'train_siglip2_compact_ranking.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
r=json.loads(Path('/tmp/sfora-smooth-ap-cpu-v1-receipt.json').read_text())
v=json.loads(Path('/tmp/sfora-smooth-ap-cpu-v1-verification.json').read_text())
cpu=json.loads((new/'authority-cpu-v1.json').read_text())
t.check_terminal_record(r,cpu,'cpu','control',179061)
assert v['pass'] and v['receipt_sha256']==sha('/tmp/sfora-smooth-ap-cpu-v1-receipt.json') and v['log_sha256']==sha('/tmp/sfora-smooth-ap-cpu-v1-original.log')
for arm in ('control','candidate'):
 a=json.loads(json.dumps(cpu));a.update(phase='mechanics',arm=arm,resource_policy=t.policy('mechanics'))
 a['selected_cpu']={'both_locks_held':True,'invocation_id':v['invocation_id'],'log':{'path':remote+'/cpu-v1.log','sha256':v['log_sha256']},'native_peak_rss_kib':r['process_peak_rss_kib'],'receipt':{'path':'/home/riomus/runs/sfora-so400-smooth-ap-cpu-v1/receipt.json','sha256':v['receipt_sha256']},'service_seconds':v['service_seconds'],'unit':v['unit']}
 ap=new/f'authority-mechanics-{arm}-v1.json';assert not ap.exists();ap.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
 t.check_launch(a,SimpleNamespace(execution_sha256=a['execution_sha256'],phase='mechanics',arm=arm,seed=179061))
 oa=old/f'authority-mechanics-{arm}-v6.json';oc=json.loads(oa.read_text())
 replacements={str('/home/riomus/runs/sfora-so400-compact-ranking-train-source-v7'):remote,f'sfora-so400-compact-ranking-mechanics-{arm}-v6':f'sfora-so400-smooth-ap-mechanics-{arm}-v1',f'authority-mechanics-{arm}-v6.json':ap.name,f'mechanics-{arm}-v6-command.sh':f'mechanics-{arm}-v1-command.sh','authority-cpu-v7.json':'authority-cpu-v1.json','sfora-so400-compact-ranking-cpu-v8':'sfora-so400-smooth-ap-cpu-v1','cpu-v8.log':'cpu-v1.log',sha(oa):sha(ap),sha(old/'execution.json'):sha(new/'execution.json'),sha(old/'authority-cpu-v7.json'):sha(new/'authority-cpu-v1.json'),oc['selected_cpu']['receipt']['sha256']:v['receipt_sha256'],oc['selected_cpu']['log']['sha256']:v['log_sha256']}
 for k in t.FILES:replacements[sha(old/k)]=sha(new/k)
 command=(old/f'mechanics-{arm}-v6-command.sh').read_text()
 for x,y in replacements.items():command=command.replace(x,y)
 cp=new/f'mechanics-{arm}-v1-command.sh';cp.write_text(command)
 launch=(old/f'mechanics-{arm}-v6-launch.sh').read_text().replace(sha(old/f'mechanics-{arm}-v6-command.sh'),sha(cp))
 for x,y in replacements.items():launch=launch.replace(x,y)
 lp=new/f'mechanics-{arm}-v1-launch.sh';lp.write_text(launch)
 for p in (cp,lp):subprocess.run(['bash','-n',str(p)],check=True)
 assert 'RuntimeMaxSec=600' in launch and '--setenv=CUDA_VISIBLE_DEVICES=0' in launch
 assert 'compact-ranking-train-source-v7' not in command+launch and 'cpu-v8' not in command+launch
 print(json.dumps({'arm':arm,'authority_sha256':sha(ap),'command_sha256':sha(cp),'launch_sha256':sha(lp)}))
