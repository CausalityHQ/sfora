import hashlib,importlib.util,json,re,sys
from decimal import Decimal
from pathlib import Path

phase,arm=sys.argv[1:3]
root=Path('/home/riomus/runs/sfora-so400-image-anchor-smooth-ap-train-source-v2')
stem=f'/tmp/sfora-image-anchor-smooth-ap-{phase}-{arm}-v1'
if phase=='cpu':stem='/tmp/sfora-image-anchor-smooth-ap-cpu-v2'
if phase=='train':stem=f'/tmp/sfora-image-anchor-smooth-ap-train-{arm}-179061-v1'
receipt=Path(stem+'-receipt.json');logpath=Path(stem+'-original.log')
r=json.loads(receipt.read_text());log=logpath.read_text()
ap=root/('authority-cpu-v2.json' if phase=='cpu' else
         f'authority-train-{arm}-179061-v1.json' if phase=='train' else
         f'authority-mechanics-{arm}-v1.json')
a=json.loads(ap.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('exact_frozen',root/'train_siglip2_compact_ranking.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
assert r['launch']==a and r['code']==json.loads((root/'execution.json').read_text())
assert r['authority_sha256']==sha(ap)
assert all(sha(root/n)==v for n,v in r['code'].items())
t.check_terminal_record(r,a,phase,arm,179061)
def one(prefix):
 rows=[json.loads(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
 assert len(rows)==1
 return rows[0]
final=one('FINAL_CGROUP ');stop=one('STOP_CGROUP ')
assert final['command_exit_status']==0 and final['invocation_id']==stop['invocation_id']
assert stop['exit_status']=='0' and stop['exit_code']=='exited' and stop['service_result']=='success'
for footer in (final,stop):
 v=footer['values'];assert int(v['memory.max'])==8589934592 and 0<int(v['memory.peak'])<=8589934592
 assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.max','memory.swap.peak'))
 assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
matches=re.findall(r'^Service runtime: (?:(\d+)min )?(\d+(?:\.\d+)?)s$',log,re.M);assert len(matches)==1
minutes,seconds=matches[0];duration=Decimal(minutes or 0)*60+Decimal(seconds)
assert 0<duration<Decimal(t.policy(phase)['seconds']) and 'Exit status: 0' in log
result={'pass':True,'phase':phase,'arm':arm,'service_seconds':float(duration),'invocation_id':stop['invocation_id'],'host_peak_bytes':int(stop['values']['memory.peak']),'rss_kib':r['process_peak_rss_kib'],'unit':('sfora-so400-image-anchor-smooth-ap-cpu-v2' if phase=='cpu' else f'sfora-so400-image-anchor-smooth-ap-{phase}-{arm}-v1'),'quality_read':r['quality_read'],'receipt_sha256':sha(receipt),'log_sha256':sha(logpath)}
Path(stem+'-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
