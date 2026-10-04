import importlib.util,json,hashlib,re
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
root=Path('/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1')
spec=importlib.util.spec_from_file_location('compact_verified',root/'evaluate_siglip2_compact_ranking.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
r=json.loads(Path('/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-full-cpu-v1/receipt.json').read_text());a=json.loads((root/'authority-full-cpu-v1.json').read_text());guards={};assert r['launch']==a
reference=e.load_authenticated('compact_reference_verified',Path(a['reference']['root'])/'evaluate_siglip2_prototype_residual.py',a['reference']['code']['evaluate_siglip2_prototype_residual.py'],guards)
records={}
for endpoint in a['endpoints']:
 rr=e.read_json(endpoint['terminal']['receipt'],guards);records[endpoint['seed'],endpoint['arm']]={**rr,'service_seconds':endpoint['terminal']['service_seconds']}
control=records[179061,'control']
context={'args':SimpleNamespace(execution_sha256=a['execution_sha256']),'code':json.loads((root/'execution.json').read_text()),'launch':a,'guards':guards,'reference':reference,'costs':e.paired_cost(records,'full'),'training_context':{'source':control['source'],'legacy':{'selected':{'source_cpu':{'numerical_flags':control['numerical_flags']}}}}}
e.check_receipt(context,r,'cpu');assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in context['code'].items());assert r['source_code']==context['code'] and r['execution_sha256']==a['execution_sha256']
log=(root/'full-cpu-v1.log').read_text();stops=[json.loads(x.split(' ',1)[1]) for x in log.splitlines() if x.startswith('STOP_CGROUP ')]
assert len(stops)==1
stop=stops[0];v=stop['values'];assert stop['service_result']=='success' and stop['exit_status']=='0' and stop['exit_code']=='exited'
finals=[json.loads(x.split(' ',1)[1]) for x in log.splitlines() if x.startswith('FINAL_CGROUP ')];assert len(finals)==1 and finals[0]['command_exit_status']==0 and finals[0]['invocation_id']==stop['invocation_id']
assert stop['invocation_id']==r['invocation']['invocation_id']
for footer in (finals[0],stop):
 fv=footer['values'];assert 0<int(fv['memory.peak'])<=8589934592 and int(fv['memory.max'])==8589934592
 assert all(int(fv[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
 assert all(int(x.split()[1])==0 for x in fv['memory.events'].splitlines())
assert int(v['memory.peak'])<=8589934592 and int(v['memory.max'])==8589934592
assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
match=re.search(r'^Service runtime: (?:(\d+)min )?(\d+(?:\.\d+)?)s$',log,re.M);assert match
seconds=float(Decimal(match[1] or '0')*60+Decimal(match[2]));assert 0<seconds<500
assert 'Exit status: 0' in log
print(json.dumps({'unit':'sfora-so400-fullfeature-residual-evaluation-full-cpu-v1','pass':True,'service_seconds':seconds,'host_peak_bytes':int(v['memory.peak']),'rss_kib':r['process_peak_rss_kib'],'invocation_id':stop['invocation_id'],'quality_read':r['quality_read'],'receipt_sha256':hashlib.sha256(Path(r['output'],'receipt.json').read_bytes()).hexdigest(),'log_sha256':hashlib.sha256(log.encode()).hexdigest()},indent=2))
