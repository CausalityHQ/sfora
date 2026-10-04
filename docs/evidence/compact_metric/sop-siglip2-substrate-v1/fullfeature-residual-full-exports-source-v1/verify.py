"""Verify original terminal export receipts with the frozen production checker."""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace

arm=sys.argv[1]
seed=int(sys.argv[2]);assert seed in (179061,179069)
assert arm in ('control','candidate')
root=Path('/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1')
stem='full-export-'+arm+'-'+str(seed)+'-v1'
spec=importlib.util.spec_from_file_location('compact_export_verified',root/'evaluate_siglip2_compact_ranking.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
a=json.loads((root/('authority-'+stem+'.json')).read_text());guards={}
rpath=Path('/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-'+stem+'/receipt.json')
r=json.loads(rpath.read_text())
reference=e.load_authenticated('compact_reference_verified',Path(a['reference']['root'])/'evaluate_siglip2_prototype_residual.py',a['reference']['code']['evaluate_siglip2_prototype_residual.py'],guards)
records={}
for endpoint in a['endpoints']:
    rr=e.read_json(endpoint['terminal']['receipt'],guards)
    records[endpoint['seed'],endpoint['arm']]={**rr,'service_seconds':endpoint['terminal']['service_seconds']}
control=records[179061,'control']
context={'args':SimpleNamespace(execution_sha256=a['execution_sha256']),
    'code':json.loads((root/'execution.json').read_text()),'launch':a,'guards':guards,'reference':reference,
    'costs':e.paired_cost(records,'full'),'cpu':e.read_json(a['selected_cpu']['receipt'],guards),
    'training_context':{'source':control['source'],'legacy':{'selected':{'source_cpu':{'numerical_flags':control['numerical_flags']}}}}}
assert r['launch']==a and r['source_code']==context['code'] and r['authority']=={'path':str(root/('authority-'+stem+'.json')),'sha256':hashlib.sha256((root/('authority-'+stem+'.json')).read_bytes()).hexdigest()}
e.check_receipt(context,r,'export',arm,seed)
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in context['code'].items())
for name,digest in r['files'].items():
    assert hashlib.sha256((rpath.parent/name).read_bytes()).hexdigest()==digest
log=(root/(stem+'.log')).read_text()
stops=[json.loads(x.split(' ',1)[1]) for x in log.splitlines() if x.startswith('STOP_CGROUP ')]
assert len(stops)==1
finals=[json.loads(x.split(' ',1)[1]) for x in log.splitlines() if x.startswith('FINAL_CGROUP ')];assert len(finals)==1 and finals[0]['command_exit_status']==0 and finals[0]['invocation_id']==stops[0]['invocation_id']
stop=stops[0];v=stop['values']
assert stop['service_result']=='success' and stop['exit_status']=='0' and stop['exit_code']=='exited'
assert stop['invocation_id']==r['invocation']['invocation_id']
for footer in (finals[0],stop):
    fv=footer['values'];assert 0<int(fv['memory.peak'])<=8589934592 and int(fv['memory.max'])==8589934592
    assert all(int(fv[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
    assert all(int(x.split()[1])==0 for x in fv['memory.events'].splitlines())
assert int(v['memory.peak'])<=8589934592 and int(v['memory.max'])==8589934592
assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
match=re.search(r'^Service runtime: (?:(\d+)min )?(\d+(?:\.\d+)?)s$',log,re.M);assert match
seconds=float(Decimal(match[1] or '0')*60+Decimal(match[2]));assert 0<seconds<900
assert 'Exit status: 0' in log
print(json.dumps({'pass':True,'arm':arm,'seed':seed,'service_seconds':seconds,'host_peak_bytes':int(v['memory.peak']),
    'rss_kib':r['process_peak_rss_kib'],'invocation_id':stop['invocation_id'],
    'receipt_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),
    'log_sha256':hashlib.sha256(log.encode()).hexdigest(),'files':r['files'],'quality_read':r['quality_read']},indent=2))
