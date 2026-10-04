"""Root terminal verification; no scoring/model execution or state reuse."""
from decimal import Decimal
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

root=Path('/home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1')
authority_path=root/'authority-first-selection-score-v1.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(sys.argv)==2 and re.fullmatch('[0-9a-f]{64}',sys.argv[1]);assert sha(authority_path)==sys.argv[1]
assert sha(root/'evaluate_siglip2_compact_ranking.py')=='8ecaa206d0343dde117cec97fa92a0179cb8a5991127d4c18dc7917a7c535563'
assert sha(root/'execution.json')=='804bde987374b1416e8cedaaf1a9a44d888e4072a26d1af86a788bdae17ef072'
spec=importlib.util.spec_from_file_location('compact_score_verified',root/'evaluate_siglip2_compact_ranking.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
a=json.loads(authority_path.read_text());guards={}
rpath=Path('/home/riomus/runs/sfora-so400-live-top1-evaluation-first-selection-score-v1/receipt.json')
r=json.loads(rpath.read_text())
assert r['launch']==a and r['authority']=={'path':str(authority_path),'sha256':sha(authority_path)}
loaded={}
for key,name in [('reference','evaluate_siglip2_prototype_residual.py'),('nearest_evaluator','evaluate_siglip2_nearest_ranking.py'),('genuine_evaluator','evaluate_siglip2_genuine_views.py')]:
    d=a[key];loaded[key]=e.load_authenticated('score_verified_'+key,Path(d['root'])/name,d['code'][name],guards)
reference=loaded['reference'];native=loaded['nearest_evaluator'];math_helper=loaded['genuine_evaluator']
records={}
for endpoint in a['endpoints']:
    rr=e.read_json(endpoint['terminal']['receipt'],guards)
    records[endpoint['seed'],endpoint['arm']]={**rr,'service_seconds':endpoint['terminal']['service_seconds']}
control=records[179061,'control']
source=e.read_json(reference.SOURCE_INVENTORY['receipt'],guards)
concat=e.read_json(native.CONCAT_TERMINAL['receipt'],guards)
context={'args':SimpleNamespace(execution_sha256=a['execution_sha256']),
 'code':json.loads((root/'execution.json').read_text()),'launch':a,'guards':guards,'reference':reference,
 'math':math_helper,'costs':e.paired_cost(records,'first'),'score_context':{'source_record':source},'concat_record':concat,
 'training_context':{'source':control['source'],'legacy':{'selected':{'source_cpu':{'numerical_flags':control['numerical_flags']}}}}}
e.check_receipt(context,r,'score')
assert all(sha(root/n)==h for n,h in context['code'].items()) and r['source_code']==context['code']
assert r['paired_seed_average_intervals']=={} and r['bootstrap_draws']==0 and r['bootstrap_seed']==179019
for unit in a['exports'].values():
    export=e.read_json(unit['receipt'],guards)
    assert export['invocation']['invocation_id']==unit['invocation_id']
    for name,digest in export['files'].items():assert sha(Path(export['output'])/name)==digest
log=(root/'first-selection-score-v1.log').read_text()
stops=[json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('STOP_CGROUP ')]
assert len(stops)==1
stop=stops[0];v=stop['values']
assert stop['service_result']=='success' and stop['exit_status']=='0' and stop['exit_code']=='exited'
assert stop['invocation_id']==r['invocation']['invocation_id']
finals=[json.loads(x.split(' ',1)[1]) for x in log.splitlines() if x.startswith('FINAL_CGROUP ')];assert len(finals)==1 and finals[0]['command_exit_status']==0 and finals[0]['invocation_id']==stop['invocation_id']
for footer in (finals[0],stop):
    fv=footer['values'];assert 0<int(fv['memory.peak'])<=8589934592 and int(fv['memory.max'])==8589934592
    assert all(int(fv[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
    assert all(int(x.split()[1])==0 for x in fv['memory.events'].splitlines())
assert int(v['memory.peak'])<=8589934592 and int(v['memory.max'])==8589934592
assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.peak','memory.swap.max'))
assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
match=re.search(r'^Service runtime: (?:(\d+)min )?(\d+(?:\.\d+)?)s$',log,re.M);assert match
seconds=float(Decimal(match[1] or '0')*60+Decimal(match[2]));assert 0<seconds<500 and 'Exit status: 0' in log
print(json.dumps({'pass':True,'decision':r['decision'],'quality':{arm:{k:r['quality']['179061'][arm][k] for k in ('recall_at_1','map_at_r')} for arm in ('control','candidate')},
 'deltas':r['mean_deltas'],'cost':r['cost'],'service_seconds':seconds,'host_peak_bytes':int(v['memory.peak']),
 'invocation_id':stop['invocation_id'],'receipt_sha256':sha(rpath),'log_sha256':sha(root/'first-selection-score-v1.log')},indent=2))
