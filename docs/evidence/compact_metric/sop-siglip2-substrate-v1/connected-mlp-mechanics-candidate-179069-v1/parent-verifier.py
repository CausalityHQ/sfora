import json,hashlib,re,importlib.util,sys,statistics
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
base=Path('/home/rb/worktrees/sfora-positive-causality')
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--receipt-sha256',required=True)
parser.add_argument('--log-sha256',required=True)
args=parser.parse_args()
prefix='/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-mechanics-candidate-179069-v1'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rpath=Path(prefix)/'receipt.json';lpath=Path(prefix)/'original.log'
assert sha(rpath)==args.receipt_sha256
assert sha(lpath)==args.log_sha256
r=json.loads(rpath.read_text());l=lpath.read_text();cpu=json.loads(Path('/tmp/sfora-connected-mlp-cpu-v6-receipt.json').read_text())
assert sha(base/'scripts/train_siglip2_connected_mlp.py')==cpu['code']['train_siglip2_connected_mlp.py']
m=load(base/'scripts/train_siglip2_connected_mlp.py','root_connected_verify')
op=Path('/tmp/sfora-identity-diversity-cpu-v5-collected/receipt.json');assert sha(op)==r['launch']['original_cpu']['terminal']['receipt']['sha256'];old=json.loads(op.read_text())
tp=Path('/tmp/sfora-identity-diversity-train-source-v5/train_siglip2_identity_diversity.py')
assert sha(tp)==old['input_guards']['/home/riomus/runs/sfora-so400-identity-diversity-train-source-v5/train_siglip2_identity_diversity.py']
t=load(tp,'root_original_verify')
context={'trainer':t,'original_cpu_record':old,'connected_args':SimpleNamespace(execution_sha256=cpu['execution_sha256']),'connected_launch':cpu['launch'],'connected_code':cpu['code'],'source':cpu['source']}
m.check_terminal(context,r,'mechanics','candidate',179069)
assert r['launch']['selected_cpu']==json.loads(Path('/tmp/sfora-connected-mlp-cpu-v6-unit.json').read_text())
f=base/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-mechanics-candidate-179069-v1-freeze'
a=json.loads((f/'authority-mechanics-candidate-179069-v1.json').read_text());assert r['launch']==a
assert r['authority_sha256']==sha(f/'authority-mechanics-candidate-179069-v1.json')
root=Path('/home/riomus/runs/sfora-connected-mlp-train-source-v6')
assert r['invocation']['argv']==m.cli(root,root/'authority-mechanics-candidate-179069-v1.json',r['authority_sha256'],r['execution_sha256'],'mechanics','candidate',179069,Path('/home/riomus/runs/sfora-connected-mlp-mechanics-candidate-179069-v1'))
assert all(r['invocation'][k]==cpu['invocation'][k] for k in ('python','python_sha256','python_version'))
foot={}
for tag in ('FINAL_CGROUP','STOP_CGROUP'):
 found=[json.loads(line[len(tag)+1:]) for line in l.splitlines() if line.startswith(tag+' ')];assert len(found)==1;foot[tag]=found[0]
 assert foot[tag]['invocation_id']==r['invocation']['invocation_id']=='cc546b2dcf2e4b9ea023c8b073ce893d'
assert foot['FINAL_CGROUP']['command_exit_status']==0
assert foot['STOP_CGROUP']['exit_code']=='exited' and foot['STOP_CGROUP']['exit_status']=='0' and foot['STOP_CGROUP']['service_result']=='success'
for c in (r['cgroup_before'],r['cgroup_after'],*foot.values()):
 v=c['values'];assert int(v['memory.max'])==8589934592 and int(v['memory.peak'])<=8589934592
 assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.max','memory.swap.peak'))
 assert all(int(line.split()[1])==0 for line in v['memory.events'].splitlines())
assert l.count('Service runtime:')==1
match=re.search(r'^Service runtime: (\d+)min (\d+\.\d+)s$',l,re.M);assert match
service=Decimal(match[1])*60+Decimal(match[2]);assert r['wall_seconds'] < float(service) < 1200
assert '\tExit status: 0' in l and 'Main processes terminated with: code=exited/status=0' in l
assert int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',l)[1])==r['process_peak_rss_kib']
assert not {'torch','numpy','transformers','PIL'} & sys.modules.keys()
unit={'receipt':{'path':'/home/riomus/runs/sfora-connected-mlp-mechanics-candidate-179069-v1/receipt.json','sha256':sha(rpath)},'log':{'path':str(root/'mechanics-candidate-179069-v1-original.log'),'sha256':sha(lpath)},'unit':'sfora-connected-mlp-mechanics-candidate-179069-v1','invocation_id':r['invocation']['invocation_id'],'service_seconds':float(service),'native_peak_rss_kib':r['process_peak_rss_kib'],'both_locks_held':True}
verification={'schema':'connected-mlp-mechanics-parent-verification-v1','pass':True,'engineering_only':True,'original_launch_exit':0,'terminal':unit,'service_seconds':float(service),'whole_unit_peak_host_bytes':int(foot['STOP_CGROUP']['values']['memory.peak']),'peak_cuda_allocated_bytes':r['peak_cuda_allocated_bytes'],'memory_events_zero':True,'swap_bytes':0,'total_training_core_seconds':r['total_training_core_seconds'],'update_core_seconds_median':statistics.median(x['core_seconds'] for x in r['result']['steps']),'full17_vs_independent8_plus9_exact':True,'strict_public_reload_raw_unit_packed':True,'both_source_and_mechanics_freezes_exit_rehashed':True,'quality_read':False,'state_reuse_eligible':False,'next_gate':'Accepted complete paired069 mechanics admits fresh control069 TRAIN128 then candidate069 with exact same-seed fresh-control UNIT and unchanged cost checks. No state reuse.'}
assert [m.diagnostic(x) for x in r['result']['steps']]==[m.diagnostic(x) for x in r['result']['replay_steps']]
(Path(prefix)/'verification.json').write_text(json.dumps(verification,sort_keys=True,indent=2)+'\n');(Path(prefix)/'unit.json').write_text(json.dumps(unit,sort_keys=True,indent=2)+'\n')
print(json.dumps(verification))
