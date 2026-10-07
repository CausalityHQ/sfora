import json,hashlib,sys,importlib.util,re
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
root=Path('/home/riomus/runs/sfora-connected-mlp-evaluation-source-v2')
rfile=Path('/home/riomus/runs/sfora-connected-mlp-evaluation-first-cpu-v1/receipt.json')
lfile=root/'first-cpu-v1-original.log'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert digest(rfile)=='5616557484a888824eb40c2feb081e2cd5338ee7d17dd69fe5de53e95b904132'
assert digest(lfile)=='63e03d8382bda34a763d5ccfa48b318a7980a4f9903a0c8b5f9b986d4e588b43'
p=root/'evaluate_siglip2_connected_mlp.py'
assert digest(p)=='ca122f669078e39a3c300246f425d441f4c53356b95a156865dfd33cbe177581'
s=importlib.util.spec_from_file_location('parent_connected_eval_verify',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=json.loads(rfile.read_bytes());log=lfile.read_text()
args=SimpleNamespace(execution_sha256=r['execution_sha256'],authority=Path(r['authority']['path']),authority_sha256=r['authority_sha256'],phase='cpu',arm=None,seed=None,output=Path('/home/riomus/runs/sfora-connected-evaluation-parent-admission-v1'))
print('PARENT_SOURCE_ADMISSION_BEGIN',flush=True)
c=m.authority(args)
print('PARENT_SOURCE_ADMISSION_END',flush=True)
m.check_receipt(c,r,'cpu')
unit={'receipt':{'path':str(rfile),'sha256':digest(rfile)},'log':{'path':str(lfile),'sha256':digest(lfile)},'unit':'sfora-connected-mlp-evaluation-first-cpu-v1','invocation_id':'282c8bd4e350400fbb7d0131dc1aeee2','service_seconds':487.465,'native_peak_rss_kib':r['process_peak_rss_kib'],'both_locks_held':True}
m.check_unit(unit)
final=c['terminal_reader'](c['training_context']['legacy']['admission'],r,unit,500,c['guards'])
stop=[json.loads(x[12:]) for x in log.splitlines() if x.startswith('STOP_CGROUP ')]
assert len(stop)==1 and stop[0]['invocation_id']==unit['invocation_id'] and stop[0]['exit_status']=='0' and stop[0]['service_result']=='success'
for x in (r['cgroup_before'],r['cgroup_after'],final,stop[0]):c['helper'].zero_events(x)
assert not {n.split('.')[0] for n in sys.modules}&m.NATIVE
assert (root/'first-cpu-v1-terminal-status.txt').read_text().strip()=='0'
assert digest(rfile)==unit['receipt']['sha256'] and digest(lfile)==unit['log']['sha256']
print('PARENT_VERIFICATION '+json.dumps({'schema':'connected-evaluator-cpu-parent-verification-v1','pass':True,'engineering_only':True,'terminal':unit,'actual_complete_authority_check_receipt_and_original_terminal_reader_pass':True,'synthetic_bootstrap_and_complete_endpoint_payload_facts_pass':True,'native_modules':[],'whole_unit_peak_host_bytes':int(stop[0]['values']['memory.peak']),'memory_events_zero':True,'swap_bytes':0,'quality_read':False,'state_reuse_eligible':False,'next_gate':'ONE prospective control061 export900 followed by candidate061 export900; first061 selection only after both complete original exports.'}),flush=True)
