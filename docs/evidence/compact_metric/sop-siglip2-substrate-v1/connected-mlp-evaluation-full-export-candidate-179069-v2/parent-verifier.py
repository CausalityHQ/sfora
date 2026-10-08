import json,hashlib,sys,re,importlib.util
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
root=Path('/home/riomus/runs/sfora-connected-mlp-evaluation-source-v9')
p=root/'evaluate_siglip2_connected_mlp.py'
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert digest(p)=='b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4'
s=importlib.util.spec_from_file_location('parent_connected_metadata',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
guards={}
rfile=Path('/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-candidate-179069-v2/receipt.json');lfile=root/'full-export-candidate-179069-v2-original.log'
r=m.read_json({'path':str(rfile),'sha256':sys.argv[1]},guards)
log=m.bound_file(guards,lfile,sys.argv[2]).read_text()
args=SimpleNamespace(execution_sha256=r['execution_sha256'],authority=Path(r['authority']['path']),authority_sha256=r['authority_sha256'],phase='export',arm='candidate',seed=179069,output=Path(r['output']))
launch=m.read_json({'path':str(args.authority),'sha256':'2562a06a9aab595600fa34be5d7c764be58fdfad27a7d2068970491ce85fb373'},guards)
assert launch==r['launch'];m.check_launch(launch,args)
code=m.closure(root,args.execution_sha256,m.FILES,guards)
loaded={}
for key,filename in (('reference','evaluate_siglip2_prototype_residual.py'),('evaluator_reference','evaluate_siglip2_identity_diversity.py')):
    d=launch[key];assert m.closure(d['root'],d['execution_sha256'],d['code'],guards)==d['code']
    loaded[key]=m.load_authenticated('parent_'+key,Path(d['root'])/filename,d['code'][filename],guards)
records={};manifests={}
for endpoint in launch['endpoints']:
    record=m.read_json(endpoint['terminal']['receipt'],guards)
    manifest=m.read_json(endpoint['bundle'],guards)
    m.check_endpoint_binding(endpoint,record,manifest,launch['training'])
    records[endpoint['seed'],endpoint['arm']]=record;manifests[endpoint['seed'],endpoint['arm']]=manifest
cost=loaded['evaluator_reference'].paired_cost({key:{'service_seconds':next(e['terminal']['service_seconds'] for e in launch['endpoints'] if (e['seed'],e['arm'])==key),'total_training_core_seconds':record['total_training_core_seconds']} for key,record in records.items()},'full')
old=m.read_json({'path':'/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/receipt.json','sha256':'d2239da896465ee004ebb95c9316713d634fa457c91c708c6c410769a8c17b03'},guards)
source=records[179061,'control']['source'];assert all(record['source']==source==old['source'] for record in records.values())
prepunit=source['warm_source']['preparation_terminal'];proof=m.read_json(prepunit['proof'],guards)
t={'source':source,'legacy':{'selected':{'source_cpu':old,'source':{'preparation_terminal':prepunit},'export_record':proof}}}
c={'args':args,'guards':guards,'code':code,'launch':launch,'training_context':t,'records':records,'manifests':manifests,'costs':cost,**loaded}
c['preparation_costs']=m.preparation_costs(c)
cpu = m.read_json(launch['selected_cpu']['receipt'],guards)
m.check_receipt(c,cpu,'cpu')
c['cpu'] = cpu
item = launch['nearest_evaluator']
assert m.closure(item['root'],item['execution_sha256'],item['code'],guards) == item['code']
native = m.load_authenticated('parent_nearest_export',Path(item['root'])/'evaluate_siglip2_nearest_ranking.py',item['code']['evaluate_siglip2_nearest_ranking.py'],guards)
archived = m.read_json(native.CONCAT_TERMINAL['receipt'],guards)
partition = m.read_json(archived['spec']['partition'],guards)
assert archived['spec']['partition']['sha256'] == source['partition_sha256']
c['score_context'] = {'partition':partition}
m.check_receipt(c,r,'export','candidate',179069)
for name,expected in r['files'].items():
    path = m.bound_file(guards,Path(r['output'])/name,expected)
    if name.endswith('.packed.bin'):
        assert path.stat().st_size == 3449*130
    else:
        import ast,struct
        with path.open('rb') as f:
            assert f.read(8) == b'\x93NUMPY\x01\x00'
            n = struct.unpack('<H',f.read(2))[0]
            assert 0 < n <= 8192
            header = ast.literal_eval(f.read(n).decode('latin1'))
        assert header == {'descr':'<f4','fortran_order':False,'shape':(3449,128)}
        assert path.stat().st_size == 10+n+3449*128*4

unit={'receipt':{'path':str(rfile),'sha256':digest(rfile)},'log':{'path':str(lfile),'sha256':digest(lfile)},'unit':'sfora-connected-mlp-evaluation-full-export-candidate-179069-v2','invocation_id':'cd02099051764ad782216d69bf858d21','service_seconds':float(sys.argv[3]),'native_peak_rss_kib':r['process_peak_rss_kib'],'both_locks_held':True}
m.check_unit(unit)
lines=log.splitlines()
required=[f"Running as unit: {unit['unit']}.service; invocation ID: {unit['invocation_id']}",'\tExit status: 0','Finished with result: success','Main processes terminated with: code=exited/status=0','\tSwaps: 0','Memory swap peak: 0B',f"\tMaximum resident set size (kbytes): {unit['native_peak_rss_kib']}"]
assert all(lines.count(x)==1 for x in required)
runtimes=[x.removeprefix('Service runtime: ') for x in lines if x.startswith('Service runtime: ')];assert len(runtimes)==1
minutes,seconds=loaded['reference'].runtime_components(runtimes[0]);duration=Decimal(minutes or '0')*60+Decimal(seconds)
assert (minutes is None or Decimal(seconds)<60) and duration==Decimal(str(unit['service_seconds']))
assert 0<r['wall_seconds']<=unit['service_seconds']<1500 and 0<r['process_peak_rss_kib']<=unit['native_peak_rss_kib']<=8*1024**2
foot={}
for tag in ('FINAL_CGROUP','STOP_CGROUP'):
    found=[json.loads(x[len(tag)+1:]) for x in lines if x.startswith(tag+' ')];assert len(found)==1 and found[0]['invocation_id']==unit['invocation_id'];foot[tag]=found[0]
assert foot['FINAL_CGROUP']['command_exit_status']==0
assert foot['STOP_CGROUP']['exit_code']=='exited' and foot['STOP_CGROUP']['exit_status']=='0' and foot['STOP_CGROUP']['service_result']=='success'
for value in (r['cgroup_before'],r['cgroup_after'],*foot.values()):
    v=value['values'];assert value['path']==foot['FINAL_CGROUP']['path'] and Path(value['path']).name==unit['unit']+'.service'
    assert v['memory.max']=='8589934592' and 0<int(v['memory.current'])<=int(v['memory.peak'])<=8589934592
    assert all(v[k]=='0' for k in ('memory.swap.current','memory.swap.peak','memory.swap.max')) and all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
assert int(foot['FINAL_CGROUP']['values']['memory.peak'])>=int(r['cgroup_after']['values']['memory.peak'])>=int(r['cgroup_before']['values']['memory.peak'])
assert (root/'full-export-candidate-179069-v2-terminal-status.txt').read_text().strip()=='0'
assert not {n.split('.')[0] for n in sys.modules}&m.NATIVE
assert all(r['invocation'][k]==old['invocation'][k] for k in ('python','python_sha256','python_version'))
assert digest(rfile)==unit['receipt']['sha256'] and digest(lfile)==unit['log']['sha256']
print('PARENT_VERIFICATION '+json.dumps({'schema':'connected-evaluator-export-parent-verification-v1','pass':True,'engineering_only':True,'terminal':unit,'actual_complete_check_receipt_on_independently_authenticated_metadata_pass':True,'original_normal_terminal_duration_rss_and_cgroup_predicates_checked':True,'two_complete_independent_passes_tails_oracles_and_typed_wires_pass':True,'native_modules':[],'whole_unit_peak_host_bytes':int(foot['STOP_CGROUP']['values']['memory.peak']),'memory_events_zero':True,'swap_bytes':0,'quality_read':False,'state_reuse_eligible':False,'complete_production_accept_unit_required_before_export_cuda':True,'original_previous_auxiliary_full_authority_audit_timeout_preserved':True,'next_gate':'ONE control069 export1500 after original candidate069 receipt acceptance. All four sequential full-stage exports must pass before full paired selection score700; no held VAL without full GO.'}),flush=True)
