"""Parent scalar metadata/terminal verification; export owns complete accept_unit before CUDA."""
import hashlib,importlib.util,json,math,pathlib,re,types
p=pathlib.Path('/tmp/sfora-probe-first-evaluator-cpu-v3-result')
freeze=pathlib.Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-evaluation-first-cpu-v3-freeze')
r=json.loads((p/'receipt.json').read_text());log=(p/'original.log').read_text()
a=json.loads((freeze/'authority-first-cpu-v3.json').read_text())
assert (p/'terminal-status.txt').read_text().strip()=='0'
def load(name,key):
    q=pathlib.Path('scripts')/name
    if key:assert hashlib.sha256(q.read_bytes()).hexdigest()==a[key]['code'][name]
    spec=importlib.util.spec_from_file_location('parent_'+name[:-3],q)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load('evaluate_siglip2_connected_probe.py',None)
assert hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()==r['source_code']['evaluate_siglip2_connected_probe.py']
reference=load('evaluate_siglip2_prototype_residual.py','reference')
oracle=load('evaluate_siglip2_identity_diversity.py','evaluator_reference')
assert (r['schema'],r['phase'],r['arm'],r['seed'],r['stage'],r['panel'])==(m.SCHEMA,'cpu',None,None,'first','selection')
assert r['launch']==a and r['execution_sha256']==a['execution_sha256']
assert r['authority']=={'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/authority-first-cpu-v3.json','sha256':hashlib.sha256((freeze/'authority-first-cpu-v3.json').read_bytes()).hexdigest()}
assert r['authority_sha256']==r['authority']['sha256']
m.check_launch(a,types.SimpleNamespace(execution_sha256=r['execution_sha256'],phase='cpu',arm=None,seed=None))
m.check_resource_facts(r,'cpu')
for k in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass','sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority','terminal_exit_and_both_locks_require_parent_receipt','updated_payloads_authenticated','malformed_inference_rejected','metadata_only'):assert r[k] is True,k
for k in ('quality_read','official_read','global_production_goal_met','public_latency_measured','product_go','cuda_initialized'):assert r[k] is False,k
assert r['files']=={} and r['invocation']['invocation_id']=='462e354ef9df42668c20b4d257baff12'
assert r['invocation']['cuda_visible_devices']=='' and type(r['invocation']['optimize']) is int and r['invocation']['optimize']==0
assert r['invocation']['cublas_workspace_config']==':4096:8' and r['cost_policy']==m.COST_POLICY
reference.check_synthetic_bootstrap(r['synthetic_bootstrap'])
context={'manifests':{},'records':{}}
for e in a['endpoints']:
    q=pathlib.Path('/tmp/sfora-probe-'+e['arm']+'-train-v1-result')
    record=json.loads((q/'receipt.json').read_text());bundle=json.loads((q/'bundle.json').read_text())
    assert hashlib.sha256((q/'receipt.json').read_bytes()).hexdigest()==e['terminal']['receipt']['sha256']
    assert hashlib.sha256((q/'bundle.json').read_bytes()).hexdigest()==e['bundle']['sha256']
    m.check_endpoint(e);m.check_endpoint_binding(e,record,bundle,m.TRAINING)
    context['manifests'][e['seed'],e['arm']]=bundle;context['records'][e['seed'],e['arm']]=record
    m.check_payload_facts(context,r['payload_facts'][m.label(e)],e)
    oracle.check_residual_oracle(r['calibration']['residual_oracles'][m.label(e)],e['arm'])
assert r['payload_facts'].keys()=={m.label(e) for e in a['endpoints']}
assert r['calibration']['residual_oracles'].keys()==r['payload_facts'].keys()
assert r['calibration']['same_role_forward_exact'] is True and r['calibration']['raw_unit_packed_exact'] is True
foot=[]
for prefix in ('FINAL_CGROUP','STOP_CGROUP'):
    lines=[x for x in log.splitlines() if x.startswith(prefix+' ')];assert len(lines)==1
    f=json.loads(lines[0].split(' ',1)[1]);assert f['invocation_id']==r['invocation']['invocation_id'];foot.append(f)
assert foot[0]['command_exit_status']==0 and foot[1]['service_result']=='success' and foot[1]['exit_status']=='0'
for g in (r['cgroup_before'],r['cgroup_after'],*foot):
    v=g['values'];assert int(v['memory.max'])==8589934592 and int(v['memory.peak'])<=8589934592
    assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.max','memory.swap.peak'))
    assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
from decimal import Decimal
def service_duration(text):
    tokens=re.findall(r'(\d+(?:\.\d+)?)(min|ms|us|h|s)',text)
    assert ''.join(n+u for n,u in tokens)==text.replace(' ','') and tokens
    scales={'h':Decimal(3600),'min':Decimal(60),'s':Decimal(1),'ms':Decimal('.001'),'us':Decimal('.000001')}
    return float(sum((Decimal(n)*scales[u] for n,u in tokens),Decimal(0)))
assert service_duration('621ms')==.621 and service_duration('4min 141ms')==240.141
t=re.findall(r'^Service runtime: (.+)$',log,re.M);assert len(t)==1
seconds=service_duration(t[0]);assert 0<r['wall_seconds']<seconds<700
rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',log);assert rss
assert 'Main processes terminated with: code=exited/status=0' in log
unit={'unit':'sfora-connected-probe-evaluation-first-cpu-v3','invocation_id':r['invocation']['invocation_id'],'both_locks_held':True,'service_seconds':seconds,'native_peak_rss_kib':int(rss[1]),'receipt':{'path':r['output']+'/receipt.json','sha256':hashlib.sha256((p/'receipt.json').read_bytes()).hexdigest()},'log':{'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/first-cpu-v3-original.log','sha256':hashlib.sha256((p/'original.log').read_bytes()).hexdigest()}}
m.check_unit(unit)
(p/'accepted-unit.json').write_text(json.dumps(unit,indent=2,sort_keys=True)+'\n')
(p/'verification.json').write_text(json.dumps({'schema':'connected-probe-evaluator-cpu-parent-verification-v1','pass':True,'terminal':unit,'quality_read':False,'engineering_only':True,'scalar_original_payload_bootstrap_resource_predicates_pass':True,'normal_terminal_full_source_exit':True,'complete_production_accept_unit_required_before_export_cuda':True,'complete_parent_check_receipt_executed':False,'whole_unit_peak_host_bytes':int(r['cgroup_after']['values']['memory.peak']),'next_gate':'ONE serial export per arm after complete production accept_unit, original source/parity/native exact four/resource gates unchanged'},indent=2,sort_keys=True)+'\n')
print(json.dumps(unit,sort_keys=True))
