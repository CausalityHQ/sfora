"""Parent scalar export receipt check; production SCORE re-admits the complete UNIT."""
import hashlib,importlib.util,json,math,pathlib,re,sys,types
from decimal import Decimal
arm,invocation=sys.argv[1:];assert arm in ('control','candidate') and re.fullmatch('[0-9a-f]{32}',invocation)
p=pathlib.Path('/tmp/sfora-probe-export-'+arm+'-179061-v1-result')
f=pathlib.Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-export-'+arm+'-179061-v1-freeze')
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
r=json.loads((p/'receipt.json').read_text());log=(p/'original.log').read_text();assert (p/'terminal-status.txt').read_text().strip()=='0'
a=json.loads((f/('authority-export-'+arm+'-179061-v1.json')).read_text())
spec=importlib.util.spec_from_file_location('parent_export','scripts/evaluate_siglip2_connected_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert sha(pathlib.Path(m.__file__))==r['source_code']['evaluate_siglip2_connected_probe.py']
assert (r['schema'],r['phase'],r['arm'],r['seed'],r['stage'],r['panel'])==(m.SCHEMA,'export',arm,179061,'first','selection')
assert r['launch']==a and r['execution_sha256']==a['execution_sha256']
assert r['authority']=={'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/authority-export-'+arm+'-179061-v1.json','sha256':sha(f/('authority-export-'+arm+'-179061-v1.json'))}
assert r['authority_sha256']==r['authority']['sha256']
args=types.SimpleNamespace(execution_sha256=r['execution_sha256'],authority=pathlib.Path(r['authority']['path']),authority_sha256=r['authority_sha256'],phase='export',arm=arm,seed=179061,output=pathlib.Path(r['output']))
m.check_launch(a,args);assert r['invocation']['argv'][1:]==m.cli(args)[1:];m.check_resource_facts(r,'export')
for key in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass','sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority','terminal_exit_and_both_locks_require_parent_receipt','strict_independent_reload_exact','full_updated_state_exact','raw_unit_packed_readback_exact','updated_source_mutants_rejected','bundle_dependency_boundary_enforced','same_role_oracle_exact','native_exact_four_post_calibration'):assert r[key] is True,key
for key in ('quality_read','official_read','global_production_goal_met','public_latency_measured','product_go'):assert r[key] is False,key
assert r['invocation']['invocation_id']==invocation and r['invocation']['cuda_visible_devices']=='0' and type(r['invocation']['optimize']) is int and r['invocation']['optimize']==0
assert r['invocation']['cublas_workspace_config']==':4096:8' and r['cost_policy']==m.COST_POLICY
cpu=pathlib.Path('/tmp/sfora-probe-first-evaluator-cpu-v3-result');cpu_unit=json.loads((cpu/'accepted-unit.json').read_text());c=json.loads((cpu/'receipt.json').read_text());assert a['selected_cpu']==cpu_unit
assert sha(cpu/'receipt.json')==cpu_unit['receipt']['sha256'] and sha(cpu/'original.log')==cpu_unit['log']['sha256']
e=next(x for x in a['endpoints'] if (x['seed'],x['arm'])==(179061,arm));m.check_endpoint(e)
assert r['payload_facts']==c['payload_facts'][m.label(e)] and r['inference_state_sha256']==e['inference_state_sha256'] and r['source']==c['source'] and r['preparation_costs']==c['preparation_costs'] and r['cost']==c['cost']
assert r['batch_sizes']=={role:m.batch_sizes(m.PANELS['selection'][i]) for role,i in [('query',1),('gallery',2)]}
assert r['batch_sizes']['query'][-1]==6 and r['batch_sizes']['gallery'][-1]==19
assert r['files'].keys()=={m.label(e)+suffix for suffix in ('.raw.npy','.unit.npy','.packed.bin')}
assert all(m.sha(value) for value in r['files'].values())
images=r['images'];assert len(images)==sum(len(v) for v in r['batch_sizes'].values())
for role in ('query','gallery'):
 facts=[x for x in images if x['role']==role];assert len(facts)==len(r['batch_sizes'][role])
 assert [len(x['rows']) for x in facts]==r['batch_sizes'][role]
 assert len({row['panel_ordinal'] for x in facts for row in x['rows']})==sum(r['batch_sizes'][role])
 for x in facts:
  assert x.keys()=={'rows','rgb_sha256','pixels_sha256','outputs_sha256','residual_oracle','role'}
  assert all(m.sha(x[k]) for k in ('rgb_sha256','pixels_sha256','outputs_sha256'))
foot=[]
for prefix in ('FINAL_CGROUP','STOP_CGROUP'):
 vals=[json.loads(x[len(prefix)+1:]) for x in log.splitlines() if x.startswith(prefix+' ')];assert len(vals)==1;foot.extend(vals)
assert all(x['invocation_id']==invocation for x in foot)
assert foot[0]['command_exit_status']==0 and foot[1]['service_result']=='success' and foot[1]['exit_status']=='0'
for g in (r['cgroup_before'],r['cgroup_after'],*foot):
 v=g['values'];assert int(v['memory.max'])==8589934592 and int(v['memory.peak'])<=8589934592
 assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.max','memory.swap.peak'))
 assert all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
t=re.findall(r'^Service runtime: (.+)$',log,re.M);assert len(t)==1
tokens=re.findall(r'(\d+(?:\.\d+)?)(min|ms|us|h|s)',t[0]);assert ''.join(n+u for n,u in tokens)==t[0].replace(' ','') and tokens
scales={'h':Decimal(3600),'min':Decimal(60),'s':Decimal(1),'ms':Decimal('.001'),'us':Decimal('.000001')};seconds=float(sum((Decimal(n)*scales[u] for n,u in tokens),Decimal(0)))
assert 0<r['wall_seconds']<seconds<1500 and 'Main processes terminated with: code=exited/status=0' in log
rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',log);assert rss
command=(f/('export-'+arm+'-179061-v1-command.sh')).read_text().split('\nHASHES\n');assert len(command)==3
before=command[0].split("<<'HASHES'\n",1)[1].splitlines();after=command[1].split("<<'HASHES'\n",1)[1].splitlines();assert before==after
expected=[row[66:]+': OK' for row in before];lines=log.splitlines();i=lines.index(expected[0]);assert lines[i:i+len(expected)]==expected;j=lines.index(expected[0],i+len(expected));assert lines[j:j+len(expected)]==expected
unit={'unit':'sfora-connected-probe-evaluation-export-'+arm+'-179061-v1','invocation_id':invocation,'both_locks_held':True,'service_seconds':seconds,'native_peak_rss_kib':int(rss[1]),'receipt':{'path':r['output']+'/receipt.json','sha256':sha(p/'receipt.json')},'log':{'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/export-'+arm+'-179061-v1-original.log','sha256':sha(p/'original.log')}}
m.check_unit(unit)
(p/'accepted-unit.json').write_text(json.dumps(unit,indent=2,sort_keys=True)+'\n')
(p/'verification.json').write_text(json.dumps({'schema':'connected-probe-export-parent-verification-v1','pass':True,'engineering_only':True,'quality_read':False,'terminal':unit,'complete_parent_check_receipt_executed':False,'complete_production_accept_unit_required_before_score_quality':True,'all_ordered_before_after_fresh_hash_checks':len(expected),'query_gallery_tail_sizes':[6,19],'host_peak_bytes':int(r['cgroup_after']['values']['memory.peak']),'cuda_peak_bytes':r['peak_cuda_allocated_bytes'],'native_wire_readback_and_independent_reload_in_original_receipt':True},indent=2,sort_keys=True)+'\n')
print(json.dumps(unit,sort_keys=True))
