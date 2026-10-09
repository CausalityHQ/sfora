"""Parent normal-terminal/scalar decision replay; original scorer owns native/wire admission."""
import hashlib,importlib.util,json,pathlib,re,sys,types
from decimal import Decimal
invocation,=sys.argv[1:];assert re.fullmatch('[0-9a-f]{32}',invocation)
prefix='first-selection-score-v1';unit_name='sfora-connected-probe-evaluation-'+prefix
p=pathlib.Path('/tmp/sfora-probe-'+prefix+'-result')
f=pathlib.Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-'+prefix+'-freeze')
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
def load(name,pin=None):
 q=pathlib.Path('scripts')/name
 if pin is not None:assert sha(q)==pin
 spec=importlib.util.spec_from_file_location('parent_'+name[:-3],q);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('evaluate_siglip2_connected_probe.py');e=load('evaluate_siglip2_identity_diversity.py',m.EVALUATOR_PINS['evaluate_siglip2_identity_diversity.py']);math_helper=load('evaluate_siglip2_genuine_views.py',m.GENUINE_PINS['evaluate_siglip2_genuine_views.py'])
r=json.loads((p/'receipt.json').read_text());log=(p/'original.log').read_text();assert (p/'terminal-status.txt').read_text().strip()=='0'
a=json.loads((f/('authority-'+prefix+'.json')).read_text())
assert sha(pathlib.Path(m.__file__))==r['source_code']['evaluate_siglip2_connected_probe.py']
assert (r['schema'],r['phase'],r['arm'],r['seed'],r['stage'],r['panel'])==(m.SCHEMA,'score',None,None,'first','selection')
assert r['launch']==a and r['execution_sha256']==a['execution_sha256']
assert r['authority']=={'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/authority-'+prefix+'.json','sha256':sha(f/('authority-'+prefix+'.json'))}
assert r['authority_sha256']==r['authority']['sha256']
args=types.SimpleNamespace(execution_sha256=r['execution_sha256'],authority=pathlib.Path(r['authority']['path']),authority_sha256=r['authority_sha256'],phase='score',arm=None,seed=None,output=pathlib.Path(r['output']))
m.check_launch(a,args);assert r['invocation']['argv'][1:]==m.cli(args)[1:];m.check_resource_facts(r,'score')
for k in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass','sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority','terminal_exit_and_both_locks_require_parent_receipt','source_archived_perquery_exact','concat_archived_perquery_exact','all_export_wires_readback_before_quality','persisted_wire_scoring_replay_exact'):assert r[k] is True,k
for k in ('official_read','global_production_goal_met','public_latency_measured','product_go'):assert r[k] is False,k
assert r['quality_read'] is True and r['files']=={} and r['readiness']==dict.fromkeys(m.READINESS,True)
assert r['bootstrap_seed']==179019 and r['bootstrap_draws']==0 and r['paired_seed_average_intervals']=={}
assert r['invocation']['invocation_id']==invocation and r['invocation']['cuda_visible_devices']=='' and type(r['invocation']['optimize']) is int and r['invocation']['optimize']==0
cpu=pathlib.Path('/tmp/sfora-probe-first-evaluator-cpu-v3-result');c=json.loads((cpu/'receipt.json').read_text());u=json.loads((cpu/'accepted-unit.json').read_text());assert a['selected_cpu']==u and sha(cpu/'receipt.json')==u['receipt']['sha256'] and sha(cpu/'original.log')==u['log']['sha256']
assert r['source']==c['source'] and r['preparation_costs']==c['preparation_costs'] and r['cost']==c['cost'] and r['cost_policy']==m.COST_POLICY
for arm in ('control','candidate'):
 root=pathlib.Path('/tmp/sfora-probe-export-'+arm+'-179061-v1-result');u=json.loads((root/'accepted-unit.json').read_text());assert a['exports'][arm+'-179061']==u and sha(root/'receipt.json')==u['receipt']['sha256'] and sha(root/'original.log')==u['log']['sha256']
decision=e.decide(math_helper,r['quality'],r['source_quality'],r['concat_quality'],'first','selection',{},r['cost']);assert all(r[k]==v for k,v in decision.items())
foot=[]
for prefix in ('FINAL_CGROUP','STOP_CGROUP'):
 vals=[json.loads(x[len(prefix)+1:]) for x in log.splitlines() if x.startswith(prefix+' ')];assert len(vals)==1;foot.extend(vals)
assert all(x['invocation_id']==invocation for x in foot) and foot[0]['command_exit_status']==0 and foot[1]['service_result']=='success' and foot[1]['exit_status']=='0'
for g in (r['cgroup_before'],r['cgroup_after'],*foot):
 v=g['values'];assert int(v['memory.max'])==8589934592 and int(v['memory.peak'])<=8589934592
 assert all(int(v[k])==0 for k in ('memory.swap.current','memory.swap.max','memory.swap.peak')) and all(int(x.split()[1])==0 for x in v['memory.events'].splitlines())
t=re.findall(r'^Service runtime: (.+)$',log,re.M);assert len(t)==1
tokens=re.findall(r'(\d+(?:\.\d+)?)(min|ms|us|h|s)',t[0]);assert tokens and ''.join(n+u for n,u in tokens)==t[0].replace(' ','')
scales={'h':Decimal(3600),'min':Decimal(60),'s':Decimal(1),'ms':Decimal('.001'),'us':Decimal('.000001')};seconds=float(sum((Decimal(n)*scales[u] for n,u in tokens),Decimal(0)))
assert 0<r['wall_seconds']<seconds<700 and 'Main processes terminated with: code=exited/status=0' in log
rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',log);assert rss
cmd=(f/'first-selection-score-v1-command.sh').read_text().split('\nHASHES\n');assert len(cmd)==3
before=cmd[0].split("<<'HASHES'\n",1)[1].splitlines();after=cmd[1].split("<<'HASHES'\n",1)[1].splitlines();assert before==after
expected=[row[66:]+': OK' for row in before];lines=log.splitlines();i=lines.index(expected[0]);assert lines[i:i+len(expected)]==expected;j=lines.index(expected[0],i+len(expected));assert lines[j:j+len(expected)]==expected
unit={'unit':unit_name,'invocation_id':invocation,'both_locks_held':True,'service_seconds':seconds,'native_peak_rss_kib':int(rss[1]),'receipt':{'path':r['output']+'/receipt.json','sha256':sha(p/'receipt.json')},'log':{'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/first-selection-score-v1-original.log','sha256':sha(p/'original.log')}};m.check_unit(unit)
(p/'accepted-unit.json').write_text(json.dumps(unit,indent=2,sort_keys=True)+'\n')
(p/'verification.json').write_text(json.dumps({'schema':'connected-probe-first-score-parent-verification-v1','pass':True,'terminal':unit,'decision_replayed_from_actual_pinned_source':decision,'quality_scope':'In-Shop official TRAIN-derived selection1734 query1715 gallery498products; seed179061 only','official_quality_qualified':False,'serving_speed_qualified':False,'complete_original_native_admission_owned_by_original_scorer':True,'host_peak_bytes':int(r['cgroup_after']['values']['memory.peak'])},indent=2,sort_keys=True)+'\n')
print(json.dumps({'terminal':unit,'decision':decision},sort_keys=True))
