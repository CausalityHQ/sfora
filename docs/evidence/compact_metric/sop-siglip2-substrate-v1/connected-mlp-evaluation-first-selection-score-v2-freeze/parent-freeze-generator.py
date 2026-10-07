"""Root-only freeze; requires both actual independently accepted exports."""
import hashlib,importlib.util,json,pathlib,shlex,sys
from types import SimpleNamespace
base=pathlib.Path('/home/rb/worktrees/sfora-positive-causality')
old=pathlib.Path('/home/rb/agents/handoffs/sfora-connected-evaluation-first-cpu-v5-preparation')
remote=pathlib.Path('/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5')
prefix='first-selection-score-v2';unit='sfora-connected-mlp-evaluation-'+prefix;output=pathlib.Path('/home/riomus/runs')/unit
prep=pathlib.Path('/home/rb/agents/handoffs/sfora-connected-'+prefix+'-preparation');prep.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(prep/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
launch=json.loads((old/'authority-first-cpu-v5.json').read_bytes());exports={};extra=[]
for arm,suffix in [('control','v2'),('candidate','v1')]:
 d=base/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'/('connected-mlp-evaluation-export-'+arm+'-179061-'+suffix)
 v=json.loads((d/'verification.json').read_bytes());r=json.loads((d/'receipt.json').read_bytes());u=v['terminal']
 assert v['pass'] is True and v['actual_complete_check_receipt_on_independently_authenticated_metadata_pass'] is True and v['two_complete_independent_passes_tails_oracles_and_typed_wires_pass'] is True
 assert sha(d/'receipt.json')==u['receipt']['sha256'] and sha(d/'original.log')==u['log']['sha256']
 assert (d/'terminal-status.txt').read_text().strip()=='0' and r['arm']==arm and r['seed']==179061
 exports[arm+'-179061']=u
 extra.extend((x['path'],x['sha256']) for x in (u['receipt'],u['log']))
 extra.extend((str(pathlib.Path(r['output'])/n),h) for n,h in r['files'].items())
cpu=json.loads((base/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-cpu-v5/unit.json').read_bytes())
verification=json.loads((base/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-cpu-v5/verification.json').read_bytes())
assert verification['pass'] is True and verification['actual_complete_check_receipt_on_independently_authenticated_metadata_pass'] is True
assert verification['terminal'] == cpu
launch.update(phase='score',arm=None,seed=None,exports=exports,selected_cpu=cpu)
name='authority-'+prefix+'.json';write(name,launch)
spec=importlib.util.spec_from_file_location('freeze_connected_score',base/'scripts/evaluate_siglip2_connected_mlp.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert sha(base/'scripts/evaluate_siglip2_connected_mlp.py')=='919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69'
assert m.policy('score')['seconds']==700 and m.policy('score')['cuda_visible_devices']==''
args=SimpleNamespace(execution_sha256=launch['execution_sha256'],authority=remote/name,authority_sha256=sha(prep/name),phase='score',arm=None,seed=None,output=output);m.check_launch(launch,args)
argv=m.cli(args);argv[0]=str(remote/'evaluate_siglip2_connected_mlp.py')
extra.extend((x['path'],x['sha256']) for x in (cpu['receipt'],cpu['log']));extra.append((str(remote/name),sha(prep/name)))
assert exports == m.ORIGINAL_EXPORT_UNITS
extra.append((m.ORIGINAL_SCORE_AUTHORITY['path'],m.ORIGINAL_SCORE_AUTHORITY['sha256']))
parts=(old/'first-cpu-v5-command.sh').read_text().split('\nHASHES\n');assert len(parts)==3
add='\n'+''.join(h+'  '+p+'\n' for p,h in extra)
command=parts[0].rstrip()+add+'HASHES\n'+shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',*argv])+'\n'+parts[1].split('\n',1)[1].rstrip()+add+'HASHES\n'
(prep/(prefix+'-command.sh')).write_text(command)
wrapper=(old/'first-cpu-v5-launch.sh').read_text().replace('first-cpu-v5',prefix).replace(sha(old/'first-cpu-v5-command.sh'),sha(prep/(prefix+'-command.sh')))
assert wrapper.count('RuntimeMaxSec=500')==1
wrapper=wrapper.replace('RuntimeMaxSec=500','RuntimeMaxSec=700')
assert wrapper.count('RuntimeMaxSec=700')==1 and wrapper.count('--setenv=CUDA_VISIBLE_DEVICES= --')==1
(prep/(prefix+'-launch.sh')).write_text(wrapper)
(prep/(prefix+'-controller.sh')).write_text((old/'first-cpu-v5-controller.sh').read_text().replace('first-cpu-v5',prefix))
assert not {'torch','numpy','transformers','PIL'} & sys.modules.keys()
write(prefix+'-freeze.json',{'schema':'connected-first-selection-score-parent-freeze-v1','source_root':str(remote),'execution_sha256':launch['execution_sha256'],'selected_cpu':cpu,'exports':exports,'unit':unit,'output':str(output),'policy':m.policy('score'),'files':{p.name:sha(p) for p in sorted(prep.iterdir())},'stop_rule':'ONE first061 selection score700; original source+concat perquery exact replay, both complete actual updated wires and readback before candidate metrics, original costs/source/normal0/8GiB/noSwap/bothlocks/full uncached exit mandatory. Immediate strict candidate R1 gain and nonnegative AP plus original floors required for CONTINUE. KILL forbids069/VAL; no CI or official read here; no reused partials or runtime extension.'})
print(json.dumps({'prep':str(prep),'files':{p.name:sha(p) for p in sorted(prep.iterdir())}}))
