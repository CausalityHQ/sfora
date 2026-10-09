"""Root preparation only: requires both original accepted exports; launches nothing."""
import copy,hashlib,importlib.util,json,pathlib,shlex,subprocess,types
p=pathlib.Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-evaluation-first-cpu-v3-freeze')
base=json.loads((p/'authority-first-cpu-v3.json').read_text())
cpu=pathlib.Path('/tmp/sfora-probe-first-evaluator-cpu-v3-result')
unit=json.loads((cpu/'accepted-unit.json').read_text()); record=json.loads((cpu/'receipt.json').read_text())
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(cpu/'receipt.json')==unit['receipt']['sha256'] and sha(cpu/'original.log')==unit['log']['sha256']
spec=importlib.util.spec_from_file_location('root_probe_score','scripts/evaluate_siglip2_connected_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
remote=pathlib.Path('/home/riomus/runs/sfora-connected-probe-evaluation-source-v1')
prefix='first-selection-score-v1'; name='sfora-connected-probe-evaluation-'+prefix
extra={str(q):h for q,h in record['input_guards'].items()};extra.update({unit[k]['path']:unit[k]['sha256'] for k in ('receipt','log')})
exports={}
for arm in ('control','candidate'):
    root=pathlib.Path('/tmp/sfora-probe-export-'+arm+'-179061-v1-result')
    u=json.loads((root/'accepted-unit.json').read_text());v=json.loads((root/'verification.json').read_text());r=json.loads((root/'receipt.json').read_text())
    assert (root/'terminal-status.txt').read_text().strip()=='0' and v['pass'] is True and v['terminal']==u and v['quality_read'] is False
    assert sha(root/'receipt.json')==u['receipt']['sha256'] and sha(root/'original.log')==u['log']['sha256']
    assert r['arm']==arm and r['seed']==179061 and r['phase']=='export'
    exports[m.label(next(e for e in base['endpoints'] if e['arm']==arm))]=u
    for path,h in {**r['input_guards'],**{u[k]['path']:u[k]['sha256'] for k in ('receipt','log')}}.items():
        assert path not in extra or extra[path]==h
        extra[path]=h
    for filename,h in r['files'].items():
        path=str(pathlib.Path(r['output'])/filename)
        assert path not in extra or extra[path]==h
        extra[path]=h
a=copy.deepcopy(base);a.update(phase='score',arm=None,seed=None,selected_cpu=unit,exports=exports)
args=types.SimpleNamespace(execution_sha256=a['execution_sha256'],authority=remote/('authority-'+prefix+'.json'),authority_sha256=None,phase='score',arm=None,seed=None,output=pathlib.Path('/home/riomus/runs')/name)
m.check_launch(a,args)
target=pathlib.Path('/tmp/sfora-probe-'+prefix+'-freeze');target.mkdir(exist_ok=False)
authority=target/args.authority.name;authority.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');args.authority_sha256=sha(authority);extra[str(args.authority)]=sha(authority)
parts=(p/'first-cpu-v3-command.sh').read_text().split('\nHASHES\n');assert len(parts)==3
add='\n'+''.join(h+'  '+q+'\n' for q,h in sorted(extra.items()))
argv=m.cli(args);argv[0]=str(remote/'evaluate_siglip2_connected_probe.py')
command=parts[0].rstrip()+add+'HASHES\n'+shlex.join(['/home/riomus/group-learning/.venv/bin/python','-B',*argv])+'\n'+parts[1].split('\n',1)[1].rstrip()+add+'HASHES\n'
assert command.count('/usr/bin/python3 -I -B -S '+str(remote/'verify_fresh_sha_table.py')+" <<'HASHES'")==2
before,after=command.split('\nHASHES\n')[:2];assert before.split("<<'HASHES'\n",1)[1].splitlines()==after.split("<<'HASHES'\n",1)[1].splitlines()
cmd=target/(prefix+'-command.sh');cmd.write_text(command)
replace=lambda text:text.replace('sfora-connected-probe-evaluation-first-cpu-v3',name).replace('first-cpu-v3',prefix)
launch=replace((p/'first-cpu-v3-launch.sh').read_text()).replace(sha(p/'first-cpu-v3-command.sh'),sha(cmd))
assert launch.count('RuntimeMaxSec=700')==1 and '--setenv=CUDA_VISIBLE_DEVICES= --' in launch
(target/(prefix+'-launch.sh')).write_text(launch);(target/(prefix+'-controller.sh')).write_text(replace((p/'first-cpu-v3-controller.sh').read_text()))
for q in target.glob('*.sh'):subprocess.run(['/bin/bash','-n',str(q)],check=True)
(target/'freeze.json').write_text(json.dumps({'schema':'connected-probe-first-score-freeze-v1','unit':name,'policy':m.policy('score'),'files':{q.name:sha(q) for q in sorted(target.iterdir())},'stop_rule':'ONE paired first selection score after both original accepted exports; original complete accept_unit before quality; frozen CONTINUE/KILL; no official, serving-speed or SOTA claim.'},indent=2,sort_keys=True)+'\n')
print('Frozen, not launched:',name)
