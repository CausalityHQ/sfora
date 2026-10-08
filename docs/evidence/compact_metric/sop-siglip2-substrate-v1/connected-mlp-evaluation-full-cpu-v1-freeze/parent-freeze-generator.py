"""Root-only prospective CPU freeze from actual reviewed first-selection-only composition source bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

base = Path('/home/rb/worktrees/sfora-positive-causality')
old = Path('/home/rb/agents/handoffs/sfora-connected-evaluation-first-cpu-v3-preparation')
remote = Path('/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5')
prefix = 'full-cpu-v1'
unit = 'sfora-connected-mlp-evaluation-' + prefix
output = Path('/home/riomus/runs') / unit
prep = Path('/home/rb/agents/handoffs/sfora-connected-evaluation-full-cpu-v1-preparation')
expected = sys.argv[1]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, value):
    (prep/name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
names = ('evaluate_siglip2_connected_mlp.py', 'test_connected_mlp_evaluation.py')
code = {n: sha(base/'scripts'/n) for n in names}
execution = (json.dumps(code, sort_keys=True, indent=2) + '\n').encode()
assert hashlib.sha256(execution).hexdigest() == expected
spec = importlib.util.spec_from_file_location('freeze_connected_cpu_v5', base/'scripts'/names[0])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert [m.policy(p)['seconds'] for p in ('cpu', 'export', 'score')] == [500, 1500, 700]
assert not {'torch', 'numpy', 'transformers', 'PIL'} & sys.modules.keys()
launch = json.loads((old/'authority-first-cpu-v3.json').read_bytes())
launch['execution_sha256'] = expected
launch['resource_policies'] = {p: m.policy(p) for p in ('cpu', 'export', 'score')}
assert launch['phase'] == 'cpu' and launch['selected_cpu'] is None and launch['exports'] == {}
evidence = base/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
launch['stage'] = 'full'
launch['first_selection'] = json.loads((evidence/'connected-mlp-evaluation-first-selection-score-v2/unit.json').read_text())
assert json.loads((evidence/'connected-mlp-evaluation-first-selection-score-v2/verification.json').read_text())['pass'] is True
for arm in ('candidate','control'):
    d = evidence/('connected-mlp-train-'+arm+'-179069-v1')
    assert json.loads((d/'verification.json').read_text())['pass'] is True
    launch['endpoints'].append(json.loads((d/'endpoint.json').read_text()))
assert [(e['seed'],e['arm']) for e in launch['endpoints']] == list(m.endpoint_order('full'))

prep.mkdir(exist_ok=False)
for name in names:
    (prep/name).write_bytes((base/'scripts'/name).read_bytes())
(prep/'execution.json').write_bytes(execution)
authority = 'authority-' + prefix + '.json'
write(authority, launch)
args = SimpleNamespace(execution_sha256=expected, authority=remote/authority,
    authority_sha256=sha(prep/authority), phase='cpu', arm=None, seed=None, output=output)
m.check_launch(launch, args)
argv = m.cli(args)
argv[0] = str(remote/names[0])
parts = (old/'first-cpu-v3-command.sh').read_text().split('\nHASHES\n')
assert len(parts) == 3
extra = [(str(remote/n), sha(prep/n)) for n in (*names, 'execution.json', authority)]
for endpoint in launch['endpoints']:
    for key in ('launch','checkpoint','bundle'):
        item=endpoint[key]; extra.append((item['path'],item['sha256']))
    for key in ('receipt','log'):
        item=endpoint['terminal'][key]; extra.append((item['path'],item['sha256']))
for key in ('receipt','log'):
    item=launch['first_selection'][key]; extra.append((item['path'],item['sha256']))
extra=list(dict.fromkeys(extra))
add = '\n' + ''.join(h + '  ' + p + '\n' for p, h in extra)
command = (parts[0].rstrip() + add + 'HASHES\n' +
    shlex.join(['/home/riomus/group-learning/.venv/bin/python', '-B', *argv]) + '\n' +
    parts[1].split('\n', 1)[1].rstrip() + add + 'HASHES\n')
# Complete historical before/after tables remain byte-identical, including old own source.
for section in (parts[0], parts[1].split('\n', 1)[1]):
    for line in section.splitlines():
        if len(line.split('  ', 1)) == 2 and len(line.split('  ', 1)[0]) == 64:
            assert command.count(line) >= 2
(prep/(prefix+'-command.sh')).write_text(command)
wrapper = (old/'first-cpu-v3-launch.sh').read_text().replace(
    str(remote.with_name('sfora-connected-mlp-evaluation-source-v3')), str(remote))
wrapper = wrapper.replace('first-cpu-v3', prefix).replace(
    sha(old/'first-cpu-v3-command.sh'), sha(prep/(prefix+'-command.sh')))
assert wrapper.count('RuntimeMaxSec=500') == 1 and wrapper.count('--setenv=CUDA_VISIBLE_DEVICES= --') == 1
(prep/(prefix+'-launch.sh')).write_text(wrapper)
controller = (old/'first-cpu-v3-controller.sh').read_text().replace(
    '/home/riomus/runs/sfora-connected-mlp-evaluation-source-v3', str(remote)).replace('first-cpu-v3', prefix)
(prep/(prefix+'-controller.sh')).write_text(controller)
write(prefix+'-freeze.json', {'schema':'connected-evaluator-cpu-prospective-freeze-v1',
    'source_root':str(remote), 'code':code, 'execution_sha256':expected,
    'unit':unit, 'output':str(output), 'policy':m.policy('cpu'),
    'files':{p.name:sha(p) for p in sorted(prep.iterdir())},
    'stop_rule':'ONE fresh CPU500, original metadata/parity/source/whole normal0/8GiB/zeroSwap/bothlocks/full uncached exit mandatory. Four fresh stage-specific exports1500 then score700 only after this full CPU PASS; historical first exports cannot serve full stage. Preserve original score500 FAIL; no quality or state reuse from partial score.'})
print(json.dumps({'prep':str(prep),'execution_sha256':expected,'source_root':str(remote)}))
