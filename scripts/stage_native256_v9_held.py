#!/usr/bin/env python3
"""Stage and launch the native256 v9 held CPU/export/score units (run on DGX after all four TRAIN100 units pass).

Everything is derived from completed v9 receipts; schemas follow docs/inshop_so400_native256_review_boundary_2026-09-30.md.
  stage SRC_DIR                 held root: v9 trainer5 + exporter/scorer/held test + two pinned references
  authority                     native256-held-authority.json from the four TRAIN endpoints
  launch cpu|export SEED ARM    command.sh + printed watchdog-wrapped systemd-run line (export binds that CPU terminal)
  cpu-terminal SEED ARM         standalone original CPU TERMINAL JSON after a successful CPU unit
  wires                         native256-held-wires.json after all four exports
  launch score                  scorer unit (CUDA hidden)
"""
import hashlib
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('stage_train', Path(__file__).with_name('stage_native256_v9.py'))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
RUNS, TRAIN_ROOT = train.RUNS, train.NEW
HELD = RUNS / 'sfora-native256-held-v9'
ORDER = ((179032, 'large'), (179032, 'so400'), (179041, 'so400'), (179041, 'large'))
PINS = {'export_siglip2_substrate_adaptation.py': '30bc6b08714821bf92f2457a7446f93c8cc18abbea0765d8229bc36a4c028194',
        'score_siglip2_substrate_adaptation.py': '47fc5c73dc8825669a41f0a9a0c2a895a49c4b750b00475e395e105bb08d6338',
        'reference_compare_inshop_sop_warmstart_100.py': '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250',
        'reference_score_inshop_crop_view_pair.py': '16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed'}
ADDED = ('export_siglip2_substrate_adaptation.py', 'score_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_held.py',
         'reference_compare_inshop_sop_warmstart_100.py', 'reference_score_inshop_crop_view_pair.py')
sha, dump = train.sha, train.dump


def unit(phase, seed=None, arm=None):
    return f'sfora-native256-held-{phase}' + (f'-{arm}-{seed}' if seed else '') + '-v9'


def exporter():
    spec = importlib.util.spec_from_file_location('held_exporter', HELD / 'export_siglip2_substrate_adaptation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(src):
    HELD.mkdir()
    for name in train.FILES:
        (HELD / name).write_bytes((TRAIN_ROOT / name).read_bytes())
    for name in ADDED:
        (HELD / name).write_bytes((Path(src) / name).read_bytes())
    code = {name: sha(HELD / name) for name in (*train.FILES, *ADDED)}
    test_mode = os.environ.get('SFORA_NATIVE256_TEST_BYTES') == '1'
    assert test_mode or all(code[n] == p for n, p in PINS.items()), 'held bytes differ from pinned v9/reference SHAs'
    assert code.items() >= json.loads((TRAIN_ROOT / 'execution.json').read_text()).items(), 'trainer5 differs from v9 root'
    print('held_execution_sha256', dump(HELD / 'execution.json', code))


def train_receipt(seed, arm):
    return RUNS / train.unit('train', arm, seed) / 'receipt.json'


def authority():
    held = exporter()
    prerequisites, endpoints, schedules = {}, [], {}
    for arm in ('large', 'so400'):
        launch = json.loads((TRAIN_ROOT / f'authority-{arm}-mechanics-v9.json').read_text())
        prerequisites[arm] = {k: launch[k] for k in ('qualifier_root', 'qualifier_execution_sha256', 'qualifier_authority')}
        prerequisites[arm].update(cpu=launch['selected_cpu'], mechanics=train.terminal('mechanics', arm, 179032))
    for seed, arm in ORDER:
        receipt = json.loads(train_receipt(seed, arm).read_text())
        launch = TRAIN_ROOT / f'authority-{arm}-train-{seed}-v9.json'
        checkpoint = train_receipt(seed, arm).parent / 'resume.pt'
        endpoints.append({'seed': seed, 'arm': arm, 'launch': {'path': str(launch), 'sha256': sha(launch)},
                          'terminal': train.terminal('train', arm, seed),
                          'checkpoint': {'path': str(checkpoint), 'sha256': sha(checkpoint)},
                          'terminal_state_sha256': receipt['terminal_state_sha256']})
        schedules.setdefault(str(seed), {'batches': [row['batch'] for row in receipt['steps']],
                                         'fact': receipt['identity']['schedule_facts'][str(seed)],
                                         'sha256': receipt['identity']['schedule_sha256']})
        fits = [p for p in receipt['input_guards'] if p.endswith('/fit.json')]
        assert len(fits) == 1, 'one FIT manifest guard expected'
        frozen = json.loads(Path(fits[0]).read_text())['original_receipt']
    value = {'schema': held.AUTHORITY_SCHEMA, 'execution_sha256': sha(HELD / 'execution.json'),
             'training_root': str(TRAIN_ROOT), 'training_execution_sha256': sha(TRAIN_ROOT / 'execution.json'),
             'prerequisites': prerequisites, 'endpoints': endpoints, 'schedules': schedules, 'frozen_split': frozen,
             'resource_policies': {p: held.policy(p) for p in ('cpu', 'export', 'score')},
             'cost_policy': held.COST_POLICY, 'both_locks_held': True}
    print('held_authority_sha256', dump(HELD / 'native256-held-authority.json', value))


def terminal(name, receipt):
    return train.terminal_from(HELD / f'{name}.log', receipt, name)


def cpu_terminal(seed, arm):
    seed = int(seed)
    path = HELD / f'cpu-terminal-{arm}-{seed}.json'
    print(path, dump(path, terminal(unit('cpu', seed, arm), RUNS / unit('cpu', seed, arm) / 'proof.json')))


def wires():
    held_authority = HELD / 'native256-held-authority.json'
    entries = []
    for seed, arm in ORDER:
        cpu = HELD / f'cpu-terminal-{arm}-{seed}.json'
        entries.append({'seed': seed, 'arm': arm,
                        'terminal': terminal(unit('export', seed, arm), RUNS / unit('export', seed, arm) / 'receipt.json'),
                        'cpu_terminal': {'path': str(cpu), 'sha256': sha(cpu)}})
    value = {'schema': 'native256-held-wires-v1', 'authority_sha256': sha(held_authority),
             'execution_sha256': sha(HELD / 'execution.json'), 'wires': entries, 'both_locks_held': True}
    print('held_wires_sha256', dump(HELD / 'native256-held-wires.json', value))


def launch(phase, seed=None, arm=None):
    seed = int(seed) if seed else None
    name = unit(phase, seed, arm)
    authority_path = HELD / 'native256-held-authority.json'
    common = ['--execution-sha256', sha(HELD / 'execution.json'), '--authority', str(authority_path),
              '--authority-sha256', sha(authority_path)]
    if phase == 'score':
        wires_path = HELD / 'native256-held-wires.json'
        script, args = 'score_siglip2_substrate_adaptation.py', [*common, '--wires-authority', str(wires_path),
                       '--wires-authority-sha256', sha(wires_path), '--output', str(RUNS / f'{name}.json')]
    else:
        script = 'export_siglip2_substrate_adaptation.py'
        args = [*common, '--phase', phase, '--seed', str(seed), '--arm', arm, '--output', str(RUNS / name)]
        if phase == 'export':
            cpu = HELD / f'cpu-terminal-{arm}-{seed}.json'
            args += ['--cpu-terminal', str(cpu), '--cpu-terminal-sha256', sha(cpu)]
    pinned = [*sorted(json.loads((HELD / 'execution.json').read_text())), 'execution.json', authority_path.name]
    if phase == 'score':
        pinned.append('native256-held-wires.json')
    hashes = ''.join(f'{sha(HELD / n)}  {n}\n' for n in pinned) + ''.join(f'{sha(f)}  {f}\n' for f in train.FOOTERS)
    check = f"sha256sum -c <<'HASHES'\n{hashes}HASHES\n"
    header = (train.OLD / 'large-mechanics-command.sh').read_text().split('sha256sum -c', 1)[0]
    header = train.re.sub(r'^cd .*$', f'cd {HELD}', header, flags=train.re.M)
    command = HELD / f'{name}-command.sh'
    command.write_text(header + check + shlex.join([train.PYTHON, '-B', str(HELD / script), *args]) + '\n' + check)
    cuda = ['--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8'] if phase == 'export' \
        else ['--setenv=CUDA_VISIBLE_DEVICES=']
    print(train.launch_line(name, HELD, command, cuda))


if __name__ == '__main__':
    {'stage': stage, 'authority': authority, 'launch': launch, 'cpu-terminal': cpu_terminal,
     'wires': wires}[sys.argv[1]](*sys.argv[2:])
