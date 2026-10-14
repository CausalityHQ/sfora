#!/usr/bin/env python3
"""Parent-only full-stage tooling (stdlib; stage/launch/admit/decision).

stage SRC_DIR consumes a flat source directory containing LOCAL files below.
It creates separate frozen code and output roots; no resume/retry/attempt option.
Run the staged driver via parent nohup setsid: Large chunks 100..2000, then
So400 chunks 100..2000, then Large score, then So400 paired score.
The original pilot still loads its original, independently hash-checked closure.
Its five files are also copied here and checked at both paths around each unit.
"""
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Checks require unoptimized Python')

RUNS = Path(os.environ.get('SFORA_NATIVE256_RUNS', '/home/riomus/runs'))
ROOT = RUNS / 'sfora-native256-full-v1'
OUTPUT = RUNS / 'sfora-native256-full-run-v1'
PILOT = RUNS / 'sfora-native256-adaptation-source-v9'
OLD = RUNS / 'sfora-native256-adaptation-source-v6'
PYTHON = '/home/riomus/group-learning/.venv/bin/python'
FOOTERS = (RUNS / 'sfora-native256-initialized-cpu-final-footer-v1.py', RUNS / 'sfora-native256-stop-cgroup-footer-v1.py')
LOCKS = ('/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock')
SECONDS, BACKSTOP = 3600, 3780
TRAINER, SCORER = 'train_siglip2_substrate_full.py', 'score_siglip2_substrate_official.py'
PINS = {
    TRAINER: 'b34089def204658f231c05b05036a300352e2b00defddfe12ff0e11e7aa6b1e9',
    SCORER: 'b17e4ab4f88f7d4754387006927870a459856361da3da8a5a724209fe44a2752',
    'test_siglip2_substrate_full.py': '39685db5e22714cb2656b81a930f300fe2005d0d33f95871da54f8a5dbdffbb7',
    'unicom_inshop.py': '635d300c66b2bebe7753707c9196d4735a97f07c7dd999f186f9b9b1251b7f1f',
    'evaluate_inshop_siglip2_official.py': '6fee591bffdb2af8d77355bd4f0c282ac52979fa2125ff46bfb31df0ea985411',
    'qualify_pe_large_inshop_official.py': 'd6d52da270f9c983ede3ccb23dc4374964a6e520e040758d73193c33ca5fa5a5',
    'score_inshop_crop_view_pair.py': '16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed',
    'train_siglip2_substrate_adaptation.py': '1b73b2a6bdc1ee0cc468109e15746c8c5cde32569ac3b5de05d07f9e2f0b573f',
    'pilot-execution.json': 'a74746f333a60d2774c2905258630458e1a80e2a2726c960de9128047477839d',
}
PILOT_FILES = ('deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py',
               'test_siglip2_substrate_adaptation.py', 'train_siglip2_substrate_adaptation.py')
LOCAL = tuple(f for f in PINS if f not in (*PILOT_FILES, 'pilot-execution.json')) + (
    'stage_siglip2_substrate_full.py', 'run_siglip2_substrate_full.sh', 'test_stage_siglip2_substrate_full.py',
    'run_with_progress_watchdog.sh')
FILES = (*LOCAL, *PILOT_FILES, 'pilot-execution.json', 'command-header.sh')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def absent(path):
    assert not path.exists() and not path.is_symlink(), f'existing path: INVALID/HOLD: {path}'


def write(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def pins(code):
    assert all(code[f] == value for f, value in PINS.items()) or os.environ.get('SFORA_NATIVE256_TEST_BYTES') == '1', 'pinned source bytes differ'


def stage(src):
    absent(ROOT)
    absent(OUTPUT)
    assert RUNS.is_absolute() and RUNS.resolve() == RUNS, 'canonical runs root required'
    data = {f: (Path(src) / f).read_bytes() for f in LOCAL}
    data.update({f: (PILOT / f).read_bytes() for f in PILOT_FILES})
    data['pilot-execution.json'] = (PILOT / 'execution.json').read_bytes()
    pilot_code = json.loads(data['pilot-execution.json'])
    assert pilot_code.keys() == set(PILOT_FILES), 'pilot closure differs'
    assert all(hashlib.sha256(data[f]).hexdigest() == pilot_code[f] for f in PILOT_FILES), 'pilot closure SHA differs'
    # Preserve the reviewed v9-held header/trap byte-for-byte, changing only cd.
    header = (OLD / 'large-mechanics-command.sh').read_text().split('sha256sum -c', 1)[0]
    assert len(re.findall(r'^cd .*$', header, re.M)) == 1 and 'set -euo pipefail' in header, 'v6 command header differs'
    assert 'trap ' in header and 'SFORA_COMMAND_STATUS' in header and str(FOOTERS[0]) in header, 'final footer trap missing'
    data['command-header.sh'] = re.sub(r'^cd .*$', f'cd {shlex.quote(str(ROOT))}', header, flags=re.M).encode()
    code = {f: hashlib.sha256(data[f]).hexdigest() for f in FILES}
    pins(code)
    for footer in FOOTERS:
        assert footer.is_file() and footer.resolve() == footer, 'canonical footer required'
    ROOT.mkdir()
    OUTPUT.mkdir()
    for arm in ('large', 'so400'):
        (OUTPUT / arm).mkdir()
    for f in FILES:
        write(ROOT / f, data[f])
    for f in ('run_siglip2_substrate_full.sh', 'run_with_progress_watchdog.sh'):
        (ROOT / f).chmod(0o755)
    write(ROOT / 'execution.json', (json.dumps(code, indent=2, sort_keys=True) + '\n').encode())
    print('full_execution_sha256', sha(ROOT / 'execution.json'))


def modules():
    code = json.loads((ROOT / 'execution.json').read_text())
    assert code.keys() == set(FILES), 'full closure differs'
    pins(code)
    for f, digest in code.items():
        path = ROOT / f
        assert path.resolve() == path and sha(path) == digest, 'frozen code SHA differs: ' + f
    assert sha(__file__) == code['stage_siglip2_substrate_full.py'], 'parent tool differs from staged tool'
    for f in PILOT_FILES:
        assert sha(PILOT / f) == code[f], 'original pilot code differs'
    assert sha(PILOT / 'execution.json') == code['pilot-execution.json'], 'original pilot execution differs'
    loaded = []
    for file in (TRAINER, SCORER):
        name = Path(file).stem
        spec = importlib.util.spec_from_file_location(name, ROOT / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        raw = (ROOT / file).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == code[file], 'code changed before import'
        exec(compile(raw, str(ROOT / file), 'exec'), vars(module))
        loaded.append(module)
    return (*loaded, code)


def target(phase, arm, end=None):
    assert arm in ('large', 'so400') and phase in ('train', 'score'), 'phase/arm differs'
    if phase == 'train':
        end = int(end)
        assert end in range(100, 2001, 100), 'chunk endpoint differs'
        return f'sfora-native256-full-train-{arm}-{end:04d}-v1', OUTPUT / arm / f'chunk-{end:04d}'
    assert end is None, 'score has no chunk argument'
    return f'sfora-native256-full-score-{arm}-v1', OUTPUT / 'official' / arm


def terminal(full, phase, arm, end=None):
    name, folder = target(phase, arm, end)
    ref = full.descriptor(folder / 'terminal.json')
    value = full.read_json(ref)
    assert value['unit'] == name and value['log']['path'] == str(ROOT / f'{name}.log'), 'original unit/log differs'
    assert value['receipt']['path'] == str(folder / 'receipt.json'), 'original receipt path differs'
    record = full.read_json(value['receipt'])
    assert record['arm'] == arm, 'original receipt arm differs'
    full.admit_terminal(record, value)
    return ref, value['receipt'], record


def source(full, score, arm):
    receipt = score.PILOT_RECEIPTS[arm]
    record = full.read_json(receipt)
    checkpoint = full.descriptor(Path(receipt['path']).parent / 'resume.pt', record['checkpoint']['sha256'])
    value = {'receipt': receipt, 'checkpoint': checkpoint}
    score.validate_source_pin(value, arm, record)
    return value


def arguments(full, score, code, phase, arm, end=None):
    _, folder = target(phase, arm, end)
    absent(folder)
    train_code = {'trainer': code[TRAINER], 'partition_parser': full.PARSER_SHA}
    args = ['--arm', arm, '--partition-parser', str(ROOT / 'unicom_inshop.py'), '--output', str(folder)]
    if phase == 'train':
        end = int(end)
        ends = full.inspect_layout(OUTPUT)
        assert ends == ({'large': end - 100, 'so400': 0} if arm == 'large' else {'large': 2000, 'so400': end - 100}), 'fixed Large-then-So400 order differs'
        absent(OUTPUT / 'official')
        selected = source(full, score, arm)
        args += ['--source-resume', selected['checkpoint']['path'], '--source-sha256', selected['checkpoint']['sha256'],
                 '--source-receipt-sha256', selected['receipt']['sha256'], '--chunk-end', str(end)]
        if end > 100:
            term, receipt, _ = terminal(full, phase, arm, end - 100)
            full.check_chain(receipt, end - 100, arm, selected, train_code, term)
            args += ['--previous', receipt['path'], '--previous-sha256', receipt['sha256'], '--previous-terminal-sha256', term['sha256']]
        return TRAINER, args
    terminals = {a: terminal(full, 'train', a, 2000)[0] for a in ('large', 'so400')}
    records, admitted = score.admit_pair(OUTPUT, {a: t['sha256'] for a, t in terminals.items()}, train_code)
    score.check_score_layout(OUTPUT, arm, folder)
    args += ['--run-root', str(OUTPUT), '--large-terminal-sha256', admitted['large']['sha256'],
             '--so400-terminal-sha256', admitted['so400']['sha256']]
    if arm == 'so400':
        control, _, record = terminal(full, 'score', 'large')
        assert record['training'] == records['large'] and record['paired_training_terminals'] == admitted, 'control chains differ'
        args += ['--control-terminal-sha256', control['sha256']]
    return SCORER, args


def launch_line(name_, root, command, cuda_env):
    # Same systemd/watchdog argv as master 0895906e stage_native256_v9.launch_line.
    log = root / f'{name_}.log'
    argv = ['systemd-run', '--user', '--wait', '--collect', f'--unit={name_}',
            '-p', f'StandardOutput=append:{log}', '-p', f'StandardError=append:{log}',
            '-p', f'WorkingDirectory={root}', '-p', f'RuntimeMaxSec={SECONDS}', '-p', 'MemoryMax=8G',
            '-p', 'MemorySwapMax=0', '-p', 'KillMode=control-group', '-p', f'ExecStopPost=/usr/bin/python3 {FOOTERS[1]}',
            *cuda_env, '/usr/bin/flock', '-n', LOCKS[0], '/usr/bin/flock', '-n', LOCKS[1],
            '/usr/bin/time', '-v', '/bin/bash', str(command)]
    liveness = f'cg=$(systemctl --user show -P ControlGroup {name_}.service); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cpu.stat"'
    return (f'LIVENESS_CMD={shlex.quote(liveness)} STOP_CMD={shlex.quote(f"systemctl --user stop {name_}.service")} '
            f'BACKSTOP_SECONDS={BACKSTOP} {shlex.quote(str(root / "run_with_progress_watchdog.sh"))} '
            f'{shlex.quote(str(log))} -- ' + shlex.join(argv))


def launch(phase, arm, end=None):
    name, _ = target(phase, arm, end)
    command = ROOT / f'{name}-command.sh'
    absent(command)
    absent(ROOT / f'{name}.log')
    full, score, code = modules()
    script, args = arguments(full, score, code, phase, arm, end)
    expected = {ROOT / f: code[f] for f in sorted(code)}
    expected[ROOT / 'execution.json'] = sha(ROOT / 'execution.json')
    expected.update({PILOT / f: code[f] for f in PILOT_FILES})
    expected[PILOT / 'execution.json'] = code['pilot-execution.json']
    expected.update({path: sha(path) for path in FOOTERS})
    hashes = ''.join(f'{digest}  {path}\n' for path, digest in expected.items())
    check = f"sha256sum -c <<'HASHES'\n{hashes}HASHES\n"
    run = shlex.join([PYTHON, '-B', str(ROOT / script), *args]) + '\n'
    write(command, ((ROOT / 'command-header.sh').read_text() + check + run + check).encode())
    print(launch_line(name, ROOT, command, ['--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8']))


def admit(phase, arm, end=None):
    name, folder = target(phase, arm, end)
    full, _, _ = modules()
    receipt = full.descriptor(folder / 'receipt.json')
    log = full.descriptor(ROOT / f'{name}.log')
    subprocess.run([PYTHON, '-B', str(ROOT / TRAINER), 'admit-terminal', '--receipt', receipt['path'],
                    '--receipt-sha256', receipt['sha256'], '--log', log['path'], '--log-sha256', log['sha256'],
                    '--both-locks-held'], check=True)
    ref, _, _ = terminal(full, phase, arm, end)
    print('ADMIT', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), name, ref['sha256'], flush=True)


def decision():
    full, score, code = modules()
    ends = {a: terminal(full, 'train', a, 2000)[0]['sha256'] for a in ('large', 'so400')}
    records, terminals = score.admit_pair(OUTPUT, ends, {'trainer': code[TRAINER], 'partition_parser': full.PARSER_SHA})
    control_ref, control_receipt, control = terminal(full, 'score', 'large')
    _, _, candidate = terminal(full, 'score', 'so400')
    assert {p.name for p in (OUTPUT / 'official').iterdir()} == {'large', 'so400'}, 'extra official attempt'
    for arm, record in (('large', control), ('so400', candidate)):
        assert record['training'] == records[arm] and record['paired_training_terminals'] == terminals, 'score chains differ'
        assert record['code']['trainer'] == code[TRAINER] and record['code']['scorer'] == code[SCORER], 'score code differs'
    assert candidate['control_terminal'] == control_ref and candidate['control'] == control_receipt, 'paired control differs'
    paired = candidate['paired']
    expected = 'GO' if score.quality_gate(paired['candidate_minus_control']) else 'KILL'
    assert paired['decision'] == expected and paired['bootstrap_draws'] == 5000 and paired['bootstrap_seed'] == 179019, 'paired decision differs'
    print('DECISION', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), json.dumps(paired, sort_keys=True), flush=True)


if __name__ == '__main__':
    {'stage': stage, 'launch': launch, 'admit': admit, 'decision': decision}[sys.argv[1]](*sys.argv[2:])
