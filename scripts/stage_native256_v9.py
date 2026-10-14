#!/usr/bin/env python3
"""Stage and launch native256 v9 So400m-vs-Large TRAIN100 units from the frozen v6 root (run on DGX, stdlib only).

v9 differs from v6 only in the trainer/test bytes (hang-backstop policy) and resource_policy.seconds in each authority.
Every other authority field (qualifier root/authority, selected CPU proof) is copied verbatim from the same-arm v6 file.

  stage SRC_DIR                          copy v9 trainer+test from SRC_DIR, other FILES from v6; write execution.json
  authority PHASE ARM [SEED]             write authority + command.sh (TRAIN binds the v9 mechanics terminal)
  launch PHASE ARM [SEED]                print the systemd-run command (run it under run_with_progress_watchdog.sh)
"""
import hashlib
import json
import os
import re
import shlex
import sys
from decimal import Decimal
from pathlib import Path

RUNS = Path(os.environ.get('SFORA_NATIVE256_RUNS', '/home/riomus/runs'))
OLD, NEW = RUNS / 'sfora-native256-adaptation-source-v6', RUNS / 'sfora-native256-adaptation-source-v9'
# A new attempt (e.g. 'r1') gets fresh unit/output/log/authority/command names; earlier attempts stay untouched.
TAG = 'v9' + os.environ.get('SFORA_NATIVE256_ATTEMPT', '')
FILES = ('deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py',
         'test_siglip2_substrate_adaptation.py', 'train_siglip2_substrate_adaptation.py')
CHANGED = {'test_siglip2_substrate_adaptation.py', 'train_siglip2_substrate_adaptation.py'}
SECONDS = 3600
BACKSTOP = SECONDS + 180  # watchdog backstop strictly after systemd RuntimeMaxSec
V9 = {'train_siglip2_substrate_adaptation.py': '1b73b2a6bdc1ee0cc468109e15746c8c5cde32569ac3b5de05d07f9e2f0b573f',
      'test_siglip2_substrate_adaptation.py': '7df1d58718709b3a1230ec6fef0cc9a806abcb9e4eceaba4c0267df5edc3e9c2'}
PYTHON = '/home/riomus/group-learning/.venv/bin/python'
FOOTERS = (RUNS / 'sfora-native256-initialized-cpu-final-footer-v1.py', RUNS / 'sfora-native256-stop-cgroup-footer-v1.py')
LOCKS = ('/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return sha(path)


def name(phase, arm, seed, version):
    return f'{arm}-{phase}' + (f'-{seed}' if phase == 'train' else '') + f'-{version}'


def unit(phase, arm, seed):
    return f'sfora-native256-{phase}-{arm}' + (f'-{seed}' if phase == 'train' else '') + f'-{TAG}'


def stage(src):
    NEW.mkdir()
    old_code = json.loads((OLD / 'execution.json').read_text())
    for file in FILES:
        (NEW / file).write_bytes(((Path(src) if file in CHANGED else OLD) / file).read_bytes())
    code = {file: sha(NEW / file) for file in FILES}
    assert all(code[f] == old_code[f] for f in FILES if f not in CHANGED), 'unchanged reference bytes differ from v6'
    assert all(code[f] == V9[f] for f in CHANGED) or os.environ.get('SFORA_NATIVE256_TEST_BYTES') == '1', 'v9 bytes differ from 79e449bf'
    print('execution_sha256', dump(NEW / 'execution.json', code))


def terminal(phase, arm, seed):
    """Descriptor of a completed v9 TRAIN-root unit, from its original log and receipt."""
    name_ = unit(phase, arm, seed)
    return terminal_from(NEW / f'{name_}.log', RUNS / name_ / 'receipt.json', name_)


def terminal_from(log, receipt, name_, runtime=None):
    """TERMINAL descriptor from an original unit log + receipt (admission re-checks everything).

    runtime=None: frozen TRAIN/mechanics parser. Held v2 passes the staged exporter's service_runtime, which its
    admit_held_terminal re-checks ('3min 81ms' = systemd omitting a zero seconds term).
    """
    text = log.read_text()
    assert not re.search(r'^watchdog:', text, re.M), 'watchdog-stopped run is invalid and never reused'
    invocation = re.search(r'^Running as unit: \S+\.service; invocation ID: ([0-9a-f]{32})$', text, re.M).group(1)
    if runtime is not None:
        lines = [line.removeprefix('Service runtime: ') for line in text.splitlines() if line.startswith('Service runtime: ')]
        assert len(lines) == 1, 'exactly one service runtime line required'
        minutes, seconds = None, str(runtime(lines[0]))
    else:
        # Admission parses only '[Nmin ]S[.f]s'; systemd may print '...ms' or omit seconds. Pre-registered: such a
        # valid run is rerun under a new unit name (not a failed state), never re-parsed.
        assert re.search(r'^Service runtime: (?:\d+min )?\d+(?:\.\d+)?s$', text, re.M), 'unadmissible runtime format: rerun'
        minutes, seconds = re.search(r'^Service runtime: (?:(\d+)min )?(\d+(?:\.\d+)?)s$', text, re.M).groups()
    rss = int(re.search(r'^\tMaximum resident set size \(kbytes\): (\d+)$', text, re.M).group(1))
    return {'both_locks_held': True, 'invocation_id': invocation, 'log': {'path': str(log), 'sha256': sha(log)},
            'native_peak_rss_kib': rss, 'receipt': {'path': str(receipt), 'sha256': sha(receipt)},
            'service_seconds': float(Decimal(minutes or '0') * 60 + Decimal(seconds)), 'unit': name_}


def command_path(phase, arm, seed):
    plain = f'{name(phase, arm, seed, TAG)}-command.sh'
    return NEW / (plain.replace('-v9-command', '-command') if TAG == 'v9' else plain)


def authority(phase, arm, seed=None):
    seed = int(seed or 179032)
    value = json.loads((OLD / f'authority-{name(phase, arm, seed, "v6")}.json').read_text())
    assert value['phase'] == phase and value['arm'] == arm and value['seed'] == seed and value['resource_policy']['seconds'] == 300
    value['execution_sha256'] = sha(NEW / 'execution.json')
    value['resource_policy'] = {**value['resource_policy'], 'seconds': SECONDS}
    value['selected_mechanics'] = None if phase == 'mechanics' else terminal('mechanics', arm, 179032)
    path = NEW / f'authority-{name(phase, arm, seed, TAG)}.json'
    digest = dump(path, value)
    old_command = (OLD / f'{name(phase, arm, seed, "v6")}-command.sh'.replace('-v6-command', '-command')).read_text()
    header = old_command.split('sha256sum -c', 1)[0]
    assert len(re.findall(r'^cd .*$', header, re.M)) == 1, 'v6 command header differs'
    header = re.sub(r'^cd .*$', f'cd {NEW}', header, flags=re.M)
    hashes = ''.join(f'{sha(NEW / f)}  {f}\n' for f in (*FILES, 'execution.json', path.name))
    hashes += ''.join(f'{sha(f)}  {f}\n' for f in FOOTERS)
    check = f"sha256sum -c <<'HASHES'\n{hashes}HASHES\n"
    run = (f'{PYTHON} -B {NEW / "train_siglip2_substrate_adaptation.py"} --execution-sha256 {value["execution_sha256"]} '
           f'--authority {path} --authority-sha256 {digest} --phase {phase} --arm {arm} --seed {seed} '
           f'--output {RUNS / unit(phase, arm, seed)}\n')
    command = command_path(phase, arm, seed)
    command.write_text(header + check + run + check)
    print(path, digest, command)


def launch(phase, arm, seed=None):
    seed = int(seed or 179032)
    command = command_path(phase, arm, seed)
    print(launch_line(unit(phase, arm, seed), NEW, command,
                      ['--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8']))


def launch_line(name_, root, command, cuda_env):
    # No --pipe: with --pipe systemd-run 255 skips waiting for the start job and can print "Running as unit"
    # without the invocation ID that admission requires (observed for both first-attempt mechanics units).
    # Unit stdout/stderr append to the same log the client writes, keeping the v6 line format.
    log = root / f'{name_}.log'
    argv = ['systemd-run', '--user', '--wait', '--collect', f'--unit={name_}',
            '-p', f'StandardOutput=append:{log}', '-p', f'StandardError=append:{log}',
            '-p', f'WorkingDirectory={root}', '-p', f'RuntimeMaxSec={SECONDS}', '-p', 'MemoryMax=8G',
            '-p', 'MemorySwapMax=0', '-p', 'KillMode=control-group', '-p', f'ExecStopPost=/usr/bin/python3 {FOOTERS[1]}',
            *cuda_env, '/usr/bin/flock', '-n', LOCKS[0], '/usr/bin/flock', '-n', LOCKS[1],
            '/usr/bin/time', '-v', '/bin/bash', str(command)]
    # Liveness = log growth OR the unit cgroup's cpu.stat moving (io controller is not delegated on DGX).
    # Empty ControlGroup (unit not yet started/already collected) must print nothing, never the root cgroup.
    liveness = f'cg=$(systemctl --user show -P ControlGroup {name_}.service); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cpu.stat"'
    return (f"LIVENESS_CMD='{liveness}' STOP_CMD='systemctl --user stop {name_}.service' "
            f'BACKSTOP_SECONDS={BACKSTOP} {Path(__file__).absolute().parent / "run_with_progress_watchdog.sh"} '
            f'{log} -- ' + shlex.join(argv))


if __name__ == '__main__':
    {'stage': stage, 'authority': authority, 'launch': launch}[sys.argv[1]](*sys.argv[2:])
