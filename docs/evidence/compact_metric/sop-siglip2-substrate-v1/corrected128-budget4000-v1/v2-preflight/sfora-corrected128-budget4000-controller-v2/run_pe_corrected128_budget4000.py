#!/usr/bin/env python3
"""DGX-only stdlib coordinator for one fixed 20-chunk corrected continuation."""
if not __debug__:
    raise SystemExit('optimized mode is forbidden')

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/riomus/runs/sfora-corrected128-budget4000-source-v2')
OWN = Path('/home/riomus/runs/sfora-corrected128-budget4000-controller-v2')
SOURCE = '4f0483cc6c45f2049118fe945eb8836388b53fe00914b5c979208562afe069f9'
CPU = 'b470f4e19cac120f20c2c6c89edd3353e8e9a6959b899c66697e2d3d107e5ba6'
CPU_ROOT = Path('/home/riomus/runs/sfora-corrected128-budget4000-cpu-v2')
PARENT = Path('/home/riomus/runs/sfora-full-valid-anchor-2000-v1')
PARENT_RECEIPT = 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
PARENT_CHECKPOINT = 'e32dd813c456a0ed319d933b74ffbc9b42886cf0641cb673e4c36945b90c43fa'
CONTROLLER = 'sfora-corrected128-budget4000-controller-v2.service'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def output(end):
    return Path(f'/home/riomus/runs/sfora-corrected128-budget4000-{end}-v2')


def command(end, previous_sha=None):
    name = f'sfora-corrected128-budget4000-{end}-v2'
    args = ['--execution-sha256', SOURCE, '--phase', 'train', '--chunk-end', str(end),
            '--cpu-proof', str(CPU_ROOT), '--cpu-sha256', CPU, '--output', str(output(end))]
    if end > 2100:
        assert previous_sha and len(previous_sha) == 64
        args += ['--previous', str(output(end - 100)), '--previous-sha256', previous_sha]
    else:
        assert previous_sha is None
    return ['systemd-run', '--user', '--wait', '--collect', '--pipe', '--unit=' + name,
            '-p', 'WorkingDirectory=' + str(ROOT), '-p', 'RuntimeMaxSec=300', '-p', 'TimeoutStopSec=1',
            '-p', 'MemoryMax=8G', '-p', 'MemorySwapMax=0', '-p', 'KillMode=control-group',
            '--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '--setenv=PYTHONPATH=' + ':'.join(map(str, (ROOT, ROOT / 'src', ROOT / 'isolated-deps'))),
            '--setenv=PATH=/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin:/usr/local/bin:/usr/bin:/bin',
            '/usr/bin/flock', '-n', '/home/riomus/runs/.sfora-siglip2-gpu.lock',
            '/usr/bin/flock', '-n', '/home/riomus/.sfora-siglip2-gpu.lock',
            '/usr/bin/time', '-v', '-o', str(ROOT / (name + '-time.txt')),
            '/usr/bin/timeout', '-k1', '299', '/home/riomus/group-learning/.venv/bin/python', '-u',
            str(ROOT / 'train_pe_corrected128_budget4000.py'), *args]


def verify():
    assert sha(ROOT / 'corrected128-budget4000-execution.json') == SOURCE
    code = json.loads((ROOT / 'corrected128-budget4000-execution.json').read_text())
    assert len(code) == 104 and all(sha(ROOT / name) == digest for name, digest in code.items())
    assert sha(CPU_ROOT / 'receipt.json') == CPU
    cpu = json.loads((CPU_ROOT / 'receipt.json').read_text())
    assert cpu['pass'] and cpu['execution_sha256'] == SOURCE and cpu['total_updates'] == 4000
    assert cpu['exact_first2000_index_and_class_prefix'] and cpu['complete_parent_state_and_identity_verified']
    assert not cpu['quality_read'] and sha(PARENT / 'receipt.json') == PARENT_RECEIPT
    assert sha(PARENT / 'resume.pt') == PARENT_CHECKPOINT
    log = (ROOT / 'cpu-v2.log').read_text()
    assert all(s in log for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    for end in range(2100, 4001, 100):
        args = command(end, 'a' * 64 if end > 2100 else None)
        assert args.count('RuntimeMaxSec=300') == args.count('MemoryMax=8G') == args.count('MemorySwapMax=0') == 1
        assert args[args.index('--chunk-end') + 1] == str(end)
    try:
        command(2200, None)
    except AssertionError:
        pass
    else:
        raise AssertionError('unbound previous receipt accepted')


def main():
    verify()
    if sys.argv[1:] == ['--check']:
        print('PASS source/actual parent/CPU proof/caps/ordered20 commands/unbound-resume rejection; no training')
        return
    assert not sys.argv[1:] and not (OWN / 'state.json').exists()
    OWN.mkdir(exist_ok=True)
    rows, latest = [], None

    def save(end, status):
        value = {'controller_pid': os.getpid(), 'controller_sha256': sha(Path(__file__)),
                 'source_sha256': SOURCE, 'cpu_sha256': CPU, 'current_end': end,
                 'status': status, 'completed': rows, 'quality_read': False}
        temporary = OWN / 'state.part'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(OWN / 'state.json')

    for end in range(2100, 4001, 100):
        assert not output(end).exists()
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        active = subprocess.check_output(['systemctl', '--user', 'list-units', '--state=running', '--plain', '--no-legend', 'sfora-*'], text=True)
        assert all(line.split()[0] == CONTROLLER for line in active.splitlines() if line.strip()), active
        save(end, 'running'); print('START', end, flush=True)
        log_path = ROOT / f'sfora-corrected128-budget4000-{end}-v2.log'
        with log_path.open('x') as log:
            result = subprocess.run(command(end, latest), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save(end, 'failed'); raise SystemExit(result.returncode)
        receipt = json.loads((output(end) / 'receipt.json').read_text())
        assert receipt['pass'] and receipt['completed_step'] == end and receipt['execution_sha256'] == SOURCE
        assert receipt['previous_receipt_sha256'] == (latest or PARENT_RECEIPT) and not receipt['quality_read']
        assert receipt['peak_cuda_allocated_bytes'] < 10_000_000_000 and sha(output(end) / 'resume.pt') == receipt['checkpoint_sha256']
        assert all(s in log_path.read_text() for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
        latest = sha(output(end) / 'receipt.json')
        rows.append({'end': end, 'receipt_sha256': latest, 'training_wall_seconds': receipt['training_wall_seconds'],
                     'images_per_second': receipt['images_per_second'], 'peak_cuda_allocated_bytes': receipt['peak_cuda_allocated_bytes']})
        save(end, 'collected'); print('COMPLETE', json.dumps(rows[-1]), flush=True)
    save(4000, 'complete'); print('ALL_FIXED4000_COLLECTED', latest, flush=True)


if __name__ == '__main__':
    main()
