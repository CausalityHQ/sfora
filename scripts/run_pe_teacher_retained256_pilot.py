#!/usr/bin/env python3
"""DGX-only stdlib coordinator for twenty capped, sequential TRAIN chunks."""
if not __debug__:
    raise SystemExit('Coordinator requires Python assertions; optimized mode is forbidden')

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/riomus/runs/sfora-teacher-retained256-source-v1')
OWN = Path('/home/riomus/runs/sfora-teacher-retained256-controller-v1')
SOURCE = 'a85dd55c516f054c4c63341b31e0c7e4e77fb6925fd9ea29adabf097b1156bcb'
CPU = '84157b3331b56f868db15a38f3498daea467e24bd0aae87d76b716e0b3d2ee23'
MECHANICS = {128: '5b03da45feafcce833609f39e38ff53dce1a49ca42371f86faf6d388833283b9',
             256: 'deb9dac4783fb5da797f34b8e855b5b47ed93d36d24a428c974adb7e110005fa'}
CONTROLLER = 'sfora-teacher-retained256-controller-v1.service'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def output(width, end):
    return Path(f'/home/riomus/runs/sfora-teacher-retained256-{width}-{end}-v1')


def mechanics(width):
    return Path(f'/home/riomus/runs/sfora-teacher-retained256-mechanics-{width}-v1')


def command(width, end, previous_sha=None, control_sha=None):
    name = f'sfora-teacher-retained256-{width}-{end}-v1'
    args = ['--execution-sha256', SOURCE, '--phase', 'train', '--width', str(width), '--chunk-end', str(end),
            '--output', str(output(width, end)), '--cpu-proof', str(ROOT / 'teacher-width-cpu-proof.json'), '--cpu-sha256', CPU,
            '--control-mechanics', str(mechanics(128)), '--control-sha256', MECHANICS[128],
            '--candidate-mechanics', str(mechanics(256)), '--candidate-sha256', MECHANICS[256]]
    if end > 100:
        assert previous_sha and len(previous_sha) == 64
        args += ['--previous', str(output(width, end - 100)), '--previous-sha256', previous_sha]
    if width == 256:
        assert control_sha and len(control_sha) == 64
        args += ['--control-chunk', str(output(128, end)), '--control-chunk-sha256', control_sha]
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
            str(ROOT / 'train_pe_teacher_retained256.py'), *args]


def verify():
    assert sha(ROOT / 'teacher-retained256-execution.json') == SOURCE
    code = json.loads((ROOT / 'teacher-retained256-execution.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in code.items())
    assert sha(ROOT / 'teacher-width-cpu-proof.json') == CPU
    for width in (128, 256):
        assert sha(mechanics(width) / 'receipt.json') == MECHANICS[width]
        d = json.loads((mechanics(width) / 'receipt.json').read_text())
        assert d['pass'] and d['width'] == width and d['execution_sha256'] == SOURCE
        assert d['native_17_equals_serialized8_plus9_exact'] and d['strict400_reload_whole_head_packed_exact'] and d['training_state_discarded']
        assert d['chunk100_admission_seconds'] <= 269 and d['median_ratio_vs_control'] <= 1.10
        log = (ROOT / f'mechanics-{width}-v1.log').read_text()
        assert all(s in log for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    for end in range(100, 1001, 100):
        for width in (128, 256):
            args = command(width, end, 'a' * 64 if end > 100 else None, 'b' * 64 if width == 256 else None)
            assert args.count('RuntimeMaxSec=300') == args.count('MemoryMax=8G') == args.count('MemorySwapMax=0') == 1
            assert args[args.index('--chunk-end') + 1] == str(end)
    try:
        command(256, 100, control_sha=None)
    except AssertionError:
        pass
    else:
        raise AssertionError('unbound control accepted')


def main():
    verify()
    if sys.argv[1:] == ['--check']:
        print('PASS actual source/CPU/two discarded mechanics/caps/ordered twenty-command plan/unbound-control negative; no training')
        return
    assert not sys.argv[1:] and not (OWN / 'state.json').exists()
    OWN.mkdir(exist_ok=True)
    rows, latest = [], {128: None, 256: None}
    def save(width, end, status):
        value = {'controller_pid': os.getpid(), 'controller_sha256': sha(Path(__file__)), 'source_sha256': SOURCE,
                 'width': width, 'current_end': end, 'status': status, 'completed': rows, 'quality_read': False}
        temporary = OWN / 'state.part'; temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(OWN / 'state.json')
    for end in range(100, 1001, 100):
        for width in (128, 256):
            assert not output(width, end).exists()
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            active = subprocess.check_output(['systemctl', '--user', 'list-units', '--state=running', '--plain', '--no-legend', 'sfora-*'], text=True)
            assert all(line.split()[0] == CONTROLLER for line in active.splitlines() if line.strip()), active
            save(width, end, 'running'); print('START', width, end, flush=True)
            log_path = ROOT / f'sfora-teacher-retained256-{width}-{end}-v1.log'
            with log_path.open('x') as log:
                result = subprocess.run(command(width, end, latest[width], latest[128] if width == 256 else None), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                save(width, end, 'failed'); raise SystemExit(result.returncode)
            d = json.loads((output(width, end) / 'receipt.json').read_text())
            assert d['pass'] and d['width'] == width and d['completed_step'] == end and d['execution_sha256'] == SOURCE and not d['quality_read']
            assert d['previous_receipt_sha256'] == latest[width] and d['peak_cuda_allocated_bytes'] < 10_000_000_000
            assert sha(output(width, end) / 'resume.pt') == d['checkpoint_sha256']
            assert all(s in log_path.read_text() for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
            latest[width] = sha(output(width, end) / 'receipt.json')
            rows.append({'width': width, 'end': end, 'receipt_sha256': latest[width],
                         'training_wall_seconds': d['training_wall_seconds'], 'images_per_second': d['images_per_second'], 'peak_cuda_allocated_bytes': d['peak_cuda_allocated_bytes']})
            save(width, end, 'collected'); print('COMPLETE', json.dumps(rows[-1]), flush=True)
    save(256, 1000, 'complete'); print('ALL_FIXED1000_COLLECTED', json.dumps(latest), flush=True)


if __name__ == '__main__':
    main()
