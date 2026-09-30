#!/usr/bin/env python3
"""One CPU allocation diagnostic; its outputs cannot admit GPU or timing."""
if not __debug__:
    raise SystemExit('Assertions required')
import argparse
import hashlib
import json
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--controller-sha256', required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == a.controller_sha256
assert not a.output.exists()
root = Path('/home/riomus/runs/sfora-full-valid-anchor-public-source-v2')
sys.path.insert(0, str(root))
import qualify_full_valid_anchor_serving as q
assert q.sha(root / q.MANIFEST) == '8c6a1412524606518d5f29ee029036d82a349f302193276f3f951aa10fc8eefb'
group = Path('/sys/fs/cgroup') / next(x.split(':', 2)[2].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
assert int((group / 'memory.max').read_text()) == q.HOST_CAP and int((group / 'memory.swap.max').read_text()) == 0
samples = []
stage = 'before_main'
start = time.perf_counter()
stop = threading.Event()
watched = {'startup', 'runtime_startup', 'native_reload', 'model_facts', 'cpu_phase', 'load_arm', 'configure', 'usage'}
def profile(frame, event, arg):
    global stage
    if event in ('call', 'return') and frame.f_code.co_name in watched:
        stage = Path(frame.f_code.co_filename).name + ':' + frame.f_code.co_name + ':' + event
        if frame.f_code.co_name == 'native_reload':
            stage += ':independent=' + str(frame.f_locals.get('independent'))
def sample():
    stats = {k: int(v) for k, v in (x.split() for x in (group / 'memory.stat').read_text().splitlines())}
    process = {k.rstrip(':'): int(v.split()[0]) * 1024 for k, v in (x.split(':', 1) for x in Path('/proc/self/status').read_text().splitlines() if x.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssFile:', 'RssShmem:', 'VmSwap:')))}
    samples.append({'seconds': time.perf_counter() - start, 'stage': stage,
                    'current': int((group / 'memory.current').read_text()), 'peak': int((group / 'memory.peak').read_text()),
                    'stat': {k: stats[k] for k in ('anon', 'file', 'kernel', 'shmem', 'file_mapped', 'inactive_file', 'active_file')},
                    'process': process})
def monitor():
    while not stop.wait(.05):
        sample()
thread = threading.Thread(target=monitor, daemon=True)
sample(); thread.start(); sys.setprofile(profile)
status, error = 0, None
sys.argv = [str(root / 'qualify_full_valid_anchor_serving.py'), '--phase', 'cpu', '--authority-sha256', '3f7b29b3034a20b76765d822b0f1d5d618d8c5aea736a541099728594d33f4c1', '--output', str(a.output.parent / 'UNADMITTED-diagnostic-proof.json')]
try:
    q.main()
except BaseException as exc:
    status, error = 1, {'type': type(exc).__name__, 'message': str(exc)}
    traceback.print_exc()
finally:
    sys.setprofile(None); stop.set(); thread.join(); sample()
    value = {'schema': 'full2000-memory-profile-v1', 'diagnostic_only': True, 'claim_eligible': False,
             'controller_sha256': a.controller_sha256, 'source_execution_sha256': '8c6a1412524606518d5f29ee029036d82a349f302193276f3f951aa10fc8eefb',
             'qualifier_exit_status': status, 'error': error, 'samples': samples, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'model_quality_or_speed_measurement': False, 'cgroup': str(group)}
    with a.output.open('x') as f:
        json.dump(value, f, indent=2)
    highest = max(samples, key=lambda x: x['current'])
    print('DIAGNOSTIC current maximum', json.dumps(highest), flush=True)
    print('DIAGNOSTIC qualifier status', status, 'no qualification admission', flush=True)
raise SystemExit(status)
