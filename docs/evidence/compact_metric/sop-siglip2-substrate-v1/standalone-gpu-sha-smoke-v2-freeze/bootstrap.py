"""Root-only smoke bootstrap: pinned helper, inherited locks, original caller, outer exit."""
import time
STARTED = time.monotonic()
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys


def read(fact):
    path = Path(fact['path'])
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError('bootstrap FILE is not canonical')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('bootstrap FILE is not regular')
        raw = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(path.stat()) or hashlib.sha256(raw).hexdigest() != fact['sha256']:
        raise ValueError('bootstrap current FILE differs')
    return raw


def resources(cgroup):
    facts = {'wall_seconds': time.monotonic() - STARTED,
             'memory_peak_bytes': int((cgroup / 'memory.peak').read_text()),
             'swap_current_bytes': int((cgroup / 'memory.swap.current').read_text()),
             'memory_events': {k: int(v) for k, v in (line.split() for line in (cgroup / 'memory.events').read_text().splitlines())}}
    if (cgroup / 'memory.max').read_text().strip() != '8589934592' or (cgroup / 'memory.swap.max').read_text().strip() != '0' or facts['memory_peak_bytes'] > 8589934592 or facts['swap_current_bytes'] != 0 or any(facts['memory_events'].values()) or facts['wall_seconds'] >= 1500:
        raise ValueError('bootstrap whole-unit resources differ: ' + json.dumps(facts, sort_keys=True))
    return facts


def main():
    authority_path, authority_sha, self_sha = sys.argv[1:]
    own = {'path': str(Path(__file__).resolve()), 'sha256': self_sha}
    read(own)
    fact = {'path': authority_path, 'sha256': authority_sha}
    raw = read(fact)
    plan = json.loads(raw)
    if plan.keys() != {'schema', 'files', 'native_authority', 'output', 'interpreter'} or plan['schema'] != 'cuda-sha256-smoke-bootstrap-v1' or plan['files'].keys() != {'driver', 'test', 'requests', 'probe_serializer', 'mlp_serializer'} or Path(sys.executable).resolve() != Path(plan['interpreter']['path']):
        raise ValueError('bootstrap schema/interpreter differs')
    read(plan['interpreter'])
    for item in plan['files'].values():
        read(item)
    helper = plan['files']['requests']
    name = 'qualify_connected_serving_requests'
    if name in sys.modules:
        raise ValueError('bootstrap helper registry is occupied')
    spec = importlib.util.spec_from_file_location(name, helper['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(read(helper), helper['path'], 'exec', dont_inherit=True), vars(module))
    source = module.Source(module, helper)
    cgroup = Path('/sys/fs/cgroup') / next(line.split(':', 2)[2].lstrip('/') for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    fds, failures = [], []
    try:
        for expected, path in zip((3, 4), ('/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock'), strict=True):
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            fds.append(fd)
            if fd != expected:
                raise ValueError('bootstrap inherited lock descriptor differs')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks = module.Locks([{'path': '/home/riomus/runs/.sfora-siglip2-gpu.lock', 'fd': 3}, {'path': '/home/riomus/.sfora-siglip2-gpu.lock', 'fd': 4}])
        source.check()
        resources(cgroup)
        native = plan['native_authority']
        sys.argv = [plan['files']['driver']['path'], 'smoke', '--authority', native['path'], '--authority-sha256', native['sha256'], '--output', plan['output']]
        runpy.run_path(sys.argv[0], run_name='__main__')
    except BaseException as error:
        failures.append(error)
    finally:
        checks = [source.check, lambda: read(own), lambda: read(fact), lambda: read(plan['interpreter'])]
        checks.extend(lambda item=item: read(item) for item in plan['files'].values())
        if 'locks' in locals():
            checks.append(locks.check)
        checks.append(lambda: resources(cgroup))
        for check in checks:
            try:
                check()
            except BaseException as error:
                failures.append(error)
        for fd in reversed(fds):
            os.close(fd)
    if failures:
        module.raise_failures(failures)
    print(json.dumps({'bootstrap_exit_pass': True, 'resources': resources(cgroup), 'normal_outer_terminal_required': True}, sort_keys=True))


if __name__ == '__main__':
    main()
