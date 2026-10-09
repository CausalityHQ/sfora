"""Root-only one-shot compile; no CUDA/library load. Unknown input => no build admission."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


def digest(path):
    path = Path(path)
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError('noncanonical build FILE: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('build FILE is not regular')
        h = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(path.stat()):
        raise ValueError('build FILE changed during hashing')
    return {'sha256': h.hexdigest(), 'size': before.st_size}


def check(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError('build input changed: ' + path)


def observed_inputs(trace, generated, devices):
    result = set()
    for line in trace.splitlines():
        if ('openat(' in line or 'open(' in line) and 'O_RDONLY' in line:
            hit = re.search(r'= \d+<(/[^>]+)>', line)
        elif 'execve(' in line and re.search(r'= 0\s*$', line):
            hit = re.search(r'execve\("(/[^"]+)"', line)
        else:
            hit = None
        if hit:
            path = Path(hit[1].split('<', 1)[0]).resolve()
            if path.is_relative_to(generated):
                continue
            path = path.resolve(strict=True)
            if path.is_dir():
                continue
            if str(path) in devices:
                info = path.stat()
                if not stat.S_ISCHR(info.st_mode) or {'mode': info.st_mode, 'rdev': info.st_rdev} != devices[str(path)]:
                    raise ValueError('compiler device identity differs')
                continue
            result.add(str(path))
    return sorted(result)


def main():
    started = time.monotonic()
    authority_path, authority_sha = sys.argv[1:]
    if digest(authority_path)['sha256'] != authority_sha:
        raise ValueError('build authority differs')
    authority = json.loads(Path(authority_path).read_text())
    inputs = authority['inputs']
    check(inputs)
    output = Path(authority['output_dir'])
    output.mkdir(mode=0o700)
    (output / 'tmp').mkdir()
    (output / 'intermediates').mkdir()
    locks = []
    try:
        for path in ['/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock']:
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(fd)
        cgroup = Path('/sys/fs/cgroup') / next(line.split(':', 2)[2].lstrip('/') for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        if (cgroup / 'memory.max').read_text().strip() != '8589934592' or (cgroup / 'memory.swap.max').read_text().strip() != '0':
            raise ValueError('compile unit caps differ')
        command = ['/usr/bin/strace', '-f', '-yy', '-e', 'trace=openat,open,execve,readlink', '-o', str(output / 'files.trace'), *authority['command']]
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            result = subprocess.run(command, env=authority['environment'], stdout=stdout, stderr=stderr, timeout=max(1, 270 - (time.monotonic() - started)))
        check(inputs)
        observed = observed_inputs((output / 'files.trace').read_text(), output, authority['device_inputs'])
        unknown = sorted(set(observed) - inputs.keys())
        library = Path(authority['library'])
        inspections = {}
        if result.returncode == 0:
            for label, args in [('elf', ['/usr/bin/readelf', '-d', str(library)]), ('symbols', ['/usr/bin/readelf', '--dyn-syms', '--wide', str(library)]), ('sass', [authority['cuobjdump'], '--list-elf', str(library)]), ('ptx', [authority['cuobjdump'], '--list-ptx', str(library)])]:
                inspected = subprocess.run(args, env=authority['environment'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                raw = inspected.stdout
                (output / (label + '.log')).write_bytes(raw)
                inspections[label] = {'exit_status': inspected.returncode, 'text': raw.decode(errors='replace')}
        resources = {'memory_peak_bytes': int((cgroup / 'memory.peak').read_text()), 'memory_events': dict(line.split() for line in (cgroup / 'memory.events').read_text().splitlines()), 'swap_current_bytes': int((cgroup / 'memory.swap.current').read_text()), 'wall_seconds': time.monotonic() - started}
        accepted = result.returncode == 0 and not unknown and all(v['exit_status'] == 0 for v in inspections.values()) and 'sm_121' in inspections['sass']['text'] and not inspections['ptx']['text'].strip() and 'sfora_sha256_occurrences' in inspections['symbols']['text'] and resources['memory_peak_bytes'] <= 8589934592 and resources['swap_current_bytes'] == 0 and all(int(v) == 0 for v in resources['memory_events'].values()) and resources['wall_seconds'] < 300
        check(inputs)
        if digest(authority_path)['sha256'] != authority_sha:
            raise ValueError('build authority changed at exit')
        record = {'schema': 'standalone-gpu-sha-build-observation-v1', 'compiler_exit': result.returncode, 'build_admitted': accepted, 'unknown_inputs': unknown, 'observed_inputs': observed, 'library': {'path': str(library), **digest(library)} if library.exists() else None, 'inspections': inspections, 'resources': resources, 'cuda_launched': False, 'library_loaded': False, 'both_locks_held': True, 'speed_go': False, 'product_go': False}
        with (output / 'observation.json').open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({'build_admitted': accepted, 'compiler_exit': result.returncode, 'unknown_inputs': unknown, 'resources': resources}))
        return 0 if accepted else 1
    finally:
        for fd in reversed(locks):
            os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
