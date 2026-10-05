#!/usr/bin/env python3
"""External read-only observer; it provides no qualification or state admission.

Parent owns the sole native job and terminates this observer in its finally.
CLI: --cgroup EXACT_SERVICE_CGROUP --log ORIGINAL_LOG --output NEW_JSONL
Fixed .5s cadence, 1200 sample cap, 550s watchdog, 16MiB exclusive output.
"""
import argparse
import json
from pathlib import Path
import signal
import time

MEMORY_FILES = ('memory.current', 'memory.peak', 'memory.max', 'memory.stat',
                'memory.events', 'memory.events.local', 'memory.swap.current',
                'memory.swap.peak', 'memory.swap.max', 'cgroup.procs', 'cgroup.events')
STATUS_FIELDS = {'Name', 'VmRSS', 'VmHWM', 'VmData', 'RssAnon', 'RssFile', 'RssShmem'}
STOP = False


def read_small(path):
    with path.open() as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError('input exceeds64KiB: '+str(path))
    return value.strip()


def snapshot(cgroup, proc=Path('/proc')):
    values, errors, pids = {}, {}, {}
    for name in MEMORY_FILES:
        try:
            values[name] = read_small(cgroup/name)
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    ids = values.get('cgroup.procs', '').split()
    if len(ids) > 256:
        errors['cgroup.procs'] = 'PID limit256 exceeded; statuses omitted'
    else:
        for pid in ids:
            if not pid.isdigit():
                errors['cgroup.procs'] = 'nondecimal PID'; continue
            try:
                status = read_small(proc/pid/'status')
                pids[pid] = {key: value.strip() for line in status.splitlines()
                             if ':' in line for key, value in [line.split(':', 1)]
                             if key in STATUS_FIELDS}
            except (OSError, ValueError) as error:
                pids[pid] = {'error': str(error)}
    return {'values': values, 'errors': errors, 'pids': pids}


class Phases:
    def __init__(self, path):
        self.path, self.offset, self.partial, self.last, self.inode = path, 0, b'', None, None

    def read(self):
        try:
            with self.path.open('rb') as stream:
                stat = self.path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if self.inode is not None and (identity != self.inode or stat.st_size < self.offset):
                    raise ValueError('original log replaced or truncated')
                self.inode = identity
                stream.seek(self.offset)
                chunk = stream.read(65536)
                self.offset += len(chunk)
        except FileNotFoundError:
            return self.last
        lines = (self.partial+chunk).split(b'\n')
        self.partial = lines.pop()
        if len(self.partial) > 65536:
            raise ValueError('phase log line exceeds64KiB')
        for raw in lines:
            try:
                line = raw.decode('utf-8')
                value = json.loads(line)
                if isinstance(value, dict) and value.get('event') == 'COMPACT_PHASE':
                    self.last = line
            except (UnicodeError, json.JSONDecodeError):
                pass
        return self.last


def emit(stream, record, used, cap=16*1024**2):
    line = json.dumps(record, sort_keys=True, allow_nan=False)+'\n'
    size = len(line.encode('utf-8'))
    if used+size > cap:
        raise ValueError('output byte cap exceeded')
    stream.write(line); stream.flush()
    return used+size


def observe(cgroup, log, output, *, clock=time.monotonic, sleep=time.sleep):
    reader = Phases(log)
    start, seen, used, count = clock(), False, 0, 0
    with output.open('x', encoding='utf-8') as stream:
        while not STOP and clock()-start < 550 and count < 1200:
            began = time.monotonic_ns()
            exists = cgroup.is_dir()
            if seen and not exists:
                reason = 'cgroup_disappeared'; break
            seen = seen or exists
            record = {'event': 'sample' if exists else 'waiting', 'sample': count,
                      'monotonic_ns': began, 'realtime_ns': time.time_ns(),
                      'cgroup': str(cgroup), 'last_COMPACT_PHASE': reader.last}
            try:
                record['last_COMPACT_PHASE'] = reader.read()
            except (OSError, ValueError) as error:
                record['phase_log_error'] = str(error)
            if exists:
                record.update(snapshot(cgroup))
            record['sample_end_monotonic_ns'] = time.monotonic_ns()
            # Reserve space for an explicit terminal record; never truncate data.
            used = emit(stream, record, used, 16*1024**2-4096)
            count += 1
            sleep(min(.5, max(0., 550-(clock()-start))))
        else:
            reason = 'signal' if STOP else 'watchdog' if clock()-start >= 550 else 'sample_cap'
        emit(stream, {'event': 'observer_terminal', 'reason': reason, 'samples': count,
                      'cgroup_seen': seen, 'monotonic_ns': time.monotonic_ns(),
                      'elapsed_seconds': clock()-start, 'qualification_eligible': False}, used)
    return reason


def stop(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cgroup', 'log', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if (not args.cgroup.is_absolute() or not args.cgroup.is_relative_to('/sys/fs/cgroup') or
            not args.cgroup.name.endswith('.service') or args.cgroup.resolve() != args.cgroup):
        parser.error('exact canonical /sys/fs/cgroup/.../unique.service required')
    for path in (args.log, args.output):
        if not path.is_absolute() or path.parent.resolve() != path.parent or path.is_symlink():
            parser.error('canonical absolute log/new output paths required')
    if args.log == args.output:
        parser.error('separate observation output required')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    reason = observe(args.cgroup, args.log, args.output)
    print(json.dumps({'observer_terminal': reason, 'output': str(args.output),
                      'qualification_eligible': False}), flush=True)


if __name__ == '__main__':
    main()
