#!/usr/bin/env python3
"""Collect SOURCE-only cuDNN wheel byte provenance; never admit native libraries.

Parent launches a SHA-pinned interpreter with -I -S -B and the fixed CLI order:
--execution-sha256 SHA --authority PATH --authority-sha256 SHA --output NEWDIR.
Adjacent execution.json maps exactly this file and test_cudnn_wheel_provenance.py
to their final SHA256s. Authority schema cudnn-wheel-provenance-authority-v1 has
exact keys: schema, execution_sha256, installed_evidence {path,sha256},
installed_site_root, official_wheel {url,sha256,size_bytes,release_url}, python
{path,sha256}, python_version, output, unit (without .service), both_locks_held,
lock_paths, resource_policy. Fixed values are constants below; paths/hashes,
interpreter version and unit are frozen by the parent before launch.

Download-only: exclusive NEWDIR holds the intact wheel and proof.json. Hash and
size precede ZIP parsing; only four library members are streamed, never loaded
or extracted. METADATA/WHEEL must match exactly. RECORD's selected six hashed
rows must match; full RECORD hashes and row differences are reported because
pip may change installation bookkeeping. Installed RECORD is independently
pinned to the original evidence, never trusted as sole library authentication.
All installed *.dist-info/RECORD files are streamed to establish unique ownership
of the four canonical targets; their bytes and the RECORD inventory are guarded
through exit. Other distributions' METADATA and library bytes are not consumed.

Both locks must surround the whole parent unit. Internal snapshots do not prove
terminal lock lifetime, normal exit or final cgroup peaks: parent original
footers remain mandatory. A proof is byte provenance only, not model/quality
qualification or amendment of the failed original native-origin authority.
Failures may leave NEW files; do not reuse them. No non-stdlib imports.
"""
if not __debug__:
    raise SystemExit('Unoptimized source-only execution required')

import argparse
import base64
import csv
import hashlib
import importlib.machinery
import io
import json
import os
import re
import resource
import signal
import stat
import sys
import time
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

CHUNK = 1024**2
FILES = {'qualify_cudnn_wheel_provenance.py', 'test_cudnn_wheel_provenance.py'}
POLICY = {'seconds': 300, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0, 'cuda_visible_devices': ''}
LOCKS = ['/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock']
DIST = 'nvidia_cudnn_cu13-9.20.0.48.dist-info/'
MEMBERS = {'nvidia/cudnn/lib/libcudnn_' + name + '.so.9' for name in
           ('engines_precompiled', 'engines_runtime_compiled', 'graph', 'heuristic')}
META = {name: DIST + name for name in ('METADATA', 'WHEEL', 'RECORD')}
WHEEL = {
    'url': 'https://files.pythonhosted.org/packages/56/c5/83384d846b2fd17c44bd499b36c75a45ed4f095fbbb2252294e89cea5c5c/nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_aarch64.whl',
    'sha256': 'e31454ae00094b0c55319d9d15b6fa2fc50a9e1c0f5c8c80fb75258234e731e1',
    'size_bytes': 444574296,
    'release_url': 'https://pypi.org/pypi/nvidia-cudnn-cu13/9.20.0.48/json',
}
AUTHORITY_KEYS = {'schema', 'execution_sha256', 'installed_evidence', 'installed_site_root',
                  'official_wheel', 'python', 'python_version', 'output', 'unit',
                  'both_locks_held', 'lock_paths', 'resource_policy'}
NATIVE = {'torch', 'torchvision', 'numpy', 'nvidia', 'safetensors', 'transformers', 'ctypes'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'lowercase SHA256 required')
    return value


def canonical(value, directory=False):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path and
            (path.is_dir() if directory else path.is_file()), 'canonical regular path required: ' + str(path))
    return path


def signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def stream_hash(stream, size, output=None):
    """Bound every read to 1MiB and reject short or oversized streams."""
    whole, count = hashlib.sha256(), 0
    while True:
        block = stream.read(min(CHUNK, size - count + 1))
        if not block:
            break
        count += len(block)
        require(count <= size, 'stream exceeds expected size')
        whole.update(block)
        if output is not None:
            output.write(block)
    require(count == size, 'stream size differs')
    return {'sha256': whole.hexdigest(), 'size_bytes': count}


def file_fact(value):
    path = canonical(value)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'regular file required')
        fact = stream_hash(stream, before.st_size)
        require(signature(before) == signature(os.fstat(stream.fileno())) == signature(path.stat()),
                'file changed during hash: ' + str(path))
    canonical(path)
    return fact


def bind(value, expected, guards, size=None):
    digest(expected)
    path = canonical(value)
    fact = file_fact(path)
    require(fact['sha256'] == expected and (size is None or fact['size_bytes'] == size),
            'bound file hash/size differs: ' + str(path))
    require(guards.setdefault(str(path), fact) == fact, 'conflicting file binding')
    return path


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('invalid JSON constant: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(type(result) is dict, 'JSON object required')
    return result


def bounded_bytes(path):
    with canonical(path).open('rb') as stream:
        raw = stream.read(CHUNK + 1)
    require(len(raw) <= CHUNK, 'metadata/JSON exceeds 1MiB')
    return raw


def bound_bytes(value, expected, guards):
    path = bind(value, expected, guards)
    raw = bounded_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == expected, 'bytes changed during read')
    return raw


def reference(value, guards):
    require(type(value) is dict and value.keys() == {'path', 'sha256'}, 'file reference differs')
    return bound_bytes(value['path'], value['sha256'], guards)


def closure(root, expected, guards):
    code = strict_json(bound_bytes(root / 'execution.json', expected, guards))
    require(code.keys() == FILES, 'execution requires exactly two source files')
    for name, sha in code.items():
        bind(root / name, sha, guards)
    return code


def new_output(value, roots):
    path = Path(value)
    require(path.is_absolute() and path.parent.resolve() == path.parent and path.parent.is_dir() and
            not path.exists() and not path.is_symlink(), 'exclusive canonical NEWDIR required')
    require(all(not path.is_relative_to(root) and not Path(root).is_relative_to(path) for root in roots),
            'output overlaps source/input')
    return path


def source_only():
    require(sys.flags.optimize == 0 and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and
            sys.gettrace() is None and sys.getprofile() is None and
            isinstance(__loader__, importlib.machinery.SourceFileLoader) and
            Path(__loader__.path) == canonical(__file__), 'unoptimized -I -S -B source-only startup required')
    require(not any(name.split('.')[0] in NATIVE for name in sys.modules), 'native package imported')
    maps = Path('/proc/self/maps').read_text()
    require(not re.search(r'/(?:libcudnn|libtorch|libcuda|libcublas)[^/\s]*\.so', maps), 'native library loaded')


def prepare(args):
    source_only()
    root = canonical(__file__).parent
    require(sys.argv == [str(root / 'qualify_cudnn_wheel_provenance.py'), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--output', str(args.output)], 'fixed canonical CLI order required')
    guards = {}
    code = closure(root, args.execution_sha256, guards)
    launch = strict_json(bound_bytes(args.authority, args.authority_sha256, guards))
    require(launch.keys() == AUTHORITY_KEYS and launch['schema'] == 'cudnn-wheel-provenance-authority-v1' and
            launch['execution_sha256'] == args.execution_sha256 and launch['official_wheel'] == WHEEL and
            type(launch['official_wheel']['size_bytes']) is int and launch['resource_policy'] == POLICY and
            all(type(launch['resource_policy'][key]) is int for key in ('seconds', 'host_bytes', 'swap_bytes')) and
            launch['both_locks_held'] is True and launch['lock_paths'] == LOCKS and
            launch['output'] == str(args.output) and isinstance(launch['unit'], str) and
            re.fullmatch('[A-Za-z0-9_-]+', launch['unit']), 'parent authority differs')
    python = launch['python']
    require(type(python) is dict and python.keys() == {'path', 'sha256'} and
            str(Path(sys.executable).resolve()) == python['path'] and launch['python_version'] == sys.version,
            'interpreter path/version differs')
    bind(python['path'], python['sha256'], guards)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'CUDA hidden/systemd invocation required')
    site = canonical(launch['installed_site_root'], directory=True)
    evidence = strict_json(reference(launch['installed_evidence'], guards))
    selected, metadata = evidence_files(evidence, site)
    new_output(args.output, [root, site, canonical(args.authority), canonical(launch['installed_evidence']['path'])])
    return {'launch': launch, 'guards': guards, 'code': code, 'selected': selected, 'metadata': metadata}


def evidence_files(evidence, site):
    require(evidence.keys() == {'schema', 'entries', 'native_imported', 'independent_upstream_authentication'} and
            evidence['schema'] == 'cudnn-installed-record-readonly-evidence-v1' and
            evidence['native_imported'] is False and evidence['independent_upstream_authentication'] is False and
            type(evidence['entries']) is list and len(evidence['entries']) == 1, 'installed evidence schema differs')
    entry = evidence['entries'][0]
    require(entry.keys() == {'files', 'metadata', 'wheel', 'record'} and
            entry['files'].keys() == {str(site / name) for name in MEMBERS}, 'exact four installed files required')
    selected = {}
    for name in sorted(MEMBERS):
        row = entry['files'][str(site / name)]
        require(row.keys() == {'observed_native_sha256', 'record_sha256', 'record_size_bytes'} and
                digest(row['observed_native_sha256']) == digest(row['record_sha256']) and
                type(row['record_size_bytes']) is int and row['record_size_bytes'] > 0, 'installed evidence file differs')
        selected[name] = {'path': str(site / name), 'sha256': row['observed_native_sha256'],
                          'size_bytes': row['record_size_bytes']}
    metadata = {}
    for kind, name in META.items():
        row = entry[kind.lower()]
        require(row.keys() == ({'path', 'sha256'} if kind == 'RECORD' else {'path', 'sha256', 'text'}) and
                row['path'] == str(site / name), 'installed metadata path/fields differ')
        digest(row['sha256'])
        if kind != 'RECORD':
            require(hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256'], 'evidence metadata text differs')
        metadata[kind] = row
    return selected, metadata


def installed_snapshot(selected, metadata, guards):
    facts = {}
    for name, row in selected.items():
        bind(row['path'], row['sha256'], guards, row['size_bytes'])
        facts[name] = {key: row[key] for key in ('sha256', 'size_bytes')}
    raw = {kind: bound_bytes(row['path'], row['sha256'], guards) for kind, row in metadata.items()}
    for kind in ('METADATA', 'WHEEL'):
        require(raw[kind] == metadata[kind]['text'].encode(), 'installed metadata text changed')
    return facts, raw


def installed_records(site):
    return [str(canonical(path)) for path in sorted(site.glob('*.dist-info/RECORD'))]


def record_ownership(site, selected, guards):
    records = installed_records(site)
    owners = {str(canonical(row['path'])): [] for row in selected.values()}
    expected = str(site / META['RECORD'])
    for name in records:
        fact = file_fact(name)
        require(guards.setdefault(name, fact) == fact, 'installed RECORD changed before ownership scan')
        with Path(name).open('r', encoding='utf-8', newline='') as stream:
            for row in csv.reader(stream, strict=True):
                require(len(row) == 3 and row[0], 'installed RECORD ownership row differs')
                target = str((site / row[0]).resolve())
                if target in owners:
                    owners[target].append(name)
    require(all(value == [expected] for value in owners.values()), 'installed RECORD owner is not unique pinned distribution')
    return {'records': records, 'owners': owners}


def download(path, wheel):
    request = urllib.request.Request(wheel['url'], headers={'Accept-Encoding': 'identity'})
    with path.open('xb') as output, urllib.request.urlopen(request, timeout=30) as response:
        require(response.status == 200 and response.geturl() == wheel['url'] and
                response.headers.get('Content-Encoding', 'identity') == 'identity', 'wheel HTTP origin/encoding differs')
        length = response.headers.get('Content-Length')
        require(length is None or length == str(wheel['size_bytes']), 'wheel HTTP size differs')
        fact = stream_hash(response, wheel['size_bytes'], output)
        require(fact['sha256'] == wheel['sha256'], 'downloaded wheel SHA256 differs')
        output.flush()
        os.fsync(output.fileno())
    return fact


def member_path(name):
    require(isinstance(name, str) and name and not any(c in name for c in ('\\', '\x00', ':')) and
            not name.startswith('/') and all(part not in ('', '.', '..') for part in name.rstrip('/').split('/')) and
            str(PurePosixPath(name)) == name.rstrip('/'), 'unsafe ZIP/RECORD path: ' + repr(name))


def record_rows(raw):
    rows = {}
    for row in csv.reader(io.StringIO(raw.decode('utf-8'), newline=''), strict=True):
        require(len(row) == 3, 'RECORD row must have three fields')
        name, encoded, size = row
        member_path(name)
        require(not name.endswith('/') and name not in rows, 'duplicate/invalid RECORD entry')
        rows[name] = [encoded, size]
    return rows


def check_record(rows, facts):
    for name, fact in facts.items():
        encoded = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(fact['sha256'])).decode().rstrip('=')
        require(rows.get(name) == [encoded, str(fact['size_bytes'])], 'RECORD selected entry hash/size differs: ' + name)
    require(rows.get(META['RECORD']) == ['', ''], 'RECORD self entry differs')


def check_metadata(raw):
    metadata, wheel = BytesParser().parsebytes(raw['METADATA']), BytesParser().parsebytes(raw['WHEEL'])
    require(metadata.get_all('Name') == ['nvidia-cudnn-cu13'] and metadata.get_all('Version') == ['9.20.0.48'],
            'METADATA name/version differs')
    require(wheel.get_all('Wheel-Version') == ['1.0'] and
            wheel.get_all('Tag') == ['py3-none-manylinux_2_27_aarch64'], 'WHEEL platform tag differs')


def inspect_wheel(path, selected, installed, wheel):
    """Authenticate the complete archive BEFORE interpreting its directory."""
    require(file_fact(path) == {key: wheel[key] for key in ('sha256', 'size_bytes')}, 'whole wheel hash/size differs')
    with zipfile.ZipFile(path) as archive:
        entries = {}
        for item in archive.infolist():
            member_path(item.filename)
            require(item.orig_filename == item.filename and item.filename.rstrip('/') not in entries,
                    'duplicate ZIP entry')
            mode = stat.S_IFMT(item.external_attr >> 16)
            require(mode in (0, stat.S_IFDIR if item.is_dir() else stat.S_IFREG) and
                    not item.flag_bits & 1, 'nonregular/encrypted ZIP entry')
            entries[item.filename.rstrip('/')] = item
        require(MEMBERS <= entries.keys() and set(META.values()) <= entries.keys() and selected.keys() == MEMBERS,
                'missing/extra selected wheel members')
        facts = {}
        for name in sorted(MEMBERS):
            info = entries[name]
            require(not info.is_dir() and info.file_size == selected[name]['size_bytes'], 'ZIP selected entry size differs')
            with archive.open(info) as stream:
                facts[name] = stream_hash(stream, info.file_size)
            require(facts[name] == selected[name], 'ZIP selected entry hash differs: ' + name)
        raw = {}
        for kind, name in META.items():
            require(not entries[name].is_dir() and entries[name].file_size <= CHUNK, 'ZIP metadata size differs')
            with archive.open(entries[name]) as stream:
                raw[kind] = stream.read(CHUNK + 1)
            require(len(raw[kind]) == entries[name].file_size, 'ZIP metadata truncated')
        check_metadata(raw)
        for kind in ('METADATA', 'WHEEL'):
            require(raw[kind] == installed[kind], 'installed/wheel metadata bytes differ: ' + kind)
            facts[META[kind]] = {'sha256': hashlib.sha256(raw[kind]).hexdigest(), 'size_bytes': len(raw[kind])}
        upstream, current = record_rows(raw['RECORD']), record_rows(installed['RECORD'])
        check_record(upstream, facts)
        check_record(current, facts)
        differences = {name: {'wheel': upstream.get(name), 'installed': current.get(name)}
                       for name in sorted(upstream.keys() | current.keys()) if upstream.get(name) != current.get(name)}
        return {'selected_members': {name: facts[name] for name in sorted(MEMBERS)},
                'metadata': {kind: {'wheel_sha256': hashlib.sha256(raw[kind]).hexdigest(),
                    'installed_sha256': hashlib.sha256(installed[kind]).hexdigest(),
                    'bytes_equal': raw[kind] == installed[kind]} for kind in META},
                'selected_record_entries_equal': True, 'record_row_differences': differences,
                'record_scope': 'Only four libraries plus METADATA/WHEEL authenticated; RECORD bookkeeping differences are reported.'}


def lock_snapshot(paths):
    lines = [line.split() for line in Path('/proc/locks').read_text().splitlines()]
    result = []
    for name in paths:
        info = canonical(name).stat()
        inode = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
        owners = [int(row[4]) for row in lines if len(row) >= 8 and
                  row[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and row[5] == inode]
        require(len(owners) == 1 and owners[0] > 0, 'parent lock not held')
        start = Path(f'/proc/{owners[0]}/stat').read_text().rsplit(')', 1)[1].split()[19]
        result.append({'path': name, 'inode': inode, 'owner_pid': owners[0], 'owner_start_ticks': start})
    return result


def resources(unit):
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(groups) == 1 and groups[0].endswith('/' + unit + '.service'), 'parent cgroup unit differs')
    group = canonical(Path('/sys/fs/cgroup') / groups[0].lstrip('/'), directory=True)
    names = ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.swap.peak', 'memory.swap.max')
    values = {name: int((group / name).read_text().strip()) for name in names}
    events = {key: int(value) for key, value in (line.split() for line in (group / 'memory.events').read_text().splitlines())}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    swap = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmSwap:'))
    require(values['memory.max'] == POLICY['host_bytes'] and 0 < values['memory.peak'] < POLICY['host_bytes'] and
            all(values[key] == 0 for key in names if key.startswith('memory.swap')) and
            {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill'} <= events.keys() and
            all(value == 0 for value in events.values()) and usage.ru_nswap == swap == 0 and
            usage.ru_maxrss * 1024 < POLICY['host_bytes'], 'resource caps/swap/events differ')
    return {'cgroup': str(group), 'values': values, 'memory_events': events,
            'process_peak_rss_kib': usage.ru_maxrss, 'process_swaps': usage.ru_nswap, 'process_swap_kib': swap}


def rehash(guards):
    for path, expected in guards.items():
        require(file_fact(path) == expected, 'exit rehash differs: ' + path)


def collect(args):
    started, phases = time.perf_counter(), []
    def timeout(signum, frame):
        raise TimeoutError('whole collector exceeds 300 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(POLICY['seconds'])
    def phase(name, fn):
        tick = time.perf_counter()
        try:
            result = fn()
        except Exception:
            phases.append({'name': name, 'seconds': time.perf_counter() - tick, 'pass': False})
            raise
        phases.append({'name': name, 'seconds': time.perf_counter() - tick, 'pass': True})
        return result
    context = phase('source_authority', lambda: prepare(args))
    launch, guards = context['launch'], context['guards']
    before = resources(launch['unit'])
    locks = lock_snapshot(LOCKS)
    args.output.mkdir()
    proof = {'schema': 'cudnn-wheel-provenance-v1', 'pass': False, 'model_qualified': False,
             'quality_read': False, 'quality_qualified': False, 'native_imported': False,
             'native_authority_admitted': False, 'original_native_authority_modified': False,
             'execution_sha256': args.execution_sha256, 'authority_sha256': args.authority_sha256,
             'authority': launch, 'code': context['code'], 'resource_policy': POLICY,
             'cgroup_before': before, 'locks_before': locks, 'phases': phases,
             'invocation': {'argv': sys.argv, 'python': launch['python'], 'python_version': sys.version,
                            'pid': os.getpid(), 'invocation_id': os.environ['INVOCATION_ID'],
                            'isolated': sys.flags.isolated, 'no_site': sys.flags.no_site,
                            'dont_write_bytecode': sys.dont_write_bytecode, 'optimize': sys.flags.optimize,
                            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
             'terminal_exit_and_both_locks_require_parent_receipt': True}
    try:
        facts, installed = phase('installed_bytes', lambda: installed_snapshot(context['selected'], context['metadata'], guards))
        site = Path(launch['installed_site_root'])
        proof['installed_record_ownership'] = phase('installed_record_ownership',
            lambda: record_ownership(site, context['selected'], guards))
        wheel_path = args.output / WHEEL['url'].rsplit('/', 1)[1]
        fact = phase('download', lambda: download(wheel_path, WHEEL))
        guards[str(wheel_path)] = fact
        proof['wheel'] = {'path': str(wheel_path), **WHEEL}
        proof['comparison'] = phase('authenticated_wheel_comparison', lambda: inspect_wheel(wheel_path, facts, installed, WHEEL))
        proof['pass'] = True
    except Exception as error:
        proof['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        try:
            phase('exit_uncached_rehash', lambda: rehash(guards))
            if 'installed_record_ownership' in proof:
                require(installed_records(site) == proof['installed_record_ownership']['records'],
                        'exit installed RECORD inventory differs')
            source_only()
            proof['exit_rehash_pass'] = True
            proof['cgroup_after'] = resources(launch['unit'])
            proof['locks_after'] = lock_snapshot(LOCKS)
            require(proof['locks_after'] == locks and proof['cgroup_after']['cgroup'] == before['cgroup'],
                    'terminal locks/cgroup changed')
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA visibility changed')
            require(time.perf_counter() - started < POLICY['seconds'], 'whole collector wall cap exceeded')
        except Exception as error:
            proof['pass'] = False
            proof['exit_error'] = type(error).__name__ + ': ' + str(error)
        proof['input_guards'] = guards
        proof['wall_seconds'] = time.perf_counter() - started
        with (args.output / 'proof.json').open('x') as stream:
            json.dump(proof, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    # Alarm stays armed through receipt serialization and process exit.
    return proof


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', required=True, type=Path)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--output', required=True, type=Path)
    return result


def main():
    proof = collect(parser().parse_args())
    print(json.dumps({'pass': proof['pass'], 'native_authority_admitted': False,
                      'wall_seconds': proof['wall_seconds']}, sort_keys=True))
    return 0 if proof['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
