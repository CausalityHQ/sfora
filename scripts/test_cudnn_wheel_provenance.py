#!/usr/bin/env python3
"""Five-second stdlib synthetic falsifiers; no real wheel download/native work."""
if not __debug__:
    raise SystemExit('Unoptimized source-only tests required')

import base64
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().with_name('qualify_cudnn_wheel_provenance.py')
spec = importlib.util.spec_from_file_location('cudnn_provenance', SOURCE)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rejected(fn, text):
    try:
        fn()
    except (ValueError, FileExistsError) as error:
        if text not in str(error):
            raise AssertionError((text, str(error))) from error
    else:
        raise AssertionError('accepted invalid input: ' + text)


def fixture(root):
    site = root / 'site'
    payload = {name: ('library bytes ' + name).encode() for name in sorted(d.MEMBERS)}
    payload[d.META['METADATA']] = b'Metadata-Version: 2.4\nName: nvidia-cudnn-cu13\nVersion: 9.20.0.48\n\ntext\n'
    payload[d.META['WHEEL']] = b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-manylinux_2_27_aarch64\n\n'
    records = []
    for name, raw in payload.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        records.append(f'{name},sha256={encoded},{len(raw)}\n')
    records.append(d.META['RECORD'] + ',,\n')
    payload[d.META['RECORD']] = ''.join(records).encode()
    for name, raw in payload.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    entry = {'files': {str(site / name): {'observed_native_sha256': sha(payload[name]),
             'record_sha256': sha(payload[name]), 'record_size_bytes': len(payload[name])} for name in d.MEMBERS}}
    for kind, name in d.META.items():
        entry[kind.lower()] = {'path': str(site / name), 'sha256': sha(payload[name])}
        if kind != 'RECORD':
            entry[kind.lower()]['text'] = payload[name].decode()
    evidence = {'schema': 'cudnn-installed-record-readonly-evidence-v1', 'entries': [entry],
                'native_imported': False, 'independent_upstream_authentication': False}
    return site, payload, evidence


def archive(path, payload, extra=()):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as wheel:
            for name, raw in list(payload.items()) + list(extra):
                wheel.writestr(name, raw)
    return dict(d.WHEEL, **d.file_fact(path))


def zip_tests(root):
    site, payload, evidence = fixture(root)
    selected, metadata = d.evidence_files(evidence, site)
    guards = {}
    facts, installed = d.installed_snapshot(selected, metadata, guards)
    ownership = d.record_ownership(site, selected, guards)
    assert ownership['records'] == [str(site / d.META['RECORD'])]
    duplicate = site / 'other-1.dist-info' / 'RECORD'
    duplicate.parent.mkdir()
    duplicate.write_text(next(iter(d.MEMBERS)) + ',,\n')
    assert d.installed_records(site) != ownership['records']
    rejected(lambda: d.record_ownership(site, selected, {}), 'owner is not unique')
    duplicate.write_text('unrelated/file,,\n')
    d.record_ownership(site, selected, guards)
    assert str(duplicate) in guards
    duplicate.write_text('changed/file,,\n')
    rejected(lambda: d.rehash(guards), 'exit rehash')
    guards.pop(str(duplicate))
    duplicate.unlink()
    duplicate.parent.rmdir()
    path = root / 'synthetic.whl'
    wheel = archive(path, payload)
    proof = d.inspect_wheel(path, facts, installed, wheel)
    assert proof['selected_record_entries_equal'] and not proof['record_row_differences']
    assert all(row['bytes_equal'] for row in proof['metadata'].values())
    extra_record = dict(installed, RECORD=installed['RECORD'] + (d.DIST + 'INSTALLER,sha256=fixture,3\n').encode())
    proof = d.inspect_wheel(path, facts, extra_record, wheel)
    assert not proof['metadata']['RECORD']['bytes_equal'] and len(proof['record_row_differences']) == 1
    with patch.object(d.zipfile, 'ZipFile', side_effect=AssertionError('ZIP before authentication')):
        rejected(lambda: d.inspect_wheel(path, facts, installed, dict(wheel, sha256='0' * 64)), 'whole wheel')
    member = sorted(d.MEMBERS)[0]
    for extra in [(member, payload[member]), ('../escape', b'x'), ('/absolute', b'x'),
                  ('a//b', b'x'), ('a/./b', b'x'), ('a\\b', b'x'), ('C:/escape', b'x')]:
        wheel = archive(path, payload, [extra])
        rejected(lambda: d.inspect_wheel(path, facts, installed, wheel),
                 'duplicate ZIP' if extra[0] == member else 'unsafe ZIP')
    link = zipfile.ZipInfo('symlink')
    link.create_system = 3
    link.external_attr = 0o120777 << 16
    wheel = archive(path, payload, [(link, b'/outside')])
    rejected(lambda: d.inspect_wheel(path, facts, installed, wheel), 'nonregular')
    changed = dict(payload, **{member: b'X' * len(payload[member])})
    wheel = archive(path, changed)
    rejected(lambda: d.inspect_wheel(path, facts, installed, wheel), 'entry hash')
    changed = dict(payload)
    changed.pop(member)
    wheel = archive(path, changed)
    rejected(lambda: d.inspect_wheel(path, facts, installed, wheel), 'missing/extra')
    wheel = archive(path, payload)
    rejected(lambda: d.inspect_wheel(path, dict(facts, extra=facts[member]), installed, wheel), 'missing/extra')
    bad_record = payload[d.META['RECORD']].replace(b'sha256=', b'sha512=', 1)
    for bad in [bad_record, payload[d.META['RECORD']] + payload[d.META['RECORD']].splitlines(keepends=True)[0],
                b'wrong,fields\n']:
        changed = dict(payload, **{d.META['RECORD']: bad})
        wheel = archive(path, changed)
        rejected(lambda: d.inspect_wheel(path, facts, installed, wheel), 'RECORD')
    wheel = archive(path, payload)
    rejected(lambda: d.inspect_wheel(path, facts, dict(installed, RECORD=bad_record), wheel), 'RECORD')
    for kind, old, new, error in [('METADATA', b'9.20.0.48', b'9.20.0.49', 'METADATA'),
                                  ('WHEEL', b'aarch64', b'x86_64', 'platform')]:
        changed = dict(payload, **{d.META[kind]: payload[d.META[kind]].replace(old, new)})
        wheel = archive(path, changed)
        rejected(lambda: d.inspect_wheel(path, facts, dict(installed, **{kind: changed[d.META[kind]]}), wheel), error)
    wheel = archive(path, payload)
    rejected(lambda: d.inspect_wheel(path, facts, dict(installed, WHEEL=b'different'), wheel), 'metadata bytes')
    # Extra/missing installed evidence paths must not enlarge the admitted set.
    entry = evidence['entries'][0]
    original = dict(entry['files'])
    entry['files']['/outside'] = next(iter(original.values()))
    rejected(lambda: d.evidence_files(evidence, site), 'exact four')
    entry['files'] = dict(original)
    entry['files'].pop(next(iter(original)))
    rejected(lambda: d.evidence_files(evidence, site), 'exact four')
    for path_string in guards:
        file = Path(path_string)
        raw = file.read_bytes()
        file.write_bytes(raw + b'mutated')
        rejected(lambda: d.rehash(guards), 'exit rehash')
        file.write_bytes(raw)
    alias = root / 'alias'
    alias.symlink_to(site / member)
    rejected(lambda: d.file_fact(alias), 'canonical')
    rejected(lambda: d.strict_json(b'{"x":1,"x":2}'), 'duplicate JSON')
    rejected(lambda: d.new_output(site, []), 'NEWDIR')
    rejected(lambda: d.new_output(site / 'new', [site]), 'overlaps')
    rejected(lambda: d.new_output(Path('relative'), []), 'NEWDIR')


def launch_tests(root):
    root.mkdir()
    site, payload, evidence = fixture(root)
    code_root = root / 'code'
    code_root.mkdir()
    for name in d.FILES:
        (code_root / name).write_bytes(SOURCE.read_bytes() if name == SOURCE.name else b'# synthetic test closure\n')
    code = {name: d.file_fact(code_root / name)['sha256'] for name in d.FILES}
    manifest = code_root / 'execution.json'
    manifest.write_text(json.dumps(code))
    execution_sha = d.file_fact(manifest)['sha256']
    evidence_path = root / 'evidence.json'
    evidence_path.write_text(json.dumps(evidence))
    python = Path(sys.executable).resolve()
    args = SimpleNamespace(execution_sha256=execution_sha, authority=root / 'authority.json',
                           authority_sha256='', output=root / 'output')
    wheel_path = root / 'synthetic.whl'
    wheel = archive(wheel_path, payload)
    launch = {'schema': 'cudnn-wheel-provenance-authority-v1', 'execution_sha256': execution_sha,
              'installed_evidence': {'path': str(evidence_path), 'sha256': d.file_fact(evidence_path)['sha256']},
              'installed_site_root': str(site), 'official_wheel': wheel,
              'python': {'path': str(python), 'sha256': d.file_fact(python)['sha256']}, 'python_version': sys.version,
              'output': str(args.output), 'unit': 'synthetic-test', 'both_locks_held': True,
              'lock_paths': d.LOCKS, 'resource_policy': d.POLICY}
    def authorize():
        args.authority.write_text(json.dumps(launch))
        args.authority_sha256 = d.file_fact(args.authority)['sha256']
        sys.argv[:] = [str(code_root / SOURCE.name), '--execution-sha256', args.execution_sha256,
                       '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
                       '--output', str(args.output)]
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Length': str(wheel['size_bytes'])}
        def geturl(self): return wheel['url']
        def read(self, count=-1):
            assert 0 < count <= d.CHUNK
            return super().read(count)
    saved_argv = sys.argv[:]
    loader = importlib.machinery.SourceFileLoader(d.__name__, str(code_root / SOURCE.name))
    try:
        with patch.object(d, '__file__', str(code_root / SOURCE.name)), patch.object(d, '__loader__', loader), \
             patch.object(d, 'WHEEL', wheel), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', INVOCATION_ID='a' * 32):
            authorize()
            context = d.prepare(args)
            for key, value in [('both_locks_held', False), ('lock_paths', []), ('resource_policy', {}),
                               ('official_wheel', dict(wheel, url='https://example.invalid/x')), ('output', '/wrong')]:
                old = launch[key]
                launch[key] = value
                authorize()
                rejected(lambda: d.prepare(args), 'authority')
                launch[key] = old
            authorize()
            sys.argv.reverse()
            rejected(lambda: d.prepare(args), 'CLI')
            authorize()
            with patch.object(d, '__loader__', None):
                rejected(d.source_only, 'source-only')
            with patch.dict(sys.modules, torch=object()):
                rejected(d.source_only, 'native package')
            # Cached facts cannot hide changes to any source/manifest/authority/evidence.
            for file in [manifest, args.authority, evidence_path, *(code_root / name for name in d.FILES)]:
                raw = file.read_bytes()
                file.write_bytes(raw + b' ')
                rejected(lambda: d.rehash(context['guards']), 'exit rehash')
                file.write_bytes(raw)
            rejected(lambda: d.closure(code_root, '0' * 64, {}), 'bound file')
            original = manifest.read_bytes()
            for invalid in [{SOURCE.name: code[SOURCE.name]}, dict(code, extra='a' * 64)]:
                manifest.write_text(json.dumps(invalid))
                rejected(lambda: d.closure(code_root, d.file_fact(manifest)['sha256'], {}), 'exactly two')
            manifest.write_bytes(original)
            # Actual downloader code, fake in-memory HTTP body only.
            with patch.object(d.urllib.request, 'urlopen', side_effect=lambda *a, **k: Response(wheel_path.read_bytes())), \
                 patch.object(d, 'resources', return_value={'cgroup': '/fixture', 'process_peak_rss_kib': 1}), \
                 patch.object(d, 'lock_snapshot', return_value=[{'fixture': True}]):
                proof = d.collect(args)
                assert proof['pass'] and proof['exit_rehash_pass']
                assert [p['name'] for p in proof['phases']] == ['source_authority', 'installed_bytes',
                                                               'installed_record_ownership', 'download',
                                                               'authenticated_wheel_comparison', 'exit_uncached_rehash']
                assert all(p['seconds'] >= 0 and p['pass'] for p in proof['phases'])
                assert proof['native_authority_admitted'] is proof['model_qualified'] is proof['quality_read'] is False
                assert json.loads((args.output / 'proof.json').read_text())['pass']
                rejected(lambda: d.collect(args), 'NEWDIR')
                args.output = root / 'mutation-output'
                launch['output'] = str(args.output)
                authorize()
                inspect = d.inspect_wheel
                def mutate(*a):
                    result = inspect(*a)
                    (site / d.META['WHEEL']).write_bytes(b'mutation after comparison')
                    return result
                with patch.object(d, 'inspect_wheel', side_effect=mutate):
                    proof = d.collect(args)
                assert not proof['pass'] and 'exit rehash' in proof['exit_error']
                (site / d.META['WHEEL']).write_bytes(payload[d.META['WHEEL']])
                args.output = root / 'inventory-mutation-output'
                launch['output'] = str(args.output)
                authorize()
                def add_record(*a):
                    result = inspect(*a)
                    extra = site / 'new-1.dist-info' / 'RECORD'
                    extra.parent.mkdir()
                    extra.write_text('unrelated,,\n')
                    return result
                with patch.object(d, 'inspect_wheel', side_effect=add_record):
                    proof = d.collect(args)
                assert not proof['pass'] and 'RECORD inventory' in proof['exit_error']
            for body, error in [(b'x', 'size differs'), (b'x' * (wheel['size_bytes'] + 1), 'exceeds'),
                                (b'x' * wheel['size_bytes'], 'SHA256')]:
                target = root / ('bad-' + str(len(body)))
                with patch.object(d.urllib.request, 'urlopen', return_value=Response(body)):
                    rejected(lambda: d.download(target, wheel), error)
    finally:
        sys.argv[:] = saved_argv
        signal.alarm(0)


def resource_tests():
    values = {'cgroup': '0::/fixture/test.service\n', 'status': 'VmSwap:\t0 kB\n',
              'memory.current': '100', 'memory.peak': '200', 'memory.max': str(d.POLICY['host_bytes']),
              'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
              'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}
    with patch.object(d, 'canonical', side_effect=lambda path, **kwargs: Path(path)), \
         patch.object(Path, 'read_text', autospec=True, side_effect=lambda path: values[path.name]), \
         patch.object(d.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=100, ru_nswap=0)):
        assert d.resources('test')['values']['memory.peak'] == 200
        for key, changed in [('memory.max', '1'), ('memory.peak', str(d.POLICY['host_bytes'])),
                             ('memory.swap.current', '1'), ('memory.swap.peak', '1'), ('memory.swap.max', '1'),
                             ('memory.events', values['memory.events'].replace('max 0', 'max 1')),
                             ('status', 'VmSwap:\t1 kB\n')]:
            original = values[key]
            values[key] = changed
            rejected(lambda: d.resources('test'), 'resource')
            values[key] = original
        rejected(lambda: d.resources('wrong-unit'), 'unit')
    class Bounded(io.BytesIO):
        def read(self, count=-1):
            assert 0 < count <= d.CHUNK
            return super().read(count)
    raw = b'x' * (2 * d.CHUNK + 1)
    assert d.stream_hash(Bounded(raw), len(raw)) == {'sha256': sha(raw), 'size_bytes': len(raw)}


def main():
    started = time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError('synthetic tests exceed five seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(5)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        zip_tests(root)
        launch_tests(root / 'launch')
    resource_tests()
    for flags in [('-I', '-S', '-B', '-O'), ('-I', '-S', '-B', '-OO'), ('-B',), ('-I', '-S')]:
        result = subprocess.run([sys.executable, *flags, str(SOURCE), '--execution-sha256', '0' * 64,
                                 '--authority', '/nonexistent', '--authority-sha256', '0' * 64,
                                 '--output', '/nonexistent-output'], capture_output=True, timeout=1)
        assert result.returncode != 0 and b'source-only' in result.stderr
    assert time.perf_counter() - started < 5
    print('PASS: synthetic ZIP/path/hash/RECORD/unique-owner/inventory/mutation/source/output/download/resources; real wheel/native UNRUN')


if __name__ == '__main__':
    main()
