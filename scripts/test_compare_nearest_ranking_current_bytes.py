#!/usr/bin/env python3
"""Five-second stdlib falsifiers. Native execution is UNRUN and parent-owned."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import weakref

PATH = Path(__file__).resolve().with_name('compare_nearest_ranking_current_bytes.py')


def rejected(action, message):
    try:
        action()
    except (ValueError, OSError) as error:
        assert message in str(error), (message, str(error))
    else:
        raise AssertionError('accepted: ' + message)


def manifest(root, names):
    code = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    raw = json.dumps(code).encode()
    (root / 'execution.json').write_bytes(raw)
    return code, hashlib.sha256(raw).hexdigest()


def startup_checks(d):
    # The stdlib source fixture owns its authority/terminal seam. It cannot
    # import native packages; real file authentication and driver predicates run.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        driver, candidate = root / 'driver', root / 'candidate'
        driver.mkdir(); candidate.mkdir()
        for name in d.FILES: (driver / name).write_text('# source\n')
        for name in d.TRAIN_FILES: (candidate / name).write_text('# source\n')
        original = root / 'original.py'
        original.write_text('import hashlib\ndef fingerprint(value, frozen=None, consumed=None):\n    return "a" * 64\n')
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        python = Path(sys.executable).resolve()
        with python.open('rb') as stream:
            python_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        prior = dict(python=str(python), python_sha256=python_sha, python_version=sys.version)
        cpu_file = root / 'cpu.json'
        cpu_launch = {'schema': 'cpu-fixture', 'phase': 'cpu', 'arm': 'control', 'seed': 179061}
        cpu_file.write_text(json.dumps(cpu_launch))
        cpu_binding = dict(path=str(cpu_file), sha256=hashlib.sha256(cpu_file.read_bytes()).hexdigest())
        trainer = candidate / 'train_siglip2_nearest_ranking.py'
        trainer.write_text('''import hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
FILES = {'train_siglip2_nearest_ranking.py', 'test_siglip2_nearest_ranking.py', 'nearest_ranking_readout.py'}
SEED = 179061
def bound_file(guards, path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError('SHA256 fixture changed')
    if guards.setdefault(str(path), expected) != expected: raise ValueError('guard fixture conflict')
    return path
def authority(args):
    assert (args.phase, args.arm, args.seed) == ('cpu', 'control', 179061)
    assert args.authority == CPU_PATH and args.authority_sha256 == CPU_SHA
    assert not args.output.exists()
    assert not any(n.split('.')[0] in {'torch', 'numpy'} for n in sys.modules)
    spec = importlib.util.spec_from_file_location('_comparison_original_fixture', ORIGINAL_PATH)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    launch = json.loads(args.authority.read_bytes())
    return dict(args=args, guards={str(ORIGINAL_PATH): ORIGINAL_SHA},
        required_guards={str(ORIGINAL_PATH): ORIGINAL_SHA}, root=Path(__file__).parent,
        fitter=SimpleNamespace(TERMINAL_SOURCE_SHA=ORIGINAL_SHA), launch=launch,
        legacy=dict(original=original, invocations={'3' * 32}, selected=dict(source_cpu=dict(invocation=PRIOR))),
        events=['authority'])
def admit_terminal(context, unit, phase, arm):
    assert (phase, arm) == ('cpu', 'control') and unit['both_locks_held'] is True
    assert DRIVER_PATH not in context['guards']
    context['events'].append('terminal')
    return dict(authority=dict(path=str(CPU_PATH), sha256=CPU_SHA), launch=context['launch'])
'''+f'CPU_PATH = Path({str(cpu_file)!r})\nCPU_SHA = {cpu_binding["sha256"]!r}\n'
            +f'ORIGINAL_PATH = Path({str(original)!r})\nORIGINAL_SHA = {original_sha!r}\n'
            +f'PRIOR = {prior!r}\nDRIVER_PATH = {str(driver / PATH.name)!r}\n')
        code, candidate_sha = manifest(candidate, d.TRAIN_FILES)
        own, execution = manifest(driver, d.FILES)
        unit = dict(receipt=dict(path=str(root / 'receipt.json'), sha256='a' * 64),
                    log=dict(path=str(root / 'cpu.log'), sha256='b' * 64), unit='cpu-fixture',
                    invocation_id='2' * 32, service_seconds=217., native_peak_rss_kib=2048,
                    both_locks_held=True)
        launch = dict(schema=d.AUTHORITY_SCHEMA, execution_sha256=execution,
            python=dict(path=str(python), sha256=python_sha), python_version=sys.version,
            candidate=dict(root=str(candidate), execution_sha256=candidate_sha, code=code),
            cpu_authority=cpu_binding, selected_cpu=unit, output=str(root / 'out'), unit='native-fixture',
            both_locks_held=True, resource_policy=d.POLICY, comparison_seconds=90,
            qualification_eligible=False, state_reuse_eligible=False, quality_read=False)
        launch_path = driver / 'authority.json'
        launch_path.write_text(json.dumps(launch))
        args = SimpleNamespace(execution_sha256=execution, authority=launch_path,
            authority_sha256=hashlib.sha256(launch_path.read_bytes()).hexdigest(), output=root / 'out')
        def unload():
            for name in ('_nearest_current_bytes_candidate', '_comparison_original_fixture'):
                sys.modules.pop(name, None)
        with patch.multiple(d, __file__=str(driver / PATH.name), ORIGINAL_SHA=original_sha), \
                patch.object(sys, 'argv', d.cli(driver, args)), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', INVOCATION_ID='1' * 32):
            try:
                c, context, admitted, actual = d.prepare(args)
                assert actual == own and admitted == launch and not args.output.exists()
                assert context['events'] == ['authority', 'terminal']
                assert context['args'].output == args.output and context['args'].execution_sha256 == candidate_sha
                assert context['args'].authority_sha256 == cpu_binding['sha256']
                assert context['guards'][str(driver / PATH.name)] == own[PATH.name]
                assert context['required_guards'] == {str(original): original_sha}
                fn = context['legacy']['original'].fingerprint
                fn.__defaults__ = ('changed', None)
                rejected(lambda: d.original_source(c, context), 'fingerprint changed')
                fn.__defaults__ = (None, None)
                context['legacy']['original'].injected_global = object()
                assert d.original_source(c, context) is fn  # Unused globals are outside the serializer dependency set.
                del context['legacy']['original'].injected_global
                context['legacy']['original'].hashlib = object()
                rejected(lambda: d.original_source(c, context), 'global changed')
                context['legacy']['original'].hashlib = hashlib
                saved = original.stat()
                original.write_bytes(b'x' * saved.st_size)
                os.utime(original, ns=(saved.st_atime_ns, saved.st_mtime_ns))
                rejected(lambda: d.original_source(c, context), 'SHA256')
                original.write_text('import hashlib\ndef fingerprint(value, frozen=None, consumed=None):\n    return "a" * 64\n')
                context['legacy']['original'].__spec__.origin = str(root / 'wrong.py')
                rejected(lambda: d.original_source(c, context), 'origin')
            finally:
                unload()
            with patch.object(sys, 'argv', d.cli(driver, args)[:-2]):
                rejected(lambda: d.prepare(args), 'CLI')
            with patch.dict(os.environ, INVOCATION_ID='2' * 32):
                rejected(lambda: d.prepare(args), 'fresh systemd')
            with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
                rejected(lambda: d.prepare(args), 'source-only')
            saved = trainer.stat()
            trainer.write_bytes(b'x' * saved.st_size)
            os.utime(trainer, ns=(saved.st_atime_ns, saved.st_mtime_ns))
            rejected(lambda: d.prepare(args), 'SHA256')
            assert '_nearest_current_bytes_candidate' not in sys.modules


def lifecycle_checks(d):
    # No native runtime: only its expensive boundaries are stand-ins. Real run,
    # cleanup ordering, fresh source union hashing and exclusive receipt writing
    # execute on success and on failures retaining model refs in tracebacks.
    class RNG:
        def clone(self): return self
    class Model:
        def named_parameters(self): return [(str(i), object()) for i in range(448)]
    rng = RNG()
    for failure in (None, 'integrity', 'fresh', 'exit', 'union'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, input_file = root / 'out', root / 'input'
            input_file.write_bytes(b'authenticated source')
            sha = hashlib.sha256(input_file.read_bytes()).hexdigest()
            events = []
            flags = dict(threads=1, interop_threads=1)
            values = {'memory.max': '8589934592', 'memory.current': '1024', 'memory.peak': '2048',
                'memory.swap.max': '0', 'memory.swap.current': '0', 'memory.swap.peak': '0',
                'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}
            group = dict(path='/sys/fs/cgroup/native-fixture.service', values=values)
            def cgroup(): events.append('resources'); return group
            source = SimpleNamespace(cgroup_memory=cgroup, numerical_flags=lambda: flags)
            reference = SimpleNamespace(admit_cgroup=lambda value, unit: d.require(unit == 'native-fixture', 'unit'))
            old = SimpleNamespace(zero_events=lambda value: None, audit_origins=lambda legacy: events.append('origins'))
            legacy = dict(source_driver=source, selected=dict(source_cpu=dict(numerical_flags=flags),
                genuine=dict(reference=reference)), origins=dict(files={str(input_file): sha}))
            launch = dict(unit='native-fixture', python=dict(path=str(Path(sys.executable).resolve()), sha256='a' * 64),
                cpu_authority=dict(path=str(root / 'cpu.json'), sha256='b' * 64), selected_cpu={})
            args = SimpleNamespace(output=output, execution_sha256='c' * 64,
                authority=root / 'authority.json', authority_sha256='d' * 64)
            cpu_args = SimpleNamespace(output=output, execution_sha256='e' * 64,
                authority=root / 'cpu.json', authority_sha256='f' * 64)
            ident = dict(static_sha256='1' * 64, initial_model_sha256='2' * 64)
            context = dict(legacy=legacy, args=cpu_args, root=root / 'candidate', guards={str(input_file): sha},
                code={}, launch={}, old=old, fit_context={}, flags=flags, phase_seconds={}, source={},
                terminals={'cpu:control': dict(identity=ident, initial_model_sha256='2' * 64)}, terminal_cgroups={})
            cuda = SimpleNamespace(is_initialized=lambda: False, is_available=lambda: True, device_count=lambda: 1,
                manual_seed_all=lambda seed: events.append('cuda_seed'), get_rng_state_all=lambda: [rng],
                max_memory_allocated=lambda: 1024)
            torch = SimpleNamespace(cuda=cuda, set_num_threads=lambda count: None, get_num_interop_threads=lambda: 1,
                random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: events.append('cpu_seed')),
                                       get_rng_state=lambda: rng), equal=lambda a, b: a is b)
            def bound_file(guards, path, expected):
                events.append('hash')
                path = Path(path)
                d.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'union SHA256 changed')
                return path
            def no_model():
                d.require(context.get('live_model', lambda: None)() is None, 'model retained')
            def fresh(ctx, arm, device):
                assert ctx is context and (arm, device) == ('control', 'cuda')
                model = Model(); context['live_model'] = weakref.ref(model); events.append('fresh')
                if failure == 'fresh': raise ValueError('fresh failed')
                return dict(model=model, counter=0)
            def integrity(ctx, state, identity):
                model = state['model']
                events.append('integrity')
                if failure == 'integrity': raise ValueError('integrity failed')
            def release(ctx, state):
                events.append('release'); state.clear(); no_model()
            def exit_rehash(ctx):
                events.append('exit'); no_model()
                if failure == 'exit': raise ValueError('exit failed')
                if failure == 'union': input_file.write_bytes(b'tampered source')
                for path, digest in ctx['guards'].items(): bound_file({}, path, digest)
            candidate = SimpleNamespace(cli=lambda *args: ['frozen', 'cpu', 'argv'], helper_guard=lambda ctx: events.append('helper'),
                prepare_native=lambda ctx: events.append('prepare'), fresh=fresh, identity=lambda ctx, state: ident,
                frozen_vision=lambda state: {str(i): None for i in range(444)}, integrity=integrity,
                calibration=lambda ctx, state, oracle: dict(oracle=oracle, raw=rng, unit=rng, codes=rng,
                    inverse_norms=rng, wire=b'packed-wire'), release=release,
                fingerprint=lambda ctx, value, **kwargs: 'a' * 64,
                require_no_model=lambda ctx: no_model(), exit_rehash=exit_rehash, bound_file=bound_file)
            def prepare(actual, owned):
                assert actual is args
                owned.update(candidate=candidate, context=context, launch=launch, code={})
                return candidate, context, launch, {}
            def compare(c, ctx, state, identity, record):
                events.append('compare'); record['exact_digests_every_pair'] = True
            def own_closure(*args): events.append('closure'); return {}
            with patch.object(d, 'prepare', prepare), patch.object(d, 'original_source', lambda c, ctx: lambda v: 'a' * 64), \
                    patch.object(d, 'cpu_probe', lambda *args: {'exact': True}), patch.object(d, 'compare_live', compare), \
                    patch.object(d, 'closure', own_closure), patch.dict(sys.modules, {'torch': torch}), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', INVOCATION_ID='1' * 32):
                if failure is None:
                    receipt = d.run(args)
                    assert receipt['pass'] and receipt['qualification_eligible'] is False and receipt['completed_step'] == 0
                else:
                    rejected(lambda: d.run(args), ('SHA256' if failure == 'union' else failure + ' failed'))
                receipt = json.loads((output / 'receipt.json').read_bytes())
                assert receipt['pass'] == (failure is None)
                assert receipt['exit_rehash_pass'] == (failure not in ('exit', 'union'))
                assert receipt['release_pass'] is True, 'tracebacks must not keep failed native models alive'
                assert receipt['resources_pass'] is True
                assert receipt['quality_read'] is False and receipt['state_reuse_eligible'] is False
                assert receipt['input_guards'] == {str(input_file): sha}
                assert events.index('exit') < events.index('closure') < len(events) - 1
                assert events[-1] == 'resources'
                assert events.index('cpu_seed') < events.index('cuda_seed') < events.index('prepare')
                if failure != 'fresh': assert events.index('release') < events.index('exit')


def check():
    assert PATH.is_file(), 'comparison driver missing'
    spec = importlib.util.spec_from_file_location('_current_bytes_test', PATH)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)
    startup_checks(d)
    lifecycle_checks(d)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        driver, candidate = root / 'driver', root / 'candidate'
        driver.mkdir(); candidate.mkdir()
        for name in d.FILES: (driver / name).write_text('# source\n')
        for name in d.TRAIN_FILES: (candidate / name).write_text('# source\n')
        own, sha = manifest(driver, d.FILES)
        guards = {}
        assert d.closure(driver, sha, d.FILES, guards) == own and len(guards) == 3
        victim = driver / PATH.name
        stat = victim.stat()
        victim.write_bytes(b'x' * stat.st_size)
        os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        rejected(lambda: d.closure(driver, sha, d.FILES, {}), 'SHA256')
        victim.write_text('# source\n')
        (driver / 'execution.json').write_text(json.dumps({**own, 'extra.py': 'a' * 64}))
        bad = hashlib.sha256((driver / 'execution.json').read_bytes()).hexdigest()
        rejected(lambda: d.closure(driver, bad, d.FILES, {}), 'closure')
        own, sha = manifest(driver, d.FILES)
        code, execution = manifest(candidate, d.TRAIN_FILES)
        rejected(lambda: d.strict_json(b'{"x":1,"x":2}'), 'duplicate')
        rejected(lambda: d.strict_json(b'{"x":NaN}'), 'nonfinite')
        rejected(lambda: d.source_bytes({'path': str(victim), 'sha256': 'A' * 64}, {}), 'FILE')
        python = Path(sys.executable).resolve()
        with python.open('rb') as stream:
            python_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        unit = dict(receipt=dict(path=str(root / 'cpu' / 'receipt.json'), sha256='a' * 64),
                    log=dict(path=str(root / 'cpu.log'), sha256='b' * 64), unit='fresh-cpu',
                    invocation_id='2' * 32, service_seconds=217., native_peak_rss_kib=2048,
                    both_locks_held=True)
        launch = dict(schema=d.AUTHORITY_SCHEMA, execution_sha256=sha,
            python=dict(path=str(python), sha256=python_sha), python_version=sys.version,
            candidate=dict(root=str(candidate), execution_sha256=execution, code=code),
            cpu_authority=dict(path=str(root / 'cpu.json'), sha256='c' * 64),
            selected_cpu=unit, output=str(root / 'out'), unit='fresh-native',
            both_locks_held=True, resource_policy=d.POLICY.copy(), comparison_seconds=90,
            qualification_eligible=False, state_reuse_eligible=False, quality_read=False)
        args = SimpleNamespace(execution_sha256=sha, output=root / 'out')
        d.check_authority(launch, args)
        for key, value in [('schema', 'bad'), ('extra', 0), ('execution_sha256', 'd' * 64),
                ('qualification_eligible', True), ('state_reuse_eligible', True), ('quality_read', True),
                ('both_locks_held', False), ('comparison_seconds', 91), ('unit', 'wrong.service'),
                ('output', str(root / 'other')), ('resource_policy', dict(d.POLICY, seconds=301)),
                ('resource_policy', dict(d.POLICY, swap_bytes=False)),
                ('selected_cpu', dict(unit, both_locks_held=False)),
                ('selected_cpu', dict(unit, service_seconds=True)),
                ('selected_cpu', dict(unit, service_seconds=501)),
                ('selected_cpu', dict(unit, native_peak_rss_kib=8388609)),
                ('python', dict(launch['python'], sha256='A' * 64)),
                ('candidate', dict(launch['candidate'], code={**code, 'extra.py': 'a' * 64}))]:
            rejected(lambda key=key, value=value: d.check_authority({**launch, key: value}, args), 'authority')
        d.exclusive_output(root / 'out', [driver, candidate])
        rejected(lambda: d.exclusive_output(driver / 'out', [driver]), 'exclusive')
        (root / 'out').mkdir()
        rejected(lambda: d.exclusive_output(root / 'out', []), 'exclusive')
        (root / 'link').symlink_to(root / 'out', target_is_directory=True)
        rejected(lambda: d.canonical(root / 'link'), 'canonical')
        receipt = root / 'receipt.json'
        d.write_receipt(receipt, {'pass': False, 'quality_read': False})
        assert json.loads(receipt.read_bytes()) == {'pass': False, 'quality_read': False}
        rejected(lambda: d.write_receipt(receipt, {'pass': True}), 'exists')

    calls, syncs, rows = [], [], []
    clock = SimpleNamespace(value=0.)
    expected = ('a' * 64, 'b' * 64)
    def now(): return clock.value
    def sync(): syncs.append(len(calls))
    def arm(name, seconds, digests=expected):
        def work():
            calls.append(name); clock.value += seconds
            return digests
        return work
    result = d.paired_samples(arm('baseline', .02), arm('candidate', .04), sync, rows, now, expected)
    assert calls == ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate']
    assert len(syncs) == 12 and len(rows) == 3
    assert result['baseline_median_seconds'] == .02 or abs(result['baseline_median_seconds'] - .02) < 1e-12
    assert abs(result['candidate_to_baseline_ratio'] - 2.) < 1e-12
    assert result['exact_digests_every_pair'] is True  # Slower exact candidate remains a valid measurement.
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01), arm('candidate', .01, ('c' * 64, 'b' * 64)),
        sync, rows, now, expected), 'digests')
    assert len(rows) == 1
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', 46.), arm('candidate', 46.), sync, rows, now, expected), '90-second')
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01, ('x', 'y')), arm('candidate', .01),
        sync, rows, now, expected), 'digests')
    rows.clear(); clock.value = 0.
    def broken(): calls.append('baseline'); raise ValueError('arm failed')
    def broken_sync():
        if calls[-1:] == ['baseline']: raise ValueError('sync failed')
    calls.clear()
    rejected(lambda: d.paired_samples(broken, arm('candidate', .01), broken_sync, rows, now, expected), 'arm failed')

    # Real comparison helpers with tiny scalar stand-ins catch alias omission,
    # stale current bytes, version shortcuts and failure to restore on error.
    class Scalar:
        def __init__(self): self.value, self._version, self.data = 3., 4, self
        def __getitem__(self, key):
            assert key == (0, 0)
            return SimpleNamespace(item=lambda: self.value)
        def __setitem__(self, key, value): assert key == (0, 0); self.value = value
    scalar = Scalar()
    value = {'aliases': [scalar, scalar], 1: 'integer key', '1': 'string key'}
    order = []
    def original(tree, **kwargs):
        def visit(item):
            if isinstance(item, Scalar):
                if kwargs.get('consumed') is not None: kwargs['consumed'](item)
                return ['Scalar', item.value]
            if type(item) is dict:
                return ['dict', [(visit(k), visit(item[k])) for k in sorted(item, key=repr)]]
            if type(item) in (list, tuple): return [type(item).__name__, [visit(x) for x in item]]
            return [type(item).__name__, item]
        return hashlib.sha256(repr(visit(tree)).encode()).hexdigest()
    def proposed(tree, **kwargs): return original(tree, **kwargs)
    parity = d.serializer_parity(original, proposed, value)
    assert parity == original(value)
    original(value, consumed=lambda item: order.append(id(item)))
    assert order == [id(scalar), id(scalar)]
    def omitted(tree, **kwargs): return original({'aliases': [scalar], 1: 'integer key', '1': 'string key'}, **kwargs)
    rejected(lambda: d.serializer_parity(original, omitted, value), 'parity')
    # A scalar supports .item for saving and numeric += through its data view.
    class Data:
        def __getitem__(self, key): return scalar.value
        def __setitem__(self, key, value): scalar.value = value
    scalar.data = Data()
    result = d.data_probe(original, proposed, value, scalar)
    assert scalar.value == 3. and scalar._version == 4 and result['restored_digest'] == original(value)
    old = original(value)
    rejected(lambda: d.data_probe(original, lambda tree, **kw: old, value, scalar), 'mutation')
    assert scalar.value == 3. and scalar._version == 4
    def corrupt(tree, **kwargs):
        if scalar.value != 3.: raise ValueError('injected digest failure')
        return original(tree, **kwargs)
    rejected(lambda: d.data_probe(original, corrupt, value, scalar), 'injected digest failure')
    assert scalar.value == 3.
    class BadRestore(Data):
        def __setitem__(self, key, value): scalar.value = 9. if value == 3. else value
    scalar.data = BadRestore()
    try:
        d.data_probe(original, corrupt, value, scalar)
    except ValueError as error:
        assert 'injected digest failure' in str(error)
        assert any('restoration' in note for note in getattr(error, '__notes__', [])), 'error path must verify exact restoration'
    else:
        raise AssertionError('injected failure missing')
    scalar.data, scalar.value = Data(), 3.
    seen = []
    primary = ValueError('primary')
    def fail(): seen.append('release'); raise RuntimeError('cleanup')
    record = {}
    assert d.cleanup_steps(primary, [('release', fail), ('exit', lambda: seen.append('exit'))], record) is primary
    assert seen == ['release', 'exit'] and record['release_pass'] is False and record['exit_pass'] is True
    assert 'cleanup' in primary.__notes__[0]
    rejected(lambda: (_ for _ in ()).throw(d.cleanup_steps(None,
        [('exit', lambda: (_ for _ in ()).throw(ValueError('fresh exit failed')))], {})), 'fresh exit')

    # Complete cgroup / lifetime peak predicates are checked at their boundaries.
    values = {'memory.max': '8589934592', 'memory.swap.max': '0', 'memory.swap.current': '0', 'memory.swap.peak': '0',
              'memory.current': '1024', 'memory.peak': '2048',
              'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}
    before = dict(path='/sys/fs/cgroup/fresh-native.service', values=values.copy())
    after = dict(path=before['path'], values=values.copy())
    d.check_resources(before, after, 299., 1024, 9999999999)
    for wall, rss, cuda in [(300., 1024, 1), (299., 8388609, 1), (299., 1024, 10000000000)]:
        rejected(lambda: d.check_resources(before, after, wall, rss, cuda), 'resource')
    for key, changed in [('memory.peak', '1023'), ('memory.swap.current', '1'),
                         ('memory.max', '8589934593'), ('memory.events', values['memory.events'].replace('oom 0', 'oom 1'))]:
        rejected(lambda: d.check_resources(before, dict(after, values={**values, key: changed}), 299., 1024, 1), 'resource')
    rejected(lambda: d.check_resources(before, dict(after, path='different'), 299., 1024, 1), 'resource')

    for flags in (['-B'], ['-B', '-O'], ['-B', '-OO']):
        result = subprocess.run([sys.executable, *flags, '-S', str(PATH), '--help'],
                                capture_output=True, text=True, timeout=2)
        assert (result.returncode == 0) == (len(flags) == 1), result.stderr
        assert '--authority-sha256' in result.stdout if len(flags) == 1 else 'optimized mode' in result.stderr
    tree = ast.parse(PATH.read_bytes())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
        (n.func.attr in {'reset_peak_memory_stats', 'reset_max_memory_allocated'} or
         isinstance(n.func.value, ast.Name) and n.func.value.id == 'candidate' and
         n.func.attr in {'update', 'gpu_run', 'cpu_witnesses', 'save', 'restore', 'save_inference'})
        for n in ast.walk(tree))
    assert not any(isinstance(n, ast.With) and 'no_grad' in ast.unparse(n.items) for n in ast.walk(tree))
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)
    print('PASS: stdlib launch/closure/source/order/alias/datarestore/resources/cleanup/receipt; native UNRUN')


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('5-second test cap')))
    signal.alarm(5)
    try:
        check()
    finally:
        signal.alarm(0)
