#!/usr/bin/env python3
"""One stdlib-only falsifier; no native, image, quality or speed qualification."""
import ast
from contextlib import contextmanager
import gc
import hashlib
import importlib.abc
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {'torch', 'numpy', 'PIL', 'sfora', 'transformers', 'torchvision', 'safetensors'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def installed_fixture(root, *, source=None, bundle=None):
    """Re-pin a temporary installed closure; historical sources are execution traps."""
    installed = Path(tempfile.mkdtemp(prefix='installed-',dir=root))
    package = ROOT / 'src/sfora'
    runtime = (package / 'connected_inference.py').read_text()
    if source is not None:
        tree, replacement = ast.parse(runtime), ast.parse(source)
        names = {n.name for n in replacement.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.name not in names]
        tree.body += replacement.body
        runtime = ast.unparse(tree) + '\n'
    (installed / 'connected_inference.py').write_text(runtime)
    (installed / 'packed_int8.py').write_text('''class PackedInt8Embeddings:
    @classmethod
    def from_bytes(cls, wire, *, count, dimensions):
        assert dimensions == 128
        return wire
''')
    ledger = {n.targets[0].id:ast.literal_eval(n.value) for n in
        ast.parse((package / '_connected_inference_authority.py').read_bytes()).body if isinstance(n, ast.Assign)}
    bundle = root / 'bundle' if bundle is None else bundle
    bundle.mkdir(exist_ok=True)
    for name, _ in ledger['HISTORICAL_CODE']:
        path = bundle / name
        if not path.exists(): path.write_text("raise AssertionError('historical code executed')\n")
    ledger.update(HISTORICAL_CODE=tuple(sorted((name, binding(bundle / name)['sha256'])
        for name, _ in ledger['HISTORICAL_CODE'])),
        RUNTIME_SHA256=binding(installed / 'connected_inference.py')['sha256'],
        PACKED_SHA256=binding(installed / 'packed_int8.py')['sha256'])
    path = installed / '_connected_inference_authority.py'
    path.write_text('\n'.join(name + ' = ' + repr(value) for name, value in ledger.items()) + '\n')
    bridge = (package / 'connected_compact_serving.py').read_text()
    old_pin = next(n for n in ast.walk(ast.parse(bridge)) if isinstance(n, ast.FunctionDef) and n.name == '_installed_authority')
    pin = next(n.value.elts[1].value for n in old_pin.body if isinstance(n, ast.Return))
    assert bridge.count(pin) == 1
    (installed / 'connected_compact_serving.py').write_text(bridge.replace(pin, binding(path)['sha256']))
    manifest = bundle / 'bundle.json'
    if manifest.exists():
        value = json.loads(manifest.read_bytes())
    else:
        value = {'schema':'siglip2-connected-mlp-bundle-v1'}
    if value.get('code') != dict(ledger['HISTORICAL_CODE']):
        value['code'] = dict(ledger['HISTORICAL_CODE'])
        manifest.write_text(json.dumps(value))
    return {role:binding(installed / name) for role, name in (
        ('bridge','connected_compact_serving.py'), ('runtime','connected_inference.py'),
        ('ledger','_connected_inference_authority.py'), ('packed','packed_int8.py'))}


def packed_standin(path):
    """Real source/origin metadata without importing a native implementation."""
    spec = importlib.util.spec_from_file_location('sfora.packed_int8', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), vars(module))
    return module


def rejects(call, message):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), (message, str(error))
    else:
        raise AssertionError('accepted ' + message)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE:
            raise AssertionError('thirdparty import: ' + fullname)


@contextmanager
def modules(stubs):
    old = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, previous in old.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class Tensor:
    """Logical byte views model aliasing and unchanged-version .data mutation."""
    widths = {'uint8':1, 'float32':4, 'int64':8, 'torch.float32':4, 'torch.int8':1, 'torch.float16':2}

    def __init__(self, raw, dtype='uint8', shape=None, indices=None):
        self.data = raw if isinstance(raw, bytearray) else bytearray(raw)
        self.dtype = dtype
        self.indices = tuple(range(len(raw) // self.widths[dtype])) if indices is None else indices
        self.shape = (len(self.indices),) if shape is None else shape
        self._version = 0
        self.device = SimpleNamespace(type='cpu')

    def data_ptr(self): return id(self.data)
    def numel(self): return len(self.indices)
    def element_size(self): return self.widths[self.dtype]
    def detach(self): return self
    def cpu(self): return self

    def contiguous(self):
        width = self.element_size()
        return Tensor(b''.join(self.data[i * width:(i + 1) * width] for i in self.indices), self.dtype, self.shape)

    def reshape(self, *args):
        assert args == (-1,)
        return Tensor(self.data, self.dtype)

    def view(self, dtype):
        assert dtype == 'uint8'
        return Tensor(self.data)

    def numpy(self): return self.data


def serializer_check(d, original):
    # Dropping/reordering occurrences, caching .data or retaining frames fails here.
    stubs = {'torch': SimpleNamespace(Tensor=Tensor, uint8='uint8')}
    sources = {'runtime': binding(Path(original.__file__))}
    shared = bytearray(b'abcdefgh')
    a = Tensor(shared, indices=(0, 2, 4, 6))
    b = Tensor(shared, indices=(1, 3, 5, 7))
    scalar = Tensor(b'12345678', 'int64', ())
    empty = Tensor(b'')
    tree = {'z': [a, b, a, empty], 'a': (scalar,), 0: True, '0': b'wire'}
    globals_before = dict(vars(original))
    with modules(stubs):
        expected = original.fingerprint(tree)
        observed = d.RequestObserver(original.fingerprint, sources)
        consumed = []
        actual = d.observe_call(observed, lambda: original.fingerprint(tree, consumed=consumed.append))
        assert actual == expected
        assert consumed == [scalar, a, b, a, empty]
        report = observed.report()
        leaves = report['tensor_occurrences']
        assert [row['bytes'] for row in leaves] == [8, 4, 4, 4, 0]
        assert [row['sha256'] for row in leaves] == [hashlib.sha256(raw).hexdigest()
            for raw in (b'12345678', b'aceg', b'bdfh', b'aceg', b'')]
        assert report['fingerprints'][0]['sha256'] == expected
        assert report['complete'] and report['callback_seconds'] >= 0
        assert all(row['inclusive_seconds'] >= row['exclusive_seconds'] >= 0 for row in report['host_events'])
        assert report['exclusive_host_phase_seconds']['tensor-sha'] > 0
        assert report['exclusive_host_phase_seconds']['fingerprint/framing'] > 0
        assert report['counter_inspection_seconds'] > 0
        assert report['cuda_seconds'] is None
        pointer, version = a.data_ptr(), a._version
        a.data[0] = ord('Z')
        again = d.RequestObserver(original.fingerprint, sources)
        changed = d.observe_call(again, lambda: original.fingerprint(tree))
        assert changed != expected and (a.data_ptr(), a._version) == (pointer, version)
        assert changed == original.fingerprint(tree)
        assert again.report()['tensor_occurrences'][1]['sha256'] == hashlib.sha256(b'Zceg').hexdigest()
        for left, right in (([scalar], (scalar,)), ({0: a}, {'0': a}),
                            (Tensor(b'abcd', shape=(4,)), Tensor(b'abcd', shape=(2, 2))),
                            (Tensor(b'abcd'), Tensor(b'abcd', 'float32', (1,)))):
            assert original.fingerprint(left) != original.fingerprint(right)
            for value in (left, right):
                probe = d.RequestObserver(original.fingerprint, sources)
                assert d.observe_call(probe, lambda: original.fingerprint(value)) == original.fingerprint(value)
        probe = d.RequestObserver(original.fingerprint, sources)
        frozen = {(scalar.data_ptr(), scalar._version, scalar.dtype, scalar.shape): ('int64', (), 'stale')}
        rejects(lambda: d.observe_call(probe, lambda: original.fingerprint(scalar, frozen=frozen)), 'cache')
        assert not probe.report()['complete']
        probe = d.RequestObserver(original.fingerprint, sources)
        def fail():
            original.fingerprint(Tensor(b'temporary'))
            raise RuntimeError('original target error')
        rejects(lambda: d.observe_call(probe, fail), 'original target error')
        assert not probe.report()['complete'] and probe.report()['target_error'] == 'RuntimeError'
        assert sys.getprofile() is None
        transient = Tensor(b'bye')
        ref = weakref.ref(transient)
        probe = d.RequestObserver(original.fingerprint, sources)
        d.observe_call(probe, lambda: original.fingerprint(transient))
        del transient
        gc.collect()
        assert ref() is None, 'observer retained a tensor/frame'
        assert all(vars(original).get(k) is value for k, value in globals_before.items())
        assert original.fingerprint.__globals__ is vars(original)
        probe = d.RequestObserver(original.fingerprint, sources)
        def previous(frame, event, arg): pass
        sys.setprofile(previous)
        try:
            rejects(lambda: d.observe_call(probe, lambda: None), 'unprofiled')
            assert sys.getprofile() is previous
        finally:
            sys.setprofile(None)



def resource_accounting_check(d, original):
    sources = {'runtime':binding(Path(original.__file__))}
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        probe = d.RequestObserver(original.fingerprint, sources)
        probe.calls = 2_000_000
        expected = original.fingerprint(Tensor(b'ab'))
        assert d.observe_call(probe, lambda:original.fingerprint(Tensor(b'ab'))) == expected
        report = probe.report()
        assert report['resource_usage']['total_calls'] > 2_000_000
        assert report['instrumentation_policy']['total_calls'] == 'diagnostic_only'
        assert report['first_failure'] is None
        for predicate, seed in (
                ('aggregate_keys', lambda p:p.events.update({('python','x',str(i),1,'other/unresolved'):[1,1,1,0] for i in range(4096)})),
                ('tensor_occurrences', lambda p:p.leaves.extend([{}]*4096)),
                ('fingerprint_records', lambda p:p.fingerprints.extend([{}]*4096)),
                ('encoded_bytes', lambda p:setattr(p,'metadata_bytes',8*1024**2))):
            probe = d.RequestObserver(original.fingerprint, sources)
            seed(probe)
            rejects(lambda:d.observe_call(probe, lambda:original.fingerprint(Tensor(b'ab'))), predicate)
            failure = probe.first_failure
            assert failure['predicate'] == predicate and failure['total_calls'] > 0
            assert failure['code'] and failure['max_depth'] >= failure['current_depth']
            assert sys.getprofile() is None and not probe.stack
            assert len(json.dumps(failure)) < 8192
        exact = d.RequestObserver(original.fingerprint, sources)
        exact._reserve(8*1024**2-exact.metadata_bytes)
        assert exact.metadata_bytes == 8*1024**2
        rejects(lambda:exact._reserve(1),'encoded_bytes')
        assert exact.metadata_bytes == 8*1024**2 and not exact.fingerprints and not exact.leaves
        probe = d.RequestObserver(original.fingerprint, sources)
        def recursive(n):
            return recursive(n-1) if n else original.fingerprint(3)
        rejects(lambda:d.observe_call(probe, lambda:recursive(600)), 'live_depth')
        assert probe.first_failure['predicate'] == 'live_depth'
        assert probe.first_failure['current_depth'] == probe.first_failure['max_depth'] == 512
        assert not probe.stack and sys.getprofile() is None
        # Repeated scalar fingerprints retain records even without tensor occurrences.
        probe = d.RequestObserver(original.fingerprint, sources)
        # A fixed short fixture filename isolates the record-count limit from the encoded-byte limit.
        scope = {'fingerprint':original.fingerprint}
        exec(compile('def repeat():\n return [fingerprint(i) for i in range(4097)]\n',
            '<observer-record-count>', 'exec'), scope)
        rejects(lambda:d.observe_call(probe, scope['repeat']), 'fingerprint_records')
        assert probe.first_failure['predicate'] == 'fingerprint_records'
        assert len(probe.fingerprints) == 4096 and not probe.leaves

        probe = d.RequestObserver(original.fingerprint, sources)
        tensor = Tensor(b'ab')
        rejects(lambda:d.observe_call(probe, lambda:original.fingerprint([tensor]*4097)), 'tensor_occurrences')
        assert len(probe.leaves) == 4096 and sys.getprofile() is None and not probe.stack
        assert [row['sha256'] for row in probe.leaves] == [hashlib.sha256(b'ab').hexdigest()]*4096
        namespace = {}
        exec(compile('\n'.join(f'def key_{i}(): pass' for i in range(4097)),'distinct-keys','exec'),namespace)
        funcs = [namespace[f'key_{i}'] for i in range(4097)]
        probe = d.RequestObserver(original.fingerprint, sources)
        def keys():
            for fn in funcs: fn()
        rejects(lambda:d.observe_call(probe,keys),'aggregate_keys')
        assert len(probe.events) == 4096 and not probe.stack and sys.getprofile() is None


def call_volume_check(d, original):
    # Real extracted observer + original serializer, one pair under the external30s/256MiB guard.
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        sources = {'runtime':binding(Path(original.__file__))}
        expected = original.fingerprint(3)
        def target():
            for i in range(2_000_010): len(())
            return original.fingerprint(3)
        probe = d.RequestObserver(original.fingerprint, sources)
        assert d.observe_call(probe,target) == expected
        report = probe.report()
        assert report['complete'] and probe.calls > 2_000_000
        assert len(probe.events) < 64 and len(probe.fingerprints) == 1 and not probe.leaves
        assert probe.fingerprints[0]['sha256'] == expected and sys.getprofile() is None
        # Restore exactly the old resource conjunct in an isolated code namespace.
        source = Path(d.__file__).read_text()
        old = "                self._bound(len(self.stack) < 512, 'live_depth')"
        assert source.count(old) == 1
        mutant = ModuleType('_observer_old_policy_inverse')
        mutant.__file__ = d.__file__
        exec(compile(source.replace(old, "                require(self.calls <= 2_000_000 and len(self.stack) < 512, 'old call cap')"),d.__file__,'exec'),vars(mutant))
        oldprobe = mutant.RequestObserver(original.fingerprint,sources)
        rejects(lambda:mutant.observe_call(oldprobe,target),'old call cap')
        assert oldprobe.calls == 2_000_001 and not oldprobe.stack and sys.getprofile() is None
        print(json.dumps({'status':'PASS','new_calls':probe.calls,'old_calls':oldprobe.calls,
            'keys':len(probe.events),'fingerprints':len(probe.fingerprints),'digest':expected}))

def fingerprint_attribution_check(d, original):
    # Wrong callers, cached bytes or retained caller frames must reject attribution.
    sources = {'runtime':binding(Path(original.__file__))}
    def first(value):
        return original.fingerprint(value)
    def second(value):
        return original.fingerprint(value)
    refs = []
    def target():
        tensor = Tensor(b'ab')
        refs.append(weakref.ref(tensor))
        tree = {'x':[tensor,tensor]}
        pointer, version = tensor.data_ptr(), tensor._version
        results = [first(tree), first(tree), second(tree)]
        tensor.data[0] = ord('Z')
        assert (tensor.data_ptr(), tensor._version) == (pointer, version)
        return results + [second(tree)]
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        tensor = Tensor(b'ab')
        expected = original.fingerprint({'x':[tensor,tensor]})
        tensor.data[0] = ord('Z')
        changed = original.fingerprint({'x':[tensor,tensor]})
        assert changed != expected
        probe = d.RequestObserver(original.fingerprint, sources)
        results = d.observe_call(probe, target)
        report = probe.report()
        assert report['complete'] and results == [expected,expected,expected,changed]
        rows = report['fingerprints']
        assert len(rows) == 4
        for row, caller, digest in zip(rows, (first,first,second,second), results):
            assert row.get('caller_filename') == caller.__code__.co_filename, 'caller filename missing/wrong'
            assert row['caller_function'] == caller.__qualname__
            assert row['caller_line'] == caller.__code__.co_firstlineno + 1
            assert row['sha256'] == digest and row['occurrences'] == 2 and row['bytes'] == 4
            assert type(row['host_seconds']) is float and math.isfinite(row['host_seconds']) and row['host_seconds'] >= 0
        leaves = report['tensor_occurrences']
        assert [row['fingerprint'] for row in leaves] == [0,0,1,1,2,2,3,3]
        assert [row['sha256'] for row in leaves] == [hashlib.sha256(b'ab').hexdigest()] * 6 + [hashlib.sha256(b'Zb').hexdigest()] * 2
        gc.collect()
        assert refs[0]() is None, 'fingerprint attribution retained a caller frame/tensor'
        assert sys.getprofile() is None and not probe.stack


def stack_check(d, original):
    sources = {'runtime':binding(Path(original.__file__))}
    class MismatchedReturn(d.RequestObserver):
        def _event(self, frame, event, arg, tick):
            super()._event(frame, event, arg, tick)
            if event == 'call' and frame.f_code is self.fingerprint_code:
                super()._event(frame.f_back, 'return', None, tick)
    class ResidualAtStop(d.RequestObserver):
        def _event(self, frame, event, arg, tick):
            super()._event(frame, event, arg, tick)
            if event == 'c_call' and arg is sys.setprofile:
                self.stack.append({'pending':True})
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        for observer_type, message in ((MismatchedReturn,'profile return stack mismatch'),
                                       (ResidualAtStop,'profile stack at stop')):
            probe = observer_type(original.fingerprint, sources)
            rejects(lambda:d.observe_call(probe, lambda:original.fingerprint(3)), message)
            assert not probe.report()['complete'] and not probe.stack
            assert sys.getprofile() is None
        unexpected = d.RequestObserver(original.fingerprint, sources)
        unexpected(sys._getframe(), 'return', None)
        assert unexpected.failures and 'profile return stack empty' in unexpected.failures[0]
        normal = d.RequestObserver(original.fingerprint, sources)
        assert d.observe_call(normal, lambda:original.fingerprint(3)) == original.fingerprint(3)
        assert normal.report()['complete'] and not normal.stack


def c_metadata_check(d, original):
    sources = {'runtime':binding(Path(original.__file__))}
    kept, values = [], [3,1,2]
    def key(value):
        kept.append(b'x'.upper)  # Retain reused bound-method allocations across C profile events.
        return value
    def target():
        values.sort(key=key)
        return original.fingerprint(3)
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        probe = d.RequestObserver(original.fingerprint, sources)
        assert d.observe_call(probe, target) == original.fingerprint(3)
        assert values == [1,2,3] and probe.report()['complete'] and sys.getprofile() is None
        wrong = d.RequestObserver(original.fingerprint, sources)
        wrong(sys._getframe(), 'c_call', [].append)
        wrong(sys._getframe(), 'c_return', [].sort)
        assert wrong.failures and 'profile return stack mismatch' in wrong.failures[0]


def predicate_check(d, original):
    class Predicate:
        def item(self): return True
    def target():
        assert Predicate().item()
        return original.fingerprint(3)
    with modules({'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8')}):
        probe = d.RequestObserver(original.fingerprint, {'runtime':binding(Path(original.__file__))})
        d.observe_call(probe, target)
        rows = [row for row in probe.report()['host_events'] if row['function'].endswith('Predicate.item')]
        assert rows and all(row['phase'] == 'predicate-sync-plus-wait' for row in rows), 'item wait mislabeled'
        assert probe.report()['cuda_seconds'] is None


def public_check(d, original, bridge, root, *, witness_only=False, cancel_only=False, transient_only=False, cleanup_only=False, bundle_only=False, ambient_only=False):
    # Real bridge authentication, registry, serializer, exact wire and release predicates.
    trainer_source = (ROOT / 'src/sfora/connected_inference.py').read_text()
    tree = ast.parse(trainer_source)
    release = ast.get_source_segment(trainer_source, next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == 'release_inference'))
    events = ModuleType('_attribution_events')
    events.original, events.Tensor = original, Tensor
    events.param = Tensor(b'LIVE')
    events.expected = None
    events.cache = events.retained = None
    events.failure = None
    events.target_error = events.close_error = None
    source = '''import gc, sys, weakref
from functools import lru_cache
from types import ModuleType
import importlib.util, time
import _attribution_events as events
def require(condition, message):
    if not condition: raise ValueError(message)
def load_authenticated(name, path): return None
def admit_bundle(directory, digest): return None
class Owner:
    def parameters(self): return ()
    def buffers(self): return ()
def _processor_cache(processor, guards):
    return object() if events.failure == 'cache_authority' else processor.cache
def load_inference(directory, digest, device):
    assert device == 'cuda'
    cache = lru_cache(maxsize=10)(lambda value: value)
    endpoint = {name:Owner() for name in ('model','processor_object','head_object','A','C','mu_train')}
    endpoint.update(modules={'runtime':sys.modules[__name__]}, guards={}, processor_cache=cache, directory=directory, manifest=__import__('json').loads((directory / 'bundle.json').read_bytes()))
    endpoint['processor_object'].cache = cache
    cache(endpoint['processor_object'])
    events.cache = cache
    events.refs = [weakref.ref(endpoint[name]) for name in ('model','processor_object','head_object','A','C','mu_train')]
    if events.failure == 'lifetime': events.retained = endpoint['model']
    return endpoint
def inference_outputs(endpoint, images):
    digest = fingerprint({'param':events.param})
    require(digest == events.expected, 'current .data rejected')
    if events.target_error is not None: raise events.target_error
    if events.failure == 'inference': raise ValueError('inference failed')
    wire = bytes(130 * len(images))
    return {'raw':events.Tensor(bytes(512*len(images)), 'torch.float32', (len(images),128)), 'unit':events.Tensor(bytes(512*len(images)), 'torch.float32', (len(images),128)),
            'codes':events.Tensor(bytes(128*len(images)), 'torch.int8', (len(images),128)), 'inverse_norms':events.Tensor(bytes(2*len(images)), 'torch.float16', (len(images),)), 'wire':wire}
''' + release + '\n'
    if transient_only:
        # A C tensor primitive leaves no Python `self` frame; mimic that boundary.
        cancellation = KeyboardInterrupt('endpoint numel cancellation')
        class CancelTensor(Tensor):
            def numel(self):
                self = None
                raise cancellation
        events.CancelTensor = CancelTensor
        source = source.replace("    endpoint['processor_object'].cache = cache", "    endpoint['A'] = events.CancelTensor(b'OWNED')\n    endpoint['processor_object'].cache = cache")
        source = source.replace("fingerprint({'param':events.param})", "fingerprint({'param':endpoint['A']})")
    bundle = root / 'bundle'
    bundle.mkdir()
    for name in bridge._FILES:
        (bundle / name).write_bytes(b'owned standin')
    manifest = {'schema':'siglip2-connected-mlp-bundle-v1',
        'code':{},
        'files':{name:binding(bundle / name)['sha256'] for name in bridge._FILES},
        **{name:{} for name in bridge._MANIFEST - {'schema', 'code', 'files'}}}
    (bundle / 'bundle.json').write_text(json.dumps(manifest))
    installed = installed_fixture(root, source=source, bundle=bundle)
    bridge = load('_serving_installed_bridge_check', Path(installed['bridge']['path']))
    sys.modules.pop(bridge.__name__)
    gallery, library = root / 'gallery.bin', root / 'native.so'
    gallery.write_bytes(b'gallery'); library.write_bytes(b'native standin')
    closes = []
    class Image:
        def close(self): closes.append('image')
    class Packed:
        @classmethod
        def from_bytes(cls, wire, *, count, dimensions):
            assert dimensions == 128
            return wire
    class Gallery:
        @classmethod
        def open_packed(cls, path, packed): return cls()
        def search_packed(self, wire, *, k):
            assert k == 10
            if events.failure == 'native': raise RuntimeError('native failed')
            # Tied scores: fixed ascending IDs survive the real bridge untouched.
            return memoryview(struct.pack('10q', *range(10))).cast('q'), memoryview(struct.pack('10f', *([1.] * 10))).cast('f')
        def close(self):
            closes.append('gallery')
            if events.close_error is not None: raise events.close_error
    stubs = {'_attribution_events':events, 'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8',
              cuda=SimpleNamespace(is_initialized=lambda:False))}
    for name in ('PIL', 'PIL.Image', 'sfora', 'sfora.joint_relational_compaction', 'sfora.cutile_int8'):
        stubs[name] = ModuleType(name)
    stubs['PIL.Image'].Image = Image
    stubs['sfora.packed_int8'] = packed_standin(Path(installed['packed']['path']))
    stubs['sfora.joint_relational_compaction'].PackedInt8Embeddings = Packed
    stubs['sfora.cutile_int8'].CutilePackedInt8Gallery = Gallery
    args = dict(bundle_dir=bundle, expected_bundle_sha256=binding(bundle / 'bundle.json')['sha256'],
        gallery_path=gallery, expected_gallery_sha256=binding(gallery)['sha256'], gallery_count=10,
        native_library_path=library, expected_native_library_sha256=binding(library)['sha256'])
    sources = {**installed, 'trainer':binding(bundle / bridge._TRAINER),
        'serializer':binding(bundle / 'train_siglip2_substrate_adaptation.py')}
    with modules(stubs):
        events.expected = original.fingerprint({'param':events.param})
        def request(index, observer=None):
            order = []
            def read(paths): order.append('decode'); return [Image() for _ in paths]
            def sync(): order.append('sync')
            result, row = d.measure_request(index, read, sync, [root / 'TRAIN.fixture'], observer=observer)
            assert order == ['sync', 'decode', 'sync']
            assert row['seconds'] > 0 and row['image_cleanup_seconds'] >= 0
            assert abs(row['seconds'] - row['read_decode_seconds'] - row['public_call_seconds'] -
                       row['completion_sync_seconds']) < 1e-9
            assert bytes(result[0]) == struct.pack('10q', *range(10))
            return row
        if ambient_only:
            ambient, close_error = KeyError('unrelated parent error'), ValueError('ambient image close')
            class BrokenImage(Image):
                def close(self): raise close_error
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            caught = None
            try: raise ambient
            except KeyError:
                try: d.measure_request(index, lambda paths:[BrokenImage()], lambda:None, [root / 'TRAIN.fixture'])
                except BaseException as error: caught = error
            assert caught is close_error, 'ambient except swallowed image close failure'
            assert not getattr(ambient,'__notes__',())
            assert index._closed and not index._owned and all(ref() is None for ref in events.refs)
            return
        if bundle_only:
            assert hasattr(d.RequestObserver, 'from_index'), 'admitted-index observer missing'
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            actual = index._endpoint['modules']['runtime']
            assert actual.fingerprint is not original.fingerprint and actual.fingerprint.__code__ is not original.fingerprint.__code__
            before = dict(vars(actual))
            probe = d.RequestObserver.from_index(index, sources)
            request(index, probe)
            assert probe.report()['complete'] and probe.fingerprints and probe.leaves and probe.output is not None
            assert all(vars(actual).get(key) is value for key,value in before.items())
            foreign = load('_foreign_equal_runtime', Path(sources['runtime']['path']))
            assert foreign.fingerprint.__code__ == actual.fingerprint.__code__
            try:
                index._endpoint['modules']['runtime'] = foreign
                rejects(lambda:d.RequestObserver.from_index(index,sources), 'runtime module owner')
                index._endpoint['modules']['runtime'] = actual
                index._endpoint['modules']['extra'] = foreign
                rejects(lambda:d.RequestObserver.from_index(index,sources), 'runtime module owner')
                del index._endpoint['modules']['extra']
                for role in ('runtime','ledger','packed'):
                    rejects(lambda:d.RequestObserver.from_index(index,sources | {
                        role:sources[role] | {'sha256':'0'*64}}), 'installed source guards')
                for role in ('trainer','serializer'):
                    rejects(lambda:d.RequestObserver.from_index(index,sources | {
                        role:binding(Path(original.__file__))}), 'bundle source FILE paths')
            finally:
                index._endpoint['modules'] = {'runtime':actual}
                sys.modules.pop(foreign.__name__)
            index.close()
            assert all(ref() is None for ref in events.refs)
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            wrong = d.RequestObserver(original.fingerprint, sources | {'runtime':binding(Path(original.__file__))})
            rejects(lambda:request(index, wrong), 'public witness')
            assert not wrong.report()['complete'] and index._closed and not index._owned
            assert all(ref() is None for ref in events.refs)
            return
        if cleanup_only:
            cases = (
                (None, [ValueError('first close'), KeyboardInterrupt('second close')], None, 1),
                (ValueError('original inference'), [KeyboardInterrupt('image cancellation')], None, 1),
                (None, [ValueError('first close'), RuntimeError('second close')], None, 0),
                (None, [ValueError('image ordinary')], SystemExit(42), 1),
            )
            for target, image_errors, index_error, preferred in cases:
                events.target_error, events.close_error = target, index_error
                index = bridge.ConnectedCompactIndex.from_bundle(**args)
                errors = ([target] if target is not None else []) + image_errors + ([index_error] if index_error is not None else [])
                done = []
                class ClosingImage(Image):
                    def __init__(self, ordinal, error): self.ordinal, self.error = ordinal, error
                    def close(self):
                        done.append(self.ordinal)
                        raise self.error
                images = [ClosingImage(i,error) for i,error in enumerate(image_errors)]
                caught = None
                try:
                    d.measure_request(index, lambda paths:images, lambda:None,
                                      [root / f'TRAIN-{i}.fixture' for i in range(len(images))])
                except BaseException as error:
                    caught = error
                assert caught is errors[preferred], 'cleanup cancellation/primary identity lost: ' + repr(caught)
                assert done == list(range(len(images))), 'later image cleanup skipped'
                others = [error for error in errors if error is not caught]
                notes = getattr(caught,'__notes__',())
                assert all(any(repr(error) in note for note in notes) for error in others), 'cleanup failure note lost'
                assert isinstance(caught.__cause__, BaseExceptionGroup)
                assert all(any(error is saved for saved in caught.__cause__.exceptions) for error in others), 'secondary exception identity lost'
                assert index._closed and not index._owned and sys.getprofile() is None
                assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
            # Body error before search still runs genuine close, with cancellation priority.
            for target, index_error in ((ValueError('decoder ordinary'), KeyboardInterrupt('index cancellation')),
                                        (KeyboardInterrupt('original cancellation'), ValueError('index ordinary'))):
                events.target_error, events.close_error = None, index_error
                index = bridge.ConnectedCompactIndex.from_bundle(**args)
                def reader(paths): raise target
                caught = None
                try: d.measure_request(index, reader, lambda:None, [root / 'TRAIN.fixture'])
                except BaseException as error: caught = error
                expected = target if isinstance(target, KeyboardInterrupt) else index_error
                other = index_error if expected is target else target
                assert caught is expected and isinstance(caught.__cause__, BaseExceptionGroup)
                assert any(error is other for error in caught.__cause__.exceptions)
                assert any(repr(other) in note for note in caught.__notes__)
                assert index._closed and not index._owned
                assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
            return
        if transient_only:
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            ref = weakref.ref(index._endpoint['A'])
            serializer = index._endpoint['modules']['runtime']
            pins = sources | {'runtime':binding(Path(serializer.__file__))}
            probe = d.RequestObserver.from_index(index, pins)
            caught = None
            try:
                request(index, probe)
            except BaseException as error:
                caught = error
            assert caught is cancellation and sys.getprofile() is None
            gc.collect()
            assert ref() is None, 'observer traceback retained endpoint-owned tensor after original frames cleared'
            assert index._closed and not index._owned and events.cache.cache_info().currsize == 0
            assert all(ref() is None for ref in events.refs)
            trace = caught.__traceback__
            names = []
            while trace is not None:
                names.append(trace.tb_frame.f_code.co_name)
                if trace.tb_frame.f_globals is vars(d):
                    for name in ('frame','arg','item','local','fact','value','raw'):
                        assert trace.tb_frame.f_locals.get(name) is None, 'observer transient survives: ' + name
                trace = trace.tb_next
            assert {'__call__','_event','numel'} <= set(names), 'cancellation traceback locations lost'
            assert not any('lifetime survived' in note for note in getattr(caught,'__notes__',()))
            return
        if cancel_only:
            class Cancelled(BaseException): pass
            for cancellation in (KeyboardInterrupt('stop'), SystemExit(37), Cancelled('stop')):
                class CancelObserver(d.RequestObserver):
                    def _event(self, frame, event, arg, tick):
                        if event == 'call' and frame.f_code is self.fingerprint_code:
                            raise cancellation
                        super()._event(frame, event, arg, tick)
                index = bridge.ConnectedCompactIndex.from_bundle(**args)
                probe = CancelObserver.from_index(index, sources)
                try:
                    request(index, probe)
                except BaseException as error:
                    assert error is cancellation, 'cancellation replaced by ' + type(error).__name__
                else:
                    raise AssertionError('cancellation swallowed')
                assert probe.target_error == type(cancellation).__name__ and not probe.report()['complete']
                assert sys.getprofile() is None and index._closed and not index._owned
                assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
                assert not any(n.startswith(('_sfora_connected_compact_', '_connected_serving_attribution_fixture')) for n in sys.modules)
            class InspectionError(d.RequestObserver):
                def _event(self, frame, event, arg, tick):
                    if event == 'call' and frame.f_code is self.fingerprint_code:
                        raise ValueError('bad counter inspection')
                    super()._event(frame, event, arg, tick)
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            probe = InspectionError.from_index(index, sources)
            rejects(lambda:request(index, probe), 'incomplete observation')
            assert index._closed and events.cache.cache_info().currsize == 0
            assert not probe.report()['complete'] and sys.getprofile() is None
            return
        if witness_only:
            # Generic scalar profiling is valid; PUBLIC profiling needs byte/output witnesses.
            scalar = d.RequestObserver(original.fingerprint, {'runtime':binding(Path(original.__file__))})
            assert d.observe_call(scalar, lambda:original.fingerprint(3)) == original.fingerprint(3)
            assert scalar.report()['complete'] and not scalar.leaves
            class SilentObserver(d.RequestObserver):
                def __call__(self, frame, event, arg): pass
            class MissingOutput(d.RequestObserver):
                def _event(self, frame, event, arg, tick):
                    super()._event(frame, event, arg, tick)
                    self.output = None
            class MissingFingerprint(d.RequestObserver):
                def _event(self, frame, event, arg, tick):
                    super()._event(frame, event, arg, tick)
                    if event == 'return' and frame.f_code is self.fingerprint_code:
                        self.fingerprints.clear()
            class MissingLeaves(d.RequestObserver):
                def _event(self, frame, event, arg, tick):
                    super()._event(frame, event, arg, tick)
                    if event == 'return' and frame.f_code is self.fingerprint_code:
                        self.leaves.clear()
            for observer_type in (SilentObserver, MissingOutput, MissingFingerprint, MissingLeaves):
                index = bridge.ConnectedCompactIndex.from_bundle(**args)
                probe = observer_type.from_index(index, sources)
                rejects(lambda: request(index, probe), 'public witness')
                assert not probe.report()['complete']
                assert index._closed and not index._owned
                assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
                assert sys.getprofile() is None
                assert not any(n.startswith(('_sfora_connected_compact_', '_connected_serving_attribution_fixture')) for n in sys.modules)
            return
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        globals_before = dict(vars(index._module))
        row = request(index)
        probe = d.RequestObserver.from_index(index, sources)
        observed = request(index, probe)
        assert row['native'] == observed['native']
        report = probe.report()
        assert report['output']['wire_hex'] == bytes(130).hex()
        assert report['output']['raw']['hex'] == bytes(512).hex()
        assert report['tensor_occurrences'][0]['sha256'] == hashlib.sha256(b'LIVE').hexdigest()
        assert 0 < report['output_capture_seconds'] <= report['callback_seconds']
        assert all(vars(index._module).get(k) is v for k, v in globals_before.items())
        index.close()
        assert events.cache.cache_info().currsize == 0 and all(ref() is None for ref in events.refs)
        assert sys.getprofile() is None
        assert not any(n.startswith(('_sfora_connected_compact_', '_connected_serving_attribution_fixture')) for n in sys.modules)
        for failure, message in (('native', 'native failed'), ('inference', 'inference failed'),
                                 ('cache_authority', 'processor teardown authority'), ('lifetime', 'lifetime survived')):
            events.failure = failure
            index = bridge.ConnectedCompactIndex.from_bundle(**args)
            probe = d.RequestObserver.from_index(index, sources)
            rejects(lambda: request(index, probe) if failure in ('native','inference') else d.observe_call(probe, index.close), message)
            index.close()
            events.retained = None
            gc.collect()
            assert sys.getprofile() is None and not index._owned
            assert events.cache.cache_info().currsize == (1 if failure == 'cache_authority' else 0)
            events.cache.cache_clear()  # Dispose ONLY the rejected standin cache after asserting it.
        events.failure = None
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        events.param.data[0] = ord('X')
        probe = d.RequestObserver.from_index(index, sources)
        rejects(lambda: request(index, probe), 'current .data rejected')
        assert index._closed and not index._owned
        index.close()
        events.param.data[0] = ord('L')
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        probe = d.RequestObserver.from_index(index, sources)
        # A retained-byte boundary failure closes the genuine endpoint too.
        probe.metadata_bytes = 8*1024**2
        rejects(lambda: request(index, probe), 'incomplete observation')
        assert index._closed and events.cache.cache_info().currsize == 0
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        trainer = bundle / bridge._TRAINER
        trainer_raw = trainer.read_bytes()
        trainer.write_bytes(trainer_raw + b'# stale source\n')
        rejects(lambda: request(index), 'current file bytes differ')
        assert index._closed and not index._owned
        trainer.write_bytes(trainer_raw)
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        def bad_reader(paths): raise RuntimeError('decode failed')
        rejects(lambda: d.measure_request(index, bad_reader, lambda:None, [root / 'TRAIN.fixture']), 'decode failed')
        assert index._closed and events.cache.cache_info().currsize == 0
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        class BadClose(Image):
            def close(self): raise RuntimeError('image cleanup failed')
        rejects(lambda: d.measure_request(index, lambda paths:[BadClose()], lambda:None,
                                          [root / 'TRAIN.fixture']), 'image cleanup failed')
        assert index._closed and events.cache.cache_info().currsize == 0
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        helper = next(iter(index._endpoint['modules'].values()))
        foreign = ModuleType(helper.__name__)
        sys.modules[helper.__name__] = foreign
        rejects(lambda: request(index), 'owned connected registry changed')
        assert sys.modules[helper.__name__] is foreign
        del sys.modules[helper.__name__]
        assert index._closed and events.cache.cache_info().currsize == 0


def preparation_check(d, root):
    # A plausible metadata-only receipt must never produce a qualification claim.
    assert d.LIMITS == {'body_seconds':300,'host_bytes':8589934592,'swap_bytes':0,
        'cuda_allocated_bytes_exclusive':10000000000}
    names = {'observer':'scripts/observe_connected_serving.py', 'test':'scripts/test_observe_connected_serving.py',
        'bridge':'src/sfora/connected_compact_serving.py', 'trainer':'scripts/train_siglip2_connected_mlp.py',
        'serializer':'scripts/train_siglip2_substrate_adaptation.py'}
    bundle = root / 'future_bundle'
    bundle.mkdir()
    sources = {role:binding(ROOT / name) for role,name in names.items()}
    installed = installed_fixture(root, bundle=bundle)
    sources.update(installed)
    for role in ('trainer','serializer'):
        sources[role] = binding(bundle / Path(names[role]).name)
    manifest = bundle / 'bundle.json'
    fixture = root / 'future.fixture'
    fixture.write_bytes(b'not native/image/terminal evidence')
    images = []
    for number in range(32):
        path = root / f'TRAIN-{number}.fixture'
        path.write_bytes(bytes([number]))
        images.append(binding(path))
    authority = {'schema':'connected-serving-attribution-authority-v2',
        'sources':sources,
        'bundle':{'directory':str(bundle), 'manifest':binding(manifest)},
        'gallery':{'file':binding(fixture),'count':10}, 'native':binding(fixture),
        'train_images':images, 'qualified_terminal':binding(fixture), 'cache_conditions':'fixed cold read, resident gallery',
        'resource_policy':{'body_seconds':300,'host_bytes':8589934592,'swap_bytes':0,
            'cuda_allocated_bytes_exclusive':10000000000,'whole_process_seconds':1500,'exit_reserve_seconds':300},
        'both_locks_held':True, 'qualification_eligible':False, 'state_reuse_eligible':False}
    path = root / 'authority.json'
    def write(value):
        path.write_text(json.dumps(value))
        return binding(path)['sha256']
    digest = write(authority)
    assert d.prepare(path, digest) == authority
    for key,value in [('body_seconds',120),('body_seconds',299),('body_seconds',301),('body_seconds',300.),
            ('whole_process_seconds',599),('whole_process_seconds',600),('exit_reserve_seconds',0),
            ('host_bytes',8589934593),('swap_bytes',1),('cuda_allocated_bytes_exclusive',10000000001)]:
        mutant = authority | {'resource_policy':authority['resource_policy'] | {key:value}}
        rejects(lambda:d.prepare(path,write(mutant)), 'required')
    headroom = authority | {'resource_policy':authority['resource_policy'] | {'whole_process_seconds':601}}
    assert d.prepare(path,write(headroom)) == headroom
    for role in ('trainer','serializer'):
        mutant = authority | {'sources':sources | {role:binding(ROOT / names[role])}}
        rejects(lambda:d.prepare(path,write(mutant)), 'bundle source FILE paths required')
    for mutant in (authority | {'qualification_eligible':True}, authority | {'extra':True},
                   authority | {'train_images':images[:-1] + images[:1]},
                   authority | {'resource_policy':authority['resource_policy'] | {'body_seconds':121}}):
        rejects(lambda: d.prepare(path, write(mutant)), 'required')
    digest = write(authority)
    fixture.write_bytes(b'stale same path')
    rejects(lambda: d.prepare(path, digest), 'SHA256 differs')
    fixture.write_bytes(b'not native/image/terminal evidence')
    output = root / 'preparation.json'
    d.main(['--authority',str(path),'--authority-sha256',digest,'--output',str(output)])
    row = json.loads(output.read_bytes())
    assert row['status'] == 'SOURCE_ONLY_PREPARED; native gate UNRUN'
    assert row['qualification_eligible'] is row['state_reuse_eligible'] is row['optimization_eligible'] is False
    assert row['attribution'] == 'UNMEASURED' and len(row['required_root_seams']) >= 7


def installed_sources_check(d, root):
    sources = installed_fixture(root)
    sources.update(observer=binding(Path(d.__file__)), test=binding(Path(__file__)),
        trainer=binding(root/'bundle/train_siglip2_connected_mlp.py'),
        serializer=binding(root/'bundle/train_siglip2_substrate_adaptation.py'))
    bundle = {'directory':str(root/'bundle'), 'manifest':binding(root/'bundle/bundle.json')}
    d.check_runtime_sources(sources,bundle)
    for path in (root/'bundle').glob('*.py'):
        assert path.read_text().startswith("raise AssertionError('historical code executed')")
    rejects(lambda:d.check_runtime_sources({k:v for k,v in sources.items() if k != 'runtime'},bundle), 'source pins')
    for role in ('runtime','ledger','packed','trainer','serializer'):
        path = Path(sources[role]['path']); raw = path.read_bytes()
        try:
            path.write_bytes(raw+b'\n# current bytes mutant\n')
            rejects(lambda:d.check_runtime_sources(sources,bundle), 'SHA256 differs')
        finally: path.write_bytes(raw)
        rejects(lambda:d.check_runtime_sources(sources | {role:sources[role] | {'sha256':'0'*64}},bundle), 'SHA256 differs')
    foreign = root/'foreign.py'; foreign.write_bytes(Path(sources['runtime']['path']).read_bytes())
    rejects(lambda:d.check_runtime_sources(sources | {'runtime':binding(foreign)},bundle), 'source sibling')
    ledger, bridge = Path(sources['ledger']['path']), Path(sources['bridge']['path'])
    ledger_raw, bridge_raw = ledger.read_text(), bridge.read_text()
    record = {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(ledger_raw).body}
    mutants = [
        (ledger_raw+'SCHEMA = "sfora-connected-inference-extraction-v1"\n','duplicate'),
        (ledger_raw+"raise AssertionError('ledger executed')\n",'literal'),
        (ledger_raw+'EXTRA = 1\n','exact installed'),
        (ledger_raw.replace(repr(record['SCHEMA']),repr('wrong-schema'),1),'exact installed'),
        (ledger_raw.replace(repr(record['RUNTIME_SHA256']),repr('0'*64),1),'ledger pins'),
        (ledger_raw.replace(repr(record['PACKED_SHA256']),repr('0'*64),1),'ledger pins'),
        (ledger_raw.replace(repr(record['HISTORICAL_CODE']),repr(record['HISTORICAL_CODE'][:-1]),1),'historical inference')]
    for raw, message in mutants:
        try:
            ledger.write_text(raw)
            pinned = sources | {'ledger':binding(ledger)}
            bridge.write_text(bridge_raw.replace(sources['ledger']['sha256'],pinned['ledger']['sha256']))
            pinned['bridge'] = binding(bridge)
            rejects(lambda:d.check_runtime_sources(pinned,bundle), message)
        finally:
            ledger.write_text(ledger_raw); bridge.write_text(bridge_raw)
    try:
        ledger.write_text(ledger_raw+'\n')
        rejects(lambda:d.check_runtime_sources(sources | {'ledger':binding(ledger)},bundle), 'bridge ledger identity')
    finally: ledger.write_text(ledger_raw)
    manifest = Path(bundle['manifest']['path']); raw = manifest.read_text(); value = json.loads(raw)
    for code in (dict(list(value['code'].items())[:-1]), value['code'] | {'extra.py':'0'*64},
                 value['code'] | {'quadratic_readout.py':'0'*64}):
        try:
            manifest.write_text(json.dumps(value | {'code':code}))
            rejects(lambda:d.check_runtime_sources(sources,bundle | {'manifest':binding(manifest)}), 'historical inference')
        finally: manifest.write_text(raw)
    path = root/'bundle/quadratic_readout.py'; raw = path.read_bytes()
    try:
        path.write_bytes(raw+b'\n')
        rejects(lambda:d.check_runtime_sources(sources,bundle), 'SHA256 differs')
    finally: path.write_bytes(raw)
    d.check_runtime_sources(sources,bundle)
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 16*1024**2


def main():
    assert __debug__ and sys.getprofile() is None
    assert not any(n.split('.')[0] in NATIVE for n in sys.modules)
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    owned = ('_serving_observer_check', '_serving_serializer_check', '_serving_bridge_check')
    try:
        path = ROOT / 'scripts/observe_connected_serving.py'
        if '--baseline-observer' in sys.argv:
            assert '--review-ambient' in sys.argv, 'baseline is only the ambient-except counterexample'
            path = Path(sys.argv[sys.argv.index('--baseline-observer') + 1])
        assert path.is_file(), 'observer implementation missing'
        d = load(owned[0], path)
        original = load(owned[1], ROOT / 'src/sfora/connected_inference.py')
        bridge = load(owned[2], ROOT / 'src/sfora/connected_compact_serving.py')
        if '--review-public' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-installed-public-') as scratch:
                public_check(d, original, bridge, Path(scratch))
            print('PASS installed public calls, historical-byte guards, fresh data and genuine cleanup')
            return
        if '--installed-sources' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-installed-sources-') as scratch:
                installed_sources_check(d, Path(scratch))
            print('PASS literal ledger, exact historical9 evidence, installed siblings and fresh source pins')
            return
        if '--review-fingerprint' in sys.argv:
            fingerprint_attribution_check(d, original)
            print('PASS original fingerprint caller attribution, inclusive duration, fresh bytes and release')
            return
        if '--review-predicate' in sys.argv:
            predicate_check(d, original)
            print('PASS item is explicit predicate-sync-plus-wait; CUDA time stays unmeasured')
            return
        if '--review-ambient' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-ambient-') as scratch:
                public_check(d, original, bridge, Path(scratch), ambient_only=True)
            print('PASS unrelated parent except cannot swallow image cleanup failure')
            return
        if '--review-c-metadata' in sys.argv:
            c_metadata_check(d, original)
            print('PASS balanced C calls use stable callable metadata; wrong C function rejected')
            return
        if '--review-prepare' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-prepare-') as scratch:
                preparation_check(d, Path(scratch))
            print('PASS trainer/serializer FILE pins require the actual canonical bundle')
            return
        if '--review-bundle' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-bundle-') as scratch:
                public_check(d, original, bridge, Path(scratch), bundle_only=True)
            print('PASS exact admitted bundle serializer/output code observed; repo owner rejected')
            return
        if '--review-cleanup' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-cleanup-') as scratch:
                public_check(d, original, bridge, Path(scratch), cleanup_only=True)
            print('PASS all cleanup failures retained; exact cancellation outranks ordinary errors')
            return
        if '--review-transient' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-transient-') as scratch:
                public_check(d, original, bridge, Path(scratch), transient_only=True)
            print('PASS numel cancellation releases only observer transients before authentic teardown')
            return
        if '--review-stack' in sys.argv:
            stack_check(d, original)
            print('PASS mismatched/unfinished profile stacks reject; valid scalar stack completes')
            return
        if '--review-witness' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-witness-') as scratch:
                public_check(d, original, bridge, Path(scratch), witness_only=True)
            print('PASS successful public request requires fingerprint/tensor/output witnesses')
            return
        if '--review-cancellation' in sys.argv:
            with tempfile.TemporaryDirectory(prefix='connected-serving-cancellation-') as scratch:
                public_check(d, original, bridge, Path(scratch), cancel_only=True)
            print('PASS exact profiler cancellation propagates with restoration/genuine cleanup')
            return
        if '--resource-bounds' in sys.argv:
            resource_accounting_check(d, original)
            print('PASS retained resource boundaries and diagnostic calls')
            return
        if '--call-volume' in sys.argv:
            call_volume_check(d, original)
            return
        resource_accounting_check(d, original)
        serializer_check(d, original)
        fingerprint_attribution_check(d, original)
        stack_check(d, original)
        c_metadata_check(d, original)
        predicate_check(d, original)
        with tempfile.TemporaryDirectory(prefix='connected-serving-observer-') as scratch:
            root = Path(scratch)
            public_check(d, original, bridge, root)
            preparation_check(d, root)
            installed = root / 'installed-source-negatives'; installed.mkdir()
            installed_sources_check(d, installed)
            witness, cancel = root / 'missing-witness', root / 'cancellation'
            transient, cleanup = root / 'transient', root / 'cleanup'
            bundle, ambient = root / 'actual-bundle', root / 'ambient-except'
            for path in (witness,cancel,transient,cleanup,bundle,ambient): path.mkdir()
            public_check(d, original, bridge, witness, witness_only=True)
            public_check(d, original, bridge, cancel, cancel_only=True)
            public_check(d, original, bridge, transient, transient_only=True)
            public_check(d, original, bridge, cleanup, cleanup_only=True)
            public_check(d, original, bridge, bundle, bundle_only=True)
            public_check(d, original, bridge, ambient, ambient_only=True)
        assert not any(n.split('.')[0] in NATIVE for n in sys.modules)
        print('PASS source-only genuine public/profiler, fresh-byte, parity and release falsifier')
    finally:
        sys.meta_path.remove(guard)
        for name in owned: sys.modules.pop(name, None)


if __name__ == '__main__':
    main()
