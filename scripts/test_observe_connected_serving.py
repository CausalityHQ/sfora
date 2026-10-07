#!/usr/bin/env python3
"""One stdlib-only falsifier; no native, image, quality or speed qualification."""
import ast
from contextlib import contextmanager
import gc
import hashlib
import importlib.abc
import importlib.util
import json
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
    widths = {'uint8': 1, 'float32': 4, 'int64': 8}

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
    sources = {'serializer': binding(Path(original.__file__))}
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


def stack_check(d, original):
    sources = {'serializer':binding(Path(original.__file__))}
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


def public_check(d, original, bridge, root, *, witness_only=False, cancel_only=False):
    # Real bridge authentication, registry, serializer, exact wire and release predicates.
    trainer_source = (ROOT / 'scripts/train_siglip2_connected_mlp.py').read_text()
    tree = ast.parse(trainer_source)
    release = ast.get_source_segment(trainer_source, next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == 'release_inference'))
    events = ModuleType('_attribution_events')
    events.original, events.Tensor = original, Tensor
    events.param = Tensor(b'LIVE')
    events.expected = None
    events.cache = events.retained = None
    events.failure = None
    source = '''import gc, sys, weakref
from functools import lru_cache
from types import ModuleType
import _attribution_events as events
def require(condition, message):
    if not condition: raise ValueError(message)
def load_authenticated(name, path): return None
class Owner:
    def parameters(self): return ()
    def buffers(self): return ()
def _processor_cache(processor, guards):
    return object() if events.failure == 'cache_authority' else processor.cache
def load_inference(directory, digest, device):
    assert device == 'cuda'
    helper = ModuleType('_connected_serving_attribution_fixture')
    sys.modules[helper.__name__] = helper
    cache = lru_cache(maxsize=10)(lambda value: value)
    endpoint = {name:Owner() for name in ('model','processor_object','head_object','A','C','mu_train')}
    endpoint.update(modules={'helper':helper}, guards={}, processor_cache=cache)
    endpoint['processor_object'].cache = cache
    cache(endpoint['processor_object'])
    events.cache = cache
    events.refs = [weakref.ref(endpoint[name]) for name in ('model','processor_object','head_object','A','C','mu_train')]
    if events.failure == 'lifetime': events.retained = endpoint['model']
    return endpoint
def inference_outputs(endpoint, images):
    digest = events.original.fingerprint({'param':events.param})
    require(digest == events.expected, 'current .data rejected')
    if events.failure == 'inference': raise ValueError('inference failed')
    wire = bytes([129]) * (130 * len(images))
    return {'raw':events.Tensor(b'raw'), 'unit':events.Tensor(b'unit'),
            'codes':events.Tensor(b'codes'), 'inverse_norms':events.Tensor(b'norm'), 'wire':wire}
''' + release + '\n'
    bundle = root / 'bundle'
    bundle.mkdir()
    for name in bridge._CODE:
        (bundle / name).write_text(source if name == bridge._TRAINER else '# standin helper\n')
    for name in bridge._FILES:
        (bundle / name).write_bytes(b'owned standin')
    manifest = {'schema':'siglip2-connected-mlp-bundle-v1',
        'code':{name:binding(bundle / name)['sha256'] for name in bridge._CODE},
        'files':{name:binding(bundle / name)['sha256'] for name in bridge._FILES},
        **{name:{} for name in bridge._MANIFEST - {'schema', 'code', 'files'}}}
    (bundle / 'bundle.json').write_text(json.dumps(manifest))
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
            return memoryview(struct.pack('10q', *range(10))), memoryview(struct.pack('10f', *([1.] * 10)))
        def close(self): closes.append('gallery')
    stubs = {'_attribution_events':events, 'torch':SimpleNamespace(Tensor=Tensor, uint8='uint8',
              cuda=SimpleNamespace(is_initialized=lambda:False))}
    for name in ('PIL', 'PIL.Image', 'sfora', 'sfora.joint_relational_compaction', 'sfora.cutile_int8'):
        stubs[name] = ModuleType(name)
    stubs['PIL.Image'].Image = Image
    stubs['sfora.joint_relational_compaction'].PackedInt8Embeddings = Packed
    stubs['sfora.cutile_int8'].CutilePackedInt8Gallery = Gallery
    args = dict(bundle_dir=bundle, expected_bundle_sha256=binding(bundle / 'bundle.json')['sha256'],
        gallery_path=gallery, expected_gallery_sha256=binding(gallery)['sha256'], gallery_count=10,
        native_library_path=library, expected_native_library_sha256=binding(library)['sha256'])
    sources = {'bridge':binding(Path(bridge.__file__)), 'trainer':binding(bundle / bridge._TRAINER),
        'serializer':binding(Path(original.__file__))}
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
        if cancel_only:
            class Cancelled(BaseException): pass
            for cancellation in (KeyboardInterrupt('stop'), SystemExit(37), Cancelled('stop')):
                class CancelObserver(d.RequestObserver):
                    def _event(self, frame, event, arg, tick):
                        if event == 'call' and frame.f_code is self.fingerprint_code:
                            raise cancellation
                        super()._event(frame, event, arg, tick)
                index = bridge.ConnectedCompactIndex.from_bundle(**args)
                probe = CancelObserver(original.fingerprint, sources)
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
            probe = InspectionError(original.fingerprint, sources)
            rejects(lambda:request(index, probe), 'incomplete observation')
            assert index._closed and events.cache.cache_info().currsize == 0
            assert not probe.report()['complete'] and sys.getprofile() is None
            return
        if witness_only:
            # Generic scalar profiling is valid; PUBLIC profiling needs byte/output witnesses.
            scalar = d.RequestObserver(original.fingerprint, sources)
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
                probe = observer_type(original.fingerprint, sources)
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
        probe = d.RequestObserver(original.fingerprint, sources)
        observed = request(index, probe)
        assert row['native'] == observed['native']
        report = probe.report()
        assert report['output']['wire_hex'] == (b'\x81' * 130).hex()
        assert report['output']['raw']['hex'] == b'raw'.hex()
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
            probe = d.RequestObserver(original.fingerprint, sources)
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
        probe = d.RequestObserver(original.fingerprint, sources)
        rejects(lambda: request(index, probe), 'current .data rejected')
        assert index._closed and not index._owned
        index.close()
        events.param.data[0] = ord('L')
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        probe = d.RequestObserver(original.fingerprint, sources)
        # A diagnostic counter failure closes the genuine endpoint too.
        probe.calls = 2_000_000
        rejects(lambda: request(index, probe), 'incomplete observation')
        assert index._closed and events.cache.cache_info().currsize == 0
        index = bridge.ConnectedCompactIndex.from_bundle(**args)
        trainer = bundle / bridge._TRAINER
        trainer.write_text(source + '# stale source\n')
        rejects(lambda: request(index), 'current file bytes differ')
        assert index._closed and not index._owned
        trainer.write_text(source)
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
    names = {'observer':'scripts/observe_connected_serving.py', 'test':'scripts/test_observe_connected_serving.py',
        'bridge':'src/sfora/connected_compact_serving.py', 'trainer':'scripts/train_siglip2_connected_mlp.py',
        'serializer':'scripts/train_siglip2_substrate_adaptation.py'}
    bundle = root / 'future_bundle'
    bundle.mkdir()
    manifest = bundle / 'bundle.json'
    manifest.write_bytes(b'prospective standin; NOT an admitted bundle')
    fixture = root / 'future.fixture'
    fixture.write_bytes(b'not native/image/terminal evidence')
    images = []
    for number in range(32):
        path = root / f'TRAIN-{number}.fixture'
        path.write_bytes(bytes([number]))
        images.append(binding(path))
    authority = {'schema':'connected-serving-attribution-authority-v1',
        'sources':{role:binding(ROOT / name) for role,name in names.items()},
        'bundle':{'directory':str(bundle), 'manifest':binding(manifest)},
        'gallery':{'file':binding(fixture),'count':10}, 'native':binding(fixture),
        'train_images':images, 'qualified_terminal':binding(fixture), 'cache_conditions':'fixed cold read, resident gallery',
        'resource_policy':{'body_seconds':120,'host_bytes':8589934592,'swap_bytes':0,
            'cuda_allocated_bytes_exclusive':10000000000,'whole_process_seconds':300,'exit_reserve_seconds':30},
        'both_locks_held':True, 'qualification_eligible':False, 'state_reuse_eligible':False}
    path = root / 'authority.json'
    def write(value):
        path.write_text(json.dumps(value))
        return binding(path)['sha256']
    digest = write(authority)
    assert d.prepare(path, digest) == authority
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


def main():
    assert __debug__ and sys.getprofile() is None
    assert not any(n.split('.')[0] in NATIVE for n in sys.modules)
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    owned = ('_serving_observer_check', '_serving_serializer_check', '_serving_bridge_check')
    try:
        path = ROOT / 'scripts/observe_connected_serving.py'
        assert path.is_file(), 'observer implementation missing'
        d = load(owned[0], path)
        original = load(owned[1], ROOT / 'scripts/train_siglip2_substrate_adaptation.py')
        bridge = load(owned[2], ROOT / 'src/sfora/connected_compact_serving.py')
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
        serializer_check(d, original)
        stack_check(d, original)
        with tempfile.TemporaryDirectory(prefix='connected-serving-observer-') as scratch:
            root = Path(scratch)
            public_check(d, original, bridge, root)
            preparation_check(d, root)
            witness, cancel = root / 'missing-witness', root / 'cancellation'
            witness.mkdir(); cancel.mkdir()
            public_check(d, original, bridge, witness, witness_only=True)
            public_check(d, original, bridge, cancel, cancel_only=True)
        assert not any(n.split('.')[0] in NATIVE for n in sys.modules)
        print('PASS source-only genuine public/profiler, fresh-byte, parity and release falsifier')
    finally:
        sys.meta_path.remove(guard)
        for name in owned: sys.modules.pop(name, None)


if __name__ == '__main__':
    main()
