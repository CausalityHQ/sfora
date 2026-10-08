#!/usr/bin/env python3
"""Stdlib-only falsifiers for the source-only connected request driver."""
import hashlib
import ast
from contextlib import contextmanager
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import struct
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
NATIVE = {'torch', 'numpy', 'PIL', 'sfora', 'transformers', 'torchvision', 'safetensors'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fact(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE:
            raise AssertionError('thirdparty import: ' + fullname)


@contextmanager
def public_fixture(root):
    """Real bridge/observer/serializer/release; native and PIL boundaries are standins."""
    helpers = load('_requests_fixture_helpers', HERE / 'test_observe_connected_serving.py')
    observer = load('_requests_fixture_observer', HERE / 'observe_connected_serving.py')
    bridge = load('_requests_fixture_bridge', HERE.parent / 'src/sfora/connected_compact_serving.py')
    serializer = load('_requests_fixture_serializer', HERE.parent / 'src/sfora/connected_inference.py')
    trainer_tree = ast.parse((HERE.parent / 'src/sfora/connected_inference.py').read_text())
    release = ast.unparse(next(n for n in trainer_tree.body if isinstance(n, ast.FunctionDef)
                              and n.name == 'release_inference'))
    test_tree = ast.parse((HERE / 'test_observe_connected_serving.py').read_text())
    public = next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'public_check')
    source = next(n.value.left.left.value for n in public.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'source' for t in n.targets)) + release + '\n'
    widths = dict(helpers.Tensor.widths)
    helpers.Tensor.widths.update({'torch.float32':4, 'torch.int8':1, 'torch.float16':2})
    events = ModuleType('_attribution_events')
    events.original, events.Tensor = serializer, helpers.Tensor
    events.param = helpers.Tensor(b'LIVE')
    events.expected = events.cache = events.retained = None
    events.failure = events.target_error = events.close_error = None
    events.owners, events.closes, events.images, events.searches = [], [], [], []
    bundle = root / 'bundle'
    bundle.mkdir()
    for name in bridge._FILES:
        (bundle / name).write_bytes(b'owned fixture')
    manifest = {'schema':'siglip2-connected-mlp-bundle-v1',
        'code':{},
        'files':{n:fact(bundle/n)['sha256'] for n in bridge._FILES},
        **{n:{} for n in bridge._MANIFEST - {'schema','code','files'}}}
    (bundle / 'bundle.json').write_text(json.dumps(manifest))
    installed = helpers.installed_fixture(root,source=source,bundle=bundle)
    bridge = load('_requests_fixture_installed_bridge',Path(installed['bridge']['path']))
    gallery, library = root/'gallery.bin', root/'native.so'
    gallery.write_bytes(bytes(130*10))
    library.write_bytes(b'fixture native')
    class Image:
        def close(self): events.images.append('closed')
    class Packed:
        @classmethod
        def from_bytes(cls, wire, *, count, dimensions):
            assert dimensions == 128 and len(wire) == 130*count
            return wire
    class Gallery:
        @classmethod
        def open_packed(cls, path, packed): return cls()
        def search_packed(self, wire, *, k):
            assert k == 10
            count = len(wire)//130
            events.searches.append(count)
            if events.failure == 'native': raise RuntimeError('native failed')
            ids = list(range(10))*count
            if events.failure == 'parity' and len(events.owners) == 2: ids[0] = 9
            return (memoryview(struct.pack('<'+'q'*len(ids), *ids)).cast('q', [count,10]),
                    memoryview(struct.pack('<'+'f'*len(ids), *([1.]*len(ids)))).cast('f', [count,10]))
        def close(self):
            events.closes.append('gallery')
            if events.close_error is not None: raise events.close_error
    stubs = {'_attribution_events':events, 'torch':SimpleNamespace(Tensor=helpers.Tensor,
        uint8='uint8', cuda=SimpleNamespace(is_initialized=lambda:False))}
    for name in ('PIL','PIL.Image','sfora','sfora.joint_relational_compaction','sfora.cutile_int8'):
        stubs[name] = ModuleType(name)
    stubs['PIL.Image'].Image = Image
    stubs['sfora.packed_int8'] = helpers.packed_standin(Path(installed['packed']['path']))
    stubs['sfora.joint_relational_compaction'].PackedInt8Embeddings = Packed
    stubs['sfora.cutile_int8'].CutilePackedInt8Gallery = Gallery
    args = {'bundle_dir':bundle, 'expected_bundle_sha256':fact(bundle/'bundle.json')['sha256'],
        'gallery_path':gallery, 'expected_gallery_sha256':fact(gallery)['sha256'], 'gallery_count':10,
        'native_library_path':library, 'expected_native_library_sha256':fact(library)['sha256']}
    pins = {**installed, 'trainer':fact(bundle/bridge._TRAINER),
            'serializer':fact(bundle/'train_siglip2_substrate_adaptation.py')}
    def factory():
        assert not any(not owner._closed for owner in events.owners), 'overlapping public owners'
        owner = bridge.ConnectedCompactIndex.from_bundle(**args)
        events.owners.append(owner)
        return owner
    try:
        with helpers.modules(stubs):
            events.expected = serializer.fingerprint({'param':events.param})
            yield SimpleNamespace(observer=observer, factory=factory, events=events, pins=pins,
                reader=lambda paths:[Image() for p in paths], paths=[root/f'image-{n}' for n in range(32)],
                packed=Packed, gallery=Gallery, native=fact(library))
            assert all(owner._closed and not owner._owned for owner in events.owners)
            assert events.cache is None or events.cache.cache_info().currsize == 0
    finally:
        helpers.Tensor.widths = widths
        for name in ('_requests_fixture_helpers','_requests_fixture_observer',
                     '_requests_fixture_bridge','_requests_fixture_installed_bridge','_requests_fixture_serializer'):
            sys.modules.pop(name, None)


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = HERE / 'qualify_connected_serving_requests.py'
        if path.is_file():
            cls.driver = load('_connected_requests_test_driver', path)

    @classmethod
    def tearDownClass(cls):
        assert sys.getprofile() is None
        assert not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native standin survived suite'

    def driver_module(self):
        self.assertTrue((HERE / 'qualify_connected_serving_requests.py').is_file(),
                        'request driver implementation missing')
        return self.driver

    def test_actual_packed_source_authenticates_legacy_metadata_and_globals(self):
        driver = self.driver_module()
        path = HERE.parent/'src/sfora/packed_int8.py'
        nn = ModuleType('torch.nn'); nn.functional = ModuleType('torch.nn.functional')
        stubs = {'numpy':ModuleType('numpy'),'torch':ModuleType('torch'),'torch.nn':nn,
            'torch.nn.functional':nn.functional}
        with patch.dict(sys.modules,stubs):
            packed = load('sfora.packed_int8',path)
            source = driver.Source(packed,fact(path))
            source.check()
            self.assertEqual(packed._unit_rows.__module__,'sfora.joint_relational_compaction')
            for name in ('_unit_rows','PackedInt8Embeddings','fixed_int8_unit_codes','pack_int8_unit_embeddings'):
                value = getattr(packed,name)
                for attr in ('__name__','__qualname__','__module__'):
                    original_attr = getattr(value,attr)
                    try:
                        setattr(value,attr,'foreign')
                        with self.assertRaisesRegex(ValueError,'packing declarations'): driver.Source(packed,fact(path))
                    finally: setattr(value,attr,original_attr)
            original = packed._unit_rows.__code__
            try:
                packed._unit_rows.__code__ = (lambda value:True).__code__
                with self.assertRaisesRegex(ValueError,'live source'):
                    source.check()
                with self.assertRaisesRegex(ValueError,'live source'):
                    driver.Source(packed,fact(path))
            finally: packed._unit_rows.__code__ = original
            source.check()
            foreign = dict(vars(packed))
            from types import FunctionType
            replacement = FunctionType(original,foreign,packed._unit_rows.__name__)
            replacement.__qualname__ = packed._unit_rows.__qualname__
            replacement.__module__ = packed._unit_rows.__module__
            with patch.object(packed,'_unit_rows',replacement),self.assertRaisesRegex(ValueError,'live source'):
                driver.Source(packed,fact(path))
            method = packed.PackedInt8Embeddings.from_bytes.__func__
            code = method.__code__
            try:
                method.__code__ = (lambda *a,**kw:None).__code__
                with self.assertRaisesRegex(ValueError,'live source'): source.check()
            finally: method.__code__ = code
            foreign_method = FunctionType(code,foreign,method.__name__)
            foreign_method.__qualname__ = method.__qualname__
            foreign_method.__module__ = 'foreign'
            with patch.object(packed.PackedInt8Embeddings,'from_bytes',classmethod(foreign_method)), \
                    self.assertRaisesRegex(ValueError,'live source'):
                driver.Source(packed,fact(path))
            getter = packed.PackedInt8Embeddings.bytes_per_vector.fget
            code = getter.__code__
            try:
                getter.__code__ = (lambda self:130).__code__
                with self.assertRaisesRegex(ValueError,'live source'): source.check()
                with self.assertRaisesRegex(ValueError,'live source'): driver.Source(packed,fact(path))
            finally: getter.__code__ = code
            descriptor = vars(packed.PackedInt8Embeddings)['to_bytes']
            try:
                del packed.PackedInt8Embeddings.to_bytes
                with self.assertRaisesRegex(ValueError,'methods missing'): driver.Source(packed,fact(path))
            finally: packed.PackedInt8Embeddings.to_bytes = descriptor
            source.check()

    def test_missing_go_never_imports_native(self):
        driver = self.driver_module()
        with self.assertRaisesRegex(ValueError, 'full selection GO'):
            driver.read_go(None)
        self.assertFalse(any(n.split('.')[0] in NATIVE for n in sys.modules))

    def test_kill_normal_exit_cannot_admit_native(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            path.write_text(json.dumps({'decision': 'KILL', 'pass': True,
                                       'stage': 'full', 'panel': 'selection'}))
            with self.assertRaisesRegex(ValueError, 'full selection GO'):
                driver.read_go({'receipt': fact(path)})
        self.assertFalse(any(n.split('.')[0] in NATIVE for n in sys.modules))

    def test_serial_twenty_two_genuine_public_requests_have_original_byte_oracles(self):
        driver = self.driver_module()
        self.assertTrue(hasattr(driver, 'request_body'), 'bounded request body missing')
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            report = driver.request_body(f.factory, f.observer, f.reader, lambda:None,
                                         f.paths, f.pins, lambda:None)
            self.assertEqual(len(f.events.owners), 2)
            self.assertEqual(f.events.searches, [1]*10 + [32]*10 + [1,32])
            self.assertEqual(len(f.events.images), 363)
            self.assertEqual(len(f.events.closes), 2)
            self.assertEqual(len(report['calls']), 22)
            self.assertEqual(sum(r['instrumented'] for r in report['calls']), 4)
            self.assertEqual([len(report['timed'][str(n)]['seconds']) for n in (1,32)], [8,8])
            self.assertTrue(report['same_group_original_byte_native_parity'])
            self.assertFalse(report['qualification_eligible'])
            self.assertFalse(report['state_reuse_eligible'])
            self.assertIsNone(sys.getprofile())

    def test_retained_instrumentation_policy_and_counters_are_strict(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            body = driver.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
            report = body['observations']['1']
            driver.validate_instrumentation(report)
            import copy
            for field,value in (('instrumentation_policy',{}),('resource_usage',{}),('first_failure',{})):
                bad = copy.deepcopy(report); bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError): driver.validate_instrumentation(bad)
            for field,value in (('max_depth',513),('aggregate_keys',4097),('fingerprint_records',4097),
                    ('tensor_occurrences',4097),('encoded_bytes',8*1024**2+1),('encoded_bytes',1),('total_calls',True)):
                bad = copy.deepcopy(report); bad['resource_usage'][field] = value
                with self.subTest(field=field,value=value), self.assertRaises(ValueError): driver.validate_instrumentation(bad)
            bad = copy.deepcopy(report); bad['instrumentation_policy']['total_calls'] = 'acceptance_cutoff'
            with self.assertRaises(ValueError): driver.validate_instrumentation(bad)

    def test_first_retained_failure_survives_genuine_cleanup_cancellation(self):
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            index = f.factory()
            probe = f.observer.RequestObserver.from_index(index,f.pins)
            probe.metadata_bytes = 8*1024**2
            cancellation = KeyboardInterrupt('genuine close cancellation')
            f.events.close_error = cancellation
            try:
                f.observer.measure_request(index,f.reader,lambda:None,f.paths[:1],observer=probe)
            except BaseException as error:
                self.assertIs(error,cancellation)
                self.assertTrue(any('observer first failure' in str(e) and 'encoded_bytes' in str(e)
                    for e in error.__cause__.exceptions))
                self.assertTrue(any('observer first failure' in note for note in error.__notes__))
            else: self.fail('accepted bound failure')
            self.assertEqual(probe.first_failure['predicate'],'encoded_bytes')
            self.assertEqual(f.events.images,['closed'])
            self.assertTrue(index._closed)
            self.assertIsNone(sys.getprofile())
            self.assertFalse(probe.stack)

    def test_typed_outputs_and_bytes_are_bounded_before_copy(self):
        helpers = load('_output_bound_helpers',HERE/'test_observe_connected_serving.py')
        observer = load('_output_bound_observer',HERE/'observe_connected_serving.py')
        copied = []
        class Bomb(helpers.Tensor):
            def detach(self): copied.append('detach'); raise AssertionError('copy attempted')
        with helpers.modules({'torch':SimpleNamespace(Tensor=helpers.Tensor,uint8='uint8')}):
            for tensor in (Bomb(bytes(512),'torch.float32',(33,128)),
                    Bomb(bytes(512),'torch.int8',(1,128)),Bomb(bytes(4),'torch.float32',(1,128))):
                with self.assertRaisesRegex(ValueError,'before copy'):
                    observer.tensor_snapshot(tensor,'torch.float32',[1,128],512)
            self.assertEqual(copied,[])
        for count in (0,33):
            ids = memoryview(bytes(count*80)).cast('q')
            scores = memoryview(bytes(count*40)).cast('f')
            with self.assertRaisesRegex(ValueError,'before copy'): observer.native_snapshot((ids,scores))
        sys.modules.pop('_output_bound_helpers',None)
        sys.modules.pop('_output_bound_observer',None)

    def test_authenticated_live_source_replacement_is_rejected(self):
        driver = self.driver_module()
        self.assertTrue(hasattr(driver, 'Source'), 'live source authentication missing')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.py'
            path.write_text('def target(): return 7\n')
            source = driver.Source.load(fact(path))
            try:
                self.assertEqual(source.module.target(), 7)
                source.module.target = lambda:8
                with self.assertRaisesRegex(ValueError, 'live source'):
                    source.check()
            finally:
                sys.modules.pop(source.module.__name__, None)

    def test_cli_missing_go_creates_no_output(self):
        driver = self.driver_module()
        self.assertTrue(hasattr(driver, 'main'), 'native-gated CLI missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = {k:None for k in driver.KEYS}
            authority['schema'] = 'connected-serving-requests-authority-v2'
            path = root/'authority.json'
            path.write_text(json.dumps(authority))
            with self.assertRaisesRegex(ValueError, 'full selection GO'):
                driver.main(['--authority',str(path),'--authority-sha256',fact(path)['sha256'],
                             '--output',str(root/'out')])
            self.assertFalse((root/'out').exists())
        self.assertFalse(any(n.split('.')[0] in NATIVE for n in sys.modules))

    def test_native_tie_oracle_uses_separate_gallery_and_ascending_ids(self):
        driver = self.driver_module()
        self.assertTrue(hasattr(driver, 'native_ties'), 'genuine native tie seam missing')
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            report = driver.native_ties(f.packed, f.gallery, f.native, f.observer)
            self.assertEqual(f.events.searches, [1,32])
            self.assertEqual(f.events.closes, ['gallery'])
            self.assertEqual(f.events.owners, [])
            self.assertTrue(report['ascending_ordinal_score_bits_exact'])

    def test_exit_reserve_and_resource_violations_never_extend_cap(self):
        driver = self.driver_module()
        self.assertTrue(hasattr(driver, 'check_resources'), 'whole-unit resource guard missing')
        policy = {'whole_process_seconds':300, 'exit_reserve_seconds':30}
        facts = {'wall_seconds':269, 'process_peak_rss_kib':12,
                 'peak_cuda_allocated_bytes':1}
        driver.check_resources(facts, policy, reserve=True)
        for key,value in (('wall_seconds',270), ('process_peak_rss_kib',8388609),
                          ('peak_cuda_allocated_bytes',10000000000), ('wall_seconds',float('nan'))):
            with self.assertRaisesRegex(ValueError, 'resource|headroom'):
                driver.check_resources(facts | {key:value}, policy, reserve=True)

    def test_success_tail_publishes_genuine_retained_resource_facts(self):
        # Execute the actual run tail without native imports; the old undefined `resources` fails here.
        driver = self.driver_module()
        tree = ast.parse(Path(driver.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run')
        start = next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
            isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].slice,ast.Constant) and
            n.targets[0].slice.value == 'full_uncached_exit_pass')
        fn = ast.parse('def tail(output,record,context,policy,final_resources): pass').body[0]
        fn.body = run.body[start:]
        namespace = dict(vars(driver))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                     '<executed original request run tail>', 'exec'),namespace)
        original = strict_fixture('connected-mlp-evaluation-full-export-candidate-179061-v2/receipt.json')
        final = {k:original[k] for k in ('wall_seconds','process_peak_rss_kib','peak_cuda_allocated_bytes')}
        helper = load('_requests_original_publisher', HERE/'export_siglip2_substrate_adaptation.py')
        self.addCleanup(sys.modules.pop, helper.__name__, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'out'
            record = {'qualification_eligible':False,'state_reuse_eligible':False}
            result = namespace['tail'](path,record,{'guards':{'input':'a'*64},'helper':helper},
                                       {'whole_process_seconds':1500,'exit_reserve_seconds':30},final)
            saved = json.loads((path/'receipt.json').read_bytes())
            self.assertEqual(saved['resources'], final)
            self.assertTrue(saved['full_uncached_exit_pass'])
            self.assertFalse(saved['qualification_eligible'])
            self.assertEqual(result, saved)
            before = (path/'receipt.json').read_bytes()
            with self.assertRaises(FileExistsError):
                namespace['tail'](path,{}, {'guards':{},'helper':helper},
                                  {'whole_process_seconds':1500,'exit_reserve_seconds':30},final)
            self.assertEqual((path/'receipt.json').read_bytes(), before)

    def test_driver_and_stdlib_dataclass_live_sources_authenticate(self):
        driver = self.driver_module()
        driver.Source(driver, fact(Path(driver.__file__))).check()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'owner.py'
            path.write_text('from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Owner:\n    value: int\n')
            source = driver.Source.load(fact(path))
            try:
                self.assertEqual(source.module.Owner(7).value, 7)
                source.check()
            finally: sys.modules.pop(source.module.__name__,None)

    def test_native_or_full_byte_parity_failure_releases_both_original_owners(self):
        driver = self.driver_module()
        for failure, message in (('native','native failed'),('parity','parity differs'),('inference','inference failed')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
                f.events.failure = failure
                with self.assertRaisesRegex((ValueError,RuntimeError), message):
                    driver.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
                self.assertTrue(all(o._closed and not o._owned for o in f.events.owners))
                self.assertIsNone(sys.getprofile())

    def test_guard_failure_and_cancellation_still_run_genuine_cleanup(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            def guard():
                if len(f.events.searches) >= 2: raise ValueError('resource predicate failed')
            with self.assertRaisesRegex(ValueError, 'resource predicate failed'):
                driver.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,guard)
            self.assertEqual(len(f.events.owners),1)
            self.assertEqual(f.events.closes,['gallery'])
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            f.events.target_error = KeyboardInterrupt('exact cancellation')
            f.events.close_error = RuntimeError('original close failed')
            with self.assertRaises(KeyboardInterrupt) as caught:
                driver.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
            self.assertIs(caught.exception, f.events.target_error)
            self.assertTrue(any('original close failed' in n for n in caught.exception.__notes__))
            self.assertIsNone(sys.getprofile())

    def test_owned_lock_descriptors_remain_exclusive_and_are_not_closed(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(n) for n in range(2)]
            for path in paths: path.touch()
            with paths[0].open('rb') as first, paths[1].open('rb') as second:
                locks = driver.Locks([{'path':str(p),'fd':f.fileno()} for p,f in zip(paths,(first,second))])
                locks.check()
                self.assertFalse(first.closed or second.closed)
                paths[0].unlink()
                paths[0].touch()
                with self.assertRaisesRegex(ValueError,'lock'):
                    locks.check()

    def test_missing_full_terminal_and_source_bytes_cannot_be_laundered_as_go(self):
        driver = self.driver_module()
        evaluator = load('_requests_original_evaluator', HERE/'evaluate_siglip2_connected_mlp.py')
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'record.json'
                path.write_text('{"decision":"GO","stage":"full","panel":"selection"}')
                metadata = {'receipt':fact(path)}
                driver.read_go(metadata)  # Early metadata is deliberately not original admission.
                with self.assertRaisesRegex(ValueError,'complete actual UNIT'):
                    evaluator.check_unit(metadata)
                path.write_text('{"decision":"GO","decision":"KILL"}')
                with self.assertRaisesRegex(ValueError,'duplicate'):
                    driver.read_go({'receipt':fact(path)})
                pin = fact(path)
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    driver.read_go({'receipt':pin})
        finally: sys.modules.pop(evaluator.__name__,None)

    def test_wrong_tie_order_and_tie_cleanup_errors_reject(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            class WrongTie(f.gallery):
                def search_packed(self, wire, *, k):
                    ids,scores = super().search_packed(wire,k=k)
                    ids = memoryview(bytearray(ids.tobytes())).cast('q',ids.shape)
                    ids[0,0] = 9
                    return ids,scores
            with self.assertRaisesRegex(ValueError,'tied-score'):
                driver.native_ties(f.packed,WrongTie,f.native,f.observer)
            self.assertEqual(f.events.closes,['gallery'])
            f.events.close_error = RuntimeError('tie close failed')
            with self.assertRaisesRegex(RuntimeError,'tie close failed'):
                driver.native_ties(f.packed,f.gallery,f.native,f.observer)

    def test_malformed_timing_and_missing_output_reject_after_real_public_call(self):
        driver = self.driver_module()
        for fault in ('seconds','raw','wire_hex'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
                def measure(*args,**kwargs):
                    result,row = f.observer.measure_request(*args,**kwargs)
                    if fault == 'seconds': row['seconds'] = -1
                    return result,row
                class Probe(f.observer.RequestObserver):
                    def report(self):
                        report = super().report()
                        if fault != 'seconds': report['output'].pop(fault)
                        return report
                proxy = SimpleNamespace(measure_request=measure,RequestObserver=Probe)
                with self.assertRaisesRegex(ValueError,'timing|witness'):
                    driver.request_body(f.factory,proxy,f.reader,lambda:None,f.paths,f.pins,lambda:None)
                self.assertEqual(f.events.searches,[1])
                self.assertEqual(f.events.closes,['gallery'])
                self.assertIsNone(sys.getprofile())

    def test_restored_original_source_hash_and_decorator_closure_are_required(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.py'
            raw = 'from contextlib import contextmanager\n@contextmanager\ndef target():\n    yield 7\n'
            path.write_text(raw)
            source = driver.Source.load(fact(path))
            try:
                source.check()
                path.write_text(raw+'# source mutation\n')
                with self.assertRaisesRegex(ValueError,'SHA256'): source.check()
                path.write_text(raw)
                source.check()
                source.module.target.__closure__[0].cell_contents = lambda:8
                with self.assertRaisesRegex(ValueError,'live source'): source.check()
            finally: sys.modules.pop(source.module.__name__,None)

    def test_raw_unit_and_inverse_norm_bits_are_required_same_group(self):
        driver = self.driver_module()
        for name in ('raw','unit','inverse_norms'):
            with self.subTest(output=name), tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
                class Probe(f.observer.RequestObserver):
                    def report(self):
                        report = super().report()
                        if len(f.events.owners) == 2:
                            report['output'][name]['hex'] = '01'+report['output'][name]['hex'][2:]
                        return report
                proxy = SimpleNamespace(measure_request=f.observer.measure_request,RequestObserver=Probe)
                with self.assertRaisesRegex(ValueError,'parity differs'):
                    driver.request_body(f.factory,proxy,f.reader,lambda:None,f.paths,f.pins,lambda:None)
                self.assertEqual(len(f.events.owners),2)
                self.assertEqual(f.events.closes,['gallery','gallery'])

    def test_full_normal_go_log_descriptors_and_frozen_export_launch_are_original_checks(self):
        evaluator = load('_requests_original_evaluator', HERE/'evaluate_siglip2_connected_mlp.py')
        try:
            record = strict_fixture('connected-mlp-evaluation-full-export-candidate-179061-v2/receipt.json')
            launch = record['launch']
            args = SimpleNamespace(execution_sha256=record['execution_sha256'],phase='export',
                                   arm='candidate',seed=179061)
            evaluator.check_launch(launch,args)
            for key,value in (('stage','first'),('both_locks_held',False),('selected_cpu',None),('exports',{})):
                if key == 'exports': continue  # Export launches genuinely have an empty export set.
                with self.subTest(key=key), self.assertRaises((ValueError,KeyError)):
                    evaluator.check_launch(launch | {key:value},args)
            unit = launch['selected_cpu']
            evaluator.check_unit(unit)
            for key,value in (('both_locks_held',False),('invocation_id','0'),('service_seconds',float('nan'))):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'complete actual UNIT'):
                    evaluator.check_unit(unit | {key:value})
        finally: sys.modules.pop(evaluator.__name__,None)

    def test_partial_decode_cleanup_retains_exact_cancellation_and_every_close_failure(self):
        driver = self.driver_module()
        original = load('_requests_original_observer', HERE/'observe_connected_serving.py')
        self.addCleanup(sys.modules.pop,original.__name__,None)
        target = KeyboardInterrupt('decode cancelled')
        opened_closes, decoded_closes = [], []
        class Decoded:
            def close(self):
                decoded_closes.append(1)
                raise RuntimeError('decoded close failed')
        class Opened:
            def __init__(self,path): self.path = path
            def convert(self,mode):
                assert mode == 'RGB'
                if self.path.name == '1': raise target
                return Decoded()
            def close(self):
                opened_closes.append(self.path.name)
                if self.path.name == '1': raise RuntimeError('opened close failed')
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(n) for n in range(3)]
            for path in paths: path.write_bytes(b'opaque fixture')
            with self.assertRaises(KeyboardInterrupt) as caught:
                driver.decode_images(original,SimpleNamespace(open=Opened),[fact(p) for p in paths],paths)
            self.assertIs(caught.exception,target)
            self.assertEqual(opened_closes,['0','1'])
            self.assertEqual(decoded_closes,[1])
            self.assertTrue(any('opened close failed' in n for n in target.__notes__))
            self.assertTrue(any('decoded close failed' in n for n in target.__notes__))

    def test_body120_stops_after_first_over_budget_call_and_releases_owner(self):
        driver = self.driver_module()
        with tempfile.TemporaryDirectory() as directory, public_fixture(Path(directory)) as f:
            def clock(): return 121. if f.events.searches else 0.
            with patch.object(driver.time,'perf_counter',clock), self.assertRaisesRegex(ValueError,'body120'):
                driver.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
            self.assertEqual(f.events.searches,[1])
            self.assertEqual(f.events.closes,['gallery'])
            self.assertIsNone(sys.getprofile())


def strict_fixture(name):
    return json.loads((HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'/name).read_bytes())


if __name__ == '__main__':
    assert __debug__ and sys.getprofile() is None
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    try:
        unittest.main()
    finally:
        sys.meta_path.remove(guard)
        sys.modules.pop('_connected_requests_test_driver', None)
