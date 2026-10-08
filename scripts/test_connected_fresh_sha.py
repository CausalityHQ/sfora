#!/usr/bin/env python3
"""Bounded stdlib falsifier:120s/AS1GiB; Torch/NumPy/native/GPU work UNRUN."""
import concurrent.futures
import copy
import hashlib
import importlib.abc
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import py_compile
import resource
import signal
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
ORIGINAL_QUALIFIER_SHA256 = '2961c2b0129b762403273c1d550b44c0ba7e4e9ec231a3e39848a9ee57d68815'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def facts(path):
    return {'path': str(path), 'sha256': sha(Path(path).read_bytes())}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fresh = load('qualify_connected_fresh_sha', HERE / 'qualify_connected_fresh_sha.py')
DRIVER = (HERE / 'qualify_connected_fresh_sha.py').read_bytes()
ORIGINAL = (HERE / 'qualify_connected_control_serving.py').read_bytes()


class Dtype:
    def __init__(self, name, size):
        self.name, self.size = name, size
    def __str__(self):
        return self.name


F32, U8 = Dtype('torch.float32', 4), Dtype('torch.uint8', 1)


def strides_of(shape):
    strides, step = [], 1
    for n in reversed(shape):
        strides.insert(0, step)
        step *= max(n, 1)
    return tuple(strides)


class Tensor:
    """Stdlib byte-faithful strided tensor; options switch on the failure mutants."""
    options = {}

    def __init__(self, store, dtype, shape, stride, offset, device, box):
        self.store, self.dtype, self.shape, self._stride = store, dtype, tuple(shape), tuple(stride)
        self.offset, self.device, self.box = offset, SimpleNamespace(type=device), box

    numel = lambda self: math.prod(self.shape)
    element_size = lambda self: self.dtype.size
    dim = lambda self: len(self.shape)
    storage_offset = lambda self: self.offset
    stride = lambda self, i=None: self._stride if i is None else self._stride[i]
    is_contiguous = lambda self: self._stride == strides_of(self.shape)
    _version = property(lambda self: self.box[0])

    def view_of(self, dtype, shape, stride, offset, box=None):
        return Tensor(self.store, dtype, shape, stride, offset, self.device.type, self.box if box is None else box)

    def positions(self):
        for index in itertools.product(*map(range, self.shape)):
            yield self.offset + sum(i * s for i, s in zip(index, self._stride))

    def tobytes(self):
        size = self.dtype.size
        return b''.join(bytes(self.store[p * size:(p + 1) * size]) for p in self.positions())

    def clone(self):
        return Tensor(bytearray(self.tobytes()), self.dtype, self.shape, strides_of(self.shape), 0, self.device.type, [0])

    def detach(self):
        return self.view_of(self.dtype, self.shape, self._stride, self.offset)

    data = property(lambda self: self.view_of(self.dtype, self.shape, self._stride, self.offset,
        self.box if self.options.get('data_shares_version') else [self.box[0]]))

    def contiguous(self):
        return self if self.is_contiguous() else self.clone()

    def t(self):
        return self.view_of(self.dtype, self.shape[::-1], self._stride[::-1], self.offset)

    def reshape(self, *shape):
        base = self.contiguous()
        shape = tuple(shape)
        if -1 in shape:
            shape = tuple(base.numel() // -math.prod(shape) if n == -1 else n for n in shape)
        return base.view_of(self.dtype, shape, strides_of(shape), base.offset)

    def view(self, dtype):
        assert self.is_contiguous()
        shape = self.shape[:-1] + (self.shape[-1] * self.dtype.size // dtype.size,)
        return self.view_of(dtype, shape, strides_of(shape), self.offset * self.dtype.size // dtype.size)

    def _get(self, key):
        key = key if isinstance(key, tuple) else (key,)
        shape, stride, offset = [], [], self.offset
        for i, (n, s) in enumerate(zip(self.shape, self._stride)):
            k = key[i] if i < len(key) else slice(None)
            if isinstance(k, int):
                offset += (k if k >= 0 else k + n) * s
            else:
                start, stop, step = k.indices(n)
                shape.append(len(range(start, stop, step)))
                stride.append(s * step)
                offset += start * s
        return self.view_of(self.dtype, shape, stride, offset)

    def __getitem__(self, key):
        view = self._get(key)
        keys = key if isinstance(key, tuple) else (key,)
        return view.clone() if self.options.get('slice_copies') and any(isinstance(k, slice) for k in keys) else view

    def __setitem__(self, key, value):
        self._get(key).copy_(value)

    def copy_(self, other):
        if self.options.get('broken_copy') and self.dim() == 1:
            return self
        size = self.dtype.size
        if isinstance(other, Tensor):
            for p, q in zip(self.positions(), other.positions()):
                self.store[p * size:(p + 1) * size] = other.store[q * size:(q + 1) * size]
        else:
            for p in self.positions():
                self.store[p * size:(p + 1) * size] = struct.pack('<f', other)
        self.box[0] += 1
        return self

    def __ixor__(self, mask):
        for p in self.positions():
            self.store[p] ^= mask
        self.box[0] += 1
        return self

    def __iadd__(self, value):
        for p in self.positions():
            self.store[p * 4:p * 4 + 4] = struct.pack('<f', struct.unpack('<f', self.store[p * 4:p * 4 + 4])[0] + value)
        self.box[0] += 1
        return self


def make(values, shape, device='cuda', dtype=F32):
    return Tensor(bytearray(struct.pack('<%df' % len(values), *values)), dtype, shape, strides_of(shape), 0, device, [0])


def zeros(shape, dtype=F32, device='cpu'):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor(bytearray(math.prod(shape) * dtype.size), dtype, shape, strides_of(shape), 0, device, [0])


def fake_torch():
    def tensor(value, dtype=F32, device='cpu'):
        return make([value], (), device)
    def arange(n, dtype=F32, device='cpu'):
        return make([float(i) for i in range(n)], (n,), device)
    return SimpleNamespace(zeros=zeros, tensor=tensor, arange=arange, float32=F32, uint8=U8, Tensor=Tensor,
        equal=lambda a, b: a.shape == b.shape and a.tobytes() == b.tobytes(), cuda=SimpleNamespace(synchronize=lambda: None))


def frame(digest, raw):
    raw = raw.encode() if isinstance(raw, str) else raw
    digest.update(str(len(raw)).encode() + b':' + raw)


def tree_digest(tree, leaf):
    digest = hashlib.sha256()
    frame(digest, 'dict'); frame(digest, str(len(tree)))
    for key in sorted(tree, key=repr):
        for part in (key, 'Tensor', str(tree[key].dtype), repr(tuple(tree[key].shape)), leaf(tree[key])):
            frame(digest, part)
    return digest.hexdigest()


def original_fingerprint(tree):
    return tree_digest(tree, lambda t: sha(t.detach().contiguous().tobytes()))


def proposed_pipeline(budget, flavor=None):
    cache, kept = {}, []
    def leaf(item):
        key = (id(item.store), item.offset, item._version, str(item.dtype), item.shape)
        if flavor == 'stale' and key in cache: return cache[key]
        raw = bytes(item.store) if flavor == 'raw_storage' else item.detach().contiguous().tobytes()
        cache[key] = result = executor.submit(sha, raw).result()
        return result
    def pipeline(tree):
        global executor
        if flavor == 'nojoin': threading.Thread(target=threading.Event().wait, args=(0.4,), daemon=True).start()
        if flavor == 'leak': kept.append(tree)
        item = None
        try:
            if not (isinstance(tree, dict) and all(type(k) is str for k in tree)):
                raise ValueError('CUDA fingerprint requires a string-to-tensor dictionary')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                for key in sorted(tree, key=repr):
                    item = tree[key]
                    if item.device.type != 'cuda': raise ValueError('unrelated' if flavor == 'wrong_cpu' else fresh.CPU_MESSAGE)
                    if item.numel() * item.element_size() > budget: raise ValueError('unrelated' if flavor == 'wrong_over' else fresh.OVER_MESSAGE)
                return tree_digest(tree, leaf)
        finally:
            if flavor != 'retain_locals': tree = item = None
    return pipeline


MLP = ('mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias')
BUDGET = 65536


def fixture(flavor=None, **options):
    Tensor.options = dict(options)
    params = {}
    for i in range(444):
        shape = (6, 8) if i % 40 == 0 else (32,)
        params['layer%03d.p' % i] = make([i + j / 7 for j in range(math.prod(shape))], shape)
    for i, name in enumerate(MLP):
        params[name] = make([1000 + i + j for j in range(32)], (32,))
    frozen = {n: p for n, p in params.items() if n not in MLP}
    model = SimpleNamespace(named_parameters=lambda: iter(params.items()), state_dict=lambda: {n: p.detach() for n, p in params.items()})
    original = SimpleNamespace(MLP=MLP, fingerprint=original_fingerprint)
    endpoint = {'model': model, 'modules': {'runtime': original}, 'vision_sha256': original_fingerprint(model.state_dict()),
        'encoder_identity': {'frozen_sha256': original_fingerprint(frozen)}}
    index = SimpleNamespace(_endpoint=endpoint, _module=original, _check_current=lambda: None, close=lambda: None)
    checks = []
    module = SimpleNamespace(MLP=MLP, hashlib=hashlib, ThreadPoolExecutor=concurrent.futures.ThreadPoolExecutor,
        _fingerprint_cuda_dict=proposed_pipeline(BUDGET, flavor))
    source = SimpleNamespace(module=module, check=lambda: checks.append(1))
    return SimpleNamespace(index=index, source=source, params=params, torch=fake_torch(), checks=checks,
        guard=lambda: {'process_peak_rss_kib': 4000000, 'peak_cuda_allocated_bytes': 2000000000})


REAL_MEASURE = fresh.measure


def measure(f):
    return REAL_MEASURE(f.index, f.torch, f.source, f.guard, BUDGET)


def snapshot(f):
    return {n: p.tobytes() for n, p in f.params.items()}, {n: p._version for n, p in f.params.items()}


class SeamInverse(unittest.TestCase):
    def test_pins_and_real_inverse(self):
        self.assertEqual(sha(ORIGINAL), ORIGINAL_QUALIFIER_SHA256)
        self.assertEqual([s[0] for s in fresh.SEAMS], ['S%d' % i for i in range(1, 9)])
        fresh.check_seams(ORIGINAL, DRIVER)

    def test_every_seam_and_unseamed_statement_is_pinned(self):
        text = DRIVER.decode()
        run_at = text.index('def run(args):')
        for sid, old, new in fresh.SEAMS:
            last = new.rstrip('\n').split('\n')[-1]
            self.assertIn(last, text[run_at:], sid)
            with self.assertRaisesRegex(ValueError, 'seam|outside the declared seams', msg=sid):
                fresh.check_seams(ORIGINAL, (text[:run_at] + text[run_at:].replace(last, last + '; pass', 1)).encode())
        for needle, replacement in (("'exclusive canonical new output required'", "'weaker'"),
                ("policy['whole_process_seconds'] <= evaluator.policy('export')['seconds'] and", ''),
                ('api.evaluator_exit(context,exit_guard)', 'pass'), ('locks.check()', 'pass')):
            self.assertIn(needle, text[run_at:], needle)
            with self.assertRaisesRegex(ValueError, 'seam|outside the declared seams'):
                fresh.check_seams(ORIGINAL, (text[:run_at] + text[run_at:].replace(needle, replacement, 1)).encode())

    def test_missing_duplicate_and_changed_original_rejected(self):
        text = DRIVER.decode()
        run_at = text.index('def run(args):')
        def edit(old, new):
            self.assertIn(old, text[run_at:])
            return (text[:run_at] + text[run_at:].replace(old, new, 1)).encode()
        with self.assertRaisesRegex(ValueError, 'seam 2 must match exactly once'):
            fresh.check_seams(ORIGINAL, edit('        require_no_native()\n', ''))
        statement = 'diagnostic = falsify(factory,guard,torch,proposed_source)\n'
        with self.assertRaisesRegex(ValueError, 'seam 5 must match exactly once'):
            fresh.check_seams(ORIGINAL, edit(statement, statement + '            ' + statement))
        with self.assertRaisesRegex(ValueError, 'outside the declared seams'):
            fresh.check_seams(ORIGINAL.replace(b"'insufficient whole-process body/exit headroom'", b"'x'"), DRIVER)
        with self.assertRaisesRegex(ValueError, 'seam 2 must match exactly once'):
            fresh.check_seams(ORIGINAL.replace(b'locks = requests.Locks(authority', b'locks = requests.Locks2(authority'), DRIVER)

    def test_driver_and_original_survive_unchanged_source_authentication(self):
        fresh.requests.Source(fresh, facts(HERE / 'qualify_connected_fresh_sha.py')).check()
        fresh.requests.Source(fresh.control, facts(HERE / 'qualify_connected_control_serving.py')).check()


class Proposal(unittest.TestCase):
    ORIGINAL_RUNTIME = (b"import hashlib\nMLP = ('a',)\n\ndef fingerprint(value, frozen=None, consumed=None):\n"
                        b"    return hashlib.sha256(repr(value).encode()).hexdigest()\n\ndef late():\n    import torch\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        (self.root / 'original_runtime.py').write_bytes(self.ORIGINAL_RUNTIME)
        self.sources = {'runtime': facts(self.root / 'original_runtime.py'),
            'control_driver': facts(HERE / 'qualify_connected_control_serving.py')}
        self.write()

    def write(self, runtime=None, ledger=None, bridge=None, mutate=None):
        runtime = runtime or self.ORIGINAL_RUNTIME + b"\ndef _fingerprint_cuda_dict(value):\n    return value\n"
        (self.root / 'runtime.py').write_bytes(runtime)
        (self.root / 'ledger.py').write_bytes(ledger or b'RUNTIME_SHA256 = "%s"\n' % sha(runtime).encode())
        (self.root / 'bridge.py').write_bytes(bridge or b'("%s",)\n' % sha((self.root / 'ledger.py').read_bytes()).encode())
        proposal = {'schema': fresh.PROPOSAL_SCHEMA, 'driver': facts(HERE / 'qualify_connected_fresh_sha.py'),
            'test': facts(HERE / 'test_connected_fresh_sha.py'), 'commit': 'a' * 40,
            'proposed': {k: facts(self.root / (k + '.py')) for k in ('runtime', 'ledger', 'bridge')}}
        if mutate: mutate(proposal)
        (self.root / 'proposal.json').write_text(json.dumps(proposal))
        return facts(self.root / 'proposal.json')

    def read(self, **kw):
        return fresh.read_proposal(self.write(**kw), self.sources)

    def test_accepts_exact_closure(self):
        proposal = self.read()
        self.assertEqual(len(fresh.proposal_files(proposal)), 5)
        source = fresh.requests.Source.load(proposal['proposed']['runtime'])
        source.check()
        self.assertTrue(callable(source.module._fingerprint_cuda_dict))
        sys.modules.pop(source.module.__name__)

    def test_rejections(self):
        def key(p): p['extra'] = 1
        def schema(p): p['schema'] = 'x'
        def commit(p): p['commit'] = 'A' * 40
        def sha_(p): p['proposed']['runtime']['sha256'] = '0' * 64
        def driver(p): p['driver']['path'] = str(self.root / 'runtime.py'); p['driver']['sha256'] = facts(self.root / 'runtime.py')['sha256']
        def alias(p): p['proposed']['ledger'] = p['proposed']['runtime']
        def original(p): p['proposed']['bridge'] = self.sources['runtime']
        for mutate in (key, schema, commit, sha_, driver, alias, original):
            with self.assertRaises(ValueError, msg=mutate.__name__):
                self.read(mutate=mutate)
        with self.assertRaisesRegex(ValueError, 'closure literals'):
            self.read(ledger=b'RUNTIME_SHA256 = "%s"\n' % (b'0' * 64))
        with self.assertRaisesRegex(ValueError, 'closure literals'):
            self.read(bridge=b'("%s",)\n' % (b'0' * 64))
        with self.assertRaisesRegex(ValueError, 'AST unchanged'):
            self.read(runtime=self.ORIGINAL_RUNTIME.replace(b'repr(value)', b'str(value)'))
        with self.assertRaisesRegex(ValueError, 'AST unchanged'):
            self.read(runtime=self.ORIGINAL_RUNTIME.replace(b'def fingerprint', b'def renamed'))

    def test_native_prefix_is_proved_statically_and_at_runtime(self):
        for source in (b'import torch\n', b'import numpy.linalg\n', b'from sfora import x\n', b'from . import x\n',
                b'try:\n    import PIL\nexcept ImportError:\n    pass\n', b'class A:\n    import safetensors\n'):
            self.assertTrue(fresh.native_imports(source), source)
        self.assertEqual(fresh.native_imports(b'import hashlib\nfrom pathlib import Path\ndef f():\n    import torch\n    return lambda: __import__("numpy")\n'), [])
        with self.assertRaisesRegex(ValueError, 'stdlib-only'):
            self.read(runtime=self.ORIGINAL_RUNTIME + b'\nimport torch\n')
        fresh.require_no_native()
        with patch.dict(sys.modules, {'numpy': SimpleNamespace()}), self.assertRaisesRegex(ValueError, 'native import preceded'):
            fresh.require_no_native()

    def test_source_load_executes_admitted_raw_bytes_not_a_poisoned_pyc(self):
        path = self.root / 'poisonmod.py'
        path.write_text('VALUE = "poison"\n')
        stat = path.stat()
        cache = Path(importlib.util.cache_from_source(str(path)))
        py_compile.compile(str(path), cfile=str(cache), doraise=True)
        path.write_text('VALUE = "rawraw"\n')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(len(path.read_bytes()), stat.st_size)
        poisoned = load('poisonmod_standard_import', path)
        self.assertEqual(poisoned.VALUE, 'poison')
        sys.modules.pop('poisonmod_standard_import')
        source = fresh.requests.Source.load(facts(path))
        self.assertEqual(source.module.VALUE, 'rawraw')
        sys.modules.pop(source.module.__name__)


class Body(unittest.TestCase):
    def tearDown(self):
        Tensor.options = {}

    def test_passes_with_exact_evidence(self):
        f = fixture()
        before, threads = snapshot(f), threading.active_count()
        result = measure(f)
        self.assertEqual(snapshot(f), before)
        falsifier = result['falsifier']
        self.assertEqual(falsifier['baseline']['full448']['original'], f.index._endpoint['vision_sha256'])
        self.assertEqual(falsifier['near_limit_transpose']['shape'], [4096, 4])
        self.assertEqual(falsifier['near_limit_transpose']['stride'], [1, 4096])
        self.assertEqual(falsifier['over_limit']['error_type'], 'ValueError')
        self.assertEqual([r['order'] for r in result['timings']['pairs']], ['OP', 'PO', 'OP'])
        self.assertEqual(falsifier['unchanged_version_byte']['steps'][0]['byte'], 0)
        self.assertEqual([v['name'] for v in falsifier['aliases']['views']], ['a_t', 'a_off', 'a_str'])
        self.assertGreater(len(f.checks), 50)
        self.assertEqual(falsifier['mlp_byte']['frozen444'], falsifier['baseline']['frozen444'])
        self.assertNotEqual(falsifier['mlp_byte']['full448'], falsifier['baseline']['full448'])
        self.assertEqual(threading.active_count(), threads)

    def test_mutants_are_rejected(self):
        for flavor, message in (('stale', 'mutated original/proposed digests differ'), ('raw_storage', 'near-limit dense transpose'),
                ('leak', 'retained error kept'), ('retain_locals', 'retained error kept'), ('nojoin', 'not joined'),
                ('wrong_over', 'exact genuine budget error'), ('wrong_cpu', 'exact genuine CPU-leaf error')):
            with self.assertRaisesRegex(ValueError, message, msg=flavor):
                measure(fixture(flavor))
            time.sleep(0.5 if flavor == 'nojoin' else 0)
        for option, message in (('data_shares_version', 'MLP byte flip precondition'), ('slice_copies', 'alias geometry'),
                ('broken_copy', 'MLP byte flip was not restored')):
            f = fixture(**{option: True})
            with self.assertRaisesRegex(ValueError, message, msg=option):
                measure(f)

    def test_unauthenticated_identity_is_rejected(self):
        f = fixture()
        f.index._endpoint['vision_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'encoder identity'):
            measure(f)
        for attr, value in (('MLP', ('x',)), ('hashlib', object()), ('ThreadPoolExecutor', object())):
            f = fixture()
            setattr(f.source.module, attr, value)
            with self.assertRaisesRegex(ValueError, 'runtime identity'):
                measure(f)
        f = fixture()
        f.source.check = lambda: (_ for _ in ()).throw(ValueError('authenticated live source changed'))
        with self.assertRaisesRegex(ValueError, 'live source changed'):
            measure(f)

    def test_failure_restores_every_byte(self):
        f = fixture()
        before = snapshot(f)
        original, calls = f.source.module._fingerprint_cuda_dict, []
        def failing(tree):
            calls.append(1)
            try:
                if len(calls) == 12: raise ValueError('injected mid-mutation failure')
                return original(tree)
            finally:
                tree = None
        f.source.module._fingerprint_cuda_dict = failing
        with self.assertRaisesRegex(ValueError, 'injected'):
            measure(f)
        self.assertEqual(len(calls), 12)
        self.assertEqual(snapshot(f), before)

    def test_falsify_owns_one_index_and_closes_it(self):
        f = fixture()
        closed = []
        f.index.close = lambda: closed.append(1)
        with patch.object(fresh, 'measure', lambda *a: measure(f)):
            result = fresh.falsify(lambda: f.index, f.guard, f.torch, f.source)
        self.assertEqual(closed, [1])
        self.assertEqual(set(result['phases']), {'admission', 'falsifier', 'timings'})
        self.assertEqual(result['forward_calls'], 0)
        f = fixture()
        f.index.close = lambda: closed.append(2)
        with self.assertRaisesRegex(ValueError, 'guard stop'):
            fresh.falsify(lambda: f.index, lambda: (_ for _ in ()).throw(ValueError('guard stop')), f.torch, f.source)
        self.assertEqual(closed, [1])


class Receipt(unittest.TestCase):
    def setUp(self):
        f = fixture()
        phases = measure(f)
        for name, size in (('near_limit_transpose', fresh.BUDGET), ('over_limit', fresh.BUDGET + 4)):
            phases['falsifier'][name]['bytes'] = size
        phases['falsifier']['near_limit_transpose']['budget_bytes'] = fresh.BUDGET
        phases['admission'] = {'driver_seconds_before_body': 500.0, 'owner_admission_seconds': 23.0, 'owner_release_seconds': 0.2}
        self.authority = {'sources': {'a': {'path': '/a', 'sha256': '1' * 64}}, 'control_export': {'x': 1}}
        self.fact, self.pfact = {'path': '/authority.json', 'sha256': '2' * 64}, {'path': '/proposal.json', 'sha256': '3' * 64}
        self.proposal = {'proposed': {'runtime': {'path': '/r', 'sha256': '4' * 64}}, 'commit': 'b' * 40, 'driver': {'path': '/driver.py'}}
        self.record = {'schema': fresh.RECEIPT_SCHEMA, 'status': 'DISCARDED_DIAGNOSTIC', 'engineering_only': True,
            'authority': self.fact, 'proposal': self.pfact, 'proposed': self.proposal['proposed'], 'proposal_commit': 'b' * 40,
            'sources': self.authority['sources'], 'control': fresh.CONTROL, 'control_export': self.authority['control_export'],
            'phases': phases, 'body_seconds': 90.0, 'forward_calls': 0, 'public_b1_b32_parity_required_later': True,
            **{k: False for k in fresh.FLAGS}, 'full_uncached_exit_pass': True, 'normal_terminal_required': True,
            'owned_cleanup_requires_terminal': True, 'whole_process_seconds': 700.0, 'output': '/out',
            'invocation': {'argv': fresh.fresh_cli(self.fact, self.pfact, '/out', '/driver.py'), 'optimize': 0,
                'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8'}}

    def check(self, record):
        fresh.validate_fresh_receipt(record, self.authority, self.fact, self.proposal, self.pfact)

    def test_accepts_and_rejects_every_mutation(self):
        self.check(self.record)
        def digest(r): r['phases']['falsifier']['baseline']['full448']['proposed'] = '0' * 64
        def stale(r): r['phases']['falsifier']['unchanged_version_byte']['steps'][1]['full448'] = r['phases']['falsifier']['baseline']['full448']
        def restored(r): r['phases']['falsifier']['restored']['leaf_bytes_equal'] = False
        def order(r): r['phases']['timings']['pairs'][1]['order'] = 'OP'
        def ratio(r): r['phases']['timings']['pairs'][0]['ratio_proposed_over_original'] = 0.5
        def speed(r): r['speed_go'] = True
        def body(r): r['body_seconds'] = 301.0
        def whole(r): r['whole_process_seconds'] = 10.0
        def forward(r): r['forward_calls'] = 1
        def argv(r): r['invocation']['argv'] = r['invocation']['argv'][:-1]
        def over(r): r['phases']['falsifier']['over_limit']['error_type'] = 'RuntimeError'
        def thread(r): r['phases']['falsifier']['retained_error']['threads_after'] = 9
        def peaks(r): r['phases']['falsifier']['near_limit_transpose']['whole_unit_peaks'].pop('after')
        def extra(r): r['phases']['falsifier']['extra'] = 1
        def message(r): r['phases']['falsifier']['over_limit']['error'] = 'unrelated'
        def cpu(r): r['phases']['falsifier']['retained_error']['error'] = 'unrelated'
        def infinite(r): r['phases']['falsifier']['over_limit']['whole_unit_peaks']['after']['process_peak_rss_kib'] = float('inf')
        def mlp_frozen(r): r['phases']['falsifier']['mlp_byte']['frozen444'] = r['phases']['falsifier']['unchanged_version_byte']['steps'][0]['frozen444']
        def mlp_same(r): r['phases']['falsifier']['mlp_byte']['full448'] = r['phases']['falsifier']['baseline']['full448']
        for mutate in (digest, stale, restored, order, ratio, speed, body, whole, forward, argv, over, thread, peaks, extra,
                message, cpu, infinite, mlp_frozen, mlp_same):
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((ValueError, KeyError), msg=mutate.__name__):
                self.check(record)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in fresh.NATIVE:
            raise ImportError('native import is outside the source-only gate: ' + name)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.alarm(120)
    sys.meta_path.insert(0, NoNative())
    unittest.main()
