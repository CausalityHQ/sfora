#!/usr/bin/env python3
"""Bounded stdlib falsifier for the bootstrap v2 driver:120s/AS1GiB; Torch/NumPy/native/GPU work UNRUN."""
import array
import ast
import concurrent.futures
import copy
import gc
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
import weakref
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
ORIGINAL_QUALIFIER_SHA256 = '2961c2b0129b762403273c1d550b44c0ba7e4e9ec231a3e39848a9ee57d68815'
# Frozen v1 driver/test, as committed and recorded by connected-fresh-sha-native-v1-freeze/freeze.json.
FROZEN_V1_SHA256 = {'qualify_connected_fresh_sha.py': 'fb8209554887c8a5cfc8a269a30c8dad619ec1cf40373a7c348fb98186f794ea',
                    'test_connected_fresh_sha.py': '32017c4c1ad6283cc62c8165124463d6bb151f9f10f3adb3bdfe439c2c65bf37'}
DRIVER_FILE, TEST_FILE = 'qualify_connected_fresh_sha_bootstrap.py', 'test_connected_fresh_sha_bootstrap.py'


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


fresh = load('qualify_connected_fresh_sha_bootstrap', HERE / DRIVER_FILE)
DRIVER = (HERE / DRIVER_FILE).read_bytes()
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


OBSERVER = load('_bootstrap_test_observer', HERE / 'observe_connected_serving.py')  # genuine stdlib-only observer module
NEAREST = ast.parse((HERE / 'train_siglip2_nearest_ranking.py').read_bytes())
FOUR = {'/site/nvidia/cudnn/lib/libcudnn_%s.so.9' % n: sha(n.encode())
        for n in ('engines_precompiled', 'engines_runtime_compiled', 'graph', 'heuristic')}
KNOWN = {'/site/torch/lib/libtorch.so': sha(b'torch'), '/site/sfora/ext.so': sha(b'ext')}
FIRST = Path('/train/img-00.png')
RESOURCES = {'process_peak_rss_kib': 4000000, 'peak_cuda_allocated_bytes': 2000000000}


def nearest_node(name, parent=None):
    scope = NEAREST.body if parent is None else next(
        n for n in NEAREST.body if isinstance(n, ast.FunctionDef) and n.name == parent).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)


def genuine_audit(world):
    """The genuine original exact-four wrapper, extracted by AST from native_source_api; only its closure is a double."""
    namespace = {}
    def private_audit(value, initial=False, admission=None):
        world.events.append('audit')
        value['origins'] = {'files': {**KNOWN, **world.mapped}, 'native_files': [*KNOWN, *world.mapped]}
    for node in (nearest_node('require'), nearest_node('audit_origins', 'native_source_api')):
        if node.name == 'audit_origins':
            namespace.update(authenticate=lambda: None, legacy=world.legacy, private_audit=private_audit,
                supplement={'files': dict(FOUR)})
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'train_siglip2_nearest_ranking.py', 'exec'), namespace)
    return namespace['audit_origins']


class Image:
    def __init__(self, events, path):
        self.events, self.path, self.closed = events, path, False
    def close(self):
        self.closed = True
        self.events.append('image_close')


def boot(flavor=None, *, maps=True, forward_error=None, read_error=None, **options):
    """Stdlib double of the owned index + admitted decoder; the genuine observer/owner/audit wrapper run on top of it."""
    f = fixture(flavor, **options)
    events = f.events = []
    f.world = world = SimpleNamespace(events=events, mapped={}, refs=[], reads=[], forwards=[], closes=[],
        legacy={'selected': {'source_cpu': {'origins': {'files': dict(KNOWN)}}}, 'warm_record': {'origins': {'files': {}}}})
    def search_images(images):
        events.append('forward')
        world.forwards.append(len(images))
        if forward_error: raise forward_error
        require_open = all(not i.closed for i in images)
        if not require_open: raise AssertionError('forward saw a closed image')
        if maps: world.mapped.update(FOUR)
        ids, scores = array.array('q', range(10)), array.array('f', [1.0 - i / 16 for i in range(10)])
        result = (memoryview(ids).cast('B').cast('q', (1, 10)), memoryview(scores).cast('B').cast('f', (1, 10)))
        world.refs.extend(weakref.ref(v) for v in result)
        return result
    def read_images(paths):
        events.append('read')
        world.reads.append([str(p) for p in paths])
        if read_error: raise read_error
        images = [Image(events, p) for p in paths]
        world.refs.extend(weakref.ref(i) for i in images)
        return images
    def close():
        events.append('close')
        world.closes.append(1)
    f.index.search_images, f.index.close, f.read_images = search_images, close, read_images
    f.torch.cuda.synchronize = lambda: events.append('sync')
    f.guard = lambda: (events.append('guard'), dict(RESOURCES))[1]
    f.api = SimpleNamespace(audit_origins=genuine_audit(world))
    f.legacy = world.legacy
    proposed, original = f.source.module._fingerprint_cuda_dict, f.index._module.fingerprint
    def hashed(call):
        def wrapper(tree):
            if 'hash' not in events: events.append('hash')
            try: return call(tree)
            finally: tree = None  # a retained error must not keep the tree alive through this frame
        return wrapper
    f.source.module._fingerprint_cuda_dict, f.index._module.fingerprint = hashed(proposed), hashed(original)
    return f


def falsify(f, *, observer=None, guard=None, paths=None, legacy=None, body=None):
    """Real fresh.falsify; only the unchanged hash body is replaced by the fixture-sized measure of the v1 suite."""
    def patched(*args):
        f.events.append('measure')
        gc.collect()
        f.alive = [ref() is not None for ref in f.world.refs]
        return measure(f) if body is None else body(f)
    with patch.object(fresh, 'measure', patched):
        return fresh.falsify(lambda: f.index, guard or f.guard, f.torch, f.source, observer or OBSERVER, f.read_images,
            [FIRST] if paths is None else paths, f.api, f.legacy if legacy is None else legacy)


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

    def test_frozen_v1_files_are_unchanged_and_only_the_body_and_receipt_seams_differ(self):
        for name, digest in FROZEN_V1_SHA256.items():
            self.assertEqual(sha((HERE / name).read_bytes()), digest, name)
        v1 = load('_frozen_v1_fresh_sha', HERE / 'qualify_connected_fresh_sha.py')
        try:
            v1.check_seams(ORIGINAL, (HERE / 'qualify_connected_fresh_sha.py').read_bytes())
            with self.assertRaisesRegex(ValueError, 'seam 5 must match exactly once'):
                fresh.check_seams(ORIGINAL, (HERE / 'qualify_connected_fresh_sha.py').read_bytes())
            self.assertEqual([(a[0], a[1]) for a in v1.SEAMS], [(a[0], a[1]) for a in fresh.SEAMS])
            self.assertEqual([a[0] for a, b in zip(v1.SEAMS, fresh.SEAMS) if a[2] != b[2]], ['S6', 'S8'])
            self.assertEqual(v1.RECEIPT_SCHEMA, 'connected-fresh-sha-falsifier-diagnostic-v1')
            self.assertEqual(fresh.RECEIPT_SCHEMA, 'connected-fresh-sha-bootstrap-falsifier-diagnostic-v2')
            self.assertNotEqual(v1.PROPOSAL_SCHEMA, fresh.PROPOSAL_SCHEMA)
            self.assertEqual((fresh.DRIVER_NAME, fresh.TEST_NAME), (DRIVER_FILE, TEST_FILE))
        finally:
            sys.modules.pop('_frozen_v1_fresh_sha', None)

    def test_unchanged_original_functions_and_old_tests_keep_their_asts(self):
        def defs(raw, kinds=(ast.FunctionDef, ast.ClassDef)):
            return {n.name: ast.dump(n) for n in ast.parse(raw).body if isinstance(n, kinds)}
        v1, v2 = defs((HERE / 'qualify_connected_fresh_sha.py').read_bytes()), defs(DRIVER)
        unchanged = ['require', 'require_no_native', 'native_imports', 'fresh_cli', 'proposal_files', 'seam_dump', 'check_seams',
                     'tree_bytes', 'same_bytes', 'capture_error', 'mutants', 'measure', 'main']
        for name in unchanged:
            self.assertEqual(v1[name], v2[name], name)
        self.assertEqual(sorted(n for n in v2 if n not in v1), ['bootstrap', 'check_bootstrap', 'seconds'])
        self.assertEqual(sorted(n for n in v1 if v1[n] != v2.get(n)), ['falsify', 'read_proposal', 'run', 'validate_fresh_receipt'])
        constants = lambda raw: {t.id: ast.dump(n.value) for n in ast.parse(raw).body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}
        c1, c2 = constants((HERE / 'qualify_connected_fresh_sha.py').read_bytes()), constants(DRIVER)
        for name in ('PROPOSAL_KEYS', 'BUDGET', 'NATIVE', 'FLAGS', 'OVER_MESSAGE', 'CPU_MESSAGE', 'PEAK_SEMANTICS'):
            self.assertEqual(c1[name], c2[name], name)
        old, new = ast.parse((HERE / 'test_connected_fresh_sha.py').read_bytes()), ast.parse(Path(__file__).read_bytes())
        top = lambda tree: {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        o, n = top(old), top(new)
        for name in ('Dtype', 'strides_of', 'Tensor', 'make', 'zeros', 'fake_torch', 'frame', 'tree_digest', 'original_fingerprint',
                     'proposed_pipeline', 'fixture', 'snapshot'):
            self.assertEqual(ast.dump(o[name]), ast.dump(n[name]), name)
        kept = {'Body': ['test_passes_with_exact_evidence', 'test_mutants_are_rejected', 'test_unauthenticated_identity_is_rejected',
                         'test_failure_restores_every_byte'],
                'Proposal': ['test_accepts_exact_closure', 'test_native_prefix_is_proved_statically_and_at_runtime',
                             'test_source_load_executes_admitted_raw_bytes_not_a_poisoned_pyc'],
                'SeamInverse': ['test_every_seam_and_unseamed_statement_is_pinned']}
        for cls, names in kept.items():
            methods = lambda node: {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            old_methods, new_methods = methods(o[cls]), methods(n[cls])
            for name in names:
                self.assertEqual(ast.dump(old_methods[name]), ast.dump(new_methods[name]), cls + '.' + name)

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
        start = text.index('diagnostic = falsify(', run_at)
        statement = text[start:text.index("['legacy'])\n", start) + len("['legacy'])\n")]
        with self.assertRaisesRegex(ValueError, 'seam 5 must match exactly once'):
            fresh.check_seams(ORIGINAL, edit(statement, statement + '            ' + statement))
        for old, new in (("observer,read_images,", "observer,lambda p: p,"), (",api,context[", ",None,context["),
                ("observation['train_images'][0]", "observation['train_images'][1]"),
                ("[Path(observation['train_images'][0]['path'])]", "[Path(f['path']) for f in observation['train_images']]"),
                ("api,context['training_context']['legacy'])", "api,None)")):
            with self.assertRaisesRegex(ValueError, 'seam 5 must match exactly once', msg=old):
                fresh.check_seams(ORIGINAL, edit(old, new))
        with self.assertRaisesRegex(ValueError, 'outside the declared seams'):
            fresh.check_seams(ORIGINAL.replace(b"'insufficient whole-process body/exit headroom'", b"'x'"), DRIVER)
        with self.assertRaisesRegex(ValueError, 'seam 2 must match exactly once'):
            fresh.check_seams(ORIGINAL.replace(b'locks = requests.Locks(authority', b'locks = requests.Locks2(authority'), DRIVER)

    def test_driver_and_original_survive_unchanged_source_authentication(self):
        fresh.requests.Source(fresh, facts(HERE / DRIVER_FILE)).check()
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
        proposal = {'schema': fresh.PROPOSAL_SCHEMA, 'driver': facts(HERE / DRIVER_FILE),
            'test': facts(HERE / TEST_FILE), 'commit': 'a' * 40,
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
        def v1_schema(p): p['schema'] = 'connected-fresh-sha-proposal-v1'
        def v1_driver(p): p['driver'] = facts(HERE / 'qualify_connected_fresh_sha.py')
        def v1_test(p): p['test'] = facts(HERE / 'test_connected_fresh_sha.py')
        def stale_test(p): p['test']['sha256'] = '0' * 64
        for mutate in (key, schema, commit, sha_, driver, alias, original, v1_schema, v1_driver, v1_test, stale_test):
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

    def test_driver_must_carry_the_new_basename_beside_its_new_test(self):
        base = fresh.read_proposal(self.write(), self.sources)['proposed']
        for driver_name, accepted in (('qualify_connected_fresh_sha_renamed.py', False), (DRIVER_FILE, True)):
            where = self.root / ('copy_' + str(accepted))
            where.mkdir()
            (where / driver_name).write_bytes(DRIVER)
            (where / TEST_FILE).write_bytes((HERE / TEST_FILE).read_bytes())
            module = load('_copied_bootstrap_driver', where / driver_name)
            try:
                proposal = {'schema': fresh.PROPOSAL_SCHEMA, 'driver': facts(where / driver_name), 'test': facts(where / TEST_FILE),
                    'commit': 'a' * 40, 'proposed': base}
                (where / 'proposal.json').write_text(json.dumps(proposal))
                if accepted:
                    module.read_proposal(facts(where / 'proposal.json'), self.sources)
                else:
                    with self.assertRaisesRegex(ValueError, 'current driver/test FILE required'):
                        module.read_proposal(facts(where / 'proposal.json'), self.sources)
            finally:
                sys.modules.pop('_copied_bootstrap_driver', None)

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
        f = boot()
        result = falsify(f)
        self.assertEqual(f.world.closes, [1])
        self.assertEqual(set(result['phases']), {'admission', 'bootstrap', 'falsifier', 'timings', 'hash_started_body_seconds'})
        self.assertEqual(result['forward_calls'], 1)
        fresh.check_bootstrap(result['phases']['bootstrap'])
        f = boot()
        with self.assertRaisesRegex(ValueError, 'guard stop'):
            falsify(f, guard=lambda: (_ for _ in ()).throw(ValueError('guard stop')))
        self.assertEqual((f.world.closes, f.events), ([], []))


class Bootstrap(unittest.TestCase):
    """Genuine observer.measure_request, requests.owner and the original audit_origins wrapper over a stdlib index."""
    PREFIX = ['guard', 'guard', 'guard', 'sync', 'read', 'forward', 'sync', 'image_close', 'audit', 'guard', 'measure']

    def test_one_genuine_b1_request_and_exact_four_audit_precede_all_hash_work(self):
        f = boot()
        result = falsify(f)
        self.assertEqual(f.events[:len(self.PREFIX) + 1], self.PREFIX + ['hash'])
        self.assertEqual((f.events.count('forward'), f.events.count('read'), f.events.count('audit'), f.events.count('image_close')),
            (1, 1, 1, 1))
        self.assertEqual((f.world.reads, f.world.forwards, f.world.closes), ([[str(FIRST)]], [1], [1]))
        boot_row = result['phases']['bootstrap']
        self.assertEqual((boot_row['batch'], boot_row['kind'], boot_row['image'], boot_row['instrumented'],
            boot_row['qualification_eligible'], boot_row['audit_exact_four']), (1, 'bootstrap', str(FIRST), False, False, True))
        self.assertEqual(boot_row['native'][0]['shape'], [1, 10])
        self.assertTrue(0 < boot_row['ended_body_seconds'] <= result['phases']['hash_started_body_seconds'] <= result['body_seconds'])
        self.assertEqual(result['forward_calls'], 1)

    def test_bootstrap_time_is_outside_every_bare_hash_clock(self):
        f = boot()
        started = []
        def body(g):
            real = time.perf_counter
            with patch.object(fresh.time, 'perf_counter', lambda: real() + 1000.0 * (not started)):
                started.append(1)
                return measure(g)
        result = falsify(f, body=body)
        for row in result['phases']['timings']['pairs']:
            self.assertLess(row['original_seconds'] + row['proposed_seconds'], 5.0)
        self.assertGreater(result['phases']['bootstrap']['seconds'], 0)

    def test_zero_forward_reproduces_the_v1_exit_failure_and_stops_hash_work(self):
        f = boot(maps=False)
        with self.assertRaisesRegex(ValueError, 'observed native difference must be exact four'):
            falsify(f)
        self.assertEqual(f.events.count('forward'), 1)
        self.assertEqual(f.events.count('audit'), 1)
        self.assertNotIn('measure', f.events)
        self.assertNotIn('hash', f.events)
        self.assertGreaterEqual(len(f.world.closes), 1)

    def test_unexpected_extra_native_origin_stops_hash_work(self):
        f = boot()
        f.index.search_images, search = None, f.index.search_images
        def leaking(images):
            f.world.mapped['/site/extra.so'] = sha(b'extra')
            return search(images)
        f.index.search_images = leaking
        with self.assertRaisesRegex(ValueError, 'exact four'):
            falsify(f)
        self.assertNotIn('hash', f.events)

    def test_audit_is_called_with_the_owned_legacy_context_only(self):
        f = boot()
        with self.assertRaisesRegex(ValueError, 'owned legacy context required'):
            falsify(f, legacy={})
        self.assertNotIn('hash', f.events)
        self.assertNotIn('audit', f.events)

    def test_forward_failure_closes_images_and_index_and_stops(self):
        f = boot(forward_error=RuntimeError('forward failed'))
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            falsify(f)
        self.assertEqual(f.events.count('image_close'), 1)
        self.assertGreaterEqual(len(f.world.closes), 1)
        self.assertTrue(all(name not in f.events for name in ('audit', 'measure', 'hash')))

    def test_decoder_failure_stops_before_any_forward(self):
        f = boot(read_error=ValueError('decode failed'))
        with self.assertRaisesRegex(ValueError, 'decode failed'):
            falsify(f)
        self.assertTrue(all(name not in f.events for name in ('forward', 'audit', 'measure', 'hash')))
        self.assertGreaterEqual(len(f.world.closes), 1)

    def test_exactly_one_admitted_image_is_required(self):
        for paths in ([], [FIRST, FIRST.with_name('img-01.png')], (FIRST,)):
            f = boot()
            with self.assertRaisesRegex(ValueError, 'exactly one admitted TRAIN image'):
                falsify(f, paths=paths)
            self.assertTrue(all(name not in f.events for name in ('read', 'forward', 'audit', 'hash')))

    def test_image_result_and_observer_aliases_are_cleared_before_hashing(self):
        f = boot()
        falsify(f)
        self.assertEqual(len(f.alive), 3)  # image plus the native ID and score views
        self.assertEqual(f.alive, [False] * 3)

    def test_guard_failure_after_the_forward_stops_hash_work(self):
        f = boot()
        calls = []
        def guard():
            calls.append(1)
            if 'audit' in f.events: raise ValueError('guard stop after forward')
            return f.guard()
        with self.assertRaisesRegex(ValueError, 'guard stop after forward'):
            falsify(f, guard=guard)
        self.assertNotIn('measure', f.events)
        self.assertNotIn('hash', f.events)

    def test_reassociated_observer_source_is_denied(self):
        source = fresh.requests.Source.load(facts(HERE / 'observe_connected_serving.py'))
        module = source.module
        try:
            guard = lambda: (source.check(), RESOURCES)[1]
            original = module.measure_request
            module.measure_request = lambda *a, **k: original(*a, **k)
            f = boot()
            with self.assertRaisesRegex(ValueError, 'authenticated live source changed'):
                falsify(f, observer=module, guard=lambda: (source.check(), f.guard())[1])
            self.assertNotIn('forward', f.events)
            module.measure_request = original
            f = boot()
            search, real = f.index.search_images, module.native_snapshot
            def hijack(images):
                module.native_snapshot = lambda result: real(result)
                return search(images)
            f.index.search_images = hijack
            with self.assertRaisesRegex(ValueError, 'authenticated live source changed'):
                falsify(f, observer=module, guard=lambda: (source.check(), f.guard())[1])
            self.assertEqual(f.events.count('forward'), 1)
            self.assertNotIn('measure', f.events)
            self.assertNotIn('hash', f.events)
            module.native_snapshot = real
            f = boot()
            falsify(f, observer=module, guard=lambda: (source.check(), f.guard())[1])
            self.assertIn('hash', f.events)
        finally:
            sys.modules.pop(module.__name__, None)

    def test_extracted_seams_are_the_genuine_signatures(self):
        measure_request = next(n for n in ast.parse((HERE / 'observe_connected_serving.py').read_bytes()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'measure_request')
        self.assertEqual([a.arg for a in measure_request.args.args], ['index', 'read_images', 'synchronize', 'paths'])
        self.assertEqual([a.arg for a in measure_request.args.kwonlyargs], ['observer'])
        audit = nearest_node('audit_origins', 'native_source_api')
        self.assertEqual([a.arg for a in audit.args.args], ['value', 'initial', 'admission'])
        self.assertEqual([a.arg for a in audit.args.kwonlyargs], ['require_exact'])
        self.assertIs(fresh.requests.owner.__wrapped__.__globals__, vars(fresh.requests))
        self.assertEqual(fresh.requests.__file__, str(HERE / 'qualify_connected_serving_requests.py'))
        self.assertEqual(OBSERVER.__file__, str(HERE / 'observe_connected_serving.py'))
        text = DRIVER.decode()
        self.assertEqual(text.count('observer.measure_request('), 1)
        self.assertEqual(text.count('api.audit_origins(legacy,require_exact=True)'), 1)
        self.assertIn('observer.measure_request(index,read_images,synchronize,paths)', text)
        self.assertIn('api.audit_origins(legacy,require_exact=True)', text)


class Receipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        for i in range(3): (root / ('img-%02d.png' % i)).write_bytes(b'image-%d' % i)
        self.observation = {'train_images': [facts(root / ('img-%02d.png' % i)) for i in range(3)]}
        self.first = self.observation['train_images'][0]
        phases = falsify(boot(), paths=[Path(self.first['path'])])['phases']
        for name, size in (('near_limit_transpose', fresh.BUDGET), ('over_limit', fresh.BUDGET + 4)):
            phases['falsifier'][name]['bytes'] = size
        phases['falsifier']['near_limit_transpose']['budget_bytes'] = fresh.BUDGET
        phases['admission'] = {'driver_seconds_before_body': 500.0, 'owner_admission_seconds': 23.0, 'owner_release_seconds': 0.2}
        self.authority = {'sources': {'a': {'path': '/a', 'sha256': '1' * 64}}, 'control_export': {'x': 1},
            'observation': {'path': '/observation.json', 'sha256': '7' * 64}}
        self.fact, self.pfact = {'path': '/authority.json', 'sha256': '2' * 64}, {'path': '/proposal.json', 'sha256': '3' * 64}
        self.proposal = {'proposed': {'runtime': {'path': '/r', 'sha256': '4' * 64}}, 'commit': 'b' * 40,
            'driver': {'path': '/run/' + DRIVER_FILE, 'sha256': '5' * 64}, 'test': {'path': '/run/' + TEST_FILE, 'sha256': '6' * 64}}
        self.record = {'schema': fresh.RECEIPT_SCHEMA, 'status': 'DISCARDED_DIAGNOSTIC', 'engineering_only': True,
            'authority': self.fact, 'proposal': self.pfact, 'proposed': self.proposal['proposed'], 'proposal_commit': 'b' * 40,
            'sources': self.authority['sources'], 'control': fresh.CONTROL, 'control_export': self.authority['control_export'],
            'observation': self.authority['observation'], 'phases': phases, 'body_seconds': 90.0, 'forward_calls': 1, 'public_b1_b32_parity_required_later': True,
            'input_guards': {f['path']: f['sha256'] for f in [*fresh.proposal_files(self.proposal), *self.observation['train_images']]},
            **{k: False for k in fresh.FLAGS}, 'full_uncached_exit_pass': True, 'normal_terminal_required': True,
            'owned_cleanup_requires_terminal': True, 'whole_process_seconds': 700.0, 'output': '/out',
            'invocation': {'argv': fresh.fresh_cli(self.fact, self.pfact, '/out', self.proposal['driver']['path']), 'optimize': 0,
                'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8'}}

    def check(self, record, observation=None):
        fresh.validate_fresh_receipt(record, self.authority, self.fact, self.proposal, self.pfact,
            self.observation if observation is None else observation)

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
        def forward(r): r['forward_calls'] = 0
        def forwards(r): r['forward_calls'] = 2
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
        for mutate in (digest, stale, restored, order, ratio, speed, body, whole, forward, forwards, argv, over, thread, peaks, extra,
                message, cpu, infinite, mlp_frozen, mlp_same):
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((ValueError, KeyError), msg=mutate.__name__):
                self.check(record)

    def test_bootstrap_evidence_is_exact_ordered_and_charged(self):
        phases = lambda r: r['phases']
        boot_ = lambda r: r['phases']['bootstrap']
        def no_bootstrap(r): phases(r).pop('bootstrap')
        def two_bootstraps(r): phases(r)['bootstrap2'] = copy.deepcopy(boot_(r))
        def no_hash_start(r): phases(r).pop('hash_started_body_seconds')
        def batch(r): boot_(r)['batch'] = 32
        def kind(r): boot_(r)['kind'] = 'warm_oracle'
        def instrumented(r): boot_(r)['instrumented'] = True
        def qualification(r): boot_(r)['qualification_eligible'] = True
        def audit_flag(r): boot_(r)['audit_exact_four'] = False
        def audit_missing(r): boot_(r).pop('audit_seconds')
        def extra(r): boot_(r)['extra'] = 1
        def image(r): boot_(r)['image'] = ''
        def shape(r): boot_(r)['native'][0]['shape'] = [32, 10]
        def truncated(r): boot_(r)['native'][1]['hex'] = boot_(r)['native'][1]['hex'][:-2]
        def fmt(r): boot_(r)['native'][1]['format'] = 'd'
        def odd_hex(r): boot_(r)['native'][0]['hex'] = 'zz' * 80
        def negative_id(r): boot_(r)['native'][0]['hex'] = (-1).to_bytes(8, 'little', signed=True).hex() + '00' * 72
        def nan_score(r): boot_(r)['native'][1]['hex'] = struct.pack('<f', float('nan')).hex() + '00' * 36
        def one_row(r): boot_(r)['native'] = boot_(r)['native'][:1]
        def timing_sum(r): boot_(r)['public_call_seconds'] += 1.0
        def nonfinite(r): boot_(r)['audit_seconds'] = float('inf')
        def negative(r): boot_(r)['image_cleanup_seconds'] = -1.0
        def after_hash(r): boot_(r)['ended_body_seconds'] = phases(r)['hash_started_body_seconds'] + 1.0
        def hash_after_body(r): phases(r)['hash_started_body_seconds'] = r['body_seconds'] + 1.0
        def uncharged_audit(r): boot_(r)['audit_seconds'] = 80.0
        def v1_schema(r): r['schema'] = 'connected-fresh-sha-falsifier-diagnostic-v1'
        def guard_missing(r): r['input_guards'].pop(self.proposal['driver']['path'])
        def guard_stale(r): r['input_guards'][self.proposal['test']['path']] = '0' * 64
        def proposed_unbound(r): r['input_guards'].pop(self.proposal['proposed']['runtime']['path'])
        def other_image(r): boot_(r)['image'] = self.observation['train_images'][1]['path']
        def unknown_image(r): boot_(r)['image'] = '/train/img-00.png'
        def image_guard_missing(r): r['input_guards'].pop(self.first['path'])
        def image_guard_stale(r): r['input_guards'][self.first['path']] = '0' * 64
        def observation_role(r): r['observation'] = {'path': '/other.json', 'sha256': '8' * 64}
        def audit_flag_string(r): boot_(r)['audit_exact_four'] = 1
        def image_type(r): boot_(r)['image'] = Path(self.first['path'])
        for mutate, message in ((no_bootstrap, 'exact measurement'), (two_bootstraps, 'exact measurement'),
                (no_hash_start, 'exact measurement'), (batch, 'bootstrap request'), (kind, 'bootstrap request'),
                (instrumented, 'bootstrap request'), (qualification, 'bootstrap request'), (audit_flag, 'bootstrap request'),
                (audit_missing, 'bootstrap evidence'), (extra, 'bootstrap evidence'), (image, 'bootstrap request'),
                (shape, 'native ID/score bytes'), (truncated, 'native ID/score bytes'), (fmt, 'native ID/score bytes'),
                (odd_hex, 'native ID/score bytes'), (negative_id, 'nonnegative ID'), (nan_score, 'finite bootstrap'),
                (one_row, 'native ID/score witness'), (timing_sum, 'bootstrap request'), (nonfinite, 'finite nonnegative'),
                (negative, 'finite nonnegative'), (after_hash, 'precede all hash work'), (hash_after_body, 'precede all hash work'),
                (uncharged_audit, 'body300'), (v1_schema, 'complete discarded'), (guard_missing, 'exact new driver'),
                (guard_stale, 'exact new driver'), (proposed_unbound, 'exact new driver'),
                (other_image, 'first SHA-authenticated'), (unknown_image, 'first SHA-authenticated'),
                (image_guard_missing, 'first SHA-authenticated'), (image_guard_stale, 'first SHA-authenticated'),
                (observation_role, 'first SHA-authenticated'), (audit_flag_string, 'bootstrap request'),
                (image_type, 'bootstrap request')):
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaisesRegex(ValueError, message, msg=mutate.__name__):
                self.check(record)

    def test_bootstrap_image_is_authenticated_against_the_actual_first_train_file(self):
        self.check(self.record)
        Path(self.first['path']).write_bytes(b'tampered after admission')
        with self.assertRaisesRegex(ValueError, 'current FILE size/SHA256 differs'):
            self.check(self.record)
        Path(self.first['path']).write_bytes(b'image-0')
        self.check(self.record)
        swapped = {'train_images': [self.observation['train_images'][1], self.first, self.observation['train_images'][2]]}
        with self.assertRaisesRegex(ValueError, 'first SHA-authenticated'):
            self.check(self.record, swapped)
        unreadable = {'train_images': [{'path': self.first['path'], 'sha256': '0' * 64}]}
        with self.assertRaisesRegex(ValueError, 'current FILE size/SHA256 differs'):
            self.check(self.record, unreadable)

    def test_receipt_must_name_the_new_driver_and_test_files(self):
        for driver, test in (('qualify_connected_fresh_sha.py', TEST_FILE), (DRIVER_FILE, 'test_connected_fresh_sha.py')):
            proposal = copy.deepcopy(self.proposal)
            proposal['driver'] = {'path': '/run/' + driver, 'sha256': '5' * 64}
            proposal['test'] = {'path': '/run/' + test, 'sha256': '6' * 64}
            record = copy.deepcopy(self.record)
            record['input_guards'] = {f['path']: f['sha256'] for f in fresh.proposal_files(proposal)}
            record['invocation']['argv'] = fresh.fresh_cli(self.fact, self.pfact, '/out', proposal['driver']['path'])
            with self.assertRaisesRegex(ValueError, 'exact new driver'):
                fresh.validate_fresh_receipt(record, self.authority, self.fact, proposal, self.pfact, self.observation)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in fresh.NATIVE:
            raise ImportError('native import is outside the source-only gate: ' + name)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.alarm(120)
    sys.meta_path.insert(0, NoNative())
    unittest.main()
