#!/usr/bin/env python3
"""Stdlib falsifiers for measure_connected_mlp_displacement.py on synthetic genuine-grammar endpoints.

No Torch/NumPy and no real archive: protocol-2 pickles are written by the real stdlib Pickler with
persistent storage IDs and torch._utils/torch.*Storage globals (stand-in modules), inside real ZIPs.
"""
import ast
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from fractions import Fraction
from pathlib import Path
import pickle
import os
import struct
import subprocess
import sys
import tempfile
import types
from unittest import mock
import warnings
import zipfile

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'measure_connected_mlp_displacement.py'
HELPER = HERE / 'pass201_pa_source_v2_contract.py'
ORIGINAL = HERE / 'train_siglip2_substrate_adaptation.py'
ORIGINAL_SHA = 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'  # manifest code pin
spec = importlib.util.spec_from_file_location('measure_connected_mlp_displacement', DRIVER)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
NAMES = m.MLP
TINY = dict(zip(NAMES, ((3, 2), (3,), (2, 3), (2,))))
OTHER = {'control': 'candidate', 'candidate': 'control'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32(values):
    return struct.pack('<%df' % len(values), *values)


def vals(raw):
    return struct.unpack('<%df' % (len(raw) // 4), raw)


def oracle_fp(shape, raw_hex):
    """Literal framing of the pinned substrate-adaptation fingerprint of one float32 CPU tensor."""
    def f(s):
        return '%d:%s' % (len(s), s)
    text = (f('Tensor') + f('tuple') + f('3') + f('str') + f(repr('torch.float32')) + f('tuple') + f(str(len(shape))) +
            ''.join(f('int') + f(repr(d)) for d in shape) + f('str') + f(repr(raw_hex)))
    return sha(text.encode())


def tiny():
    return mock.patch.dict(m.SHAPES, TINY)


def default_raws():
    control = {NAMES[0]: f32([1, -2, 3, .5, -.25, 8]), NAMES[1]: f32([.125, 0, -1]),
               NAMES[2]: f32([2, 4, -6, 1, 1, 1e-3]), NAMES[3]: f32([-.5, 7])}
    candidate = {n: f32([v + (i + 1) * .0625 for i, v in enumerate(vals(raw))]) for n, raw in control.items()}
    return {'control': control, 'candidate': candidate}


# ---- genuine pickle grammar ------------------------------------------------------------------
class Blob:
    pass


class Storage:
    def __init__(self, cls, key, location, numel):
        self.cls, self.key, self.location, self.numel = cls, key, location, numel


class Tensor:
    def __init__(self, rebuild, storage, offset, size, stride):
        self.args = (rebuild, (storage, offset, size, stride, False, collections.OrderedDict()))

    def __reduce_ex__(self, protocol):
        return self.args


class TorchPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ('storage', obj.cls, obj.key, obj.location, obj.numel)
        return None


@contextlib.contextmanager
def fake_torch():
    torch, utils = types.ModuleType('torch'), types.ModuleType('torch._utils')
    for name in ('FloatStorage', 'HalfStorage', 'DoubleStorage'):
        setattr(torch, name, type(name, (), {'__module__': 'torch'}))

    def _rebuild_tensor_v2(*args):
        raise AssertionError('never called')
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = 'torch._utils', '_rebuild_tensor_v2'
    utils._rebuild_tensor_v2, torch._utils = _rebuild_tensor_v2, utils

    class Dtype:
        __module__ = 'torch'

        def __reduce__(self):
            return 'float32'
    torch.float32 = Dtype()
    with mock.patch.dict(sys.modules, {'torch': torch, 'torch._utils': utils}):
        yield torch, utils


def make_endpoint(path, arm, raws, *, base='b' * 64, vision='c' * 64, initial=None, tensor_edit=None, root_edit=None,
                  zip_edit=None, prefix='archive'):
    initial = initial or {n: oracle_fp(m.SHAPES[n], sha(raws[n])) for n in NAMES}
    with fake_torch() as (torch, utils):
        def tensor(spec):
            return Tensor(utils._rebuild_tensor_v2, Storage(getattr(torch, spec['cls']), spec['key'], spec['location'], spec['numel']),
                          spec['offset'], spec['shape'], spec['stride'])
        storage, encoder = {}, {}
        for i, name in enumerate(NAMES):
            shape = m.SHAPES[name]
            item = {'cls': 'FloatStorage', 'key': str(i), 'location': 'cpu', 'numel': math.prod(shape), 'shape': shape,
                    'stride': m.contiguous(shape), 'offset': 0, 'data': raws[name]}
            if tensor_edit:
                tensor_edit(name, item)
            encoder[name] = tensor(item)
            storage.setdefault('%s/data/%s' % (prefix, item['key']), item['data'])
        other = {'cls': 'FloatStorage', 'location': 'cpu', 'numel': 2, 'shape': (2,), 'stride': (1,), 'offset': 0}
        extra = {}
        for key in ('A', 'C'):
            extra[key] = tensor({**other, 'key': 'x' + key})
            storage['%s/data/x%s' % (prefix, key)] = b'NOT-FLOATS-NEVER-READ'
        root = {'schema': m.INFERENCE_SCHEMA, 'source': {'k': 'v'}, 'arm': arm, 'config': {'dtype': None, 'x': [1, 2.5, True, (1, 2)]},
                'buffers': {}, 'processor': {'p': 'q'}, 'head': {}, 'A': extra['A'], 'C': extra['C'], 'means': {},
                'mu_train': {}, 'mu_train_provenance': {}, 'scope': {'arm': 'control', 'payload': {'class_names': ['a']}},
                'common_statistics': {}, 'numerical_flags': {'tf32': False}, 'base_vision': {'sha256': base},
                'encoder': encoder, 'encoder_identity': {'initial_four_sha256': dict(initial),
                                                         'inventory': {n: list(m.SHAPES[n]) for n in NAMES}},
                'vision_sha256': vision, 'fixed_sha256': 'f' * 64}
        if root_edit:
            root_edit(root, torch, utils)
        buffer = io.BytesIO()
        TorchPickler(buffer, protocol=2).dump(root)
    entries = ([[prefix + '/data.pkl', buffer.getvalue(), zipfile.ZIP_STORED], [prefix + '/byteorder', b'little', zipfile.ZIP_STORED]] +
               [[n, d, zipfile.ZIP_STORED] for n, d in storage.items()] +
               [[prefix + '/version', b'3\n', zipfile.ZIP_STORED], [prefix + '/.data/serialization_id', b'7' * 40, zipfile.ZIP_STORED]])
    if zip_edit:
        zip_edit(entries)
    with warnings.catch_warnings(), zipfile.ZipFile(path, 'w') as archive:
        warnings.simplefilter('ignore')
        for name, data, kind in entries:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = kind
            archive.writestr(info, data)
    return initial


class Case:
    """One tmp tree: endpoints, bundles, authority; every pin is computed from the files as written."""

    def __init__(self, root, raws=None, own=None):
        self.root = Path(root).resolve()
        self.raws = raws or default_raws()
        self.p = {k: self.root / v for k, v in {
            'control_endpoint': 'control.pt', 'candidate_endpoint': 'candidate.pt', 'control_bundle': 'control-bundle.json',
            'candidate_bundle': 'candidate-bundle.json', 'helper': 'helper.py', 'own': 'own.py', 'interp': 'interp.bin',
            'authority': 'authority.json'}.items()}
        self.out = self.root / 'result.json'
        self.own, self.interp = own or self.p['own'], self.p['interp']

    def write(self, hooks=None, bundle=None, authority=None, post=None, initial=None):
        hooks, bundle, post = hooks or {}, bundle or {}, post or {}
        self.p['helper'].write_bytes(HELPER.read_bytes())  # fresh copies each time; tests then mutate the copies
        self.p['own'].write_bytes(DRIVER.read_bytes())
        self.p['interp'].write_bytes(b'not-executed: the reader only admits and hashes the interpreter file\n')
        first = {n: oracle_fp(m.SHAPES[n], sha(self.raws['control'][n])) for n in NAMES}
        vision = {'control': 'b' * 64, 'candidate': 'c' * 64}
        pins = {}
        for arm in m.ARMS:
            make_endpoint(self.p[arm + '_endpoint'], arm, self.raws[arm], initial=initial or first, vision=vision[arm], **hooks.get(arm, {}))
            if arm in post:
                post[arm](self.p[arm + '_endpoint'])
            endpoint_sha = sha(self.p[arm + '_endpoint'].read_bytes())
            obj = {'schema': m.BUNDLE_SCHEMA, 'code': {}, 'endpoint_state_sha256': '1' * 64, 'environment': {}, 'scope': {},
                   'files': {'vision.pt': 'd' * 64, 'endpoint.pt': endpoint_sha, 'processor.json': 'e' * 64},
                   'encoder_identity': {'initial_four_sha256': dict(initial or first), 'inventory': {n: list(m.SHAPES[n]) for n in NAMES}},
                   'base_vision_sha256': 'b' * 64, 'vision_sha256': vision[arm]}
            if arm in bundle:
                bundle[arm](obj)
            self.p[arm + '_bundle'].write_text(json.dumps(obj))
            pins[arm] = {'bundle': {'path': str(self.p[arm + '_bundle']), 'sha256': sha(self.p[arm + '_bundle'].read_bytes())},
                         'endpoint': {'path': str(self.p[arm + '_endpoint']), 'sha256': endpoint_sha}}
        obj = {'schema': m.AUTHORITY_SCHEMA, 'own_source_sha256': sha(self.own.read_bytes()),
               'helper_source': {'path': str(self.p['helper']), 'sha256': sha(self.p['helper'].read_bytes())}, 'arms': pins}
        if authority:
            authority(obj)
        self.p['authority'].write_text(json.dumps(obj))
        return self

    def argv(self, sha_override=None):
        return ['--authority', str(self.p['authority']), '--authority-sha256', sha_override or sha(self.p['authority'].read_bytes()),
                '--output', str(self.out)]

    def run(self, argv=None):
        error, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(error), contextlib.redirect_stdout(out):
            code = m.main(argv or self.argv(), own=self.own, interpreter=self.interp)
        self.stdout = out.getvalue()
        assert m.HELPER_MODULE not in sys.modules or getattr(self, 'foreign_ok', False), 'helper registry entry leaked'
        return code, error.getvalue()

    def terminal_ok(self, code):
        """A normal terminal is exit 0 plus one stdout line whose sha256/bytes match the published file."""
        try:
            line = json.loads(self.stdout)
            data = self.out.read_bytes()
            return code == 0 and line == {'output': str(self.out), 'sha256': sha(data), 'bytes': len(data)}
        except (ValueError, OSError):
            return False

    def ok(self):
        code, error = self.run()
        assert code == 0 and self.terminal_ok(code), error
        return json.loads(self.out.read_text())

    def rejects(self, text, argv=None):
        code, error = self.run(argv)
        assert code == 1 and text in error, (text, code, error)
        assert self.stdout == '' and not self.terminal_ok(code)
        leftovers = [q.name for q in self.root.iterdir() if q.name.endswith('.tmp') or q == self.out]
        assert not leftovers, 'result/temp published on error: %s' % leftovers


def case(**kw):
    directory = tempfile.TemporaryDirectory()
    return directory, Case(directory.name, **kw)


# ---- fingerprint -----------------------------------------------------------------------------
def original_fingerprint():
    source = ORIGINAL.read_bytes()
    assert sha(source) == ORIGINAL_SHA, 'original fingerprint source is not the manifest-pinned file'
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')

    class Tensor:
        dtype, _version = type('D', (), {'__str__': lambda self: 'torch.float32'})(), 0

        def __init__(self, raw, shape):
            self.raw, self.shape = raw, shape

        def data_ptr(self): return id(self)
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def reshape(self, *a): return self
        def view(self, kind): return self
        def numpy(self): return memoryview(self.raw)
    torch = types.SimpleNamespace(Tensor=Tensor, uint8='uint8')
    namespace = {'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ORIGINAL), 'exec'), namespace)

    def call(raw, shape):
        with mock.patch.dict(sys.modules, {'torch': torch}):  # the original does `import torch` when called
            return namespace['fingerprint'](Tensor(raw, shape))
    return call


def test_fingerprint_matches_pinned_original_and_literal():
    original = original_fingerprint()
    one_two = f32([1.0, 2.0])
    for shape, raw in (((2,), one_two), ((4304,), b'\x00' * 8), ((4304, 1152), b'\x01\x02'), ((1152, 4304), b''), ((3, 2), one_two * 3)):
        want = original(raw, shape)
        assert m.typed_fingerprint(shape, sha(raw)) == want == oracle_fp(shape, sha(raw)), shape
    assert original(one_two, (2,)) == m.typed_fingerprint((2,), sha(one_two))
    assert m.typed_fingerprint((2,), sha(one_two)) != m.typed_fingerprint((1, 2), sha(one_two))
    assert m.typed_fingerprint((2,), sha(one_two)) != m.typed_fingerprint((2,), sha(one_two + b'\x00'))
    # literals produced once by the manifest-pinned original function on [1.0, 2.0] (shape (2,)) and its x3 (3, 2) tile
    assert m.typed_fingerprint((2,), sha(one_two)) == '7cac81503f33eb9802581d4688b017b1142c74e4b6672df3f089e8a8a6031c6e'
    assert m.typed_fingerprint((3, 2), sha(one_two * 3)) == '68bf92f592612e83beecff9c3406b3a796f7c75393569dd9db95e624899c7eca'


# ---- genuine run + oracle norms ------------------------------------------------------------------
def exact(raws, name):
    c, k = vals(raws['control'][name]), vals(raws['candidate'][name])
    return (sum(Fraction(x) ** 2 for x in c), sum(Fraction(x) ** 2 for x in k), sum((Fraction(b) - Fraction(a)) ** 2 for a, b in zip(c, k)))


def close(got, ss):
    want = math.sqrt(ss)
    assert math.isclose(got, want, rel_tol=1e-12, abs_tol=0), (got, want)


def check_result(result, raws, shapes):
    assert set(result) == {'schema', 'descriptive_only', 'inputs', 'base_vision_sha256', 'tensors', 'combined'}
    assert result['schema'] == m.RESULT_SCHEMA and result['descriptive_only'] is True
    assert set(result['tensors']) == set(NAMES)
    row_keys = {'shape', 'numel', 'control_raw_sha256', 'candidate_raw_sha256', 'control_matches_initial',
                'candidate_matches_initial', 'control_norm', 'candidate_norm', 'difference_norm', 'relative_l2', 'baseline_zero'}
    assert all(set(r) == row_keys for r in result['tensors'].values())
    assert set(result['combined']) == {'numel', 'control_norm', 'candidate_norm', 'difference_norm', 'relative_l2', 'baseline_zero'}
    totals = [Fraction(0)] * 3
    for name in NAMES:
        row, ss = result['tensors'][name], exact(raws, name)
        totals = [a + b for a, b in zip(totals, ss)]
        assert row['shape'] == list(shapes[name]) and row['numel'] == math.prod(shapes[name])
        for key, value in zip(('control_norm', 'candidate_norm', 'difference_norm'), ss):
            close(row[key], value)
        assert row['control_raw_sha256'] == sha(raws['control'][name]) and row['candidate_raw_sha256'] == sha(raws['candidate'][name])
        assert row['control_matches_initial'] is True and row['candidate_matches_initial'] is (raws['candidate'][name] == raws['control'][name])
        assert row['baseline_zero'] is (ss[0] == 0)
        if ss[0] == 0:
            assert row['relative_l2'] is None
        else:
            close(row['relative_l2'], ss[2] / ss[0])
    combined = result['combined']
    assert combined['numel'] == sum(math.prod(s) for s in shapes.values())
    for key, value in zip(('control_norm', 'candidate_norm', 'difference_norm'), totals):
        close(combined[key], value)
    assert combined['baseline_zero'] is (totals[0] == 0)
    if totals[0] == 0:
        assert combined['relative_l2'] is None
    else:
        close(combined['relative_l2'], totals[2] / totals[0])


def test_genuine_grammar_run_matches_exact_oracle_and_reads_only_selected_members():
    opened, real = [], zipfile.ZipFile.open

    def spy(self, member, *a, **k):
        opened.append(member.filename if isinstance(member, zipfile.ZipInfo) else member)
        return real(self, member, *a, **k)
    with tiny():
        directory, c = case()
        with directory:
            c.write()
            with mock.patch.object(zipfile.ZipFile, 'open', spy):
                result = c.ok()
            check_result(result, c.raws, TINY)
            assert set(opened) == {'archive/data.pkl', 'archive/byteorder'} | {'archive/data/%d' % i for i in range(4)}, opened
            assert opened.count('archive/data/0') == 2  # one control + one candidate read, no dedup/cache
            assert result['inputs']['control_endpoint']['sha256'] == sha(c.p['control_endpoint'].read_bytes())
            assert set(result['inputs']) == {'authority', 'own_source', 'interpreter', 'helper_source', 'control_bundle',
                                             'candidate_bundle', 'control_endpoint', 'candidate_endpoint'}
            assert result['inputs']['interpreter']['sha256'] == sha(c.interp.read_bytes())
            text = c.out.read_text()
            assert 'NaN' not in text and 'Infinity' not in text
            assert (c.out.stat().st_mode & 0o222) == 0 and sorted(q.name for q in c.root.iterdir() if q.name.endswith('.tmp')) == []


def test_extreme_magnitudes_and_zero_baseline():
    big, tiny_v, sub = 3.0e38, 1.5e-38, 1e-45
    raws = {'control': {NAMES[0]: f32([big, -big, 1, 0, sub, -sub]), NAMES[1]: f32([0, 0, 0]), NAMES[2]: f32([tiny_v] * 6),
                        NAMES[3]: f32([0, 0])},
            'candidate': {NAMES[0]: f32([-big, big, 1.0000001, 0, 0, sub]), NAMES[1]: f32([0, 0, 0]), NAMES[2]: f32([0] * 6),
                          NAMES[3]: f32([1, 2])}}
    with tiny():
        directory, c = case(raws=raws)
        with directory:
            result = c.write().ok()
            check_result(result, raws, TINY)
            zero, same = result['tensors'][NAMES[3]], result['tensors'][NAMES[1]]
            assert zero['baseline_zero'] is True and zero['relative_l2'] is None and zero['difference_norm'] > 0
            assert same['baseline_zero'] is True and same['relative_l2'] is None and same['difference_norm'] == 0
            assert result['tensors'][NAMES[1]]['candidate_matches_initial'] is True
            assert result['combined']['baseline_zero'] is False
    zeros = {arm: {n: bytes(4 * math.prod(TINY[n])) for n in NAMES} for arm in m.ARMS}
    zeros['candidate'][NAMES[1]] = f32([0, 0, 2])
    with tiny():
        directory, c = case(raws=zeros)
        with directory:
            result = c.write().ok()
            check_result(result, zeros, TINY)
            assert result['combined']['baseline_zero'] is True and result['combined']['relative_l2'] is None
            assert result['combined']['control_norm'] == 0.0 and result['combined']['difference_norm'] == 2.0
    identical = {arm: default_raws()['control'] for arm in m.ARMS}
    with tiny():
        directory, c = case(raws=identical)
        with directory:
            result = c.write().ok()
            assert result['combined']['difference_norm'] == 0.0 and result['combined']['relative_l2'] == 0.0
            assert all(r['candidate_matches_initial'] for r in result['tensors'].values())


def test_nonfinite_values_are_rejected_even_when_hashes_agree():
    for arm, bad in (('control', float('nan')), ('control', float('inf')), ('candidate', float('nan')), ('candidate', float('-inf'))):
        raws = default_raws()
        raws[arm][NAMES[2]] = f32([bad, 1, 2, 3, 4, 5])
        with tiny():
            directory, c = case(raws=raws)
            with directory:
                c.write()  # initial typed hashes derive from the (non-finite) control data, so only the finite guard can fire
                c.rejects('non-finite')


# ---- ZIP / pickle / tensor-grammar falsifiers ----------------------------------------------------
def flip_stored_byte(path):
    data = bytearray(path.read_bytes())
    at = data.index(default_raws()['control'][NAMES[0]])  # the stored data/0 payload
    data[at + 1] ^= 0x55
    path.write_bytes(data)


def zip_drop(name):
    return lambda entries: entries.__setitem__(slice(None), [e for e in entries if e[0] != name])


def zip_set(name, index, value):
    def edit(entries):
        for entry in entries:
            if entry[0] == name:
                entry[index] = value
    return edit


def swap_encoder(root, torch, utils):
    root['encoder'][NAMES[0]], root['encoder'][NAMES[2]] = root['encoder'][NAMES[2]], root['encoder'][NAMES[0]]


def tensor_mutants():
    first = lambda fn: (lambda name, item: fn(item) if name == NAMES[0] else None)
    second = lambda fn: (lambda name, item: fn(item) if name == NAMES[1] else None)
    grown = lambda extra: first(lambda i: i.update(numel=i['numel'] + extra, data=i['data'] + bytes(4 * extra)))
    yield 'shape/stride differ', first(lambda i: i.update(shape=(2, 3), stride=(3, 1)))
    yield 'shape/stride differ', first(lambda i: i.update(stride=(1, 3)))  # transposed view: same shape, non-contiguous
    yield 'storage span', first(lambda i: i.update(offset=1, numel=i['numel'] + 1, data=i['data'] + bytes(4)))
    yield 'storage span', grown(2)
    yield 'malformed tensor rebuild', first(lambda i: i.update(numel=i['numel'] - 1))  # rejected by the pinned helper itself
    yield 'CPU FloatStorage', first(lambda i: i.update(cls='HalfStorage'))
    yield 'CPU FloatStorage', first(lambda i: i.update(cls='DoubleStorage'))
    yield 'CPU FloatStorage', first(lambda i: i.update(location='cuda:0'))
    yield 'storage key differs', first(lambda i: i.update(key='a1'))
    yield 'unsafe member name', first(lambda i: i.update(key='../0'))
    yield 'share a storage', second(lambda i: i.update(key='0'))
    yield 'size/compression differs', first(lambda i: i.update(data=i['data'][:-4]))
    yield 'size/compression differs', first(lambda i: i.update(data=i['data'] + bytes(4)))


def test_selected_tensor_grammar_and_storage_spans():
    with tiny():
        for arm in m.ARMS:
            for text, edit in tensor_mutants():
                directory, c = case()
                with directory:
                    c.write(hooks={arm: {'tensor_edit': edit}})
                    c.rejects(text)


def zip_mutants():
    yield 'ZIP member missing', {'zip_edit': zip_drop('archive/data/2')}
    yield 'size/compression differs', {'zip_edit': zip_set('archive/data/0', 2, zipfile.ZIP_DEFLATED)}
    yield 'byteorder', {'zip_edit': zip_set('archive/byteorder', 1, b'big\x00\x00\x00')}
    yield 'ZIP member missing', {'zip_edit': zip_drop('archive/byteorder')}
    yield 'duplicate', {'zip_edit': lambda e: e.append(list(e[2]))}
    yield 'unsafe member name', {'zip_edit': lambda e: e.append(['archive/../escape', b'x', zipfile.ZIP_STORED])}
    yield 'unsafe member name', {'zip_edit': lambda e: e.append(['/abs', b'x', zipfile.ZIP_STORED])}
    yield 'exactly one', {'zip_edit': lambda e: e.append(['other/data.pkl', b'x', zipfile.ZIP_STORED])}
    yield 'exactly one', {'zip_edit': zip_drop('archive/data.pkl')}
    yield 'exactly one', {'zip_edit': zip_set('archive/data.pkl', 0, 'data.pkl')}  # no archive directory
    yield 'exactly one', {'zip_edit': zip_set('archive/data.pkl', 0, 'a/b/data.pkl')}
    yield 'encoder must be exactly', {'root_edit': (lambda r, t, u: r['encoder'].pop(NAMES[3]))}
    yield 'encoder must be exactly', {'root_edit': (lambda r, t, u: r['encoder'].update({'encoder.layers.26.mlp.fc3.bias': r['encoder'][NAMES[3]]}))}
    yield 'tensor metadata differs', {'root_edit': (lambda r, t, u: r['encoder'].update({NAMES[1]: 1.5}))}
    yield 'shape/stride differ', {'root_edit': (swap_encoder)}
    yield 'root keys differ', {'root_edit': (lambda r, t, u: r.update(extra=1))}
    yield 'root keys differ', {'root_edit': (lambda r, t, u: r.pop('fixed_sha256'))}
    yield 'schema/arm differs', {'root_edit': (lambda r, t, u: r.update(schema='x'))}
    yield 'vision_sha256 differs', {'root_edit': (lambda r, t, u: r.update(vision_sha256='9' * 64))}
    yield 'base_vision differs', {'root_edit': (lambda r, t, u: r.update(base_vision={'sha256': '9' * 64}))}
    yield 'initial_four_sha256 differs', {'root_edit': (lambda r, t, u: r['encoder_identity']['initial_four_sha256'].update({NAMES[0]: '9' * 64}))}
    # exact-blocker path: helper's closed grammar must reject, with no broadening
    yield 'forbidden pickle global', {'root_edit': (lambda r, t, u: r['config'].update(x=os.getcwd))}
    yield 'forbidden pickle global: torch.float32', {'root_edit': (lambda r, t, u: r['config'].update(x=t.float32))}
    yield 'stateful pickle opcode', {'root_edit': (lambda r, t, u: r['config'].update(x=Blob()))}


def test_zip_pickle_and_root_falsifiers():
    with tiny():
        for arm in m.ARMS:
            for text, hooks in zip_mutants():
                directory, c = case()
                with directory:
                    c.write(hooks={arm: hooks})
                    c.rejects(text)
        # tampering: stored byte under an unchanged ZIP CRC (pins recomputed) vs a changed file under the old pin
        directory, c = case()
        with directory:
            c.write(post={'control': flip_stored_byte})
            c.rejects('CRC')
        directory, c = case()
        with directory:
            c.write()
            data = bytearray(c.p['candidate_endpoint'].read_bytes())
            data[len(data) // 2] ^= 1
            c.p['candidate_endpoint'].write_bytes(data)
            c.rejects('endpoint SHA-256 differs from pin')


def test_real_shapes_are_required_not_the_test_shapes():
    directory, c = case()  # TINY endpoints but production shapes
    with directory:
        with tiny():
            c.write()
        c.rejects('inventory shape differs')


def test_manifest_and_cross_arm_falsifiers():
    cases = {
        'bundle schema': ({'control': lambda o: o.update(schema='x')}, 'exact keys required|schema differs'),
        'extra bundle key': ({'candidate': lambda o: o.update(extra=1)}, 'exact keys required'),
        'endpoint pin vs bundle': ({'control': lambda o: o['files'].update({'endpoint.pt': '9' * 64})}, 'bundle endpoint.pt SHA differs'),
        'inventory shape': ({'candidate': lambda o: o['encoder_identity']['inventory'].update({NAMES[2]: [4304, 1152]})}, 'inventory shape differs'),
        'initial differs across arms': ({'candidate': lambda o: o['encoder_identity']['initial_four_sha256'].update({NAMES[3]: '9' * 64})}, 'differ in initial'),
        'base differs across arms': ({'candidate': lambda o: o.update(base_vision_sha256='9' * 64)}, 'differ in base_vision_sha256'),
        'vision.pt differs across arms': ({'candidate': lambda o: o['files'].update({'vision.pt': '9' * 64})}, 'differ in vision_pt_sha256'),
        'initial names': ({'control': lambda o: o['encoder_identity']['initial_four_sha256'].pop(NAMES[0])}, 'exact keys required'),
    }
    with tiny():
        for label, (bundle, text) in cases.items():
            directory, c = case()
            with directory:
                c.write(bundle=bundle)
                code, error = c.run()
                assert code == 1 and any(t in error for t in text.split('|')), (label, error)
                assert not c.out.exists()
        # control data differing from the guarded initial typed hash (all four are checked)
        for victim in NAMES:
            directory, c = case()
            with directory:
                c.write(initial={**{n: oracle_fp(TINY[n], sha(c.raws['control'][n])) for n in NAMES}, victim: '9' * 64})
                c.rejects('differs from the guarded initial typed hash')
        # arms swapped: control file offered as the candidate
        directory, c = case()
        with directory:
            c.write()
            os.replace(c.p['control_endpoint'], c.root / 'tmp.pt')
            os.replace(c.p['candidate_endpoint'], c.p['control_endpoint'])
            os.replace(c.root / 'tmp.pt', c.p['candidate_endpoint'])
            for arm in m.ARMS:  # re-pin so only the semantic arm check can object
                for obj_path in (c.p[arm + '_bundle'],):
                    obj = json.loads(obj_path.read_text())
                    obj['files']['endpoint.pt'] = sha(c.p[arm + '_endpoint'].read_bytes())
                    obj_path.write_text(json.dumps(obj))
            auth = json.loads(c.p['authority'].read_text())
            for arm in m.ARMS:
                auth['arms'][arm]['bundle']['sha256'] = sha(c.p[arm + '_bundle'].read_bytes())
                auth['arms'][arm]['endpoint']['sha256'] = sha(c.p[arm + '_endpoint'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('schema/arm differs')
        # the same file twice
        directory, c = case()
        with directory:
            c.write(authority=lambda o: o['arms']['candidate']['endpoint'].update(o['arms']['control']['endpoint']))
            c.rejects('four distinct paths')


# ---- authority / CLI / source-pin / API falsifiers -----------------------------------------------------
def test_authority_file_schema_and_pin_order():
    marker = None
    with tiny():
        directory, c = case()
        with directory:
            c.write()
            marker = c.root / 'marker'
            c.rejects('authority SHA-256 differs', c.argv(sha_override='0' * 64))
            c.rejects('lowercase SHA-256', c.argv(sha_override='ABC'))
            link = c.root / 'link.json'
            link.symlink_to(c.p['authority'])
            c.rejects('canonical path', ['--authority', str(link), '--authority-sha256', sha(c.p['authority'].read_bytes()), '--output', str(c.out)])
            c.rejects('absolute path', ['--authority', 'authority.json', '--authority-sha256', '0' * 64, '--output', str(c.out)])
            for text, edit in (
                    ('exact keys', lambda o: o.update(extra=1)), ('exact keys', lambda o: o.pop('arms')),
                    ('schema differs', lambda o: o.update(schema='x')), ('lowercase SHA-256', lambda o: o.update(own_source_sha256='G' * 64)),
                    ('exact keys', lambda o: o['arms'].pop('candidate')), ('exact keys', lambda o: o['arms']['control']['endpoint'].update(x=1)),
                    ('absolute path', lambda o: o['helper_source'].update(path='helper.py'))):
                c.write(authority=edit)
                c.rejects(text)
            c.write(authority=lambda o: None)
            c.p['authority'].write_text(c.p['authority'].read_text().replace('"schema"', '"schema": "dup", "schema"', 1))
            c.rejects('duplicate JSON key')
            c.p['authority'].write_text('{"schema": NaN}')
            c.rejects('non-finite JSON constant')
            # source pins are checked before the helper's code runs at all
            c.write()
            auth = json.loads(c.p['authority'].read_text())
            c.p['helper'].write_text('open(%r, "w").write("ran")\n' % str(marker))
            auth['helper_source']['sha256'] = '0' * 64
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('helper source SHA-256 differs')
            assert not marker.exists(), 'unpinned helper code was executed'
            auth['helper_source']['sha256'] = sha(c.p['helper'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('helper API differs')  # pinned but not the real helper: code runs only now, then API check refuses
            assert marker.exists()
            c.write()
            auth = json.loads(c.p['authority'].read_text())
            auth['own_source_sha256'] = '0' * 64
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('own source SHA-256 differs')
            # drifted helper API with a matching pin
            drifted = HELPER.read_text().replace('    storage_offset: int\n', '    storage_start: int\n', 1)
            assert drifted != HELPER.read_text()
            c.write()
            c.p['helper'].write_text(drifted)
            auth = json.loads(c.p['authority'].read_text())
            auth['helper_source']['sha256'] = sha(c.p['helper'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('helper API differs: _TensorStub')
            # wrong sha for the bundle / endpoint
            c.write()
            auth = json.loads(c.p['authority'].read_text())
            auth['arms']['control']['bundle']['sha256'] = '0' * 64
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('control bundle SHA-256 differs')


def test_canonical_regular_file_admission_for_own_source_interpreter_helper_and_inputs():
    with tiny():
        def symlinked(label, key, text):
            directory, c = case()
            with directory:
                c.write()
                real = c.root / ('real-' + c.p[key].name)
                os.replace(c.p[key], real)
                c.p[key].symlink_to(real)
                if key in ('own', 'interp'):
                    setattr(c, 'own' if key == 'own' else 'interp', c.p[key])
                c.rejects(text)
        for label, key in (('own', 'own'), ('interpreter', 'interp'), ('helper', 'helper'),
                           ('endpoint', 'candidate_endpoint'), ('bundle', 'control_bundle')):
            symlinked(label, key, 'canonical path required')
        # a symlinked directory component is refused as well (no weakening to realpath)
        directory, c = case()
        with directory:
            c.write()
            (c.root / 'real-dir').mkdir()
            (c.root / 'link-dir').symlink_to(c.root / 'real-dir')
            for key in ('own', 'interp'):
                (c.root / 'real-dir' / c.p[key].name).write_bytes(c.p[key].read_bytes())
            c.own = c.root / 'link-dir' / 'own.py'
            c.rejects('own source: canonical path required')
            c.own = c.p['own']
            c.interp = c.root / 'link-dir' / 'interp.bin'
            c.rejects('interpreter (launch with the canonical python path): canonical path required')
            c.interp = c.p['interp']
            # not a regular file
            c.interp = c.root / 'real-dir'
            c.rejects('regular file within size limit required')
            c.interp = c.p['interp']
            assert c.ok()['inputs']['interpreter']['path'] == str(c.p['interp'])


def test_helper_registry_ownership_is_never_overwritten_or_deleted():
    with tiny():
        foreign = types.ModuleType(m.HELPER_MODULE)
        marker_code = 'open(%r, "w").write("ran")\n'
        # 1. preexisting name: rejected before any overwrite, helper code never runs, foreign entry untouched
        directory, c = case()
        with directory:
            c.write()
            marker = c.root / 'marker'
            c.p['helper'].write_text(marker_code % str(marker))
            auth = json.loads(c.p['authority'].read_text())
            auth['helper_source']['sha256'] = sha(c.p['helper'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            sys.modules[m.HELPER_MODULE] = foreign
            try:
                c.foreign_ok = True
                c.rejects('already registered')
                assert sys.modules[m.HELPER_MODULE] is foreign and not marker.exists()
            finally:
                del sys.modules[m.HELPER_MODULE]
                c.foreign_ok = False
            # control: with the name free the same pinned helper code does run (then fails the API check) and cleans up
            c.rejects('helper API differs')
            assert marker.exists() and m.HELPER_MODULE not in sys.modules
        # 2. foreign replacement during the run (after load, before cleanup): preserved and rejected, nothing published
        directory, c = case()
        with directory:
            c.write()
            real, replacement = m.measure, types.ModuleType('foreign')

            def swapping(*a, **k):
                result = real(*a, **k)
                sys.modules[m.HELPER_MODULE] = replacement
                return result
            try:
                c.foreign_ok = True
                with mock.patch.object(m, 'measure', swapping):
                    c.rejects('replaced by a foreign module')
                assert sys.modules[m.HELPER_MODULE] is replacement, 'foreign replacement was deleted'
            finally:
                sys.modules.pop(m.HELPER_MODULE, None)
                c.foreign_ok = False
            assert c.ok()['descriptive_only'] is True  # a clean registry afterwards works again
        # 3. helper code that replaces its own registry entry while executing
        directory, c = case()
        with directory:
            c.write()
            c.p['helper'].write_text('import sys, types\nsys.modules[__name__] = types.ModuleType("evil")\n')
            auth = json.loads(c.p['authority'].read_text())
            auth['helper_source']['sha256'] = sha(c.p['helper'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            try:
                c.foreign_ok = True
                c.rejects('replaced by a foreign module')
                assert sys.modules[m.HELPER_MODULE].__name__ == 'evil'
            finally:
                sys.modules.pop(m.HELPER_MODULE, None)
                c.foreign_ok = False
        # 4. helper code that raises while executing: our own entry is removed, nothing else touched
        directory, c = case()
        with directory:
            c.write()
            c.p['helper'].write_text('raise ValueError("boom while loading")\n')
            auth = json.loads(c.p['authority'].read_text())
            auth['helper_source']['sha256'] = sha(c.p['helper'].read_bytes())
            c.p['authority'].write_text(json.dumps(auth))
            c.rejects('boom while loading')
            assert m.HELPER_MODULE not in sys.modules


def test_helper_read_must_equal_the_hashed_file_handle():
    with tiny():
        directory, c = case()
        with directory:
            c.write()
            real = m.load_helper

            def diverging(path, data):
                helper = real(path, data)
                original = helper._read_checkpoint_data_pickle
                helper._read_checkpoint_data_pickle = lambda path: original(path) + b'N'  # any byte difference must be caught
                return helper
            with mock.patch.object(m, 'load_helper', diverging):
                c.rejects('differs between helper and hashed file')


def test_post_link_publication_failures_are_never_a_normal_terminal():
    real_fsync, real_unlink = os.fsync, os.unlink
    with tiny():
        # success: exactly one stdout line binds the file
        directory, c = case()
        with directory:
            c.write()
            assert c.ok() and c.stdout.count('\n') == 1
        # directory fsync fails after the hard link: output retracted, run fails, no success line
        directory, c = case()
        with directory:
            c.write()
            calls = []

            def failing_fsync(fd):
                calls.append(fd)
                if len(calls) == 2:  # 1st = temp file, 2nd = parent directory (after os.link)
                    raise OSError(5, 'injected directory fsync failure')
                return real_fsync(fd)
            with mock.patch.object(m.os, 'fsync', failing_fsync):
                code, error = c.run()
            assert code == 1 and 'injected directory fsync failure' in error and c.stdout == ''
            assert not c.out.exists() and not [q for q in c.root.iterdir() if q.name.endswith('.tmp')]
        # same failure and the retraction itself fails: the survivor must be rejected by the terminal contract
        directory, c = case()
        with directory:
            c.write()
            calls = []

            def failing_unlink(path, *a, **k):
                if Path(path) == c.out:
                    raise OSError(1, 'injected retraction failure')
                return real_unlink(path, *a, **k)

            def failing_fsync2(fd):
                calls.append(fd)
                if len(calls) == 2:
                    raise OSError(5, 'injected directory fsync failure')
                return real_fsync(fd)
            with mock.patch.object(m.os, 'fsync', failing_fsync2), mock.patch.object(m.os, 'unlink', failing_unlink):
                code, error = c.run()
            assert code == 1 and c.out.exists() and json.loads(c.out.read_text())['descriptive_only'] is True
            assert c.stdout == '' and not c.terminal_ok(code), 'survivor of a failed run accepted as a normal terminal'
        # temp cleanup fails after the link: run fails, output retracted
        directory, c = case()
        with directory:
            c.write()

            def failing_temp_unlink(path, *a, **k):
                if str(path).endswith('.tmp'):
                    raise OSError(1, 'injected temp cleanup failure')
                return real_unlink(path, *a, **k)
            with mock.patch.object(m.os, 'unlink', failing_temp_unlink):
                code, error = c.run()
            assert code == 1 and 'injected temp cleanup failure' in error and c.stdout == '' and not c.out.exists()


def test_cli_strictness_and_output_policy():
    with tiny():
        directory, c = case()
        with directory:
            c.write()
            argv = c.argv()
            for bad in ([], argv[:4], argv + ['--extra', 'x'], ['--auth', argv[1]] + argv[2:], argv + ['stray']):
                with contextlib.redirect_stderr(io.StringIO()):
                    try:
                        m.main(bad, own=c.own)
                    except SystemExit as stop:
                        assert stop.code == 2, bad
                    else:
                        raise AssertionError('accepted malformed CLI %r' % bad)
            c.out.write_text('precious')
            code, error = c.run()
            assert code == 1 and 'new file' in error and c.out.read_text() == 'precious'
            c.out.unlink()
            c.out.symlink_to(c.root / 'nowhere')
            code, error = c.run()
            assert code == 1 and 'new file' in error
            c.out.unlink()
            for bad in ('relative.json', str(c.root / 'out.txt'), str(c.root / 'missing-dir' / 'out.json')):
                code, error = c.run(c.argv()[:4] + ['--output', bad])
                assert code == 1 and not (c.root / 'out.txt').exists()
            # output appearing during the run is never overwritten
            real = m.measure

            def racing(*a, **k):
                result = real(*a, **k)
                c.out.write_text('raced')
                return result
            with mock.patch.object(m, 'measure', racing):
                code, error = c.run()
            assert code == 1 and 'already exists' in error and c.out.read_text() == 'raced'
            assert not [q for q in c.root.iterdir() if q.name.endswith('.tmp')]


def test_complete_uncached_rehash_before_publication():
    def mutate(case_, key):
        path = case_.p[key]
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(data)
    labels = {'authority': 'authority', 'own_source': 'own', 'interpreter': 'interp', 'helper_source': 'helper', 'control_bundle': 'control_bundle',
              'candidate_bundle': 'candidate_bundle', 'control_endpoint': 'control_endpoint', 'candidate_endpoint': 'candidate_endpoint'}
    with tiny():
        for label, key in labels.items():
            directory, c = case()
            with directory:
                c.write()
                real = m.measure

                def tampering(*a, **k):
                    result = real(*a, **k)
                    mutate(c, key)
                    return result
                with mock.patch.object(m, 'measure', tampering):
                    c.rejects(label + ' changed before publication')


def test_resource_and_host_guards():
    with tiny():
        directory, c = case()
        with directory:
            c.write()
            with mock.patch.object(m, 'MAX_ENDPOINT', 100):
                c.rejects('size limit')
            with mock.patch.object(m.sys, 'byteorder', 'big'):
                c.rejects('little-endian')
            clock = iter([0.0, 1000.0] + [2000.0] * 100)
            with mock.patch.object(m.time, 'monotonic', lambda: next(clock)):
                c.rejects('budget')
            assert c.ok()['descriptive_only'] is True


# ---- production shapes through the real CLI ---------------------------------------------------------------
def test_production_shapes_through_cli_subprocess():
    pattern_control = [(i + 1) * .25 for i in range(16)]
    pattern_candidate = [v + .0625 * (i % 3) for i, v in enumerate(pattern_control)]
    raws = {'control': {n: f32(pattern_control) * (math.prod(m.SHAPES[n]) // 16) for n in NAMES},
            'candidate': {n: f32(pattern_candidate) * (math.prod(m.SHAPES[n]) // 16) for n in NAMES}}
    directory = tempfile.TemporaryDirectory()
    with directory:
        c = Case(directory.name, raws=raws, own=DRIVER)
        c.write()
        python = str(Path(sys.executable).resolve())  # the reader admits only a canonical interpreter path
        process = subprocess.run([python, '-I', str(DRIVER)] + c.argv(), capture_output=True, text=True, timeout=110)
        assert process.returncode == 0, process.stderr
        line = json.loads(process.stdout)
        assert line == {'output': str(c.out), 'sha256': sha(c.out.read_bytes()), 'bytes': c.out.stat().st_size}
        result = json.loads(c.out.read_text())
        shapes = {n: m.SHAPES[n] for n in NAMES}
        # pattern-periodic data: the exact oracle is reps * one pattern's sums
        for name in NAMES:
            reps = math.prod(shapes[name]) // 16
            ss = tuple(reps * s for s in exact({'control': {name: f32(pattern_control)}, 'candidate': {name: f32(pattern_candidate)}}, name))
            row = result['tensors'][name]
            for key, value in zip(('control_norm', 'candidate_norm', 'difference_norm'), ss):
                close(row[key], value)
            close(row['relative_l2'], ss[2] / ss[0])
            assert row['shape'] == list(shapes[name]) and row['control_matches_initial'] and not row['candidate_matches_initial']
        assert result['combined']['numel'] == sum(math.prod(s) for s in shapes.values())
        assert shapes[NAMES[0]] == (4304, 1152) and shapes[NAMES[1]] == (4304,) and shapes[NAMES[2]] == (1152, 4304) and shapes[NAMES[3]] == (1152,)
        again = subprocess.run([python, '-I', str(DRIVER)] + c.argv(), capture_output=True, text=True, timeout=110)
        assert again.returncode == 1 and 'new file' in again.stderr  # exclusive: no overwrite on rerun


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, test in tests:
        started = __import__('time').perf_counter()
        test()
        print('ok %s %.1fs' % (name, __import__('time').perf_counter() - started))
    print('ALL %d OK' % len(tests))
