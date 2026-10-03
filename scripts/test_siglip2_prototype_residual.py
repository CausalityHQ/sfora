#!/usr/bin/env python3
"""One bounded stdlib falsifier; native TRAIN/resource/quality checks remain UNRUN."""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('_prototype_test_driver', HERE / 'fit_siglip2_prototype_residual.py')
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
spec.loader.exec_module(fit)


def rejects(call, message=None):
    try:
        call()
    except (ValueError, OSError) as error:
        if message is not None:
            assert message in str(error), str(error)
        return
    raise AssertionError('corruption was accepted')


class Mask:
    def __init__(self, values):
        self.values = values
    def all(self):
        return all(self.values)
    def any(self):
        return any(self.values)


class Index:
    dtype = 'int64'
    ndim = 1
    def __init__(self, values):
        self.values = values
    def numel(self):
        return len(self.values)
    def __lt__(self, value):
        return Mask([v < value for v in self.values])
    def __ge__(self, value):
        return Mask([v >= value for v in self.values])


class Matrix:
    """Tiny independent double-precision oracle for the AUTHENTICATED solver AST."""
    dtype = 'float32'
    device = 'cpu'
    ndim = 2
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))
    def __getitem__(self, indexes):
        return Matrix([self.rows[i] for i in indexes.values])
    def mean(self, dim, keepdim=False):
        assert dim == 0
        return Matrix([[sum(row[j] for row in self.rows) / self.shape[0] for j in range(self.shape[1])]])
    @property
    def T(self):
        return Matrix(zip(*self.rows))
    def __sub__(self, other):
        return Matrix([[v - other.rows[0 if other.shape[0] == 1 else i][j]
                        for j, v in enumerate(row)] for i, row in enumerate(self.rows)])
    def __add__(self, other):
        return Matrix([[v + other.rows[0 if other.shape[0] == 1 else i][j] for j, v in enumerate(row)] for i, row in enumerate(self.rows)])
    def __rmul__(self, value):
        return Matrix([[value * v for v in row] for row in self.rows])
    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        return Matrix([[sum(row[k] * other.rows[k][j] for k in range(self.shape[1]))
                        for j in range(other.shape[1])] for row in self.rows])


def solve(matrix, rhs):
    # Independent Gauss-Jordan oracle with pivoting, not the native implementation.
    n, outputs = matrix.shape[0], rhs.shape[1]
    augmented = [row[:] + rhs.rows[i][:] for i, row in enumerate(matrix.rows)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda i: abs(augmented[i][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        assert divisor != 0
        augmented[column] = [v / divisor for v in augmented[column]]
        for i in range(n):
            if i != column:
                multiplier = augmented[i][column]
                augmented[i] = [a - multiplier * b for a, b in zip(augmented[i], augmented[column])]
    return Matrix([row[n:n + outputs] for row in augmented])


def near(left, right):
    assert left.shape == right.shape
    assert max(abs(a - b) for ra, rb in zip(left.rows, right.rows) for a, b in zip(ra, rb)) < 1e-10


def typed_hash(value):
    # Small typed metadata/byte oracle. Production uses the pinned original serializer.
    def frames(item):
        if type(item) is dict:
            return b'd' + b''.join(frames(k) + frames(item[k]) for k in sorted(item, key=repr))
        if type(item) is list:
            return b'l' + b''.join(frames(v) for v in item)
        if type(item) is bytes:
            return b'b' + struct.pack('!I', len(item)) + item
        if type(item) is float:
            return b'f' + struct.pack('!d', item)
        if type(item) is int:
            return b'i' + str(item).encode()
        if type(item) is str:
            return b's' + item.encode()
        raise ValueError('unsupported oracle type')
    return hashlib.sha256(frames(value)).hexdigest()


def extracted(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def check_repeat_preparation(evidence):
    """Real pinned loader/preparation prefix; stop before native checkpoint I/O."""
    class NativeBoundary(Exception):
        pass

    def stop(*args, **kwargs):
        raise NativeBoundary

    def native_import(name, *args, **kwargs):
        if name == 'torch':
            return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
                set_num_threads=lambda n: None, get_num_interop_threads=lambda: 1)
        return builtins.__import__(name, *args, **kwargs)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        rank, packing = root / 'rank.py', root / 'packing.py'
        rank.write_text('def smooth_ap_bank_loss(): return 1\n')
        packing.write_text('def pack_int8_unit_embeddings(): return 2\n')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        namespace = dict(Path=Path, sys=sys, ast=ast, hashlib=hashlib, importlib=__import__('importlib'),
                         require=fit.require, bound_file=fit.bound_file)
        original = SimpleNamespace(**extracted(HERE / 'train_siglip2_substrate_adaptation.py',
            {'load_bare'}, dict(namespace)))
        # Preserve the actual reference_math body; its imports are the native
        # boundary. Tiny references exercise the original selected-AST loader.
        reference = root / 'reference.py'
        reference.write_text('def member_bank_positive_ordinals(): return 3\n')
        reference_tree = ast.parse(reference.read_bytes())
        original.REFERENCES = {'reference.py': {'source': digest(reference),
            'ast': hashlib.sha256(ast.dump(reference_tree, include_attributes=False).encode()).hexdigest(),
            'names': ('member_bank_positive_ordinals',)}}
        original.RANK_SHA256 = digest(rank)
        (root / 'deployed_code_rank.py').write_bytes(rank.read_bytes())
        import __future__
        reference_namespace = dict(namespace, __future__=__future__)
        extracted(HERE / 'train_siglip2_substrate_adaptation.py', {'selected_ast'}, reference_namespace)
        original.selected_ast = reference_namespace['selected_ast']
        reference_namespace.update(load_bare=original.load_bare, RANK_SHA256=original.RANK_SHA256,
            REFERENCES=original.REFERENCES, SimpleNamespace=SimpleNamespace,
            torch=None, np=None, F=None, nn=None, math=None, Dataset=object,
            transforms=None, Image=None, selected_ast=reference_namespace['selected_ast'])
        node = next(n for n in ast.parse((HERE / 'train_siglip2_substrate_adaptation.py').read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'reference_math')
        node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'reference_math.py', 'exec'), reference_namespace)
        original.reference_math = reference_namespace['reference_math']
        genuine = SimpleNamespace(**extracted(HERE / 'train_siglip2_genuine_views.py',
            {'load_helper'}, dict(namespace)))
        flags = {'threads': 1, 'interop_threads': 1}
        legacy = dict(genuine=genuine, original=original, guards={}, prior={},
            selected={'source_cpu': {'numerical_flags': flags}, 'packages': {},
                'math_context': {'root': root}, 'launch': {'helpers': {'packing':
                {'path': str(packing), 'sha256': digest(packing)}}},
                'source': {'caches': {'canonical': {'path': str(rank), 'sha256': digest(rank)}}}},
            source_driver=SimpleNamespace(numerical_flags=lambda: flags))
        old_namespace = dict(namespace, __builtins__=dict(vars(builtins), __import__=native_import),
            audit_origins=lambda *a, **k: None, admitted_file=stop,
            WARM_CHECKPOINT={'path': str(rank), 'sha256': digest(rank)})
        path = evidence / 'train_siglip2_quadratic_readout.py'
        # Execute the genuine lifecycle up to the fresh warm-load boundary.
        # First preparation completes only this stdlib prefix in this fixture;
        # repeat executes the production adapter's full retained function.
        node = next(n for n in ast.parse(path.read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'prepare_native')
        node.body = node.body[:14]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), old_namespace)
        old = SimpleNamespace(**old_namespace, __file__=str(path),
            require_no_model=lambda c: None, owned_encoder=lambda c: {'checkpoint':
            {'path': str(rank), 'sha256': digest(rank)}})
        context = dict(old=old, legacy=legacy, guards={})
        driver = dict(vars(fit), __builtins__=dict(vars(builtins), __import__=lambda name, *a, **k:
            stop() if name == 'torch' else builtins.__import__(name, *a, **k)))
        extracted(HERE / 'fit_siglip2_prototype_residual.py', {'prepare_native'}, driver)
        def reaches_warm():
            try:
                driver['prepare_native'](context)
            except NativeBoundary:
                return
            raise AssertionError('native boundary was not reached')
        partial = dict(context, legacy=dict(legacy, ref=object()))
        rejects(lambda: driver['prepare_native'](partial), 'partial original preparation')
        reaches_warm()
        ref, pack = legacy['ref'], legacy['packing']
        reaches_warm()  # RED: genuine rank loader rejects the second load.
        assert legacy['ref'] is ref and legacy['packing'] is pack
        # Independently enumerate the two removed statements in source-v7:
        # all other ASTs, including every nested require/data check, survive.
        original_node = next(n for n in ast.parse(path.read_bytes()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'prepare_native')
        original_node.body = [n for i, n in enumerate(original_node.body) if i not in (11, 13)]
        assert context['original_preparation']['ast'] == ast.dump(
            ast.Module(body=[original_node], type_ignores=[]), include_attributes=False)
        rank_module = sys.modules['_siglip2_pinned_adaptation_rank']
        for name, module in (('_siglip2_pinned_adaptation_rank', rank_module), ('_quadratic_packing', pack)):
            sys.modules[name] = SimpleNamespace(__file__=module.__file__, __spec__=module.__spec__)
            try:
                rejects(lambda: driver['prepare_native'](context), 'helper object/origin')
            finally:
                sys.modules[name] = module
        origin = pack.__spec__.origin
        pack.__spec__.origin = str(rank)
        try:
            rejects(lambda: driver['prepare_native'](context), 'helper object/origin')
        finally:
            pack.__spec__.origin = origin
        function = rank_module.smooth_ap_bank_loss
        rank_module.smooth_ap_bank_loss = lambda: 9
        try:
            rejects(lambda: driver['prepare_native'](context), 'helper function/code')
        finally:
            rank_module.smooth_ap_bank_loss = function
        code = function.__code__
        function.__code__ = (lambda: 9).__code__
        try:
            rejects(lambda: driver['prepare_native'](context), 'helper function/code')
        finally:
            function.__code__ = code
        values = original.reference_math.__globals__
        loader = values['load_bare']
        values['load_bare'] = lambda *args: None
        try:
            rejects(lambda: driver['prepare_native'](context), 'helper globals')
        finally:
            values['load_bare'] = loader
        reference_function = ref.member_bank_positive_ordinals
        ref.member_bank_positive_ordinals = lambda: 9
        try:
            rejects(lambda: driver['prepare_native'](context), 'reference metadata')
        finally:
            ref.member_bank_positive_ordinals = reference_function
        adapter = context['original_preparation']['prepare']
        context['original_preparation']['prepare'] = FunctionType(adapter.__code__, dict(adapter.__globals__))
        try:
            rejects(lambda: driver['prepare_native'](context), 'preparation AST/code')
        finally:
            context['original_preparation']['prepare'] = adapter
        code = adapter.__code__
        adapter.__code__ = (lambda c: None).__code__
        try:
            rejects(lambda: driver['prepare_native'](context), 'preparation AST/code')
        finally:
            adapter.__code__ = code
        for victim in (root / 'deployed_code_rank.py', packing, reference):
            raw, stat = victim.read_bytes(), victim.stat()
            victim.write_bytes(raw.replace(b'return', b'Return', 1))
            os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            try:
                rejects(lambda: driver['prepare_native'](context), 'SHA256')
            finally:
                victim.write_bytes(raw)
        reaches_warm()
        rejects(lambda: original.reference_math({'root': root}), 'helper already loaded')
        rejects(lambda: genuine.load_helper('_quadratic_packing', str(packing), digest(packing), {}),
                'helper already loaded')


def check_signed_readout():
    """Exercise the v2 helper with stdlib layout/algebra mocks, not native FP32."""
    path = HERE / 'prototype_residual_readout.py'
    context = {'root': HERE, 'code': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}, 'guards': {}}
    helper = fit.prepare_readout(context)
    fake = SimpleNamespace(cat=lambda blocks, dim: Matrix([a + b for a, b in
                           zip(blocks[0].rows, blocks[1].rows, strict=True)]))
    namespace = dict(vars(helper), __builtins__=dict(vars(builtins), __import__=lambda name, *args, **kw:
                     fake if name == 'torch' else builtins.__import__(name, *args, **kw)))
    extracted(path, {'basis'}, namespace)
    Z = Matrix([[float(i) for i in range(32)], [-float(i) for i in range(32)]])
    H0 = Matrix([[100. + i for i in range(128)], [200. + i for i in range(128)]])
    V = namespace['basis'](Z, H0, 'concat')
    assert V.rows == [z + h for z, h in zip(Z.rows, H0.rows, strict=True)]
    assert namespace['basis'](Z, H0, 'linear') is Z
    rejects(lambda: namespace['basis'](H0, Z, 'concat'), 'widths/order')
    rejects(lambda: namespace['basis'](Z, H0, 'quadratic'), 'arm')
    def verify_basis(value):
        fit.require(value.rows == V.rows, 'concat block order differs')
    rejects(lambda: verify_basis(Matrix([h + z for z, h in zip(Z.rows, H0.rows, strict=True)])), 'block order')
    class Parameter:
        requires_grad = is_leaf = True
        grad_fn = None
        def __init__(self, shape):
            self.shape = shape
    def check_tensor(value, shape, device, **kw):
        fit.require(value.shape == shape, 'tensor shape differs')
    primitive = SimpleNamespace(_check_tensor=check_tensor, _require=fit.require)
    fake.nn = SimpleNamespace(Parameter=Parameter)
    extracted(path, {'check_weight'}, namespace)
    namespace['check_weight'](Parameter((128, 160)), 'cpu', 'concat', primitive)
    rejects(lambda: namespace['check_weight'](Parameter((128, 32)), 'cpu', 'concat', primitive), 'shape')
    rejects(lambda: namespace['check_weight'](Parameter((128, 160)), 'cpu', 'linear', primitive), 'shape')
    bad_leaf = Parameter((128, 160)); bad_leaf.is_leaf = False
    rejects(lambda: namespace['check_weight'](bad_leaf, 'cpu', 'concat', primitive), 'leaf')
    coefficients = [struct.pack('<f', 1.) for _ in range(160)]
    digest = typed_hash(coefficients)
    coefficients[32] = struct.pack('<f', 9.)
    rejects(lambda: fit.verify_digest(typed_hash, coefficients, digest, 'H0 coefficient changed'), 'H0')
    # Execute the production mutation's indexed target against a standin whose
    # reshape deliberately returns a copy, as the strided H0 view would do.
    class Data:
        def __init__(self):
            self.values = [[0.] * 160]
        def __getitem__(self, index):
            return SimpleNamespace(item=lambda: self.values[index[0]][index[1]])
        def __setitem__(self, index, value):
            self.values[index[0]][index[1]] = value
        def reshape(self, *shape):
            return [v for row in self.values for v in row]
    mutation = next(n for n in ast.walk(ast.parse((HERE / 'fit_siglip2_prototype_residual.py').read_bytes()))
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'tensor.data[index]' and
        isinstance(n.value, ast.IfExp))
    data = Data()
    exec(compile(ast.Module(body=[mutation], type_ignores=[]), '<H0 direct mutation>', 'exec'),
         {'tensor': SimpleNamespace(data=data), 'index': (0, 32), 'value': 0.})
    assert data.values[0][32] == 1. and data.values[0][0] == 0.
    functions = [(name, fn) for name, fn in vars(helper).items() if isinstance(fn, FunctionType)]
    name, fn = functions[0]
    code = fn.__code__
    fn.__code__ = (lambda: None).__code__
    try:
        rejects(lambda: fit.prepare_readout(context), 'function/code')
    finally:
        fn.__code__ = code
    vars(helper)[name] = lambda: None
    try:
        rejects(lambda: fit.prepare_readout(context), 'globals')
    finally:
        vars(helper)[name] = fn
    spec = helper.__spec__
    origin = spec.origin
    spec.origin = str(HERE / 'fit_siglip2_prototype_residual.py')
    try:
        rejects(lambda: fit.prepare_readout(context), 'origin')
    finally:
        spec.origin = origin
    fit.prepare_readout(context)
    # Both actual consumer schema guards reject cross-version payloads before
    # reaching native arithmetic or any fake metadata/layout validator.
    historical = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/train-source-v3/fit_siglip2_prototype_residual.py'
    for source_path, consumer_schema, rejected_schema in (
        (HERE / 'fit_siglip2_prototype_residual.py', fit.SCHEMA, 'siglip2-prototype-residual-ridge-v1'),
        (historical, 'siglip2-prototype-residual-ridge-v1', fit.SCHEMA)):
        node = next(n for n in ast.parse(source_path.read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'check_payload')
        node.body = node.body[:3]  # Exact real schema admission prefix.
        namespace = dict(vars(fit), SCHEMA=consumer_schema,
            __builtins__=dict(vars(builtins), __import__=lambda name, *a, **kw:
                fake if name == 'torch' else builtins.__import__(name, *a, **kw)))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
        saved = {k: None for k in fit.PAYLOAD_KEYS}; saved['schema'] = rejected_schema
        rejects(lambda: namespace['check_payload']({'old': None, 'legacy': None}, saved, {}), 'typed payload')


def check_adapter(path, torch, source, target):
    """Exact AST correspondence and double oracle; native audit stays root-owned."""
    original = fit.solver_definitions(path.read_bytes())
    adapted = fit.solver_definitions(path.read_bytes(), adapted=True)
    fn = next(n for n in adapted if isinstance(n, ast.FunctionDef))
    assert [a.arg for a in fn.args.kwonlyargs] == ['regularization', 'reference_scale']
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and
                      isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scale')
    assert ast.unparse(assignment.value) == 'reference_scale'
    fn.args.kwonlyargs.pop(); fn.args.kw_defaults.pop()
    assignment.value = ast.parse('torch.trace(gram) / gram.shape[0]', mode='eval').body
    assert ast.dump(ast.Module(body=adapted, type_ignores=[])) == ast.dump(ast.Module(body=original, type_ignores=[]))
    changed = path.read_bytes().replace(b'x.T @ y)', b'x.T @ y + 1)', 1)
    assert changed != path.read_bytes()
    rejects(lambda: fit.solver_definitions(changed, adapted=True), 'solver AST')
    x = source - source.mean(0)
    reference = torch.trace(x.T @ x) / 2
    adapted_solver = fit.extract_solver(path, fit.SOLVER_SHA, torch, adapted=True)
    original_solver = fit.extract_solver(path, fit.SOLVER_SHA, torch)
    left = original_solver(source, target, Index([0, 1, 2, 3]), regularization=.1)
    right = adapted_solver(source, target, Index([0, 1, 2, 3]), regularization=.1, reference_scale=reference)
    assert all(getattr(left, k).rows == getattr(right, k).rows for k in ('weight', 'source_mean', 'target_mean'))
    wrong = adapted_solver(source, target, Index([0, 1, 2, 3]), regularization=.1, reference_scale=reference / 5)
    assert wrong.weight.rows != right.weight.rows
    # Operand capture only; Matrix arithmetic remains double precision.
    class Scalar:
        def __init__(self, value):
            self.value = struct.unpack('<f', struct.pack('<f', value))[0]
        def item(self):
            return self.value
        def __rmul__(self, other):
            return Scalar(other * self.value)
        def __mul__(self, other):
            return self.value * other
    fake = SimpleNamespace(**vars(torch))
    fake.equal = lambda a, b: a.rows == b.rows
    fake.isfinite = lambda value: SimpleNamespace(item=lambda: True) if isinstance(value, Scalar) else torch.isfinite(value)
    def check_tensor(value, shape, device, **kw):
        fit.require(isinstance(value, Scalar) and shape == () and device == 'cpu' and kw['frozen'], 'scalar differs')
    audit = {}
    checked = fit.checked_solver(path, SimpleNamespace(_check_tensor=check_tensor), fake, audit)
    checked(source, target, Index([0, 1, 2, 3]), regularization=.1, reference_scale=Scalar(reference))
    assert audit['lambda_bits'] == struct.pack('<f', .1 * Scalar(reference).item()).hex()
    assert audit['lambda_value'] == struct.unpack('<f', bytes.fromhex(audit['lambda_bits']))[0]
    rejects(lambda: checked(source, target, Index([0, 1, 2, 3]), regularization=.2,
                            reference_scale=Scalar(reference)), 'penalty')
    assert audit['lambda_bits'] != struct.pack('<f', .1 * reference / 5).hex()


def check_terminal_reader():
    """Real original predicates and archived 3min 745ms, no native import."""
    def load(name, filename):
        path = HERE / filename
        return fit.load_authenticated(name, path, hashlib.sha256(path.read_bytes()).hexdigest(), {})
    original = load('_signed_terminal_original', 'train_siglip2_substrate_adaptation.py')
    initializer = load('_signed_terminal_initializer', 'initialize_siglip2_substrate_fit.py')
    admission = original.FlatAdmission(); admission.init = initializer
    context = {'legacy': {'original': original, 'admission': admission},
               'guards': {original.__file__: fit.TERMINAL_SOURCE_SHA}}
    reader = fit.original_terminal_reader(context)
    node = copy.deepcopy(reader.__terminal_ast__)
    original_node = fit.terminal_ast(original, fit.TERMINAL_SOURCE_SHA, fit.TERMINAL_AST_SHA, {}, 'FlatAdmission')
    index = next(i for i, n in enumerate(original_node.body) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'match')
    node.body[index:index + 1] = copy.deepcopy(original_node.body[index:index + 3])
    assert ast.dump(node) == ast.dump(original_node)
    archive = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1'
    log = archive / 'fit-linear-v1.log'
    record = json.loads((archive / 'fit-linear-v1-receipt.json').read_bytes())
    proof = copy.deepcopy(fit.HISTORICAL_LINEAR['endpoint']['terminal'])
    proof['log']['path'] = str(log)
    lines = log.read_text().splitlines()
    assert lines.count('Service runtime: 3min 745ms') == 1
    assert fit.Decimal(3) * 60 + fit.Decimal(fit.runtime_components('3min 745ms')[1]) == fit.Decimal('180.745')
    rejects(lambda: admission.admit_terminal(record, proof, 300, {}), 'runtime format')
    footer = json.loads(next(line.removeprefix('FINAL_CGROUP ') for line in lines if line.startswith('FINAL_CGROUP ')))
    assert reader(admission, record, proof, 300, {}) == footer
    rejects(lambda: reader(admission, record, dict(proof, service_seconds=180.746), 300, {}), 'numeric binding')
    rejects(lambda: reader(admission, record, dict(proof, both_locks_held=False), 300, {}), 'both locks')
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'log'
        for content, message in (
            (log.read_text() + 'Service runtime: 3min 745ms\n', 'runtime line'),
            (log.read_text() + 'FINAL_CGROUP ' + json.dumps(footer) + '\n', 'footer'),
            (log.read_text().replace('\tExit status: 0', '\tExit status: 1'), 'normal-exit'),
            (log.read_text().replace('3min 745ms', '3min 744ms'), 'numeric binding'),
            (log.read_text().replace('3min 745ms', '3min 60000ms'), 'numeric binding'),
            (log.read_text().replace('3min 745ms', '3min 0.745ms'), 'runtime format'),
            (log.read_text().replace('"memory.swap.peak": "0"', '"memory.swap.peak": "1"'), 'memory/swap')):
            path.write_text(content)
            changed = dict(proof, log={'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            candidate_admission = original.FlatAdmission(); candidate_admission.init = initializer
            rejects(lambda: reader(candidate_admission, record, changed, 300, {}), message)
    function, code = original.FlatAdmission.admit_terminal, original.FlatAdmission.admit_terminal.__code__
    function.__code__ = (lambda *args: None).__code__
    try:
        rejects(lambda: fit.original_terminal_reader(context), 'function/body')
    finally:
        function.__code__ = code


def check_retention_correspondence():
    """Freeze all old predicates and the complete core timer, allowing only instrumentation/lifetime."""
    frozen = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/signed-concat-fit-source-v1'
    pins = json.loads((frozen / 'execution.json').read_bytes())
    for name in fit.FILES:
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == pins[name]
    assert pins['fit_siglip2_prototype_residual.py'] == '83e47dc0534d83e7e2848f453077109ec9f713a2efcba01a7f6e2e47f6a91ece'
    original = ast.parse((frozen / 'fit_siglip2_prototype_residual.py').read_bytes())
    current = ast.parse((HERE / 'fit_siglip2_prototype_residual.py').read_bytes())
    funcs = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
    old_funcs = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    assert funcs.keys() - old_funcs.keys() == {'retain_encoder', 'mark_phase', 'check_run_metadata'}

    class Instrumentation(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in (
                    'mark_phase', 'retain_encoder', 'check_run_metadata', 'context.update'):
                return None
            if (isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'require' and
                    isinstance(node.value.args[-1], ast.Constant) and
                    node.value.args[-1].value == 'encoder cleanup included cap exceeded'):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == 'progress':
                return None
            return self.generic_visit(node)

        def visit_Try(self, node):
            assert not node.handlers and not node.orelse
            assert ast.unparse(ast.Module(body=node.finalbody, type_ignores=[])) == (
                "mapping = context.pop('_encoder_mapping', None)\nif mapping is not None:\n    mapping.close()")
            return [n for item in node.body if (n := self.visit(item)) is not None]

        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not (isinstance(k, ast.Constant) and k.value in ('encoder_retention', 'phase_seconds'))]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)

    changed = {'run', 'fit_run', 'exit_rehash', 'check_terminal_record'}
    for name, node in old_funcs.items():
        candidate = copy.deepcopy(funcs[name])
        if name in changed:
            candidate = Instrumentation().visit(candidate)
        assert ast.dump(candidate) == ast.dump(node), name
    # Each marker brackets its original operation, never entering fresh()'s core timer.
    loop = next(n for n in funcs['fit_run'].body if isinstance(n, ast.For))
    operations = [(i, n) for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and
                  isinstance(n.value, ast.Call) and ast.unparse(n.value.func) in ('fresh', 'reload')]
    assert [ast.unparse(n.value.func) for _, n in operations] == ['fresh', 'fresh', 'reload']
    for (index, _), label in zip(operations, ('fresh_fit1', 'fresh_fit2', 'reload')):
        assert ast.unparse(loop.body[index - 1]) == "mark_phase(context, arm + '_" + label + "_begin')"
        assert ast.unparse(loop.body[index + 1]) == "mark_phase(context, arm + '_" + label + "_end')"
    # Existing constants, pins, imports, and entrypoint stay byte-for-byte AST identical.
    additions = {'ENCODER_CHECKPOINT', 'ENCODER_CHECKPOINT_BYTES', 'ENCODER_RETENTION_MAX_BYTES'}
    filtered = [n for n in current.body if not isinstance(n, ast.FunctionDef) and
                not (isinstance(n, ast.Import) and ast.unparse(n) == 'import mmap') and
                not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in additions)]
    assert ast.dump(ast.Module(body=filtered, type_ignores=[])) == ast.dump(ast.Module(
        body=[n for n in original.body if not isinstance(n, ast.FunctionDef)], type_ignores=[]))


def check_encoder_retention():
    """Real small readonly mappings: catch expanded hotset, stale SHA, partial touch and leaks."""
    assert fit.ENCODER_CHECKPOINT == {
        'path': '/home/riomus/runs/sfora-native256-source-cpu-so400-v4/fresh_vision.pt',
        'sha256': '5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189'}
    assert fit.ENCODER_CHECKPOINT_BYTES == 1711945083 and fit.ENCODER_RETENTION_MAX_BYTES == 2 * 1024**3
    page = os.sysconf('SC_PAGESIZE'); size = 1024**2 + page + 1
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'fresh_vision.pt'; raw = b'a' * size; path.write_bytes(raw)
        fact = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        mappings, opened = [], []
        fail_population = False

        class Pages:
            def __init__(self, fd, length, access):
                assert Path(os.readlink('/proc/self/fd/' + str(fd))) == path
                assert length == size and access == mmap.ACCESS_READ
                self.real = mmap.mmap(fd, length, access=access); self.offsets = []
                opened.append(fd); mappings.append(self)

            def __getitem__(self, offset):
                self.offsets.append(offset)
                if fail_population and offset == page:
                    raise OSError('population failed')
                return self.real[offset]

            def close(self):
                self.real.close()

        namespace = {**vars(fit), 'ENCODER_CHECKPOINT': fact, 'ENCODER_CHECKPOINT_BYTES': size,
            'mmap': SimpleNamespace(mmap=Pages, ACCESS_READ=mmap.ACCESS_READ)}
        retain = FunctionType(fit.retain_encoder.__code__, namespace)

        def context(descriptor=fact):
            legacy = {}
            def owned(value):
                assert value is legacy
                return SimpleNamespace(materialize=lambda: {'checkpoint': copy.deepcopy(descriptor)})
            return {'legacy': legacy, 'old': SimpleNamespace(owned_encoder=owned), 'guards': {}}

        c = context(); retain(c); kept = c['_encoder_mapping']
        assert mappings == [kept] and kept.offsets == list(range(0, size, page))
        assert kept.real[-1] == ord('a') and c['guards'] == {str(path): fact['sha256']}
        assert c['encoder_retention'] == {**fact, 'bytes': size, 'max_bytes': 2 * 1024**3,
            'page_bytes': page, 'pages_populated': (size + page - 1) // page, 'access': 'read-only'}
        rejects(lambda: os.fstat(opened[-1]))
        try:
            kept.real[0] = ord('b')
        except TypeError:
            pass
        else:
            raise AssertionError('retained encoder is writable')
        rejects(lambda: retain(c), 'already retained')
        # Same inode/size/restored mtime, including a byte beyond the first SHA chunk.
        stat = path.stat()
        for offset in (0, page + 7, size - 1):
            with path.open('r+b') as stream:
                stream.seek(offset); stream.write(b'b')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert path.stat().st_mtime_ns == stat.st_mtime_ns
            rejects(lambda: fit.bound_file(c['guards'], path, fact['sha256']), 'SHA256')
            assert not kept.real.closed
            path.write_bytes(raw)
        kept.close()
        for descriptor in (dict(fact, sha256='0' * 64), dict(fact, extra=True),
                           dict(fact, path='fresh_vision.pt'), dict(fact, path=str(path.parent / 'warm.pt')),
                           dict(fact, path=str(path.parent / 'resume.pt'))):
            rejects(lambda: retain(context(descriptor)), 'original encoder checkpoint')
        link = path.parent / 'alias.pt'; link.symlink_to(path)
        rejects(lambda: retain(context(dict(fact, path=str(link)))), 'original encoder checkpoint')
        actual = path.parent / 'actual.pt'; path.rename(actual); path.symlink_to(actual)
        rejects(lambda: retain(context()), 'canonical file')
        path.unlink(); actual.rename(path)
        for actual_size in (0, size - 1, size + 1):
            path.write_bytes(b'a' * actual_size)
            descriptor = dict(fact, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            resized = FunctionType(retain.__code__, {**namespace, 'ENCODER_CHECKPOINT': descriptor})
            rejects(lambda: resized(context(descriptor)), 'retention size')
        path.write_bytes(raw)
        limited = FunctionType(retain.__code__, {**namespace, 'ENCODER_RETENTION_MAX_BYTES': size - 1})
        rejects(lambda: limited(context()), 'retention size')
        path.write_bytes(b'b' * size)
        rejects(lambda: retain(context()), 'SHA256')
        path.write_bytes(raw)
        assert len(mappings) == 1, 'rejection created an extra mapping'
        fail_population = True
        failed = context(); rejects(lambda: retain(failed), 'population failed')
        assert mappings[-1].real.closed and '_encoder_mapping' not in failed
        rejects(lambda: os.fstat(opened[-1]))
        fail_population = False

        def no_mapping(fd, length, access):
            opened.append(fd)
            raise OSError('mapping failed')
        unavailable = FunctionType(retain.__code__, {**namespace,
            'mmap': SimpleNamespace(mmap=no_mapping, ACCESS_READ=mmap.ACCESS_READ)})
        failed = context(); rejects(lambda: unavailable(failed), 'mapping failed')
        assert '_encoder_mapping' not in failed and len(mappings) == 2
        rejects(lambda: os.fstat(opened[-1]))
        check_retention_lifetime(namespace, retain, context, mappings, path)


def check_retention_lifetime(namespace, retain, make_context, mappings, path):
    """Execute real run/exit/phase orchestration with native boundaries stubbed; never qualification."""
    phases = ['authority_begin', 'authority_end', 'encoder_retention_begin', 'encoder_retention_end']
    for arm in ('linear', 'concat'):
        for part in ('fresh_fit1', 'fresh_fit2', 'reload'):
            phases += [arm + '_' + part + '_begin', arm + '_' + part + '_end']
    phases += ['old_exit_begin', 'old_exit_end', 'own_union_begin', 'own_union_end', 'closure_begin', 'closure_end']
    for failure in (None, 'authority', 'cgroup', 'interpreter', 'body', 'publish', 'peak'):
        c = make_context(); events, messages = [], []
        original_count = len(mappings)
        args = SimpleNamespace(authority=path, authority_sha256='a' * 64, execution_sha256='b' * 64,
            phase='cpu', arm='linear', output=path.parent / ('output-' + str(failure)))
        prior = {'python': str(Path(sys.executable).resolve()), 'python_sha256': 'c' * 64, 'python_version': sys.version}

        def boundary(name):
            events.append(name)
            if failure == name:
                raise ValueError(name)

        def live(name):
            boundary(name)
            assert not c['_encoder_mapping'].real.closed

        def authority(value):
            assert value is args
            boundary('authority')
            return c

        def cgroup():
            boundary('cgroup')
            peak = '1' if failure == 'peak' and '_encoder_mapping' in c else '100'
            return {'path': '/sys/fs/cgroup/test.service', 'values': {'memory.peak': peak}}

        def interpreter(value):
            boundary('interpreter')
            return prior['python_sha256']

        def body(value):
            assert value is c
            live('body')
            for name in phases[4:-6]:
                fit.mark_phase(c, name)
            return {}

        def publish(extract, target, receipt):
            assert target == args.output / 'receipt.json'
            live('publish')

        c['legacy'].update(source_driver=SimpleNamespace(cgroup_memory=cgroup), guards={}, origins={}, terminal_cgroups={},
            selected={'source_cpu': {'invocation': prior}}, invocations=set(), extract=SimpleNamespace(sha=interpreter),
            admission=SimpleNamespace(init=SimpleNamespace(admit_cgroup=lambda *a: None, write_json=publish)))
        c['old'].zero_events = lambda value: None
        c['old'].exit_rehash = lambda value: live('old_exit')
        c.update(args=args, root=path.parent, code={}, launch={}, source={}, flags={}, terminal_cgroups={})
        fake_torch = SimpleNamespace(random=SimpleNamespace(get_rng_state=lambda: SimpleNamespace(clone=lambda: 1)),
            equal=lambda *a: True, cuda=SimpleNamespace(is_initialized=lambda: False))
        def native_boundary(name, *a, **kw):
            return fake_torch if name == 'torch' else builtins.__import__(name, *a, **kw)
        env = {'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': '1' * 32}
        ns = {**namespace, 'authority': authority, 'cli': lambda *a: sys.argv, 'retain_encoder': retain,
            'fit_run': body, 'os': SimpleNamespace(environ=env, getpid=os.getpid),
            '__builtins__': {**vars(builtins), '__import__': native_boundary},
            'prepare_readout': lambda value: live('readout'), 'original_terminal_reader': lambda value: live('reader'),
            'closure': lambda *a: {}, 'print': lambda value, **kw: messages.append((json.loads(value), kw))}
        ns['mark_phase'] = FunctionType(fit.mark_phase.__code__, ns)
        # Body stub still uses the production emitter via its own isolated namespace.
        ns['fit_run'] = FunctionType(body.__code__, {**body.__globals__, 'fit': SimpleNamespace(mark_phase=ns['mark_phase'])},
                                     closure=body.__closure__)
        ns['exit_rehash'] = FunctionType(fit.exit_rehash.__code__, ns)
        validator = FunctionType(fit.check_run_metadata.__code__, ns)
        ns['check_terminal_record'] = lambda record, launch, phase, arm: validator(record, phase, arm)
        run = FunctionType(fit.run.__code__, ns)
        if failure is None:
            record = run(args)
            assert list(record['phase_seconds']) == phases
            assert [m['phase'] for m, kw in messages] == phases and all(kw == {'flush': True} for m, kw in messages)
            assert all(m['event'] == 'PROTOTYPE_PHASE' for m, kw in messages)
            assert [m['seconds'] for m, kw in messages] == list(record['phase_seconds'].values())
            rejects(lambda: ns['mark_phase'](c, 'authority_begin'), 'duplicate phase')
            validator(json.loads(json.dumps(record, sort_keys=True)), 'cpu', 'linear')
            for mutate in (
                lambda r: r['phase_seconds'].pop('old_exit_end'),
                lambda r: r['phase_seconds'].update(unexpected=0.),
                lambda r: r['phase_seconds'].update(authority_end=-1.),
                lambda r: r['phase_seconds'].update(authority_end=float('nan')),
                lambda r: r['phase_seconds'].update(closure_end=r['wall_seconds'] + 1),
                lambda r: r['encoder_retention'].update(pages_populated=1),
                lambda r: r['encoder_retention'].update(path='/warm.pt'),
                lambda r: r['encoder_retention'].update(sha256='0' * 64),
                lambda r: r['encoder_retention'].update(bytes=0),
                lambda r: r['encoder_retention'].update(max_bytes=3 * 1024**3),
                lambda r: r['encoder_retention'].update(access='write'),
                lambda r: r['input_guards'].clear(),
            ):
                bad = copy.deepcopy(record); mutate(bad); rejects(lambda: validator(bad, 'cpu', 'linear'))
        else:
            rejects(lambda: run(args), 'resource cap' if failure == 'peak' else failure)
        if failure in ('authority', 'cgroup', 'interpreter'):
            assert len(mappings) == original_count
        else:
            assert len(mappings) == original_count + 1 and mappings[-1].real.closed
            assert '_encoder_mapping' not in c
    assert not any(n.split('.')[0] in fit.NATIVE for n in sys.modules)


def check():
    assert not any(n.split('.')[0] in fit.NATIVE for n in sys.modules)
    evidence = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7'
    for name, sha in fit.ORIGINAL_CODE.items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == sha
    assert hashlib.sha256((evidence / 'execution.json').read_bytes()).hexdigest() == fit.ORIGINAL_REFERENCE['execution_sha256']
    assert hashlib.sha256((evidence / 'authority-cpu-v7.json').read_bytes()).hexdigest() == fit.ORIGINAL_CPU['authority']['sha256']
    assert hashlib.sha256((evidence / 'cpu-v7-receipt.json').read_bytes()).hexdigest() == fit.ORIGINAL_CPU['terminal']['receipt']['sha256']
    check_repeat_preparation(evidence)
    check_signed_readout()
    check_terminal_reader()
    check_retention_correspondence()
    check_encoder_retention()
    original_launch = json.loads((evidence / 'authority-cpu-v7.json').read_bytes())
    launch = dict(schema=fit.AUTHORITY_SCHEMA, execution_sha256='a' * 64, phase='cpu', arm='linear',
                  original_reference=copy.deepcopy(fit.ORIGINAL_REFERENCE), original_cpu=copy.deepcopy(fit.ORIGINAL_CPU),
                  ridge_solver={'path': str(ROOT / 'src/sfora/foundation_adapter.py'), 'sha256': fit.SOLVER_SHA},
                  warm_start=original_launch['warm_start'], partition=original_launch['partition'],
                  historical_linear=copy.deepcopy(fit.HISTORICAL_LINEAR),
                  recipe=copy.deepcopy(fit.RECIPE), resource_policy=fit.policy('cpu'), both_locks_held=True, selected_cpu=None)
    args = SimpleNamespace(execution_sha256='a' * 64, phase='cpu', arm='linear')
    fit.check_launch(launch, args)
    for mutate in (
        lambda v: v.update(schema='siglip2-quadratic-readout-launch-v1'),
        lambda v: v.update(schema='siglip2-prototype-residual-launch-v1'),
        lambda v: v.update(arm='quadratic'),
        lambda v: v['historical_linear']['endpoint']['checkpoint'].update(sha256='0' * 64),
        lambda v: v['recipe']['basis'].update(concat='[H0raw128,Z32]'),
        lambda v: v.update(seed=179061),
        lambda v: v.update(phase='mechanics'),
        lambda v: v['original_cpu']['authority'].update(sha256='0' * 64),
        lambda v: v['original_reference'].update(execution_sha256='0' * 64),
        lambda v: v['ridge_solver'].update(sha256='0' * 64),
        lambda v: v['recipe'].update(rows=6354),
        lambda v: v['recipe'].update(intercept=True),
        lambda v: v.update(both_locks_held=False),
        lambda v: v['resource_policy'].update(swap_bytes=1),
    ):
        bad = copy.deepcopy(launch)
        mutate(bad)
        rejects(lambda: fit.check_launch(bad, args))
    fit_launch = dict(launch, phase='fit', resource_policy=fit.policy('fit'), selected_cpu=fit.ORIGINAL_CPU['terminal'])
    rejects(lambda: fit.check_launch(fit_launch, SimpleNamespace(execution_sha256='a' * 64, phase='fit', arm='linear')),
            'old CPU cannot qualify')
    rejects(lambda: fit.strict_json('{"schema":1,"schema":2}'))
    rejects(lambda: fit.strict_json('{"value":NaN}'))
    witness = dict(rows=6355, classes=1008, coefficients=4096, feature_width=32, arm='linear', feature_energy=4.,
                   reference_energy=4., reference_scale=.125, lambda_bits=struct.pack('<f', .0125).hex(),
                   actual_solve_checked=True, **{'lambda': struct.unpack('<f', struct.pack('<f', .0125))[0]}, stationarity_numerator=0., stationarity_denominator=3.,
                   normalized_stationarity=0., A_nonzero=True, target='raw member-inclusive prototypes', intercept=False)
    fit.check_fit_witness(witness, 'linear')
    for fields in ({'feature_energy': 0.}, {'reference_energy': 0.}, {'reference_scale': .2},
                   {'lambda_bits': '00000000'}, {'actual_solve_checked': False}, {'feature_width': 160},
                   {'lambda': .0125 * 32 / 160}, {'A_nonzero': False}, {'intercept': True},
                   {'stationarity_denominator': 0.}, {'lambda': float('nan')},
                   {'stationarity_numerator': 1., 'normalized_stationarity': 1. / 3}):
        rejects(lambda: fit.check_fit_witness(dict(witness, **fields), 'linear'))

    torch = SimpleNamespace(float32='float32', int64='int64', is_tensor=lambda v: isinstance(v, (Matrix, Index)),
        isfinite=lambda v: Mask([True for row in v.rows for _ in row]), unique=lambda v: Index(sorted(set(v.values))),
        trace=lambda v: sum(v.rows[i][i] for i in range(v.shape[0])),
        eye=lambda n, **kw: Matrix([[float(i == j) for j in range(n)] for i in range(n)]),
        linalg=SimpleNamespace(solve=solve))
    solver_path = ROOT / 'src/sfora/foundation_adapter.py'
    before = solver_path.read_bytes()
    solver = fit.extract_solver(solver_path, fit.SOLVER_SHA, torch)
    source = Matrix([[1., 2.], [3., 1.], [2., 5.], [4., 4.]])
    target = Matrix([[10., 8.], [12., 9.], [9., 10.], [13., 11.]])
    model = solver(source, target, Index([0, 1, 2, 3]), regularization=0.1)
    x, y = source - source.mean(0), target - target.mean(0)
    gram = x.T @ x
    regularized = gram + (0.1 * torch.trace(gram) / 2) * torch.eye(2)
    near(regularized @ model.weight, x.T @ y)
    A = model.weight.T
    near(x @ A.T, x @ model.weight)
    assert model.target_mean.rows == [[11., 9.5]]
    # Execute the v2 signed primitive correction expression, with fake F;
    # a nonzero target mean exposes accidental model.transform/intercept use.
    primitive = ast.parse((HERE / 'prototype_residual_readout.py').read_bytes())
    raw = next(n for n in primitive.body if isinstance(n, ast.FunctionDef) and n.name == 'raw_features')
    assignment = next(n for n in ast.walk(raw) if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets))
    h0 = Matrix([[2., 3.], [4., 5.], [6., 7.], [8., 9.]])
    corrected = eval(compile(ast.Expression(assignment.value), '<qualified primitive>', 'eval'),
                     {'h0': h0, 'phi': SimpleNamespace(detach=lambda: x), 'A': A,
                      'F': SimpleNamespace(linear=lambda value, weight: value @ weight.T)})
    near(corrected, h0 + x @ model.weight)
    assert corrected.rows != (h0 + model.transform(source)).rows
    assert solver_path.read_bytes() == before
    rejects(lambda: fit.extract_solver(solver_path, '0' * 64, torch))
    rejects(lambda: solver(source, target, Index([0, 0, 2, 3]), regularization=0.1))
    rejects(lambda: solver(source, target, Index([0, 1, 2, 3]), regularization=0.))

    tree = ast.parse((HERE / 'fit_siglip2_prototype_residual.py').read_bytes())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = lambda name: [ast.unparse(n.func) for n in ast.walk(functions[name]) if isinstance(n, ast.Call)]
    assert 'solver' in calls('fit_prototype_residual')
    assert 'check_reference_scale' in calls('fit_prototype_residual')
    assert 'historical_linear_parity' in calls('fit_run')
    assert 'prepare_readout' in calls('exit_rehash')
    assert 'prepare_readout' in calls('raw_features')
    assert 'prepare_readout' in calls('zero_source_witness')
    assert 'prepare_readout' in calls('integrity')
    assert 'prepare_readout' in calls('check_payload')
    assert 'observed_solve' in ast.unparse(functions['checked_solver'])
    assert 'actual_lambda' in ast.unparse(functions['checked_solver'])
    assert "tensor.data[index]" in ast.unparse(functions['tamper_witness'])
    assert "tensor.data.reshape" not in ast.unparse(functions['tamper_witness'])
    assert not any('transform' in name for name in calls('fit_prototype_residual'))
    assert 'reconstruct' in calls('reload') and not {'fresh', 'fit_prototype_residual', 'extract_solver'}.intersection(calls('reload'))
    assert 'extract_solver' not in calls('prepare_native')
    assert 'torch.optim' not in ast.unparse(tree)
    assert not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and
                   isinstance(t.value, ast.Name) and t.value.id == 'old' for t in n.targets) for n in ast.walk(tree))
    assert not any(n.split('.')[0] in fit.NATIVE for n in sys.modules)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / 'bytes'
        path.write_bytes(b'accepted source')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        guards = {}
        fit.bound_file(guards, path, sha)
        path.write_bytes(b'changed! source')
        rejects(lambda: fit.bound_file(guards, path, sha), 'current file SHA256')
        # Run actual authority against an exact new closure and a corrupt launch:
        # rejection precedes all original helper/native imports.
        for name in fit.FILES:
            (root / name).write_bytes((HERE / name).read_bytes())
        code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in fit.FILES}
        (root / 'execution.json').write_text(json.dumps(code))
        execution_sha = hashlib.sha256((root / 'execution.json').read_bytes()).hexdigest()
        malformed = dict(launch, execution_sha256=execution_sha, selected_cpu=fit.ORIGINAL_CPU['terminal'])
        (root / 'authority.json').write_text(json.dumps(malformed))
        authority_sha = hashlib.sha256((root / 'authority.json').read_bytes()).hexdigest()
        original_file = fit.__file__
        try:
            fit.__file__ = str(root / 'fit_siglip2_prototype_residual.py')
            rejects(lambda: fit.authority(SimpleNamespace(execution_sha256=execution_sha,
                authority=root / 'authority.json', authority_sha256=authority_sha, phase='cpu', arm='linear',
                output=root / 'new-output')), 'prerequisite')
        finally:
            fit.__file__ = original_file
    assert '_prototype_original' not in sys.modules
    saved = {'A': struct.pack('!f', 1.), 'source_mean': [2.], 'target_mean': [3.],
             'prototypes': [4.], 'counts': [5], 'head': b'complete warm head', 'labels': [0, 1]}
    digest = typed_hash(saved)
    fit.verify_digest(typed_hash, copy.deepcopy(saved), digest, 'reload/current bytes changed')
    for key, value in [('A', struct.pack('!f', 9.)), ('source_mean', [9.]), ('target_mean', [9.]),
                       ('prototypes', [9.]), ('counts', [9]), ('head', b'corrupt'), ('labels', [1, 0])]:
        bad = copy.deepcopy(saved)
        bad[key] = value
        rejects(lambda: fit.verify_digest(typed_hash, bad, digest, 'reload/current bytes changed'))
    assert typed_hash({'label': 1}) != typed_hash({'label': 1.})
    check_adapter(solver_path, torch, source, target)
    print('PASS: signed basis/width/schema/lambda/H0-byte/target-mean/adapter falsifiers, historical duration/predicates, helper integrity, readonly encoder retention/lifetime/phase metadata and frozen AST checks; native FP32 UNRUN')


if __name__ == '__main__':
    check()
