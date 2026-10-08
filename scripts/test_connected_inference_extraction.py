#!/usr/bin/env python3
"""Source-only extraction oracle; native runtime and performance remain unqualified."""

import ast
import hashlib
import importlib.abc
import importlib.util
import json
import subprocess
import symtable
import builtins
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/sfora/connected_inference.py'
CLOSURE = {
    'scripts/train_siglip2_connected_mlp.py': (
        'require', 'parameter_roles', 'mapping_absent', 'strict_json', 'file_fact',
        'bound_file', 'batch_bound_files', 'read_json', 'fullfeature_raw_features',
        '_processor_cache', 'apply_overlay', 'model_structure', 'construct_encoder',
        'encoder_facts', 'admit_bundle', 'inference_readout_tree', 'owned_copy',
        'load_inference', 'inference_outputs', 'release_inference'),
    'scripts/train_siglip2_substrate_adaptation.py': ('fingerprint', 'CheckpointPages', 'load_vision'),
    'scripts/qualify_siglip2_substrate_cpu.py': (
        'numerical_flags', 'loaded_module_origin', 'loaded_origins', 'module_origin', 'construct'),
    'scripts/train_siglip2_cached_readout.py': ('head_from',),
    'scripts/prototype_residual_readout.py': ('feature_width', 'basis', 'check_weight', 'check_means', 'raw_features'),
    'scripts/quadratic_readout.py': ('_check_tensor', '_check_base', '_check_features', '_finite'),
}


# These finite substitutions are the complete permitted difference from the sources.
PLUMBING = (
    ('original.fingerprint', 'fingerprint'), ('original.CheckpointPages', 'CheckpointPages'),
    ('original.load_vision', 'load_vision'), ('source.module_origin', 'module_origin'),
    ('source.construct', 'construct'), ('source.numerical_flags', 'numerical_flags'),
    ('primitive._check_tensor', '_check_tensor'), ('primitive._check_features', '_check_features'),
    ('primitive._check_base', '_check_base'), ('primitive._finite', '_finite'),
    ('primitive._require', 'require'), ('readout.raw_features', 'raw_features'),
    ('_require(', 'require('),
    ('def model_structure(model, source, packages):', 'def model_structure(model, packages):'),
    ('model_structure(model,source,', 'model_structure(model,'),
    ('def encoder_facts(state, original, source, packages, *, serving=False):',
     'def encoder_facts(state, packages, *, serving=False):'),
    ('encoder_facts(endpoint,original,source,', 'encoder_facts(endpoint,'),
    ('def construct_encoder(source, original, construct_context, config, buffers, processor_config, base,',
     'def construct_encoder(construct_context, config, buffers, processor_config, base,'),
    ('overlay=None, source_runtime=None):', 'overlay=None):'),
    ('construct_encoder(source,original,construct_context,', 'construct_encoder(construct_context,'),
    ('def fullfeature_raw_features(features, head, A, means, C, mu_train, arm, primitive, readout):',
     'def fullfeature_raw_features(features, head, A, means, C, mu_train, arm):'),
    ("raw_features(features, head, A, means, 'concat', primitive)",
     "raw_features(features, head, A, means, 'concat')"),
    ('def check_weight(A, device, arm, primitive):', 'def check_weight(A, device, arm):'),
    ('def check_means(means, device, primitive):', 'def check_means(means, device):'),
    ('def raw_features(features, base, A, means, arm, primitive):',
     'def raw_features(features, base, A, means, arm):'),
    ('check_weight(A, device, arm, primitive)', 'check_weight(A, device, arm)'),
    ('check_means(means, device, primitive)', 'check_means(means, device)'),
)


def original_definition(path, name):
    source = (ROOT / path).read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    return ast.get_source_segment(source, node)


def expected_definition(path, name):
    source = original_definition(path, name)
    if name == 'construct_encoder':
        start = source.index('        if source_runtime is not None:')
        end = source.index("        require('position_ids'", start)
        source = source[:start] + source[end:]
        start = source.index('        require(source_runtime is None or structure')
        end = source.index("        base =", start)
        source = source[:start] + source[end:]
    if name == 'construct':
        source = source.replace(
            "path in context['guards'] and context['extract'].sha(path) == context['guards'][path]",
            "path in context['guards']")
        source = source.replace(
            "                'loaded vision/config source hash differs')",
            "                'loaded vision/config source hash differs')\n        bound_file({},path,context['guards'][path])")
    if name == 'admit_bundle':
        source = source.replace("    directory,guards = Path(directory),{}",
            "    _check_runtime()\n    directory,guards = Path(directory),{}")
        source = source.replace(
            "require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest['code']['train_siglip2_connected_mlp.py'],\n            'current copied public loader source differs')",
            "require(tuple(sorted(manifest['code'].items())) == _binding[0],\n            'unsupported historical inference closure')")
        source = source.replace("    return manifest,guards",
            "    for path,sha in _binding[1]:\n        bound_file(guards,path,sha)\n    return manifest,guards")
    if name == 'load_inference':
        start = source.index("    modules,prefix =")
        end = source.index('    import torch', start)
        source = source[:start] + "    modules = {'runtime':sys.modules[__name__]}\n" + source[end:]
        source = source.replace(
            "{'packages':env['packages'],'guards':guards,'extract':extract,",
            "{'packages':env['packages'],'guards':guards,")
        source = source.replace("modules['train_siglip2_cached_readout.py'].head_from", 'head_from')
        source = source.replace("'manifest':manifest}", "'manifest':manifest,'directory':directory}")
        source = source.replace('Copied public loader uses only owned bundle files and installed packages.',
                                'Installed public loader uses authenticated historical evidence without executing it.')
    if name == 'inference_outputs':
        source = source.replace("    original,source = modules['train_siglip2_substrate_adaptation.py'],modules['qualify_siglip2_substrate_cpu.py']\n", "    _check_runtime()\n")
        source = source.replace("    require(0 < len(images)",
            "    for filename in SERVING_FILES | {'joint_relational_compaction.py'}:\n        path = endpoint['directory']/filename\n        bound_file({},path,endpoint['guards'][str(path)])\n    require(0 < len(images)")
        source = source.replace("endpoint['mu_train'],endpoint['arm'],modules['quadratic_readout.py'],modules['prototype_residual_readout.py'])",
                                "endpoint['mu_train'],endpoint['arm'])")
        source = source.replace("modules['joint_relational_compaction.py'].pack_int8_unit_embeddings(unit.cpu())",
                                "_pack(unit.cpu())")
    if name == '_check_base':
        source = source.replace(
            'Path(method.__code__.co_filename).name == "train_siglip2_cached_readout.py"',
            'method.__code__ == _head_method_code(name) and method.__globals__ is globals() and\n                 method.__defaults__ is None and method.__kwdefaults__ is None and\n                 Path(method.__code__.co_filename) == Path(__file__)')
    for before, after in PLUMBING:
        source = source.replace(before, after)
    return source


def dependency_check():
    source = RUNTIME.read_text()
    tree = ast.parse(source)
    allowed_imports = {'copy', 'gc', 'hashlib', 'inspect', 'json', 'math', 'os', 're',
                       'sys', 'weakref', 'concurrent.futures', 'functools', 'pathlib', 'types'}
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef, ast.Expr))
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed_imports for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed_imports
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    table = symtable.symtable(source, str(RUNTIME), 'exec')
    defined = set(table.get_identifiers()) | {'__file__', '__name__'} | set(vars(builtins))
    def walk(scope):
        for symbol in scope.get_symbols():
            assert not symbol.is_referenced() or not symbol.is_global() or symbol.get_name() in defined, symbol.get_name()
        for child in scope.get_children():
            walk(child)
    walk(table)
    for module_name, path in (('_bridge_import_subject', ROOT / 'src/sfora/connected_compact_serving.py'),):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    # The shared packed module is imported by both public serving paths, never copied.
    bridge = ast.parse((ROOT / 'src/sfora/connected_compact_serving.py').read_text())
    imports = [node for node in ast.walk(bridge) if isinstance(node, ast.ImportFrom)
               and any(alias.name == 'PackedInt8Embeddings' for alias in node.names)]
    assert len(imports) == 2 and all(node.module == 'sfora.packed_int8' for node in imports)


def correspondence():
    nodes = {node.name: node for node in ast.parse(RUNTIME.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    names = {name for symbols in CLOSURE.values() for name in symbols}
    assert nodes.keys() == names | {'_bind_runtime', '_check_runtime', '_head_method_code', '_pack'}
    for path, symbols in CLOSURE.items():
        for name in symbols:
            expected = ast.parse(expected_definition(path, name)).body[0]
            assert ast.dump(nodes[name], include_attributes=False) == ast.dump(expected, include_attributes=False), name


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision'}:
            raise AssertionError('native import during source check: ' + fullname)
        if 'train_siglip2' in fullname or 'qualify_siglip2' in fullname:
            raise AssertionError('historical execution: ' + fullname)


def subject():
    assert RUNTIME.is_file(), 'packaged inference closure is missing'
    spec = importlib.util.spec_from_file_location('_extraction_subject', RUNTIME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reject(call, fragment=None):
    try:
        call()
    except (ValueError, OSError, TypeError) as error:
        assert fragment is None or fragment in str(error), str(error)
        return error
    raise AssertionError('invalid inference authority accepted')


def ledger_check():
    source = (ROOT / 'src/sfora/_connected_inference_authority.py').read_text()
    tree = ast.parse(source)
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)}
    assert all(isinstance(node, (ast.Assign, ast.Expr)) for node in tree.body)
    assert values['SCHEMA'] == 'sfora-connected-inference-extraction-v1'
    runtime = RUNTIME.read_text()
    nodes = {node.name: node for node in ast.parse(runtime).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    ledger = values['SOURCE_SYMBOLS']
    assert {(path, name) for path, file_sha, name, *rest in ledger} == {
        (path, name) for path, symbols in CLOSURE.items() for name in symbols}
    import difflib
    differences = []
    for path, file_sha, name, source_sha, source_ast_sha, runtime_sha, runtime_ast_sha in ledger:
        original = original_definition(path, name)
        extracted = ast.get_source_segment(runtime, nodes[name])
        astsha = lambda text: sha(ast.dump(ast.parse(text), include_attributes=False).encode())
        assert sha((ROOT / path).read_bytes()) == file_sha
        assert (sha(original.encode()), astsha(original), sha(extracted.encode()), astsha(extracted)) == (
            source_sha, source_ast_sha, runtime_sha, runtime_ast_sha), name
        if original != extracted:
            differences.append((name, ''.join(difflib.unified_diff(
                original.splitlines(True), extracted.splitlines(True), fromfile=path+':'+name,
                tofile='connected_inference.py:'+name))))
    assert values['SUBSTITUTIONS'] == tuple(differences)
    assert 'BRIDGE_SHA256' not in values, 'authority cycle'
    if 'RUNTIME_SHA256' in values:
        assert values['RUNTIME_SHA256'] == sha(RUNTIME.read_bytes())
        assert values['PACKED_SHA256'] == sha((ROOT / 'src/sfora/packed_int8.py').read_bytes())
    assert 'RUNTIME_SHA256' not in runtime and 'AUTHORITY_SHA256' not in runtime
    return values


def admission_check(values):
    with tempfile.TemporaryDirectory(prefix='connected-extraction-') as scratch:
        root = Path(scratch)
        installed = root / 'installed'
        installed.mkdir()
        for name in ('connected_inference.py', '_connected_inference_authority.py', 'packed_int8.py'):
            (installed / name).write_bytes((ROOT / 'src/sfora' / name).read_bytes())
        path = installed / 'connected_inference.py'
        spec = importlib.util.spec_from_file_location('_extraction_admission', path)
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        try:
            spec.loader.exec_module(runtime)
            guards = tuple((str(installed / name), sha((installed / name).read_bytes()))
                           for name in ('connected_inference.py', '_connected_inference_authority.py', 'packed_int8.py'))
            bad = ((guards[0][0], '0' * 64), *guards[1:])
            reject(lambda: runtime._bind_runtime(values['HISTORICAL_CODE'], bad), 'current FILE bytes differ')
            assert runtime._binding is None
            runtime._bind_runtime(values['HISTORICAL_CODE'], guards)
            reject(lambda: runtime._bind_runtime(values['HISTORICAL_CODE'], guards), 'binding required')
            bundle = root / 'bundle'
            bundle.mkdir()
            old = subprocess.check_output(['git', 'show', 'c6eb09e3^:src/sfora/joint_relational_compaction.py'], cwd=ROOT)
            for name, digest in values['HISTORICAL_CODE']:
                raw = old if name == 'joint_relational_compaction.py' else (ROOT / 'scripts' / name).read_bytes()
                assert sha(raw) == digest
                (bundle / name).write_bytes(raw)
            for name in ('vision.pt', 'endpoint.pt', 'processor.json'):
                (bundle / name).write_bytes(b'owned evidence')
            constructor = installed / 'constructor.py'
            constructor.write_bytes(b'# constructor evidence\n')
            manifest = {'schema': runtime.BUNDLE_SCHEMA, 'code': dict(values['HISTORICAL_CODE']),
                'files': {name: sha((bundle / name).read_bytes()) for name in ('vision.pt', 'endpoint.pt', 'processor.json')},
                'endpoint_state_sha256': '1' * 64, 'base_vision_sha256': '2' * 64, 'vision_sha256': '3' * 64,
                'encoder_identity': {}, 'scope': {'arm': 'control', 'manifest_sha256': runtime.SCOPE_SHA256,
                    'arm_sha256': runtime.CONTROL_SHA256},
                'environment': {'packages': {name: {'root': str(installed)} for name in runtime.NATIVE - {'sfora'}},
                    'files': {str(constructor): sha(constructor.read_bytes())}, 'native_files': {},
                    'vision_constructor': str(constructor)}}
            manifest_path = bundle / 'bundle.json'
            def write(value):
                raw = json.dumps(value).encode()
                manifest_path.write_bytes(raw)
                return sha(raw)
            digest = write(manifest)
            admitted, checked = runtime.admit_bundle(bundle, digest)
            assert admitted == manifest and all(p in checked for p, h in guards)
            for filename in ('connected_inference.py', '_connected_inference_authority.py', 'packed_int8.py'):
                file = installed / filename
                raw = file.read_bytes()
                file.write_bytes(raw + b'# changed\n')
                reject(lambda: runtime.admit_bundle(bundle, digest), 'current FILE bytes differ')
                file.write_bytes(raw)
            name = 'train_siglip2_connected_mlp.py'
            raw = (bundle / name).read_bytes()
            (bundle / name).write_bytes(raw + b"\nraise AssertionError('historical execution')\n")
            changed = manifest | {'code': manifest['code'] | {name: sha((bundle / name).read_bytes())}}
            bad_digest = write(changed)
            reject(lambda: runtime.load_inference(bundle, bad_digest, 'cuda'), 'unsupported historical inference closure')
            (bundle / name).write_bytes(raw)
            digest = write(manifest)
            reject(lambda: runtime.admit_bundle(bundle, '0' * 64), 'current FILE bytes differ')
            for changed in (manifest | {'extra': True}, manifest | {'schema': 'unknown'},
                            manifest | {'code': {}}, manifest | {'scope': {}}, manifest | {'environment': {}}):
                bad_digest = write(changed)
                reject(lambda: runtime.admit_bundle(bundle, bad_digest))
            write(manifest)
            reject(lambda: runtime.strict_json('{"a":1,"a":2}'), 'duplicate JSON key')
            reject(lambda: runtime.strict_json('{"a":NaN}'), 'nonfinite JSON')
            # A valid source-only manifest reaches the first native import, never historical execution.
            try:
                runtime.load_inference(bundle, digest, 'cuda')
            except AssertionError as error:
                assert str(error) == 'native import during source check: torch'
            else:
                raise AssertionError('source-only fixture constructed native runtime')
            head_check(runtime)
            serializer_check(runtime)
        finally:
            sys.modules.pop(spec.name, None)


def head_check(runtime):
    def cell(value):
        return (lambda: value).__closure__[0]
    methods = {}
    for name in ('forward', 'residual'):
        code = runtime._head_method_code(name)
        closure = tuple(cell('control' if key == 'arm' else object()) for key in code.co_freevars)
        methods[name] = FunctionType(code, vars(runtime), name, None, closure)
    shapes = {'primary.weight': (128, 1152), 'primary.bias': (128,),
              'down.weight': (32, 1152), 'up.weight': (128, 32)}
    tensor = lambda shape: SimpleNamespace(shape=shape, dtype='torch.float32', device='cpu',
        layout='torch.strided', requires_grad=False, grad_fn=None)
    params = {name: tensor(shape) for name, shape in shapes.items()}
    buffers = {'center': tensor((1152,)), 'preactivation_std': tensor(())}
    cls = type('Residual', (), methods | {'__qualname__': 'head_from.<locals>.Residual',
        'named_parameters': lambda self: params.items(), 'named_buffers': lambda self: buffers.items()})
    head = cls()
    assert runtime._check_base(head, 'cpu') == [*params.values(), *buffers.values()]
    head.forward = lambda value: value
    reject(lambda: runtime._check_base(head, 'cpu'), 'original control factory required')
    del head.forward
    residual = cls.residual
    closure = dict(zip(residual.__code__.co_freevars, residual.__closure__, strict=True))
    closure['arm'].cell_contents = 'candidate'
    reject(lambda: runtime._check_base(head, 'cpu'), 'control source factory required')
    closure['arm'].cell_contents = 'control'
    cls.residual = FunctionType(residual.__code__, dict(vars(runtime)), 'residual', None, residual.__closure__)
    reject(lambda: runtime._check_base(head, 'cpu'), 'original source method required')
    cls.residual = residual
    for name in methods:
        original = methods[name].__code__
        methods[name].__code__ = (lambda self, value: value).__code__.replace(
            co_filename=original.co_filename, co_qualname=original.co_qualname,
            co_freevars=original.co_freevars)
        reject(lambda: runtime._check_base(head, 'cpu'), 'original source method required')
        methods[name].__code__ = original
    params['primary.weight'].requires_grad = True
    reject(lambda: runtime._check_base(head, 'cpu'), 'frozen detached tensor required')
    params['primary.weight'].requires_grad = False
    buffers.pop('center')
    reject(lambda: runtime._check_base(head, 'cpu'), 'complete source parameters/buffers required')


def serializer_check(runtime):
    class Tensor:
        def __init__(self):
            self.raw, self.copies = bytearray(b'ab'), 0
            self._version, self.dtype, self.shape = 0, 'torch.float32', (2,)
        def data_ptr(self): return 17
        def detach(self): return self
        def cpu(self):
            self.copies += 1
            return self
        def contiguous(self): return self
        def reshape(self, width): return self
        def view(self, dtype): return self
        def numpy(self): return self.raw
    torch = ModuleType('torch')
    torch.Tensor, torch.uint8 = Tensor, object()
    tensor = Tensor()
    with patch.dict(sys.modules, {'torch': torch}):
        first = runtime.fingerprint({'a': tensor, 'b': [tensor]})
        assert tensor.copies == 2
        tensor.raw[0] = ord('z')
        second = runtime.fingerprint({'a': tensor, 'b': [tensor]})
        assert tensor.copies == 4 and first != second, 'current-byte fingerprint was cached'
        consumed = []
        assert runtime.fingerprint([tensor, tensor], consumed=consumed.append)
        assert consumed == [tensor, tensor] and tensor.copies == 6
        assert runtime.fingerprint((1, 2)) != runtime.fingerprint([1, 2])


def literal_pin_check():
    spec = importlib.util.spec_from_file_location('_literal_pin_bridge', ROOT / 'src/sfora/connected_compact_serving.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    path, digest = bridge._installed_authority()
    assert path == ROOT / 'src/sfora/_connected_inference_authority.py'
    assert path.is_absolute() and path.resolve() == path
    assert digest == '538291c1cf14ead854760ad9400ee01dacc2ad73dec5b4ae56677d18bfd9412e'
    raw = bridge._read_checked(path, digest)
    with tempfile.TemporaryDirectory(prefix='connected-pin-mutation-') as scratch:
        file = Path(scratch) / path.name
        file.write_bytes(raw)
        assert bridge._read_checked(file, digest) == raw
        file.write_bytes(raw + b'# mutated authority\n')
        reject(lambda: bridge._read_checked(file, digest), 'current file bytes differ')
        file.write_bytes(raw)
        alias = file.parent / 'authority-alias.py'
        alias.symlink_to(file)
        reject(lambda: bridge._read_checked(alias, digest), 'canonical regular file required')


def packing_snapshot_check():
    # Execute the original package class only with inert import stubs; no tensor
    # construction or numerical/type-identity qualification is claimed here.
    spec = importlib.util.spec_from_file_location('_packing_bridge_subject', ROOT / 'src/sfora/connected_compact_serving.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    packed_path = ROOT / 'src/sfora/packed_int8.py'
    packed_spec = importlib.util.spec_from_file_location('sfora.packed_int8', packed_path)
    packed = importlib.util.module_from_spec(packed_spec)
    stubs = {name: ModuleType(name) for name in ('torch', 'torch.nn', 'numpy')}
    stubs['torch.nn'].functional = ModuleType('torch.nn.functional')
    stubs['sfora.packed_int8'] = packed
    with patch.dict(sys.modules, stubs):
        packed_spec.loader.exec_module(packed)
        index = bridge.ConnectedCompactIndex()
        owner = ModuleType('_packing_snapshot_owner')
        owner.__file__ = str(RUNTIME)
        owner.__spec__ = importlib.util.spec_from_file_location(owner.__name__, RUNTIME)
        sys.modules[owner.__name__] = owner
        index._module = owner
        index._owned[owner.__name__] = owner
        index._shared = (packed,)
        authority = ROOT / 'src/sfora/_connected_inference_authority.py'
        index._guards = tuple((path, sha(path.read_bytes()), False) for path in (RUNTIME, authority, packed_path))
        index._snapshot(packed, packed_path.read_bytes())
        index._check_current()
        fn = packed.PackedInt8Embeddings.to_bytes
        original = fn.__code__
        fn.__code__ = (lambda self: b'').__code__
        reject(index._check_current, 'callable state changed')
        fn.__code__ = original
        original_torch = packed.torch
        packed.torch = object()
        reject(index._check_current, 'globals changed')
        packed.torch = original_torch
        original_method = packed.PackedInt8Embeddings.to_bytes
        packed.PackedInt8Embeddings.to_bytes = lambda self: b''
        reject(index._check_current, 'globals changed')
        packed.PackedInt8Embeddings.to_bytes = original_method
        index._check_current()
        index.close()
        assert sys.modules['sfora.packed_int8'] is packed and owner.__name__ not in sys.modules


def main():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    try:
        dependency_check()
        correspondence()
        values = ledger_check()
        admission_check(values)
        packing_snapshot_check()
        literal_pin_check()
        module = subject()
        try:
            for api, names in (
                ('load_inference', ('directory', 'bundle_sha256', 'device')),
                ('inference_outputs', ('endpoint', 'images')),
                ('release_inference', ('endpoint',)),
            ):
                fn = getattr(module, api)
                assert fn.__code__.co_varnames[:fn.__code__.co_argcount] == names
            try:
                module.load_inference(Path('/missing'), '0' * 64, 'cuda')
            except ValueError as error:
                assert 'authenticated installed runtime' in str(error)
            else:
                raise AssertionError('unbound runtime admitted')
        finally:
            sys.modules.pop(module.__name__, None)
    finally:
        sys.meta_path.remove(guard)
    print('connected inference source-only checks passed')


if __name__ == '__main__':
    main()
