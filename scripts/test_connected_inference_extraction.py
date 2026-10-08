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


def pipeline_record():
    tree = ast.parse((ROOT / 'src/sfora/_connected_inference_authority.py').read_text())
    substitutions = next(ast.literal_eval(n.value) for n in tree.body
                         if isinstance(n, ast.Assign) and n.targets[0].id == 'SUBSTITUTIONS')
    assert substitutions[-1][0] == '_fresh_cpu_sha_pipeline'
    return substitutions[-1][1]


def pipeline_inverse(source):
    record = pipeline_record()
    start = source.index('\n\ndef _sha_cpu_bytes(')
    end = source.index('\n\ndef fingerprint(', start)
    # The blind deletion below may remove exactly the two helper defs and nothing else.
    removed = ast.parse(source[start:end]).body
    assert len(removed) == 2 and all(type(n) is ast.FunctionDef for n in removed)
    assert [(n.name, n.decorator_list) for n in removed] == [
        ('_sha_cpu_bytes', []), ('_fingerprint_cuda_dict', [])]
    source = source[:start] + source[end:]
    for before, after in record['replacements']:
        assert source.count(after) == 1
        source = source.replace(after, before)
    return source


BRIDGE_FACTORY_INVERSE = (('def _literal_state(', 'def _installed_probe_authority() -> tuple[Path, str]:\n    return (\n        Path(__file__).absolute().parent / "_connected_probe_inference_authority.py",\n        "0e989dd087614499096512a22948f4e8d3f9bb2840a487f2e9fe7e2d9f0371ab",\n    )\n\n\ndef _literal_state(', 1), ('    @classmethod\n    def from_bundle(\n        cls,\n        *,\n        bundle_dir: Path,\n        expected_bundle_sha256: str,\n        gallery_path: Path,\n        expected_gallery_sha256: str,\n        gallery_count: int,\n        native_library_path: Path,\n        expected_native_library_sha256: str,\n    ) -> ConnectedCompactIndex:\n        """Load explicit caller-pinned bytes; device, dimensions and k are fixed."""\n', '    @classmethod\n    def from_bundle(\n        cls,\n        *,\n        bundle_dir: Path,\n        expected_bundle_sha256: str,\n        gallery_path: Path,\n        expected_gallery_sha256: str,\n        gallery_count: int,\n        native_library_path: Path,\n        expected_native_library_sha256: str,\n    ) -> ConnectedCompactIndex:\n        """Load an explicit MLP bundle; device, dimensions and k are fixed."""\n        return cls._from_bundle(\n            _probe=False,\n            bundle_dir=bundle_dir,\n            expected_bundle_sha256=expected_bundle_sha256,\n            gallery_path=gallery_path,\n            expected_gallery_sha256=expected_gallery_sha256,\n            gallery_count=gallery_count,\n            native_library_path=native_library_path,\n            expected_native_library_sha256=expected_native_library_sha256,\n        )\n\n    @classmethod\n    def from_probe_bundle(\n        cls,\n        *,\n        bundle_dir: Path,\n        expected_bundle_sha256: str,\n        gallery_path: Path,\n        expected_gallery_sha256: str,\n        gallery_count: int,\n        native_library_path: Path,\n        expected_native_library_sha256: str,\n    ) -> ConnectedCompactIndex:\n        """Load an explicit probe bundle; installed/native parity remains unqualified."""\n        return cls._from_bundle(\n            _probe=True,\n            bundle_dir=bundle_dir,\n            expected_bundle_sha256=expected_bundle_sha256,\n            gallery_path=gallery_path,\n            expected_gallery_sha256=expected_gallery_sha256,\n            gallery_count=gallery_count,\n            native_library_path=native_library_path,\n            expected_native_library_sha256=expected_native_library_sha256,\n        )\n\n    @classmethod\n    def _from_bundle(\n        cls,\n        *,\n        _probe: bool,\n        bundle_dir: Path,\n        expected_bundle_sha256: str,\n        gallery_path: Path,\n        expected_gallery_sha256: str,\n        gallery_count: int,\n        native_library_path: Path,\n        expected_native_library_sha256: str,\n    ) -> ConnectedCompactIndex:\n        """Share the original lifecycle across exactly two fixed installed bindings."""\n', 1), ('        self = cls()\n        try:\n', '        self = cls()\n        try:\n            _require(type(_probe) is bool, "fixed internal connected binding required")\n            if _probe:\n                schema = "siglip2-connected-probe-bundle-v1"\n                code_names = (_CODE - {_TRAINER, "test_siglip2_connected_mlp.py"}) | {\n                    "train_siglip2_connected_probe.py", "test_siglip2_connected_probe.py"\n                }\n                authority_factory = _installed_probe_authority\n                authority_schema = "sfora-connected-probe-inference-extraction-v1"\n                runtime_filename = "connected_probe_inference.py"\n            else:\n                schema = "siglip2-connected-mlp-bundle-v1"\n                code_names = _CODE\n                authority_factory = _installed_authority\n                authority_schema = "sfora-connected-inference-extraction-v1"\n                runtime_filename = "connected_inference.py"\n', 1), ('manifest["schema"] == "siglip2-connected-mlp-bundle-v1"', 'manifest["schema"] == schema', 1), ('manifest["code"].keys() == _CODE', 'manifest["code"].keys() == code_names', 1), ('authority_path, authority_sha = _installed_authority()', 'authority_path, authority_sha = authority_factory()', 1), ('record["SCHEMA"] == "sfora-connected-inference-extraction-v1"', 'record["SCHEMA"] == authority_schema', 1), ('runtime_path = Path(__file__).resolve().parent / "connected_inference.py"', 'runtime_path = Path(__file__).resolve().parent / runtime_filename', 1))


def bridge_factory_inverse(source):
    for before, after, count in reversed(BRIDGE_FACTORY_INVERSE):
        assert source.count(after) == count, 'finite bridge factory inverse occurrence differs'
        source = source.replace(after, before)
    assert sha(source.encode()) == '3d395a8fd7779bdae9145a3195785fc428b74fb53a8c0e2e47237fd8b22ed3ce', 'original whole bridge bytes differ'
    return source


def pipeline_contract_check():
    import difflib
    record = pipeline_record()
    source = RUNTIME.read_text()
    base = pipeline_inverse(source)
    # Historical whole-file hashes and every prior extraction assertion survive.
    assert sha(base.encode()) == record['base_runtime_sha256'] == '52afd638cd120dc69d2f9a7764f3574bd4ce5259a865f372d15d1fad14292512'
    # An extra top-level statement hidden in the deleted helper region must not be inverted away.
    for marker in ('\n\ndef _fingerprint_cuda_dict(', '\n\ndef fingerprint('):
        smuggled = source.replace(marker, '\n\nSMUGGLED = 1' + marker, 1)
        try:
            pipeline_inverse(smuggled)
        except AssertionError:
            pass
        else:
            raise AssertionError('pipeline inverse deleted an extra top-level statement')
    nodes = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    astsha = lambda text: sha(ast.dump(ast.parse(text), include_attributes=False).encode())
    assert tuple(n for n, *_ in record['helpers']) == ('_sha_cpu_bytes', '_fingerprint_cuda_dict')
    for name, byte_sha, ast_sha in record['helpers']:
        assert (sha(nodes[name].encode()), astsha(nodes[name])) == (byte_sha, ast_sha)
    assert (sha(nodes['encoder_facts'].encode()), astsha(nodes['encoder_facts'])) == record['encoder']
    original = next(ast.get_source_segment(base, n) for n in ast.parse(base).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'encoder_facts')
    assert ''.join(difflib.unified_diff(original.splitlines(True), nodes['encoder_facts'].splitlines(True),
                fromfile='original:encoder_facts', tofile='fresh-sha:encoder_facts')) == record['encoder_diff']
    authority = (ROOT / 'src/sfora/_connected_inference_authority.py').read_text()
    start = authority.index('\n    (\n        "_fresh_cpu_sha_pipeline",')
    end = authority.index('\n)\n\nRUNTIME_SHA256', start)
    historical = authority[:start] + authority[end:]
    historical = historical.replace(sha(source.encode()), record['base_runtime_sha256'])
    assert sha(historical.encode()) == record['base_authority_sha256'] == '538291c1cf14ead854760ad9400ee01dacc2ad73dec5b4ae56677d18bfd9412e'
    bridge = bridge_factory_inverse((ROOT / 'src/sfora/connected_compact_serving.py').read_text())
    bridge = bridge.replace(sha(authority.encode()), record['base_authority_sha256'])
    assert sha(bridge.encode()) == record['base_bridge_sha256']


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
    assert nodes.keys() == names | {'_bind_runtime', '_check_runtime', '_head_method_code', '_pack',
                                          '_sha_cpu_bytes', '_fingerprint_cuda_dict'}
    historical_nodes = {node.name: node for node in ast.parse(pipeline_inverse(RUNTIME.read_text())).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for path, symbols in CLOSURE.items():
        for name in symbols:
            expected = ast.parse(expected_definition(path, name)).body[0]
            assert ast.dump(historical_nodes[name], include_attributes=False) == ast.dump(expected, include_attributes=False), name


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
    historical_runtime = pipeline_inverse(runtime)
    nodes = {node.name: node for node in ast.parse(historical_runtime).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    ledger = values['SOURCE_SYMBOLS']
    assert {(path, name) for path, file_sha, name, *rest in ledger} == {
        (path, name) for path, symbols in CLOSURE.items() for name in symbols}
    import difflib
    differences = []
    for path, file_sha, name, source_sha, source_ast_sha, runtime_sha, runtime_ast_sha in ledger:
        original = original_definition(path, name)
        extracted = ast.get_source_segment(historical_runtime, nodes[name])
        astsha = lambda text: sha(ast.dump(ast.parse(text), include_attributes=False).encode())
        assert sha((ROOT / path).read_bytes()) == file_sha
        assert (sha(original.encode()), astsha(original), sha(extracted.encode()), astsha(extracted)) == (
            source_sha, source_ast_sha, runtime_sha, runtime_ast_sha), name
        if original != extracted:
            differences.append((name, ''.join(difflib.unified_diff(
                original.splitlines(True), extracted.splitlines(True), fromfile=path+':'+name,
                tofile='connected_inference.py:'+name))))
    assert values['SUBSTITUTIONS'][:-1] == tuple(differences)
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
            fresh_sha_check(runtime)
            pipeline_failure_context_check(runtime)
            submit_lifetime_check(runtime)
            pipeline_routing_check(runtime)
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


def pipeline_routing_check(runtime):
    torch = ModuleType('torch')
    torch.float32 = object()
    params = {name: SimpleNamespace(shape=(1,), dtype=torch.float32, device=SimpleNamespace(type='cuda'),
                    grad=None, is_leaf=True, grad_fn=None, requires_grad=False)
              for name in (*runtime.MLP, *(str(i) for i in range(444)))}
    processor = SimpleNamespace(to_json_string=lambda: '{}', backend='backend')
    buffers = {'embeddings.position_ids': object()}
    model = SimpleNamespace(named_parameters=lambda: params.items(), state_dict=lambda: params,
            named_modules=lambda: [('embeddings', SimpleNamespace(_non_persistent_buffers_set={'position_ids'}))],
            named_buffers=lambda: buffers.items())
    ident = {'inventory': {name: [1] for name in params}, 'nonpersistent': {'embeddings': ['position_ids']},
             'buffers_sha256': 'buffers', 'runtime': 'runtime', 'frozen_sha256': 'frozen'}
    state = {'model': model, 'encoder_identity': ident, 'device': 'cuda', 'arm': 'control',
             'processor_object': processor, 'processor': {'config': {}, 'backend': 'backend', 'origin': 'origin'},
             'guards': {}, 'processor_cache': object()}
    serial, parallel = [], []
    def hash_dict(value):
        if value is buffers or isinstance(value, dict) and value.keys() == buffers.keys(): return 'buffers'
        if isinstance(value, dict): return 'frozen' if len(value) == 444 else 'vision'
        return 'mlp'
    def serial_hash(value):
        serial.append(value)
        return hash_dict(value)
    def parallel_hash(value):
        parallel.append(value)
        return hash_dict(value)
    with patch.dict(sys.modules, {'torch': torch}), patch.object(runtime, 'fingerprint', serial_hash), \
         patch.object(runtime, '_fingerprint_cuda_dict', parallel_hash), \
         patch.object(runtime, 'model_structure', return_value='runtime'), \
         patch.object(runtime, 'module_origin', return_value='origin'), \
         patch.object(runtime, '_processor_cache', return_value=state['processor_cache']):
        for device, serving, arm in [('cuda', True, 'control'), ('cpu', True, 'control'),
                                     ('cuda', False, 'control'), ('cpu', False, 'candidate')]:
            state.update(device=device, arm=arm)
            for name, tensor in params.items():
                tensor.device.type = device
                tensor.requires_grad = not serving and arm == 'candidate' and name in runtime.MLP
            serial.clear()
            parallel.clear()
            result = runtime.encoder_facts(state, {}, serving=serving)
            assert result == {'vision_sha256': 'vision', 'encoder': dict.fromkeys(runtime.MLP, 'mlp')}
            assert len(parallel) == (2 if device == 'cuda' and serving else 0)
            assert len(serial) == (5 if parallel else 7)
            assert list(serial[0]) == ['embeddings.position_ids']
            assert all(serial[-4 + i] is params[name] for i, name in enumerate(runtime.MLP))
        state.update(device='cuda', arm='control')
        for tensor in params.values(): tensor.device.type, tensor.requires_grad = 'cuda', False
        ident['frozen_sha256'] = 'mutated'
        reject(lambda: runtime.encoder_facts(state, {}, serving=True), 'current frozen444 bytes differ')


def fresh_sha_check(runtime):
    import threading
    import weakref
    import gc
    from array import array

    caller = threading.get_ident()
    copies, owners = [], []
    class Tensor:
        def __init__(self, storage, indices=None):
            self.storage = storage
            self.indices = tuple(range(len(storage))) if indices is None else indices
            self.dtype, self.shape, self._version = 'torch.float32', (len(self.indices),), 0
            self.device = SimpleNamespace(type='cuda')
            self.fail = None
        def data_ptr(self): return id(self.storage)
        def numel(self): return len(self.indices)
        def element_size(self): return 1
        def detach(self): return self
        def cpu(self):
            assert threading.get_ident() == caller, 'tensor copy left caller thread'
            copies.append(self)
            if self.fail: raise ValueError(self.fail)
            self.snapshot = array('B', (self.storage[i] for i in self.indices))
            owners.append(weakref.ref(self.snapshot))
            return self
        def contiguous(self): return self
        def reshape(self, width):
            assert width == -1
            return self
        def view(self, dtype): return self
        def numpy(self):
            raw, self.snapshot = self.snapshot, None
            return raw
    torch = ModuleType('torch')
    torch.Tensor, torch.uint8 = Tensor, object()
    with patch.dict(sys.modules, {'torch': torch}):
        storage = bytearray(b'abcdefghijk')
        a, b = Tensor(storage, (7, 3, 1)), Tensor(storage, (2, 3, 4))
        value = {'z': a, 'a': b, 'alias': a}
        expected = runtime.fingerprint(value)
        assert runtime._fingerprint_cuda_dict(value) == expected
        assert len(copies) == 6
        assert runtime._fingerprint_cuda_dict(dict(reversed(list(value.items())))) == expected
        storage[3] = ord('Z')  # .data-like mutation: pointer/version stay unchanged.
        assert a._version == b._version == 0
        changed = runtime.fingerprint(value)
        assert changed != expected and runtime._fingerprint_cuda_dict(value) == changed
        assert len(copies) == 15, 'each occurrence must copy fresh bytes'
        # Feed a stale-fact mutant to the exact same current-byte digest oracle.
        def oracle():
            assert runtime._fingerprint_cuda_dict(value) == runtime.fingerprint(value)
        oracle()
        with patch.object(runtime, '_fingerprint_cuda_dict', return_value=expected):
            try:
                oracle()
            except AssertionError:
                pass
            else:
                raise AssertionError('stale digest mutant survived')
        assert runtime._fingerprint_cuda_dict({}) == runtime.fingerprint({})
        from collections import OrderedDict
        ordered = OrderedDict(reversed(list(value.items())))
        assert runtime._fingerprint_cuda_dict(ordered) == runtime.fingerprint(ordered)
        gc.collect()
        assert all(ref() is None for ref in owners), 'CPU snapshots escaped'


        # Four independent real SHA calls must overlap, with at most four snapshots.
        barrier = threading.Barrier(4, timeout=2)
        lock = threading.Lock()
        active = peak = calls = live_peak = 0
        real_sha = hashlib.sha256
        def overlap_sha(raw=b''):
            nonlocal active, peak, calls, live_peak
            if not isinstance(raw, memoryview):
                return real_sha(raw)
            assert threading.get_ident() != caller and raw.format == 'B'
            with lock:
                active += 1
                calls += 1
                ordinal = calls
                peak = max(peak, active)
                live_peak = max(live_peak, sum(ref() is not None for ref in owners))
            try:
                if ordinal <= 4:
                    barrier.wait()
                return real_sha(raw)
            finally:
                with lock: active -= 1
        batch = {str(i): Tensor(bytearray([i] * 4096)) for i in range(8)}
        expected = runtime.fingerprint(batch)
        before = len(copies)
        with patch.object(runtime.hashlib, 'sha256', overlap_sha):
            assert runtime._fingerprint_cuda_dict(batch) == expected
        assert calls == 8 and peak == 4 and live_peak <= 4
        assert len(copies) - before == 8
        gc.collect()
        assert all(ref() is None for ref in owners)

        # A 33 MiB leaf cannot be copied while two older 33 MiB leaves are retained.
        byte_cap = 96 * 1024**2
        block = threading.Event()
        started = threading.Event()
        draining = threading.Event()
        pending_sizes = []
        budget_peak = 0
        class BudgetTensor(Tensor):
            def __init__(self, ordinal, size):
                super().__init__(bytearray())
                self.ordinal, self.size = ordinal, size
            def numel(self): return self.size
            def cpu(self):
                nonlocal budget_peak
                assert threading.get_ident() == caller
                with lock:
                    assert sum(pending_sizes) + self.size <= byte_cap
                    pending_sizes.append(self.size)
                    budget_peak = max(budget_peak, sum(pending_sizes))
                return self
            def numpy(self):
                raw = array('B', [self.ordinal]) * self.size
                owners.append(weakref.ref(raw))
                return raw
        def budget_sha(raw=b''):
            if not isinstance(raw, memoryview): return real_sha(raw)
            if raw[0] == 1: started.set()
            assert block.wait(2), 'budget drain did not unblock'
            try:
                return real_sha(raw)
            finally:
                with lock: pending_sizes.remove(raw.nbytes)
        from concurrent.futures import ThreadPoolExecutor
        class BudgetExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                future = super().submit(fn, *args, **kwargs)
                original_result = future.result
                def result(*args, **kwargs):
                    draining.set()
                    return original_result(*args, **kwargs)
                future.result = result
                return future
        def release_budget():
            if started.wait(2) and draining.wait(2): block.set()
        releaser = threading.Thread(target=release_budget)
        releaser.start()
        try:
            with patch.object(runtime.hashlib, 'sha256', budget_sha), \
                 patch.object(runtime, 'ThreadPoolExecutor', BudgetExecutor):
                runtime._fingerprint_cuda_dict({str(i): BudgetTensor(i, 33 * 1024**2) for i in range(3)})
        finally:
            block.set()
            releaser.join(2)
        assert not releaser.is_alive() and draining.is_set() and not pending_sizes
        assert budget_peak == 66 * 1024**2 <= byte_cap
        gc.collect()
        assert all(ref() is None for ref in owners)
        huge = BudgetTensor(0, byte_cap + 1)
        reject(lambda: runtime._fingerprint_cuda_dict({'huge': huge}), 'exceeds byte budget')
        assert not pending_sizes, 'oversize leaf copied before rejection'
        reject(lambda: runtime._fingerprint_cuda_dict({1: a}), 'string-to-tensor')
        cpu = Tensor(bytearray(b'x'))
        cpu.device.type = 'cpu'
        before = len(copies)
        reject(lambda: runtime._fingerprint_cuda_dict({'cpu': cpu}), 'CUDA tensor leaves')
        assert len(copies) == before

        # Join all workers on failure; report the earliest input failure, not completion order.
        joined = []
        failure_barrier = threading.Barrier(3, timeout=2)
        failure_owners = []
        def failing_sha(raw=b''):
            if not isinstance(raw, memoryview): return real_sha(raw)
            failure_owners.append(weakref.ref(raw.obj))
            ordinal = raw[0]
            try:
                failure_barrier.wait()
                if ordinal in (0, 1): raise ValueError('sha-' + str(ordinal))
                return real_sha(raw)
            finally:
                joined.append(ordinal)
        broken = {str(i): Tensor(bytearray([i])) for i in range(3)}
        broken['3'] = Tensor(bytearray(b'x'))
        broken['3'].fail = 'copy-3'
        with patch.object(runtime.hashlib, 'sha256', failing_sha):
            error = reject(lambda: runtime._fingerprint_cuda_dict(broken), 'sha-0')
        assert sorted(joined) == [0, 1, 2]
        gc.collect()
        assert all(ref() is None for ref in failure_owners), 'failure retained CPU snapshot'
        trace = error.__traceback__
        while trace:
            if trace.tb_frame.f_code.co_name in {'_sha_cpu_bytes', '_fingerprint_cuda_dict', 'drain'}:
                assert all(value is None for name, value in trace.tb_frame.f_locals.items()
                           if name in {'raw', 'view', 'item', 'entry', 'value'})
            trace = trace.tb_next
        # Copy failure is retained when every earlier SHA succeeds.
        joined.clear()
        with patch.object(runtime.hashlib, 'sha256', overlap_sha):
            # Avoid the four-task barrier: these three SHA tasks are already beyond it.
            reject(lambda: runtime._fingerprint_cuda_dict(broken), 'copy-3')
        gc.collect()
        assert all(ref() is None for ref in owners), 'copy failure retained CPU snapshot'


def pipeline_failure_context_check(runtime):
    import threading
    import weakref
    import gc
    from array import array
    release = threading.Event()
    refs, snapshot_refs = [], []
    caller = threading.get_ident()
    class Tensor:
        def __init__(self, ordinal):
            self.ordinal, self.dtype, self.shape, self._version = ordinal, 'torch.float32', (1,), 0
            self.device = SimpleNamespace(type='cuda')
        def data_ptr(self): return id(self)
        def numel(self): return 1
        def element_size(self): return 1
        def detach(self): return self
        def cpu(self):
            assert threading.get_ident() == caller
            if self.ordinal:
                release.set()
                raise ValueError('later copy failure')
            return self
        def contiguous(self): return self
        def reshape(self, width): return self
        def view(self, dtype): return self
        def numpy(self):
            raw = array('B', [0])
            snapshot_refs.append(weakref.ref(raw))
            return raw
    torch = ModuleType('torch')
    torch.Tensor, torch.uint8 = Tensor, object()
    def inputs():
        tensors = [Tensor(0), Tensor(1)]
        refs.extend(weakref.ref(tensor) for tensor in tensors)
        return dict(zip(('a', 'b'), tensors))
    real_sha = hashlib.sha256
    def sha_failure(raw=b''):
        if not isinstance(raw, memoryview): return real_sha(raw)
        real_sha(raw)
        assert release.wait(2)
        raise ValueError('earlier SHA failure')
    with patch.dict(sys.modules, {'torch': torch}), patch.object(runtime.hashlib, 'sha256', sha_failure):
        error = reject(lambda: runtime._fingerprint_cuda_dict(inputs()), 'earlier SHA failure')
    assert error.__context__ is None, 'later copy failure chained onto earlier SHA failure'
    gc.collect()
    assert all(ref() is None for ref in refs), 'new exception chain retained tensor aliases'
    assert all(ref() is None for ref in snapshot_refs), 'new exception chain retained CPU snapshots'


def submit_lifetime_check(runtime):
    """A submit that raises after queueing must not release the view before the executor joins."""
    import threading
    import weakref
    import gc
    from array import array
    from concurrent.futures import ThreadPoolExecutor

    real_sha = hashlib.sha256
    expected = real_sha(bytes([1])).hexdigest()
    torch = ModuleType('torch')
    for mode in ('adjust_thread_count', 'after_submit_returned'):
        submitted, outcome, refs = [], [], []
        joining, entered, proceed = threading.Event(), threading.Event(), threading.Event()

        class Tensor:
            def __init__(self, ordinal):
                self.ordinal, self.dtype, self.shape, self._version = ordinal, 'torch.float32', (1,), 0
                self.device = SimpleNamespace(type='cuda')
            def data_ptr(self): return id(self)
            def numel(self): return 1
            def element_size(self): return 1
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def reshape(self, width): return self
            def view(self, dtype): return self
            def numpy(self):
                raw = array('B', [self.ordinal])
                refs.append(weakref.ref(raw))
                return raw
        torch.Tensor, torch.uint8 = Tensor, object()

        class QueueThenRaise(ThreadPoolExecutor):
            # The stdlib queues the work item before _adjust_thread_count; the second leaf's
            # future is therefore queued yet never reaches the caller's pending list.
            def submit(self, fn, *args, **kwargs):
                submitted.append(args[0])
                future = super().submit(fn, *args, **kwargs)
                if mode == 'after_submit_returned' and len(submitted) == 2:
                    raise KeyboardInterrupt('signal between submit and pending.append')
                return future
            def _adjust_thread_count(self):
                super()._adjust_thread_count()
                if mode == 'adjust_thread_count' and len(submitted) == 2:
                    raise RuntimeError("can't start new thread")
            def shutdown(self, *args, **kwargs):
                joining.set()
                return super().shutdown(*args, **kwargs)

        def gated_sha(raw=b''):
            if not (isinstance(raw, memoryview) and len(submitted) == 2 and raw is submitted[1]):
                return real_sha(raw)
            entered.set()
            assert proceed.wait(2), 'queued SHA task was never released'
            try:
                outcome.append(real_sha(raw).hexdigest())
            except ValueError as error:
                outcome.append('view released before join: ' + str(error))
                raise
            return real_sha(raw)

        def release_after_join_starts():
            joining.wait(2)
            proceed.set()
        releaser = threading.Thread(target=release_after_join_starts)
        releaser.start()
        tensors = [Tensor(0), Tensor(1)]
        refs.extend(weakref.ref(tensor) for tensor in tensors)
        caught = None
        try:
            with patch.dict(sys.modules, {'torch': torch}), \
                 patch.object(runtime.hashlib, 'sha256', gated_sha), \
                 patch.object(runtime, 'ThreadPoolExecutor', QueueThenRaise):
                runtime._fingerprint_cuda_dict(dict(zip('ab', tensors)))
        except BaseException as error:
            caught = error
        finally:
            proceed.set()
            releaser.join(2)
        assert not releaser.is_alive() and entered.is_set() and joining.is_set(), mode
        expected_error = (RuntimeError, "can't start new thread") if mode == 'adjust_thread_count' \
            else (KeyboardInterrupt, 'signal between submit and pending.append')
        assert type(caught) is expected_error[0] and str(caught) == expected_error[1], (mode, caught)
        assert caught.__context__ is None, mode
        assert outcome == [expected], (mode, outcome)
        try:
            submitted[1].tobytes()
        except ValueError:
            pass
        else:
            raise AssertionError('queued view survived the call unreleased: ' + mode)
        # The raised error stays live: it may keep the released view in submit's args, but no Tensor or snapshot.
        tensors = None
        submitted.clear()
        gc.collect()
        assert all(ref() is None for ref in refs), 'live submit failure retained tensors or CPU snapshots: ' + mode
        caught = None


def literal_pin_check():
    spec = importlib.util.spec_from_file_location('_literal_pin_bridge', ROOT / 'src/sfora/connected_compact_serving.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    path, digest = bridge._installed_authority()
    assert path == ROOT / 'src/sfora/_connected_inference_authority.py'
    assert path.is_absolute() and path.resolve() == path
    historical_digest = pipeline_record()['base_authority_sha256']
    assert historical_digest == '538291c1cf14ead854760ad9400ee01dacc2ad73dec5b4ae56677d18bfd9412e'
    assert digest == sha(path.read_bytes())
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
        pipeline_contract_check()
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
