#!/usr/bin/env python3
"""Independent stdlib probe extraction oracle; installed/native parity is unrun."""

import ast
import hashlib
import importlib.util
import inspect
import json
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/sfora/connected_probe_inference.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def api_check():
    bridge = load('_probe_bridge_api', ROOT / 'src/sfora/connected_compact_serving.py')
    assert hasattr(bridge.ConnectedCompactIndex, 'from_probe_bundle'), 'explicit probe factory missing'
    assert inspect.signature(bridge.ConnectedCompactIndex.from_probe_bundle) == inspect.signature(
        bridge.ConnectedCompactIndex.from_bundle)
    assert RUNTIME.is_file(), 'separate probe runtime missing'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def definitions(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert len(nodes) == len({n.name for n in nodes}), 'duplicate definitions'
    return {n.name: ast.get_source_segment(source, n) for n in nodes}


def record():
    raw = (ROOT / 'src/sfora/_connected_probe_inference_authority.py').read_text()
    tree = ast.parse(raw)
    assert all(type(n) is ast.Assign or type(n) is ast.Expr and type(n.value) is ast.Constant
               and type(n.value.value) is str for n in tree.body), 'nonliteral probe authority'
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    assert values.keys() == {'SCHEMA', 'HISTORICAL_CODE', 'SOURCE_SYMBOLS', 'PACKED_SOURCE_SYMBOLS',
                             'SUBSTITUTIONS', 'RUNTIME_SHA256', 'PACKED_SHA256'}
    assert values['SCHEMA'] == 'sfora-connected-probe-inference-extraction-v1'
    return values


def correspondence(oracle, source):
    """Reconstruct from the prior independent oracle, never from the generator."""
    closure = {p.replace('connected_mlp.py', 'connected_probe.py'): names for p, names in oracle.CLOSURE.items()}
    closure['scripts/train_siglip2_connected_probe.py'] += ('_probe_decorators', '_probe_vision_forward', 'probe_source')
    # The old independent test owns the allowed source transformations. Only the
    # exact copied-loader filename predicate differs for this extraction.
    fn_source = inspect.getsource(oracle.expected_definition).replace('connected_mlp.py', 'connected_probe.py')
    namespace = {'original_definition': oracle.original_definition, 'PLUMBING': oracle.PLUMBING}
    exec(compile(fn_source, '<independent probe correspondence>', 'exec'), namespace)
    expected = {}
    for path, names in closure.items():
        for name in names:
            expected[name] = namespace['expected_definition'](path, name)
    current = oracle.RUNTIME.read_text()
    original_defs = definitions(current)
    for name in ('_bind_runtime', '_check_runtime', '_head_method_code', '_pack', '_sha_cpu_bytes', '_fingerprint_cuda_dict'):
        expected[name] = original_defs[name]
    expected['_bind_runtime'] = expected['_bind_runtime'].replace('"connected_inference.py"', '"connected_probe_inference.py"').replace(
        '"_connected_inference_authority.py"', '"_connected_probe_inference_authority.py"')
    # Preserve exactly the current two dict routes; derive them independently of the new ledger.
    expected['encoder_facts'] = expected['encoder_facts'].replace(
        '    frozen = fingerprint({n:p for n,p in params.items() if n not in PROBE})',
        '    tensor_hash = _fingerprint_cuda_dict if serving and state["device"] == "cuda" else fingerprint\n    frozen = tensor_hash({n:p for n,p in params.items() if n not in PROBE})').replace(
        "return {'vision_sha256':fingerprint(model.state_dict()),", "return {'vision_sha256':tensor_hash(model.state_dict()),")
    actual = definitions(source)
    assert actual.keys() == expected.keys(), 'missing/extra probe guard or runtime helper'
    for name in expected:
        assert actual[name] == expected[name], 'source/decorator/AST correspondence differs: ' + name
    # The entire header is fixed, including imports and mutable literal globals.
    header = current[:current.index('def _bind_runtime(')].replace('siglip2-connected-mlp-', 'siglip2-connected-probe-').replace(
        'train_siglip2_connected_mlp.py', 'train_siglip2_connected_probe.py').replace(
        'test_siglip2_connected_mlp.py', 'test_siglip2_connected_probe.py')
    start, end = header.index('MLP ='), header.index('BACKEND_SHA =')
    header = header[:start] + "PROBE = ('head.probe',)\n\n\nPROBE_SHAPES = [[1,1,1152]]\n\n\nPROBE_SOURCE_SHA = '274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31'\n\n\n" + header[end:]
    order = list(original_defs)
    order[order.index('model_structure'):order.index('model_structure')] = ('_probe_decorators', '_probe_vision_forward', 'probe_source')
    assert source == header + '\n\n\n'.join(expected[name] for name in order) + '\n', 'whole runtime statements/imports/literals differ'
    with patch.object(oracle, 'RUNTIME', RUNTIME):
        oracle.dependency_check()
    return closure


def ledger_check(oracle, closure, values):
    old = oracle.ledger_check()
    expected_code = tuple(sorted([(n, h) for n, h in old['HISTORICAL_CODE']
                                  if n not in {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}] + [
        ('train_siglip2_connected_probe.py', 'e2496033a958cc83bf8d3204c87b4d3b8284528f03f44c1cb64517df04c8cc1e'),
        ('test_siglip2_connected_probe.py', '4accfd6c276ef2ca3d85b25c98ead7d013972d260656266b1227dd59648f0120')]))
    assert values['HISTORICAL_CODE'] == expected_code and len(expected_code) == 9
    assert values['PACKED_SOURCE_SYMBOLS'] == old['PACKED_SOURCE_SYMBOLS']
    assert values['PACKED_SHA256'] == old['PACKED_SHA256']
    source = RUNTIME.read_text()
    assert values['RUNTIME_SHA256'] == sha(source.encode())
    nodes = definitions(source)
    pipeline = values['SUBSTITUTIONS'][-1]
    assert pipeline[0] == '_fresh_cpu_sha_pipeline'
    assert pipeline[1]['helpers'] == old['SUBSTITUTIONS'][-1][1]['helpers']
    base = source
    for name in ('_sha_cpu_bytes', '_fingerprint_cuda_dict'):
        base = base.replace(nodes[name] + '\n\n\n', '', 1)
    for before, after in pipeline[1]['replacements']:
        assert base.count(after) == 1
        base = base.replace(after, before)
    assert sha(base.encode()) == pipeline[1]['base_runtime_sha256']
    original_nodes = definitions(base)
    assert {(p, n) for p, h, n, *_ in values['SOURCE_SYMBOLS']} == {(p, n) for p, names in closure.items() for n in names}
    astsha = lambda text: sha(ast.dump(ast.parse(text), include_attributes=False).encode())
    differences = []
    for path, file_sha, name, source_sha, source_ast, output_sha, output_ast in values['SOURCE_SYMBOLS']:
        raw = (ROOT / path).read_bytes()
        original = definitions(raw.decode())[name]
        output = original_nodes[name]
        assert (sha(raw), sha(original.encode()), astsha(original), sha(output.encode()), astsha(output)) == (
            file_sha, source_sha, source_ast, output_sha, output_ast)
        if original != output:
            differences.append((name, ''.join(difflib.unified_diff(original.splitlines(True), output.splitlines(True),
                               fromfile=path+':'+name, tofile='connected_probe_inference.py:'+name))))
    assert tuple(differences) == values['SUBSTITUTIONS'][:-2]
    label, counted = values['SUBSTITUTIONS'][-2]
    assert label == '_counted_plumbing' and {n for n, _ in counted} == original_nodes.keys() - {
        '_bind_runtime', '_check_runtime', '_head_method_code', '_pack'}
    by_name = {n: p for p, names in closure.items() for n in names}
    for name, replacements in counted:
        raw = oracle.original_definition(by_name[name], name)
        for before, after, count in replacements:
            assert raw.count(before) == count, 'substitution count differs'
            raw = raw.replace(before, after)
        assert raw == original_nodes[name]


def falsifiers(oracle):
    source = RUNTIME.read_text()
    node = definitions(source)['probe_source']
    mutants = (
        source.replace(node + '\n\n\n', '', 1),
        source.replace('== 447', '== 444'),
        source.replace("('head.probe',)", "('encoder.layers.26.mlp.fc1.weight',)"),
        source.replace('PROBE_SHAPES = [[1,1,1152]]', 'PROBE_SHAPES = [[1152]]'),
        source.replace("probe_source(model,packages,state['guards'])", 'pass'),
        source.replace("guards.get(str(path)) == PROBE_SOURCE_SHA", 'True'),
        source + '\nSMUGGLED = 1\n',
        source.replace('import gc', 'import torch'),
        source.replace('BACKEND_SHA =', 'VERSION = "same"\nBACKEND_SHA ='),
        source.replace("{'tie_last_hidden_states':False}", "{'tie_last_hidden_states':True}"),
    )
    for mutant in mutants:
        # Avoid an ineffective mutant silently passing its own negative test.
        assert mutant != source, 'ineffective extraction negative'
        try:
            correspondence(oracle, mutant)
        except AssertionError:
            pass
        else:
            raise AssertionError('independent correspondence accepted deliberate mutant')
    bridge = (ROOT / 'src/sfora/connected_compact_serving.py').read_text()
    oracle.bridge_factory_inverse(bridge)
    for mutant in (bridge + '# unchanged version, changed bytes\n', bridge.replace('gallery_count >= 10', 'gallery_count >= 9'),
                   bridge.replace('                schema = "siglip2-connected-probe-bundle-v1"', '                schema = "unknown"')):
        try:
            oracle.bridge_factory_inverse(mutant)
        except AssertionError:
            pass
        else:
            raise AssertionError('finite bridge inverse accepted changed bytes')


def admission_check(oracle, values):
    with tempfile.TemporaryDirectory(prefix='probe-extraction-') as scratch:
        root = Path(scratch)
        installed, bundle = root / 'installed', root / 'bundle'
        installed.mkdir(); bundle.mkdir()
        names = ('connected_probe_inference.py', '_connected_probe_inference_authority.py', 'packed_int8.py')
        for name in names:
            (installed / name).write_bytes((ROOT / 'src/sfora' / name).read_bytes())
        runtime = load('_probe_admission', installed / names[0])
        try:
            guards = tuple((str(installed / n), sha((installed / n).read_bytes())) for n in names)
            oracle.reject(lambda: runtime._bind_runtime(values['HISTORICAL_CODE'], ((guards[0][0], '0'*64), *guards[1:])), 'current FILE bytes differ')
            old = oracle.ledger_check()['HISTORICAL_CODE']
            oracle.reject(lambda: runtime._bind_runtime(old, guards), 'exact historical inference authority')
            runtime._bind_runtime(values['HISTORICAL_CODE'], guards)
            oracle.reject(lambda: runtime._bind_runtime(values['HISTORICAL_CODE'], guards), 'binding required')
            historical_packing = subprocess.check_output(['git', 'show', 'c6eb09e3^:src/sfora/joint_relational_compaction.py'], cwd=ROOT)
            for name, digest in values['HISTORICAL_CODE']:
                raw = historical_packing if name == 'joint_relational_compaction.py' else (ROOT / 'scripts' / name).read_bytes()
                assert sha(raw) == digest
                (bundle / name).write_bytes(raw)
            for name in ('vision.pt', 'endpoint.pt', 'processor.json'):
                (bundle / name).write_bytes(b'owned source fixture')
            constructor = installed / 'constructor.py'
            constructor.write_bytes(b'VERSION = "unchanged"\n')
            manifest = {'schema': runtime.BUNDLE_SCHEMA, 'code': dict(values['HISTORICAL_CODE']),
                'files': {n: sha((bundle/n).read_bytes()) for n in ('vision.pt', 'endpoint.pt', 'processor.json')},
                'endpoint_state_sha256':'1'*64, 'base_vision_sha256':'2'*64, 'vision_sha256':'3'*64, 'encoder_identity':{},
                'scope': {'arm':'control', 'manifest_sha256':runtime.SCOPE_SHA256, 'arm_sha256':runtime.CONTROL_SHA256},
                'environment': {'packages':{n:{'root':str(installed)} for n in runtime.NATIVE-{'sfora'}},
                    'files':{str(constructor):sha(constructor.read_bytes())}, 'native_files':{}, 'vision_constructor':str(constructor)}}
            path = bundle/'bundle.json'
            def write(value):
                raw = json.dumps(value).encode(); path.write_bytes(raw); return sha(raw)
            digest = write(manifest)
            admitted, checked = runtime.admit_bundle(bundle, digest)
            assert admitted == manifest and all(p in checked for p, _ in guards)
            for images in ([], [object()]*33):
                torch, nn = ModuleType('torch'), ModuleType('torch.nn')
                nn.functional = ModuleType('torch.nn.functional')
                endpoint = {'modules':{'runtime':runtime}, 'device':'cpu', 'directory':bundle,
                            'guards':checked, 'flags':{}}
                with patch.dict(sys.modules, {'torch':torch, 'torch.nn':nn}), patch.object(runtime,'numerical_flags',lambda:{}):
                    oracle.reject(lambda:runtime.inference_outputs(endpoint,images),'serving batch/numerics differ')
            oracle.reject(lambda:runtime.strict_json('{"x":1,"x":2}'),'duplicate JSON key')
            oracle.reject(lambda:runtime.strict_json('{"x":NaN}'),'nonfinite JSON')
            for file in (*[installed/n for n in names], *[bundle/n for n in manifest['code']|manifest['files']], constructor):
                raw = file.read_bytes()
                file.write_bytes(raw+b'# same version, different bytes\n')
                oracle.reject(lambda:runtime.admit_bundle(bundle,digest), 'current FILE bytes differ')
                file.write_bytes(raw)
            for mutation in ({'schema':'siglip2-connected-mlp-bundle-v1'}, {'code':dict(old)},
                             {'code':manifest['code']|{'train_siglip2_connected_mlp.py':'0'*64}},
                             {'scope':{}}, {'environment':{}}, {'extra':True}):
                bad = write(manifest|mutation)
                oracle.reject(lambda:runtime.admit_bundle(bundle,bad))
            digest = write(manifest)
            try:
                runtime.load_inference(bundle,digest,'cuda')
            except AssertionError as error:
                assert str(error) == 'native import during source check: torch'
            else:
                raise AssertionError('native runtime constructed in source-only test')
            oracle.head_check(runtime)
            oracle.serializer_check(runtime)
            oracle.fresh_sha_check(runtime)
            oracle.pipeline_failure_context_check(runtime)
            oracle.submit_lifetime_check(runtime)
            # CPU-only workers are source-identical; routing separately executes the probe encoder.
        finally:
            sys.modules.pop(runtime.__name__, None)


def probe_guard_checks(oracle):
    fixture = load('_probe_source_fixtures', ROOT/'scripts/test_siglip2_connected_probe.py')
    runtime = load('_probe_guard_subject', RUNTIME)
    try:
        for api, names in (
            ('_bind_runtime', ('historical_code', 'runtime_guards')),
            ('admit_bundle', ('directory', 'digest')),
            ('load_inference', ('directory', 'bundle_sha256', 'device')),
            ('inference_outputs', ('endpoint', 'images')),
            ('release_inference', ('endpoint',)),
        ):
            fn = getattr(runtime,api)
            assert fn.__code__.co_varnames[:fn.__code__.co_argcount] == names
        genuine = runtime.probe_source
        checked = []
        def extra_negatives(model, packages, guards):
            genuine(model, packages, guards)
            if checked:
                return
            checked.append(True)
            # All changes retain the same source/version and executable names.
            raw = type(model).forward.__wrapped__.__wrapped__
            with patch.object(raw, '__defaults__', (0,)):
                oracle.reject(lambda:genuine(model,packages,guards),'base forward')
            for fn in (type(model).forward, type(model.head).forward, type(model.head).__init__):
                with patch.object(fn, '__defaults__', (None,)):
                    oracle.reject(lambda:genuine(model,packages,guards),'live probe circuit')
            fn = type(model.head).forward
            from types import FunctionType
            foreign = FunctionType(fn.__code__,dict(fn.__globals__),fn.__name__,fn.__defaults__,fn.__closure__)
            with patch.object(type(model.head),'forward',foreign):
                oracle.reject(lambda:genuine(model,packages,guards),'live probe circuit')
            module = sys.modules['transformers.models.siglip.modeling_siglip']
            path = Path(module.__file__)
            read = Path.read_bytes
            def changed_bytes(file):
                raw = read(file)
                return raw+b'# version unchanged\n' if file == path else raw
            with patch.object(Path,'read_bytes',changed_bytes):
                oracle.reject(lambda:genuine(model,packages,guards),'source changed before compilation')
        with patch.object(runtime,'probe_source',extra_negatives):
            fixture.probe_native_source_seam(runtime)
        assert checked
        with fixture.tensor_seam():
            params = {'head.probe':fixture.Tensor((1,1,1152))}
            params.update({f'frozen.{i}':fixture.Tensor((1,)) for i in range(447)})
            model = SimpleNamespace(named_parameters=lambda:params.items(), state_dict=lambda:params)
            ids = {n:id(p) for n,p in params.items()}
            runtime.apply_overlay(model, {'head.probe':fixture.Tensor((1,1,1152), 3.)})
            assert ids == {n:id(p) for n,p in params.items()} and params['head.probe'].value == 3.
            assert all(p.value == 0. for n,p in params.items() if n != 'head.probe')
            for bad in ({}, {'head.mlp.fc2.bias':fixture.Tensor((1152,))},
                        {'head.probe':fixture.Tensor((1152,))}, {'head.probe':fixture.Tensor((1,1,1152),dtype='float16')},
                        {'head.probe':fixture.Tensor((1,1,1152),float('inf'))}):
                oracle.reject(lambda:runtime.apply_overlay(model,bad))
            # Execute the complete generated encoder facts with stand-ins, including a
            # historical MLP leaf INSIDE frozen447 and both real large-dict call routes.
            frozen_leaf = params.pop('frozen.0')
            params['encoder.layers.26.mlp.fc1.weight'] = frozen_leaf
            for p in params.values():
                p.grad = None; p.is_leaf = True
            model.named_modules = lambda:[('',SimpleNamespace(_non_persistent_buffers_set=set()))]
            buffers = {'embeddings.position_ids':fixture.Tensor((1,256), dtype='int64')}
            model.named_buffers = lambda:buffers.items()
            model.named_modules = lambda:[('embeddings',SimpleNamespace(_non_persistent_buffers_set={'position_ids'}))]
            def fp(value):
                if isinstance(value,fixture.Tensor): return repr((value.shape,value.dtype,value.value))
                return repr([(n,fp(p)) for n,p in sorted(value.items())])
            processor = SimpleNamespace(to_json_string=lambda:'{}',backend='torchvision')
            identity = {'inventory':{n:list(p.shape) for n,p in params.items()},'nonpersistent':{'embeddings':['position_ids']},
                        'buffers_sha256':fp(buffers),'runtime':{},'frozen_sha256':fp({n:p for n,p in params.items() if n not in runtime.PROBE})}
            state = {'model':model,'encoder_identity':identity,'device':'cpu','arm':'candidate','guards':{},
                     'processor_object':processor,'processor':{'config':{},'backend':'torchvision','origin':{}},'processor_cache':'cache'}
            routes = []
            with patch.object(runtime,'probe_source',lambda *a:None), patch.object(runtime,'model_structure',lambda *a:{}),\
                 patch.object(runtime,'module_origin',lambda *a:{}), patch.object(runtime,'_processor_cache',lambda *a:'cache'),\
                 patch.object(runtime,'fingerprint',fp), patch.object(runtime,'_fingerprint_cuda_dict',lambda v:routes.append(tuple(v)) or fp(v)):
                runtime.encoder_facts(state,{},serving=True)
                frozen_leaf.value = 1.
                oracle.reject(lambda:runtime.encoder_facts(state,{},serving=True),'frozen447')
                frozen_leaf.value = 0.
                for p in params.values(): p.device.type='cuda'
                state['device']='cuda'
                runtime.encoder_facts(state,{},serving=True)
                assert len(routes)==2 and len(routes[0])==447 and len(routes[1])==448
                assert 'encoder.layers.26.mlp.fc1.weight' in routes[0] and 'head.probe' not in routes[0]
    finally:
        sys.modules.pop(runtime.__name__,None)
        sys.modules.pop(fixture.__name__,None)


def main():
    if not __debug__:
        raise SystemExit('source check requires assertions')
    oracle = load('_old_extraction_oracle', ROOT/'scripts/test_connected_inference_extraction.py')
    guard = oracle.NoNative()
    sys.meta_path.insert(0, guard)
    try:
        api_check()
        closure = correspondence(oracle, RUNTIME.read_text())
        values = record()
        ledger_check(oracle, closure, values)
        falsifiers(oracle)
        admission_check(oracle, values)
        probe_guard_checks(oracle)
    finally:
        sys.meta_path.remove(guard)
    print('connected probe inference source-only checks passed; native UNRUN')


if __name__ == '__main__':
    main()
