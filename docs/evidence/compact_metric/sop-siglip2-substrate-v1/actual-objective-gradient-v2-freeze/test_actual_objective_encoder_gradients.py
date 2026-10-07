#!/usr/bin/env python3
"""Bounded stdlib admission/source checks; native connectivity remains UNRUN."""
import ast
import contextlib
from functools import lru_cache
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
import weakref

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'qualify_actual_objective_encoder_gradients.py'
EVIDENCE = HERE.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/identity-diversity-v1'
BACKEND_SOURCE = EVIDENCE.parent / 'actual-objective-processor-cache-source/original-image_processing_backends.py'


def rejects(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), (message, str(error))
    else:
        raise AssertionError('accepted: ' + message)


def processor_cache_lifecycle(d):
    """Execute the pinned backend method AST, including its real LRU ownership."""
    raw = BACKEND_SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads((EVIDENCE / 'cpu-v5/receipt.json').read_text())
    original = '/home/riomus/group-learning/.venv/lib/python3.13/site-packages/transformers/image_processing_backends.py'
    assert digest == receipt['input_guards'][original] == '250394884a9f90845cf87b6fc0cf3341337b193428556c6ff61ed2b04bedb692'
    method = '_fuse_mean_std_and_rescale_factor'
    backend_name = 'transformers.image_processing_backends'
    siglip_name = 'transformers.models.siglip.image_processing_siglip'
    class Tensor(tuple):
        device = 'cpu'
        def __mul__(self, scalar): return Tensor(x * scalar for x in self)
        def to(self, **kwargs): return self
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'image_processing_backends.py'
        path.write_bytes(raw)
        backend = ModuleType(backend_name)
        backend.__file__ = str(path)
        backend.torch = SimpleNamespace(tensor=lambda x, device: Tensor(x), float32='float32')
        backend.tvF = SimpleNamespace(normalize=lambda x, mean, std: Tensor(
            (value - m) / s for value, m, s in zip(x, mean, std)))
        backend.group_images_by_shape = lambda images, **kw: ({'shape': images[0]}, None)
        backend.reorder_images = lambda grouped, index: list(grouped.values())
        backend.BatchFeature = lambda data, tensor_type: data
        # Compile the complete AST but execute ONLY these finite method bodies.
        code = compile(ast.parse(raw), str(path), 'exec', dont_inherit=True)
        cls_code = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'TorchvisionBackend')
        methods = {c.co_name: FunctionType(c, vars(backend)) for c in cls_code.co_consts
                   if isinstance(c, CodeType) and c.co_name in
                   (method, '_preprocess', 'rescale_and_normalize', 'normalize')}
        function = methods[method]
        function.__defaults__ = (None,) * 6
        wrapper = lru_cache(maxsize=10)(function)
        methods[method] = wrapper
        backend.TorchvisionBackend = type('TorchvisionBackend', (), {'__module__': backend_name, **methods})
        siglip = ModuleType(siglip_name)
        siglip.SiglipImageProcessor = type('SiglipImageProcessor', (backend.TorchvisionBackend,), {'__module__': siglip_name})
        previous = {name: sys.modules.get(name) for name in (backend_name, siglip_name)}
        sys.modules.update({backend_name: backend, siglip_name: siglip})
        def process(owner):
            result = owner._preprocess([Tensor((0., 127.5, 255.))], False, None, None,
                False, None, True, 1/255, True, (.5,)*3, (.5,)*3, False, None, False, 'pt')
            assert result == {'pixel_values': [(-1., 0., 1.)]}
        guards = {str(path): digest}
        try:
            # Root cause: the actual fused method retains self until genuine cache_clear.
            owner = siglip.SiglipImageProcessor()
            ref = weakref.ref(owner)
            process(owner)
            del owner
            gc.collect()
            assert ref() is not None and wrapper.cache_info().currsize == 1
            wrapper.cache_clear()
            gc.collect()
            assert ref() is None

            # Run the actual outer cleanup on success and on an original rejection.
            tree = ast.parse(DRIVER.read_text())
            outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'qualify')
            block = next(n for n in outer.body if isinstance(n, ast.Try))
            exercise = ast.parse('def exercise(owned, released, guards, reject=False):\n'
                ' failure = None\n release_names = {}\n processor_cache = None\n'
                ' context = {"guards": guards}\n').body[0]
            admission = [n for n in block.body if isinstance(n, ast.Assign) and
                         isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == '_processor_cache']
            body = admission + ast.parse('process(owned["processor"])\n'
                'if reject: raise ValueError("original source rejection")').body
            exercise.body.append(ast.Try(body=body, handlers=block.handlers, orelse=[], finalbody=block.finalbody))
            exercise.body += ast.parse('if failure is not None: raise failure').body
            exits = []
            namespace = {**d, 'process': process, 'torch': SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
                         'trainer': SimpleNamespace(exit_rehash=lambda context: exits.append('rehash'))}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[exercise], type_ignores=[])),
                         '<actual processor cleanup>', 'exec'), namespace)
            for reject in (False, True):
                owned = {'processor': siglip.SiglipImageProcessor()}
                ref = weakref.ref(owned['processor'])
                with contextlib.redirect_stderr(io.StringIO()):
                    if reject:
                        rejects(lambda: namespace['exercise'](owned, [ref], guards, True), 'original source rejection')
                    else:
                        namespace['exercise'](owned, [ref], guards)
                assert ref() is None and not owned and wrapper.cache_info().currsize == 0
            assert exits == ['rehash', 'rehash']

            owner = siglip.SiglipImageProcessor()
            admit = lambda: d['_processor_cache'](owner, guards, empty=True)
            assert admit() is wrapper
            path.write_bytes(raw + b'\n# changed\n')
            rejects(admit, 'source')
            path.write_bytes(raw)
            rejects(lambda: d['_processor_cache'](owner, {str(path): '0'*64}, empty=True), 'source')
            process(owner)
            before = wrapper.cache_info()
            rejects(admit, 'empty')
            assert wrapper.cache_info() == before, 'nonempty cache was changed by admission'
            owned = {'processor': siglip.SiglipImageProcessor()}
            new_ref = weakref.ref(owned['processor'])
            with contextlib.redirect_stderr(io.StringIO()):
                rejects(lambda: namespace['exercise'](owned, [], guards), 'initially be empty')
            assert not owned and new_ref() is None and wrapper.cache_info() == before
            wrapper.cache_clear()
            # Genuine wrapper with a decoy __wrapped__ must also be rejected.
            for bad in (function, lru_cache(maxsize=9)(function), lru_cache(maxsize=10, typed=True)(function),
                        lru_cache(maxsize=10)(lambda *a, **k: None)):
                if hasattr(bad, '__wrapped__'): bad.__wrapped__ = function
                backend.TorchvisionBackend._fuse_mean_std_and_rescale_factor = bad
                rejects(admit, 'wrapper')
            backend.TorchvisionBackend._fuse_mean_std_and_rescale_factor = wrapper
            old_code = function.__code__
            function.__code__ = (lambda *a, **k: None).__code__
            rejects(admit, 'code')
            function.__code__ = old_code
            owner._fuse_mean_std_and_rescale_factor = lru_cache(maxsize=10)(function)
            rejects(admit, 'wrapper')
            del owner._fuse_mean_std_and_rescale_factor
            wrapper.cache_clear = lambda: None
            rejects(admit, 'wrapper')
            del wrapper.cache_clear

            # Failed cleanup still pops owned objects and leaves the release gate intact.
            def break_cleanup(owner):
                process(owner)
                wrapper.cache_clear = lambda: None
            namespace['process'] = break_cleanup
            owned = {'processor': siglip.SiglipImageProcessor()}
            ref = weakref.ref(owned['processor'])
            log = io.StringIO()
            with contextlib.redirect_stderr(log):
                rejects(lambda: namespace['exercise'](owned, [ref], guards, True), 'tensor lifetime survived release')
            assert not owned and ref() is not None and len(exits) == 3
            assert 'original source rejection' in log.getvalue() and 'wrapper' in log.getvalue()
            del wrapper.cache_clear
            wrapper.cache_clear()
            gc.collect()
            assert ref() is None
        finally:
            type(wrapper).cache_clear(wrapper)
            for name, value in previous.items():
                if value is None: sys.modules.pop(name, None)
                else: sys.modules[name] = value


def cleanup_diagnostics(d):
    """Execute the actual handler/finally, with stdlib objects in place of tensors."""
    tree = ast.parse(DRIVER.read_text())
    view = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'view_probe')
    block = next(n for n in view.body if isinstance(n, ast.Try))
    wrapper = ast.parse("def exercise(temporary, released, chained):\n failure = None\n view = 'canonical'\n").body[0]
    wrapper.body.append(ast.Try(body=ast.parse("features = temporary['features']\nreject(features, chained)").body,
        handlers=block.handlers, orelse=block.orelse, finalbody=block.finalbody))
    wrapper.body += ast.parse('if failure is not None: raise failure').body
    class Tensor: pass
    def reject(tensor, chained):
        if not chained:
            raise ValueError('original source rejection')
        try:
            raise ValueError('original source rejection')
        except ValueError as error:
            raise RuntimeError('wrapped rejection') from error
    namespace = {**d, 'reject': reject, 'torch': SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 '<actual source cleanup>', 'exec'), namespace)
    for chained in (False, True):
        temporary = {'features': Tensor()}
        ref = weakref.ref(temporary['features'])
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            try:
                namespace['exercise'](temporary, [ref], chained)
            except ValueError as error:
                expected = 'lifetime survived release' if chained else 'original source rejection'
                assert expected in str(error)
                assert (ref() is not None) is chained
            else:
                raise AssertionError('cleanup rejection disappeared')
        assert 'ValueError: original source rejection' in log.getvalue(), log.getvalue()
        if chained:
            assert 'RuntimeError: wrapped rejection' in log.getvalue()
            row = next(json.loads(line) for line in log.getvalue().splitlines() if line.startswith('{'))
            assert row['surviving'][0]['name'] == 'released[0]'
            assert row['surviving'][0]['type'].endswith('.Tensor')
        gc.collect()
        assert ref() is None, 'diagnostic retained an object after exception release'
    # Run the outer source cleanup too: chained frames may retain the model.
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'qualify')
    block = next(n for n in outer.body if isinstance(n, ast.Try))
    wrapper = ast.parse('def exercise_outer(owned, released, release_names, chained):\n failure = None\n processor_cache = None\n').body[0]
    wrapper.body.append(ast.Try(body=ast.parse("reject(owned['model'], chained)").body,
        handlers=block.handlers, orelse=block.orelse, finalbody=block.finalbody))
    wrapper.body += ast.parse('if failure is not None: raise failure').body
    exits = []
    namespace.update(torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
                     trainer=SimpleNamespace(exit_rehash=lambda context: exits.append('rehash')), context={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 '<actual outer source cleanup>', 'exec'), namespace)
    for chained in (False, True):
        owned = {'model': Tensor(), 'processor': Tensor()}
        ref = weakref.ref(owned['model'])
        names = {id(owned['model']): 'model'}
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            try:
                namespace['exercise_outer'](owned, [ref], names, chained)
            except ValueError as error:
                expected = 'encoder/processor tensor lifetime' if chained else 'original source rejection'
                assert expected in str(error)
                assert (ref() is not None) is chained
            else:
                raise AssertionError('outer rejection disappeared')
        assert 'ValueError: original source rejection' in log.getvalue()
        if chained:
            row = next(json.loads(line) for line in log.getvalue().splitlines() if line.startswith('{'))
            assert row['surviving'][0]['name'] == 'model'
        gc.collect()
        assert ref() is None
    assert exits == ['rehash'], 'successful release skipped original exit reader'
    tensor = Tensor()
    named_ref = weakref.ref(tensor)
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        d['_diagnose_release']([named_ref], 'encoder', {id(tensor): 'model.encoder.fc1.weight'})
    assert json.loads(log.getvalue())['surviving'][0]['name'] == 'model.encoder.fc1.weight'
    del tensor
    gc.collect()
    assert named_ref() is None, 'named diagnostic retained an object'


def main():
    assert DRIVER.is_file(), 'missing actual-objective qualifier'
    d = runpy.run_path(str(DRIVER))
    processor_cache_lifecycle(d)
    cleanup_diagnostics(d)
    assert not {'torch', 'numpy', 'transformers', 'PIL', 'safetensors'} & sys.modules.keys()
    receipt_path = EVIDENCE / 'cpu-v5/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == d['CPU_UNIT']['receipt']['sha256']
    assert d['CPU_UNIT'] == json.loads((EVIDENCE / 'cpu-v5/terminal.json').read_text())
    assert d['TRAIN_CODE'] == json.loads((EVIDENCE / 'cpu-v5-freeze/execution.json').read_text())
    assert d['TRAIN_EXECUTION'] == 'bd081a02a49f1f0dd8305b78bb3f8aa91f5bb7331f3045325a185971fb7c04a8'
    assert d['CPU_AUTHORITY_SHA'] == '8fcfb9ae55c33ed842c68ab01b59fea378f1258b4b784d5874e40e9f49801eac'
    qualification = d['select_initializer'](receipt)
    assert qualification['checkpoint'] == d['INITIALIZER']
    assert qualification['terminal_state_sha256'] == d['PAYLOAD_SHA']
    for change in ({'arm': 'candidate'}, {'seed': 179069}, {'identity': {'device': 'cuda'}},
                   {'checkpoint': {**d['INITIALIZER'], 'sha256': '0'*64}},
                   {'terminal_state_sha256': '0'*64}):
        bad = {**qualification, **change}
        rejects(lambda: d['select_initializer']({'qualifications': [bad]}), 'initializer')
    rejects(lambda: d['select_initializer']({'qualifications': [qualification]*2}), 'initializer')

    source = ast.parse((HERE / 'train_siglip2_identity_diversity.py').read_text())
    # Bound API correspondence to the source-v5 trainer, independently of current file bytes.
    original_ast = {
        'check_payload':'66a74a23cff758fde54ee715bf52aeac24f092dc316deb2eddf819d294db9f37',
        'restore':'38274c5421b261fd7db23796bd553f7e901034d93fe053088ecc5c63a4703f1b',
        'loss_denominators':'ce86de123238eee428db8c62ff8ed4c09c2cde5ad6fbe52a85139623802a78e5',
        'ranking_bank':'bc2b9bb03c0e4437e68e2b3e67b34dd0a8bbef6b7c2fbeee7ee613197e14905c',
        'ranking_membership':'c56dd67441cc81a47e8fcfb71032e4c751b7d5b70f2b6df19ca6236b9301813f',
        'ranking_gallery':'80a6ab2cf778da8b8f6c78ccb183cd877085c4b62d1ac83655aa6f86ba69e3bd',
        'loss_terms':'d9cdceea0ac03e072e6a33f86053823a786e12a3cf180e645eebe002edcffc26'}
    assert {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in source.body if isinstance(n,ast.FunctionDef) and n.name in original_ast} == original_ast
    selected = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in
                ('ranking_bank', 'json_sha256', 'ranking_membership', 'loss_denominators')]
    trainer = {'require': d['require'], 'hashlib': hashlib, 'json': json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original membership>', 'exec'), trainer)
    trainer = SimpleNamespace(**trainer)
    class Batch(list):
        def tolist(self):
            return list(self)
    state = {'schedules': {'179061': [Batch(qualification['scope_schedule'][0])]},
             'ranking_bank': qualification['ranking_bank']}
    batch, anchors, valid = d['first_microbatch'](trainer, state)
    assert batch == qualification['scope_schedule'][0] and anchors == list(d['ANCHORS'])
    assert valid == 63 and trainer.loss_denominators(valid) == (128, 126)
    assert trainer.ranking_membership(state['ranking_bank'], anchors)['valid'] == 16
    state['schedules']['179061'][0] = Batch(batch[:16])
    rejects(lambda: d['first_microbatch'](trainer, state), 'B64')
    state['schedules']['179061'][0] = Batch(batch)
    bank = trainer.ranking_bank([0, 0, 0, 1], [4, 9, 12, 17])
    membership = trainer.ranking_membership(bank, [0, 3])
    assert membership['positive'] == [[1, 2], []] and membership['eligible_counts'] == [3, 3]
    state['ranking_bank'] = {**state['ranking_bank'], 'sha256': '0'*64}
    rejects(lambda: d['first_microbatch'](trainer, state), 'membership')

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        for name in d['FILES']:
            shutil.copyfile(HERE / name, root / name)
        local = runpy.run_path(str(root / DRIVER.name))
        manifest = {'schema': d['AUTHORITY_SCHEMA'],
                    'files': {n: d['sha'](root / n) for n in d['FILES']},
                    'python': {'path': str(Path(sys.executable).resolve()),
                               'sha256': d['sha'](Path(sys.executable).resolve())},
                    'output': str(root / 'result'), 'resource_policy': d['POLICY'],
                    'both_locks_held': True}
        authority = root / 'authority.json'
        def admit(value=manifest):
            authority.write_text(json.dumps(value))
            return local['authenticate'](authority, d['sha'](authority), root / 'result')
        assert len(admit()) == len(d['FILES']) + 2
        for key, bad in [('schema', 'other'), ('both_locks_held', 1),
                         ('resource_policy', {**d['POLICY'], 'seconds': 301}),
                         ('resource_policy', {**d['POLICY'], 'swap_bytes': False}),
                         ('output', str(root / 'other')),
                         ('files', {**manifest['files'], 'extra.py': '0'*64}),
                         ('python', {**manifest['python'], 'sha256': '0'*64})]:
            rejects(lambda: admit({**manifest, key: bad}), 'authority')
        for name in d['FILES']:
            path, old = root / name, (root / name).read_bytes()
            path.write_bytes(old + b'\n# stale\n')
            rejects(admit, 'source')
            path.write_bytes(old)
        authority.write_text('{"schema":1,"schema":2}')
        rejects(lambda: local['authenticate'](authority, d['sha'](authority), root / 'result'), 'duplicate')
        rejects(lambda: local['authenticate'](authority, '0'*64, root / 'result'), 'SHA256')
        (root / 'result').mkdir()
        rejects(admit, 'output')

    class Scalar:
        def __init__(self, value): self.value = value
        def all(self): return self
        def item(self): return self.value
    class Gradient(list):
        def norm(self): return math.sqrt(sum(x*x for x in self))
    torch = SimpleNamespace(isfinite=lambda g: Scalar(all(math.isfinite(x) for x in g)),
                            count_nonzero=lambda g: Scalar(sum(x != 0 for x in g)))
    assert d['gradient_fact'](torch, Gradient([3., 4.]), 'rank') == {'norm': 5., 'nonzero': 2}
    for gradient in (None, Gradient([0., 0.]), Gradient([math.inf]), Gradient([math.nan])):
        rejects(lambda: d['gradient_fact'](torch, gradient, 'rank'), 'gradient')
    assert d['gradient_fact'](torch, Gradient([0.]), 'regression', nonzero=False) == {'norm':0.,'nonzero':0}
    class Parameter:
        def __init__(self):
            self.requires_grad, self.grad = True, None
            self.shape, self.dtype, self.device = (1,), 'torch.float32', SimpleNamespace(type='cpu')
        def requires_grad_(self, value): self.requires_grad = value
    parameters = {name:Parameter() for name in d['MLP']}
    parameters.update({f'frozen.{i}':Parameter() for i in range(444)})
    model = SimpleNamespace(named_parameters=lambda: iter(parameters.items()), state_dict=lambda:parameters)
    expected = {name:[1] for name in parameters}
    d['legacy_probe'].select_mlp(model, expected)
    assert {name for name,p in parameters.items() if p.requires_grad} == set(d['MLP'])
    rejects(lambda:d['legacy_probe'].select_mlp(model, dict(list(expected.items())[:-1])), '448')
    parameters[d['MLP'][0]].grad = object()
    rejects(lambda:d['legacy_probe'].select_mlp(model, expected), 'source parameter')

    tree = ast.parse(DRIVER.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = lambda fn: [ast.unparse(n.func) for n in ast.walk(functions[fn]) if isinstance(n, ast.Call)]
    all_calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(n.endswith(('.prepare_native', '.prepare_scope', '.prepare_original',
                              '.step', '.save', '.reset_peak_memory_stats', '.half')) for n in all_calls)
    assert calls('qualify').count('source.fresh_source') == 1
    assert 'trainer.loss_terms' in calls('view_probe') and 'torch.autograd.grad' in all_calls
    assert 'trainer.restore' in calls('_load_initializer') and 'trainer.check_payload' in calls('_load_initializer')
    loader = ast.unparse(functions['_load_initializer'])
    assert 'weights_only=True' in loader and 'mmap=True' in loader and "map_location='cpu'" in loader
    assert 'CheckpointPages' in loader and 'consumed=pages.consume' in loader
    assert "copy.deepcopy(disk['identity'])" in loader
    qualify = ast.unparse(functions['qualify'])
    assert qualify.index('trainer.admit_terminal') < qualify.index('context[\'guards\'].update')
    assert qualify.index('trainer.canonical_initial_witness') < qualify.index('source.fresh_source')
    assert qualify.index("owned.pop('model'") < qualify.rindex('trainer.integrity') < qualify.index('trainer.exit_rehash')
    assert qualify.index("owned['model'].to('cpu')") < qualify.rindex('source.model_facts')
    assert 'finally:' in qualify and 'weakref.ref' in all_calls
    probe = ast.unparse(functions['view_probe'])
    assert 'features.detach()' in probe and 'allow_unused=True' in ast.unparse(functions['_gradients'])
    assert 'query' in probe and 'gallery' in probe and 'tied' in probe
    assert "nonzero=label != 'regression'" in probe and "grads[:4]" in probe
    assert "torch.equal" in probe and 'global CPU RNG' in ast.unparse(functions['pixels_for'])
    assert 'source.numerical_flags' in calls('view_probe') and 'torch.random.set_rng_state' in all_calls
    assert d['MLP'] == tuple(f'encoder.layers.26.mlp.{layer}.{field}'
                             for layer in ('fc1', 'fc2') for field in ('weight', 'bias'))
    assert d['POLICY'] == {'seconds': 300, 'host_bytes': 8589934592, 'swap_bytes': 0,
                           'cuda_allocated_bytes_exclusive': 10000000000}
    assert not {'torch', 'numpy', 'transformers', 'PIL', 'safetensors'} & sys.modules.keys()
    print('PASS stdlib initializer/authority, B64 K63/micro16, membership, gradient rejection, '
          'source/lifetime diagnostics, authenticated processor cache lifecycle; native UNRUN')


if __name__ == '__main__':
    main()
