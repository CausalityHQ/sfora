#!/usr/bin/env python3
"""Stdlib source/authority contract only; never qualifies native gradients."""
import ast
import json
import math
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'qualify_connected_encoder_gradients.py'


def rejects(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), (message, str(error))
    else:
        raise AssertionError('accepted: ' + message)


def main():
    assert DRIVER.is_file(), 'missing bounded encoder-gradient probe'
    d = runpy.run_path(str(DRIVER))
    assert not {'torch', 'numpy', 'transformers', 'PIL', 'safetensors'} & sys.modules.keys()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        for name in d['FILES']:
            shutil.copyfile(HERE / name, root / name)
        local = runpy.run_path(str(root / DRIVER.name))
        manifest = {'schema': 'connected-encoder-gradients-authority-v1',
                    'files': {n: d['sha'](root / n) for n in d['FILES']},
                    'python': {'path': str(Path(sys.executable).resolve()),
                               'sha256': d['sha'](Path(sys.executable).resolve())},
                    'output': str(root / 'result'), 'resource_policy': d['POLICY'],
                    'both_locks_held': True}
        authority = root / 'authority.json'

        def admit(value=manifest):
            authority.write_text(json.dumps(value))
            return local['authenticate'](authority, d['sha'](authority), root / 'result')

        guards = admit()
        assert len(guards) == len(d['FILES']) + 2
        for key, bad in [('schema', 'other'), ('both_locks_held', 1),
                         ('resource_policy', {**d['POLICY'], 'seconds': 301}),
                         ('resource_policy', {**d['POLICY'], 'swap_bytes': False}),
                         ('output', str(root / 'other')),
                         ('files', {**manifest['files'], 'extra.py': 'a' * 64}),
                         ('python', {**manifest['python'], 'sha256': 'a' * 64})]:
            rejects(lambda: admit({**manifest, key: bad}),
                    'interpreter' if key == 'python' else 'authority')
        for name in d['FILES']:
            path, old = root / name, (root / name).read_bytes()
            path.write_bytes(old + b'\n# changed\n')
            rejects(admit, 'source')
            path.write_bytes(old)
        authority.write_text('{"schema":1,"schema":2}')
        rejects(lambda: local['authenticate'](authority, d['sha'](authority), root / 'result'),
                'duplicate')
        rejects(lambda: local['authenticate'](authority, 'a' * 64, root / 'result'), 'SHA256')
        (root / 'result').mkdir()
        rejects(admit, 'new output')

    class Parameter:
        def __init__(self):
            self.requires_grad = True
            self.grad = None
            self.shape = (1,)
            self.dtype = 'torch.float32'
            self.device = SimpleNamespace(type='cpu')

        def requires_grad_(self, value):
            self.requires_grad = value

    params = {name: Parameter() for name in d['MLP']}
    params.update({f'frozen.{i}': Parameter() for i in range(444)})
    model = SimpleNamespace(named_parameters=lambda: iter(params.items()), state_dict=lambda: params)
    expected = {name: [1] for name in params}
    d['select_mlp'](model, expected)
    assert {name for name, p in params.items() if p.requires_grad} == set(d['MLP'])
    one = next(iter(params.values()))
    one.grad = object()
    rejects(lambda: d['select_mlp'](model, expected), 'source parameter')
    one.grad = None
    rejects(lambda: d['select_mlp'](model, dict(list(expected.items())[:-1])), '448')

    class Scalar:
        def __init__(self, value):
            self.value = value

        def all(self):
            return self

        def item(self):
            return self.value

    class Gradient(list):
        def norm(self):
            return math.sqrt(sum(x*x for x in self))

    torch = SimpleNamespace(isfinite=lambda g: Scalar(all(math.isfinite(x) for x in g)),
                            count_nonzero=lambda g: Scalar(sum(x != 0 for x in g)))
    one.grad = Gradient([3., 4.])
    assert d['gradient_fact'](torch, one, 'probe') == {'norm': 5., 'nonzero': 2}
    for grad in (None, Gradient([0., 0.]), Gradient([math.inf]), Gradient([math.nan])):
        one.grad = grad
        rejects(lambda: d['gradient_fact'](torch, one, 'probe'), 'gradient')

    tree = ast.parse(DRIVER.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # Execute the real source-context TRAIN predicate against independently bound rows.
    witness_loop = next(n for n in functions['source_context'].body
                        if isinstance(n, ast.For) and ast.unparse(n.target) == 'row')
    witness_code = compile(ast.Module(body=[witness_loop], type_ignores=[]), str(DRIVER), 'exec')
    rows = [{'relative_path': 'Img/img/a.jpg', 'product': 'a', 'image_sha256': 'a'*64, 'train_row': 3},
            {'relative_path': 'Img/img/b.jpg', 'product': 'b', 'image_sha256': 'b'*64, 'train_row': 9}]
    control = [{**{k: v for k, v in r.items() if k != 'train_row'}, 'original_train_row': r['train_row']}
               for r in rows]
    scope = {'control': {'rows': control}}
    ns = {'context': {'fit': {'rows': rows}}, 'scope': scope, 'require': d['require']}
    exec(witness_code, ns)
    for key in ('relative_path', 'product', 'image_sha256', 'original_train_row'):
        saved = control[1][key]
        control[1][key] = 'substitution'
        rejects(lambda: exec(witness_code, ns), 'frozen control TRAIN')
        control[1][key] = saved

    profile_path = HERE.parent / ('docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
                                 'new-runtime-encoder-profile-v1/driver.py')
    profile = ast.parse(profile_path.read_text())
    source_constants = {}
    for n in profile.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ('SOURCE_E', 'PINS'):
                source_constants[n.targets[0].id] = ast.literal_eval(n.value)
    assert d['SOURCE_E'] == source_constants['SOURCE_E']
    assert d['SOURCE_PINS'] == source_constants['PINS']
    for name, digest in d['SOURCE_PINS'].items():
        assert d['sha'](HERE / name) == digest
    calls = [n for n in ast.walk(functions['qualify']) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    names = [ast.unparse(n.func) for n in calls]
    ordered = ['authenticate', 'source_context', 'q.cgroup_memory', 'q.package_origins',
               'q.fresh_source', 'q.pixels_and_raw', 'select_mlp', 'model.to', 'probe',
               'q.imported_origins', 'q.rehash', 'q.cgroup_memory']
    position = 0
    for name in ordered:
        position = names.index(name, position) + 1
    assert names.count('q.fresh_source') == 1
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith(
        ('.save', '.load', '.step', '.reset_peak_memory_stats', '.half')) for n in ast.walk(tree))
    deletes = {s.lineno: {n.id for n in ast.walk(s) if isinstance(n, ast.Name)}
               for s in ast.walk(functions['qualify']) if isinstance(s, ast.Delete)}
    rehash = next(c.lineno for c in calls if ast.unparse(c.func) == 'q.rehash')
    assert any({'model', 'processor', 'pixels', 'cpu_raw'} <= ns and line < rehash
               for line, ns in deletes.items())
    probe = ast.unparse(functions['probe'])
    assert 'pixels.repeat(8, 1, 1, 1)' in probe
    assert "head_from('control', tensors=tensors)" in probe
    assert probe.count('.backward()') == 2
    assert 'features.retain_grad()' in probe and 'model.zero_grad(set_to_none=True)' in probe
    assert 'features.detach()' in probe and 'torch.equal(raw.detach(), mutant.detach())' in probe
    assert 'p.grad is None for p in model.parameters()' in probe
    factory = ast.parse((HERE / 'train_siglip2_cached_readout.py').read_text())
    factory = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == 'head_from')
    assert [a.arg for a in factory.args.args] == ['arm', 'tensors', 'values', 'features']
    assert not {'torch', 'numpy', 'transformers', 'PIL', 'safetensors'} & sys.modules.keys()
    print('PASS stdlib admission, role/gradient predicates, original API, ordering and lifetime; '
          'native encoder gradients NOT executed or qualified')


if __name__ == '__main__':
    main()
