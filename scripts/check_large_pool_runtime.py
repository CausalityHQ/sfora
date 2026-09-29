"""Stdlib regression for live config versus JSON proof equality, without Torch."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

tree = ast.parse(Path(__file__).with_name('export_large_pool_checkpoint.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'runtime')
value = {'config': {'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}},
         'buffer_devices': {'position_ids': 'cpu'}}
scope = {'native': SimpleNamespace(runtime_identity=lambda _: value), 'json': json}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'runtime', 'exec'), scope)
model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cpu')]))
expected = json.loads(json.dumps({'config': value['config']}))
assert scope['runtime'](model) == expected
value['config']['id2label'][0] = 'MUTATED'
assert scope['runtime'](model) != expected
value['buffer_devices']['position_ids'] = 'cuda:0'
try:
    scope['runtime'](model)
except AssertionError:
    pass
else:
    raise AssertionError('wrong buffer device accepted')
print('PASS JSON config equality; actual config mutation and wrong buffer device rejected')
