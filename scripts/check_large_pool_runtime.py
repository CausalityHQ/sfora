"""Stdlib regressions for JSON proof equality and the CPU-only wire boundary."""
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

class Descriptor:
    def __init__(self, device):
        self.device = device
    def detach(self):
        return self
    def cpu(self):
        return Descriptor('cpu')

devices = []
def pack(value):
    devices.append(value.device)
    assert value.device == 'cpu', 'CPU-only wire packer received CUDA descriptor'
    return SimpleNamespace(codes='codes', inverse_norms='norms')

fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'packed_equal')
scope = {'pack_int8_unit_embeddings': pack,
         'np': SimpleNamespace(array_equal=lambda a, b: a == b)}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'packed_equal', 'exec'), scope)
scope['packed_equal'](Descriptor('cuda'), Descriptor('cuda'))
assert devices == ['cpu', 'cpu']
print('PASS both CUDA descriptors cross the CPU-only wire boundary')
