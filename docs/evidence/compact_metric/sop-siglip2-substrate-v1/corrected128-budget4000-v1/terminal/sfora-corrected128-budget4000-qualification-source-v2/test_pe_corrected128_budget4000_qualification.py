#!/usr/bin/env python3
"""Stdlib-only frozen official spending gate; no Torch/model on devbox."""
import ast
import copy
import math
from pathlib import Path
import subprocess
import sys

path = Path(__file__).with_name('qualify_pe_corrected128_budget4000.py')
tree = ast.parse(path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'survives')
scope = {'math': math}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
endpoint = {'map_at_r': .818825927188}
positive = {'recall_at_1': .967000001, 'map_at_r': endpoint['map_at_r']}
intervals = {'per_query_r1': {'product_lower95': .00001}}
assert scope['survives'](positive, endpoint, intervals)
for key, value in [('recall_at_1', .967), ('map_at_r', endpoint['map_at_r'] - 1e-8),
                   ('recall_at_1', float('nan')), ('map_at_r', float('nan'))]:
    damaged = copy.deepcopy(positive); damaged[key] = value
    assert not scope['survives'](damaged, endpoint, intervals)
for value in (0., -.0001, float('nan')):
    assert not scope['survives'](positive, endpoint, {'per_query_r1': {'product_lower95': value}})
for flag in ('-O', '-OO'):
    result = subprocess.run([sys.executable, flag, str(path), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
print('PASS official point/paired-product/mAP floors and optimized rejection; no model')
