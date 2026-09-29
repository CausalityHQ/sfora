#!/usr/bin/env python3
"""Stdlib-only spending-gate/entry checks; no Torch or models on devbox."""
import ast
import copy
import math
from pathlib import Path
import subprocess
import sys

path = Path(__file__).with_name('qualify_pe_teacher_retained256.py')
tree = ast.parse(path.read_text())
gate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'survives')
scope = {'math': math}
exec(compile(ast.Module(body=[gate], type_ignores=[]), str(path), 'exec'), scope)
positive = {'per_query_r1': {'mean_difference': .005, 'product_lower95': .001},
            'per_query_ap': {'mean_difference': .01, 'product_lower95': .002}}
costs = {'training_wall_ratio': 1.10, 'median_step_ratio': 1.10}
assert scope['survives'](positive, costs)
for metric, field, value in [('per_query_r1', 'mean_difference', .004999),
                             ('per_query_ap', 'mean_difference', .009999),
                             ('per_query_r1', 'product_lower95', 0.),
                             ('per_query_ap', 'product_lower95', 0.),
                             ('per_query_r1', 'mean_difference', float('nan'))]:
    damaged = copy.deepcopy(positive); damaged[metric][field] = value
    assert not scope['survives'](damaged, costs)
for field in costs:
    for value in (0., 1.100001, float('nan')):
        assert not scope['survives'](positive, {**costs, field: value})
for flags in (['-O'], ['-OO']):
    result = subprocess.run([sys.executable, *flags, str(path), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
print('PASS exact spending floors/zero interval/nonfinite/invalid costs/optimized rejection; no model quality claim')
