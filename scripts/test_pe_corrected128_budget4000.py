#!/usr/bin/env python3
"""Stdlib-only identity boundary check; no Torch/model on devbox."""
import ast
import copy
from pathlib import Path
import subprocess
import sys

path = Path(__file__).with_name('train_pe_corrected128_budget4000.py')
tree = ast.parse(path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rebase')
scope = {'OLD': 'a' * 64, 'METHOD': 'corrected128-budget4000-v1', 'UPDATES': 4000}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
old = {'schema': 'native-optimization-resume-v1', 'arm': 'full', 'total_updates': 2000,
       'global_step': 2000, 'seed': 179032, 'execution_sha256': 'a' * 64,
       'schedule_sha256': 'b' * 64, 'intervention': 'full-valid-anchor-v1',
       'parameter_names': ['vision.x', 'compact_head.weight', 'classifier'],
       'optimizer_groups': [{'lr': 1e-5}], 'buffers_sha256': 'd' * 64,
       'frozen_prefix_sha256': 'e' * 64, 'source_checkpoint_sha256': 'f' * 64,
       'precision': 'cuda_fp16'}
new = {**old, 'total_updates': 4000, 'execution_sha256': 'c' * 64,
       'schedule_sha256': '1' * 64, 'intervention': 'corrected128-budget4000-v1'}
del new['global_step']
assert scope['rebase'](old, new) == new
for name, value in [('total_updates', 3999), ('schedule_sha256', 'b' * 64),
                    ('execution_sha256', 'a' * 64), ('intervention', 'other'),
                    ('seed', 0), ('parameter_names', []), ('buffers_sha256', '0' * 64),
                    ('optimizer_groups', []), ('precision', 'cpu_float32')]:
    damaged = copy.deepcopy(new); damaged[name] = value
    try:
        scope['rebase'](old, damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity mutation accepted: ' + name)
for name, value in [('global_step', 1999), ('total_updates', 4000),
                    ('execution_sha256', '0' * 64)]:
    damaged = copy.deepcopy(old); damaged[name] = value
    try:
        scope['rebase'](damaged, new)
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid parent accepted: ' + name)
for flag in ('-O', '-OO'):
    result = subprocess.run([sys.executable, flag, str(path), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
print('PASS strict2000→4000 identity-only rebase and optimized rejection; no model')
