#!/usr/bin/env python3
"""Full-rule admission rejects failed TRAIN evidence and optimized Python."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

path = Path(__file__).with_name('qualify_pe_full_native_trajectory_average.py')
assert path.is_file(), 'full fixed average driver missing'
import qualify_pe_full_native_trajectory_average as driver

value = json.loads(Path('/home/riomus/runs/sfora-native-trajectory-average-decision-v1/decision.json').read_text())
driver.require_train_go(value)
for mutate in (
    lambda r: r.__setitem__('decision', 'KILL'),
    lambda r: r.__setitem__('integrity_resource_pass', False),
    lambda r: r.__setitem__('additional_training_updates', 1),
    lambda r: r['comparison']['per_query_r1'].__setitem__('product_lower95', 0.),
    lambda r: r['comparison']['per_query_ap'].__setitem__('mean_difference', -.001),
):
    r = copy.deepcopy(value); mutate(r)
    try:
        driver.require_train_go(r)
    except AssertionError:
        pass
    else:
        raise AssertionError('failed TRAIN admission accepted')
for mode in ('-O', '-OO', 'env'):
    env = dict(os.environ)
    if mode == 'env':
        env['PYTHONOPTIMIZE'] = '1'
    p = subprocess.run([sys.executable] + ([] if mode == 'env' else [mode]) + [str(path), '--help'], env=env, capture_output=True, text=True, timeout=10)
    assert p.returncode and 'optimized mode is forbidden' in p.stderr
print('PASS actual TRAIN GO admission, five damaged gate rejections and optimized modes')
