#!/usr/bin/env python3
"""Focused fixed-rule averaging invariants; run Torch only on DGX."""
import copy
import os
import subprocess
import sys
from pathlib import Path

path = Path(__file__).with_name('qualify_pe_native_trajectory_average.py')
assert path.is_file(), 'fixed trajectory averaging driver missing'
import torch
import qualify_pe_native_trajectory_average as driver
from qualify_pe_native_trajectory_average import average_native


def main():
    assert hasattr(driver, 'survives'), 'prospective decision gate missing'
    effect = {'per_query_r1': {'product_lower95': .001}, 'per_query_ap': {'mean_difference': 0.}}
    assert driver.survives(effect)
    effect['per_query_r1']['product_lower95'] = 0.
    assert not driver.survives(effect)
    effect['per_query_r1']['product_lower95'] = .001
    effect['per_query_ap']['mean_difference'] = -.0001
    assert not driver.survives(effect)
    states = []
    for step in range(1100, 2001, 100):
        states.append({'identity': {'global_step': step, 'model_roles': [('frozen', False), ('train', True)]},
                       'vision': {'frozen': torch.tensor([-0., 3.]), 'train': torch.tensor([step / 100., -step / 300.])},
                       'head': {'weight': torch.tensor([[step / 400.]]), 'bias': torch.tensor([step / 500.])},
                       'buffers': {'integer': torch.tensor([2]), 'float': torch.tensor([-0., 4.])}})
    result = average_native(tuple(states))
    assert set(result) == {'vision', 'head'}
    assert torch.equal(result['vision']['frozen'].view(torch.int32), states[-1]['vision']['frozen'].view(torch.int32))
    for role, name in (('vision', 'train'), ('head', 'weight'), ('head', 'bias')):
        expected = torch.stack([s[role][name].double() for s in states]).mean(0).float()
        assert torch.equal(result[role][name], expected)
    for mutate in (
        lambda s: s[0]['vision']['frozen'].add_(1),
        lambda s: s[0]['buffers']['integer'].add_(1),
        lambda s: s[0]['buffers']['float'].fill_(0),
        lambda s: s[0]['vision']['train'].fill_(float('nan')),
        lambda s: s[0]['identity'].__setitem__('global_step', 1000),
        lambda s: s[0]['identity'].__setitem__('model_roles', [('frozen', True), ('train', True)]),
    ):
        damaged = copy.deepcopy(states)
        mutate(damaged)
        try:
            average_native(tuple(damaged))
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('invalid averaging input accepted')
    for mode in ('-O', '-OO', 'env'):
        env = dict(os.environ)
        if mode == 'env':
            env['PYTHONOPTIMIZE'] = '1'
        args = [sys.executable] + ([] if mode == 'env' else [mode]) + [str(path), '--help']
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
        assert p.returncode and 'optimized mode is forbidden' in p.stderr
    print('PASS fixed FP64 trainables / frozen bytes / buffer-role-step-finite rejection / optimized modes')


if __name__ == '__main__':
    main()
