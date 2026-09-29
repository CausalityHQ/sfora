"""Numpy-only scheduler regression; run on DGX, not the devbox."""
import ast
from pathlib import Path

import numpy as np

from large_hard_schedule import margins, schedule


def main():
    target = np.repeat(np.arange(256, dtype=np.int64), 4)
    rng = np.random.default_rng(7)
    bank = rng.normal(size=(len(target), 16)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    got = margins(target, bank)
    scores = bank @ bank.T
    np.fill_diagonal(scores, -np.inf)
    expected = np.array([scores[i, target == c].max() - scores[i, target != c].max()
                         for i, c in enumerate(target)])
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
    module = ast.parse(Path(__file__).with_name('pe_large_coverage.py').read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    fn.args.defaults = [ast.Constant(0)]
    scope = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'uniform', 'exec'), scope)
    for seed in (179032, 179041):
        uniform = scope['schedule'](target, seed)
        assert np.array_equal(schedule(target, got, seed, hard_slots=0), uniform)
        a = schedule(target, got, seed)
        assert np.array_equal(a, schedule(target, got, seed))
        assert a.shape == (100, 64) and all(len(set(target[b])) == 64 for b in a)
        assert set(target[a.ravel()]) == set(target)
        hard = a[:, 32:].ravel()
        assert (got[hard] < 0).all() and np.bincount(hard, minlength=len(target)).max() <= 10
    single = margins(np.arange(4, dtype=np.int64), np.eye(4, dtype=np.float32))
    assert np.isposinf(single).all()
    for bad in (np.ones_like(got), np.where(target < 32, -1., 1.)):
        try:
            schedule(target, bad, 7)
        except ValueError:
            pass
        else:
            raise AssertionError('infeasible hard pool accepted')
    print('PASS margins/self/singleton masks, uniform equality, deterministic coverage and finite hard-repeat cap')


if __name__ == '__main__':
    main()
