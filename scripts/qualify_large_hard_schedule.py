"""Fit-only CPU feasibility gate; no model construction or held quality read."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from large_hard_schedule import margins, schedule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / 'hard-schedule-execution.json'
    assert sha(manifest) == args.execution_sha256 and not args.output.exists()
    code = json.loads(manifest.read_text())
    previous = json.loads((root / 'dense-pilot-execution.json').read_text())
    assert len(previous) == 100 and len(code) == 103
    assert set(code) == set(previous) | {'large_hard_schedule.py', 'check_large_hard_schedule.py', 'qualify_large_hard_schedule.py'}
    assert all(code[n] == h for n, h in previous.items()) and all(sha(root / n) == h for n, h in code.items())
    cpu = Path('/home/riomus/runs/sfora-large-coverage-cpu-v2')
    assert sha(cpu / 'receipt.json') == 'fbb88035f2bdebaa243cdf4e3e2011fc91ec15a0d6722422b448b2ff4b88c61b'
    proof = json.loads((cpu / 'receipt.json').read_text())
    half = proof['arms']['half']
    assert proof['pass'] and sha(cpu / 'half.npz') == half['initializers_sha256']
    fit = half['rows']
    held = [r for r in proof['arms']['full']['rows'] if r['product'] not in set(half['classes'])]
    assert len(fit) == 13283 and len(held) == 12599
    assert set(r['product'] for r in fit).isdisjoint(r['product'] for r in held)
    with np.load(cpu / 'half.npz', allow_pickle=False) as values:
        target, bank = values['target'].copy(), values['bank'].copy()
    assert target.dtype == np.int64 and len(set(target)) == 2004 and bank.shape == (13283, 128)
    assert [half['classes'][int(i)] for i in target] == [r['product'] for r in fit]
    started = time.perf_counter()
    margin = margins(target, bank)
    pool = np.flatnonzero(margin < 0)
    # The reference is loaded as an AST function, without importing Torch/models.
    tree = ast.parse((root / 'pe_large_coverage.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    fn.args.defaults = [ast.Constant(0)]
    scope = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'authenticated-uniform', 'exec'), scope)
    results, schedules = {}, {}
    for seed in (179032, 179041):
        uniform = scope['schedule'](target, seed)
        assert np.array_equal(schedule(target, margin, seed, hard_slots=0), uniform)
        baseline = float((margin[uniform] < 0).mean())
        try:
            candidate = schedule(target, margin, seed)
        except ValueError as exc:
            results[seed] = {'feasible': False, 'reason': str(exc), 'uniform_hard_fraction': baseline}
            continue
        assert np.array_equal(candidate, schedule(target, margin, seed))
        fraction = float((margin[candidate] < 0).mean())
        enrichment = fraction / baseline if baseline else float('inf')
        results[seed] = {'feasible': True, 'uniform_hard_fraction': baseline,
            'candidate_hard_fraction': fraction, 'hard_enrichment': enrichment,
            'maximum_hard_slot_exposure': int(np.bincount(candidate[:, 32:].ravel(), minlength=len(target)).max()),
            'schedule_sha256': hashlib.sha256(candidate.tobytes()).hexdigest()}
        schedules[str(seed)] = candidate
    admitted = all(v['feasible'] and v['hard_enrichment'] >= 1.5 for v in results.values())
    assert all(sha(root / n) == h for n, h in code.items())
    assert sha(cpu / 'half.npz') == half['initializers_sha256']
    args.output.mkdir()
    np.savez(args.output / 'schedule.npz', margin=margin, **schedules)
    result = {'pass': True, 'pilot_admitted': admitted, 'decision': 'ADMIT QUALIFICATION' if admitted else 'KILL SAMPLING PROCEDURE',
        'execution_sha256': args.execution_sha256, 'source_code': code, 'fit_initializers_sha256': half['initializers_sha256'],
        'dataset': 'DeepFashion In-Shop', 'split': 'TRAIN-fit only', 'fit_images': len(target), 'fit_products': 2004,
        'hard_rows': len(pool), 'hard_products': len(set(target[pool])), 'seeds': results,
        'schedule_artifact_sha256': sha(args.output / 'schedule.npz'), 'seconds': time.perf_counter() - started,
        'uniform_reference_exact': True, 'fit_held_products_disjoint': True, 'optimizer_updates': 0,
        'model_constructed': False, 'held_quality_read': False,
        'frozen_feasibility': '>=320 hard rows; >=64 hard products; complete100x64/coverage; <=10 hard-slot repeats; >=1.5x realized hard-anchor share for bothseeds',
        'global_production_goal_met': False}
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_code'}), flush=True)


if __name__ == '__main__':
    main()
