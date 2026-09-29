#!/usr/bin/env python3
"""One exact-state corrected full128 continuation, steps2001..4000 on DGX."""
if not __debug__:
    raise SystemExit('optimized mode is forbidden')

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import train_pe_full_valid_anchor as previous

driver, lane, pair = previous.driver, previous.lane, previous.pair
OLD = '0da63376f73c9bc55c4c5f6e8c0f3f80fab7efe211ef6234a72af2ce4ed3ac36'
METHOD = 'corrected128-budget4000-v1'
PARENT = Path('/home/riomus/runs/sfora-full-valid-anchor-2000-v1')
PARENT_RECEIPT = 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
PARENT_CHECKPOINT = 'e32dd813c456a0ed319d933b74ffbc9b42886cf0641cb673e4c36945b90c43fa'
UPDATES = 4000


def rebase(old, new):
    assert old['global_step'] == old['total_updates'] == 2000
    assert old['execution_sha256'] == OLD and old['intervention'] == 'full-valid-anchor-v1'
    assert new['total_updates'] == UPDATES and new['intervention'] == METHOD
    assert new['execution_sha256'] != old['execution_sha256'] and new['schedule_sha256'] != old['schedule_sha256']
    assert set(new) == set(old) - {'global_step'}
    allowed = {'total_updates', 'execution_sha256', 'schedule_sha256', 'intervention', 'global_step'}
    assert all(new[k] == v for k, v in old.items() if k not in allowed)
    return new


def source_guard(root, expected):
    path = root / 'corrected128-budget4000-execution.json'
    assert pair.sha(path) == expected
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'budget4000 source differs'
    original = root / 'full-valid-anchor-training-execution.json'
    assert pair.sha(original) == OLD
    old = json.loads(original.read_text())
    assert len(old) == 101 and all(code[n] == h for n, h in old.items())
    assert set(code) - set(old) == {'train_pe_corrected128_budget4000.py', 'test_pe_corrected128_budget4000.py',
                                    'inshop_corrected128_budget4000_gate_2026-09-29.md'}
    return code, old


def authority(root, expected):
    code, old = source_guard(root, expected)
    selected = previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, inherited, baseline = previous.startup(root, OLD)
    assert inherited == old and len(baseline) == 20
    assert pair.sha(PARENT / 'receipt.json') == PARENT_RECEIPT
    parent = json.loads((PARENT / 'receipt.json').read_text())
    assert parent['pass'] and parent['intervention'] == 'full-valid-anchor-v1' and parent['arm'] == 'full'
    assert parent['completed_step'] == parent['total_updates'] == 2000 and parent['execution_sha256'] == OLD
    assert parent['checkpoint_sha256'] == PARENT_CHECKPOINT and not parent['quality_read']
    return control, source, prior, proof, code, parent


def schedules(proof, parent):
    target = driver.initializers(proof, 'full')['target'].numpy()
    with patch.object(driver, 'TOTAL_UPDATES', 2000):
        old = driver.schedule(target)
    old_sha = pair.smoke.digest({'batches': torch.from_numpy(old)})
    assert old_sha == parent['schedule_sha256']
    with patch.object(driver, 'TOTAL_UPDATES', UPDATES):
        new = driver.schedule(target)
    assert new.shape == (UPDATES, 64) and np.array_equal(new[:2000], old)
    assert np.array_equal(target[new[:2000]], target[old])
    counts = np.bincount(target, minlength=3997)
    assert len(counts) == 3997 and counts.min() >= 1
    exposure = np.bincount(target[new.ravel()], minlength=3997)
    assert exposure.min() >= 1 and exposure.sum() == UPDATES * 64
    eligible = int(np.all(counts[target[new[2000:]]] > 1, axis=1).sum())
    return pair.smoke.digest({'batches': torch.from_numpy(new)}), int(exposure.min()), int(exposure.max()), eligible


def cpu(root, args, proof, code, parent):
    assert not torch.cuda.is_available() and not args.output.exists()
    schedule_sha, minimum, maximum, eligible = schedules(proof, parent)
    assert pair.sha(PARENT / 'resume.pt') == PARENT_CHECKPOINT
    saved = torch.load(PARENT / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    assert set(saved) == {'identity', 'vision', 'buffers', 'head', 'classifier', 'bank', 'optimizer', 'scaler', 'cpu_rng', 'cuda_rng'}
    identity = saved['identity']
    assert identity['global_step'] == identity['total_updates'] == 2000 and identity['execution_sha256'] == OLD
    assert identity['intervention'] == 'full-valid-anchor-v1' and identity['arm'] == 'full'
    assert identity['schedule_sha256'] == parent['schedule_sha256']
    assert len(saved['vision']) == 400 and saved['head']['weight'].shape == (128, 1024)
    assert saved['classifier'].shape == (3997, 128) and saved['bank'].shape == (25882, 128)
    assert len(saved['optimizer']['state']) == len(identity['parameter_names'])
    assert all(int(s['step']) == 2000 for s in saved['optimizer']['state'].values())
    assert driver.fingerprint(saved['buffers']) == identity['buffers_sha256']
    future = {k: v for k, v in identity.items() if k != 'global_step'}
    future.update(total_updates=UPDATES, execution_sha256=args.execution_sha256,
                  schedule_sha256=schedule_sha, intervention=METHOD)
    rebase(identity, future)
    original = pair.sha
    with patch.object(pair, 'sha', lambda p: 'altered' if Path(p).resolve() == Path(__file__).resolve() else original(p)):
        try:
            source_guard(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'budget4000 source differs'
        else:
            raise AssertionError('changed continuation accepted')
    assert all(pair.sha(root / n) == h for n, h in code.items())
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'execution_sha256': args.execution_sha256,
        'code': code, 'parent_receipt_sha256': PARENT_RECEIPT, 'parent_checkpoint_sha256': PARENT_CHECKPOINT,
        'old_schedule_sha256': parent['schedule_sha256'], 'schedule_sha256': schedule_sha,
        'exact_first2000_index_and_class_prefix': True, 'total_updates': UPDATES,
        'fit_rows': 25882, 'classes': 3997, 'minimum_exposure': minimum, 'maximum_exposure': maximum,
        'extension_rank_eligible_updates': eligible,
        'complete_parent_state_and_identity_verified': True, 'allowed_identity_rebase_only': True,
        'changed_source_rejected': True, 'optimizer_updates': 0, 'quality_read': False})
    print('PASS authenticated native parent/schedule-prefix/identity admission; no update or quality')


def train(root, args, control, source, prior, proof, code, parent, cpu_proof):
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert args.chunk_end in range(2100, 4001, 100) and not args.output.exists()
    assert driver.coverage.teacher.qualified.numerical_flags() == prior['numerical_flags']
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state = driver.fresh(control, source, proof, 'full', 'cuda')
    initial = driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k]
                                  for k in ('model', 'head', 'classifier', 'bank')})
    assert initial == parent['initial_state_sha256']
    target = state['target'].cpu().numpy()
    with patch.object(driver, 'TOTAL_UPDATES', 2000):
        old_batches = driver.schedule(target)
        old_sha = pair.smoke.digest({'batches': torch.from_numpy(old_batches)})
    assert old_sha == parent['schedule_sha256'] == cpu_proof['old_schedule_sha256']
    with patch.object(driver, 'TOTAL_UPDATES', UPDATES):
        batches = driver.schedule(target)
        assert np.array_equal(batches[:2000], old_batches)
        schedule_sha = pair.smoke.digest({'batches': torch.from_numpy(batches)})
        assert schedule_sha == cpu_proof['schedule_sha256']
        identity = driver.identity(state, proof, 'full', args.execution_sha256, schedule_sha)
        identity['intervention'] = METHOD
        start = args.chunk_end - 100
        if start == 2000:
            assert not args.previous and not args.previous_sha256
            with patch.object(driver, 'TOTAL_UPDATES', 2000):
                old_identity = driver.identity(state, proof, 'full', OLD, old_sha)
                old_identity['intervention'] = 'full-valid-anchor-v1'
                driver.restore(state, old_identity, PARENT / 'resume.pt', PARENT_CHECKPOINT, 2000)
            rebase({**old_identity, 'global_step': 2000}, identity)
            previous_sha = PARENT_RECEIPT
        else:
            assert args.previous and args.previous_sha256 and pair.sha(args.previous / 'receipt.json') == args.previous_sha256
            previous_receipt = json.loads((args.previous / 'receipt.json').read_text())
            assert previous_receipt['pass'] and previous_receipt['completed_step'] == start
            assert previous_receipt['intervention'] == METHOD and previous_receipt['execution_sha256'] == args.execution_sha256
            assert previous_receipt['schedule_sha256'] == schedule_sha and previous_receipt['initial_state_sha256'] == initial
            driver.restore(state, identity, args.previous / 'resume.pt', previous_receipt['checkpoint_sha256'], start)
            previous_sha = args.previous_sha256
        assert state['counter'] == start
        counts = np.bincount(target)
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        rows = []
        torch.cuda.synchronize(); tick = time.perf_counter()
        for step in range(start + 1, args.chunk_end + 1):
            torch.cuda.synchronize(); moment = time.perf_counter()
            batch = tuple(batches[step - 1].tolist())
            images, rgb = pair.augmented_images(control.dataset_root, proof['arms']['full']['rows'], batch, step)
            pixels = pair.pixels(state['processor'], images, 'large')
            pixel_sha = pair.smoke.digest({'pixels': pixels})
            active = bool((counts[target[list(batch)]] > 1).all())
            with patch.object(driver.coverage, 'terms', lane.terms):
                row = driver.step(state, pixels, batch, active)
            assert state['counter'] == step and row['rank'] >= 0 and torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize()
            row.update(rgb_sha256=rgb, pixels_sha256=pixel_sha, rank_active_before=active,
                       recovered_rank_update=not active, seconds=time.perf_counter() - moment)
            rows.append(row); print(json.dumps(row), flush=True)
        wall = time.perf_counter() - tick
        assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
        assert driver.fingerprint(dict(state['model'].named_buffers())) == identity['buffers_sha256']
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
        assert driver.coverage.teacher.qualified.numerical_flags() == prior['numerical_flags']
        assert all(pair.sha(root / n) == h for n, h in code.items())
        args.output.mkdir(exist_ok=False)
        checkpoint_sha = driver.save(state, identity, args.output / 'resume.pt')
        pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'intervention': METHOD, 'arm': 'full', 'seed': pair.SEED,
            'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': args.cpu_sha256,
            'parent_receipt_sha256': PARENT_RECEIPT, 'parent_checkpoint_sha256': PARENT_CHECKPOINT,
            'initial_state_sha256': initial, 'schedule_sha256': schedule_sha,
            'previous_receipt_sha256': previous_sha, 'chunk_start': start, 'completed_step': args.chunk_end,
            'total_updates': UPDATES, 'checkpoint_sha256': checkpoint_sha, 'training_wall_seconds': wall,
            'images_per_second': 6400 / wall, 'steps': rows, 'recovered_rank_updates': sum(r['recovered_rank_update'] for r in rows),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'total_seconds': time.perf_counter() - started,
            'frozen_source_code_environment_rng_preserved': True, 'quality_read': False})
    print('PASS corrected full128 100-update continuation; no quality read', flush=True)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True); p.add_argument('--phase', choices=('cpu', 'train'), required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--chunk-end', type=int)
    p.add_argument('--cpu-proof', type=Path); p.add_argument('--cpu-sha256')
    p.add_argument('--previous', type=Path); p.add_argument('--previous-sha256')
    args = p.parse_args(); root = Path(__file__).resolve().parent
    torch.set_num_threads(8); torch.manual_seed(pair.SEED)
    control, source, prior, proof, code, parent = authority(root, args.execution_sha256)
    if args.phase == 'cpu':
        assert args.chunk_end is None and not args.cpu_proof and not args.previous
        cpu(root, args, proof, code, parent)
    else:
        assert args.cpu_proof and pair.sha(args.cpu_proof / 'receipt.json') == args.cpu_sha256
        admission = json.loads((args.cpu_proof / 'receipt.json').read_text())
        assert admission['pass'] and admission['execution_sha256'] == args.execution_sha256
        assert admission['complete_parent_state_and_identity_verified'] and admission['exact_first2000_index_and_class_prefix']
        assert admission['parent_checkpoint_sha256'] == PARENT_CHECKPOINT and not admission['quality_read']
        train(root, args, control, source, prior, proof, code, parent, admission)


if __name__ == '__main__':
    main()
