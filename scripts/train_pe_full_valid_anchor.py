#!/usr/bin/env python3
"""One discarded full17 mechanics or one100 chunk of a fixed full2000 candidate."""
import argparse
import copy
import gc
import json
import os
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import qualify_pe_full_valid_anchor_cpu as cpu

lane, driver = cpu.lane, cpu.driver
pair = driver.pair
CPU_CODE = 'd709b5d49e4bab4d5de8060c8d72f96bc8fbb2cad0860a29bc5da4a586f7764d'
CPU_ROOT = Path('/home/riomus/runs/sfora-full-valid-anchor-cpu-v1')
CPU_SHA = '1c78abcc8554bc69444712f24b3a15bd2d26fd073bd86b1ce6652916b74bf342'


def startup(root, expected):
    path = root / 'full-valid-anchor-training-execution.json'
    assert pair.sha(path) == expected
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'full training source differs'
    control, source, prior, proof, old = cpu.authority(root, CPU_CODE, allowed=code)
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    assert pair.sha(CPU_ROOT / 'full-valid-anchor-cpu-proof.json') == CPU_SHA
    qualified = json.loads((CPU_ROOT / 'full-valid-anchor-cpu-proof.json').read_text())
    assert qualified['pass'] and qualified['code'] == old and qualified['fit_rows'] == 25882 and qualified['classes'] == 3997
    assert qualified['same_CE_raw_active_rank_exact'] and qualified['individual_and_micro_denominator_exact']
    assert qualified['singleton_rank_gradient_zero_valid_native_gradient_nonzero'] and qualified['frozen_whole_head_bank_classifier_preserved']
    assert qualified['changed_driver_rejected'] and qualified['CPU_rng_preserved'] and qualified['optimizer_updates'] == 0 and not qualified['quality_read']
    log = (CPU_ROOT / 'full-valid-anchor-cpu.log').read_text()
    assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
    baselines = {}
    last_sha = None
    for end in range(100, 2001, 100):
        path = Path(f'/home/riomus/runs/sfora-large-optimization-full-{end}-v1/receipt.json')
        row = json.loads(path.read_text())
        assert row['advance'] and row['completed_step'] == end and row['chunk_start'] == end - 100 and row['arm'] == 'full'
        assert row['previous_receipt_sha256'] == last_sha and not row['quality_read']
        baselines[end] = row
        last_sha = pair.sha(path)
    assert last_sha == cpu.FULL_CONTROL_SHA
    assert sum(not v['rank_active'] for r in baselines.values() for v in r['steps']) == 375
    return control, source, prior, proof, code, baselines


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mechanics', action='store_true')
    p.add_argument('--startup-only', action='store_true')
    p.add_argument('--chunk-end', type=int)
    p.add_argument('--mechanics-proof', type=Path)
    p.add_argument('--mechanics-sha256')
    p.add_argument('--previous', type=Path)
    p.add_argument('--previous-sha256')
    args = p.parse_args()
    assert sum((args.mechanics, args.startup_only, args.chunk_end is not None)) == 1 and not args.output.exists()
    if args.chunk_end is not None:
        assert 100 <= args.chunk_end <= 2000 and args.chunk_end % 100 == 0
        assert bool(args.chunk_end - 100) == bool(args.previous) == bool(args.previous_sha256)
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, source, prior, proof, code, baseline = startup(root, args.execution_sha256)
    if args.startup_only:
        assert not torch.cuda.is_available()
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'full training source differs'
            else:
                raise AssertionError('changed full training driver accepted')
        pair.smoke.save(args.output, {'pass': True, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': CPU_SHA, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False})
        return
    mechanics = None
    if args.chunk_end is not None:
        assert args.mechanics_proof and pair.sha(args.mechanics_proof / 'receipt.json') == args.mechanics_sha256
        mechanics = json.loads((args.mechanics_proof / 'receipt.json').read_text())
        assert mechanics['pass'] and mechanics['execution_sha256'] == args.execution_sha256 and mechanics['cpu_authority_sha256'] == CPU_SHA
        assert mechanics['training_state_discarded'] and mechanics['native_17_equals_serialized8_plus9_exact'] and mechanics['strict400_reload_whole_head_packed_exact']
        assert mechanics['chunk100_admission_seconds'] <= 269 and mechanics['median_ratio_vs_archived_control'] <= 1.10
        assert mechanics['peak_cuda_allocated_bytes'] < 10_000_000_000 and not mechanics['quality_read']
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = driver.coverage.teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state = driver.fresh(control, source, proof, 'full', 'cuda')
    initial_sha = driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    arm = proof['arms']['full']
    target = state['target'].cpu().numpy()
    batches = driver.schedule(target)
    schedule_sha = pair.smoke.digest({'batches': torch.from_numpy(batches)})
    assert all(r['schedule_sha256'] == schedule_sha for r in baseline.values())
    identity = driver.identity(state, proof, 'full', args.execution_sha256, schedule_sha)
    identity['intervention'] = 'full-valid-anchor-v1'
    counts = np.bincount(target)
    start = 0 if args.mechanics else args.chunk_end - 100
    previous = None
    if start:
        assert pair.sha(args.previous / 'receipt.json') == args.previous_sha256
        previous = json.loads((args.previous / 'receipt.json').read_text())
        assert previous['pass'] and previous['completed_step'] == start and previous['intervention'] == identity['intervention']
        assert previous['execution_sha256'] == args.execution_sha256 and previous['mechanics_sha256'] == args.mechanics_sha256
        assert previous['initial_state_sha256'] == initial_sha and previous['schedule_sha256'] == schedule_sha
        driver.restore(state, identity, args.previous / 'resume.pt', previous['checkpoint_sha256'], expected_step=start)
    else:
        assert state['counter'] == 0 and not state['optimizer'].state
    original_terms = driver.coverage.terms
    assert original_terms is lane.original_terms

    def update(s, step):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        batch = tuple(batches[step - 1].tolist())
        images, rgb = pair.augmented_images(control.dataset_root, arm['rows'], batch, step)
        pixels = pair.pixels(s['processor'], images, 'large')
        pixel_sha = pair.smoke.digest({'pixels': pixels})
        active = bool((counts[target[list(batch)]] > 1).all())
        reference = baseline[((step - 1) // 100 + 1) * 100]['steps'][(step - 1) % 100]
        assert reference['step'] == step and reference['rank_active'] == active
        assert reference['rgb_sha256'] == rgb and reference['pixels_sha256'] == pixel_sha
        with patch.object(driver.coverage, 'terms', lane.terms):
            row = driver.step(s, pixels, batch, active)
        assert driver.coverage.terms is original_terms and s['counter'] == step
        assert row['rank'] >= 0
        if not active and step <= 17:
            assert row['rank'] > 0  # Actual frozen mechanics must exercise recovered supervision.
        row.update(rgb_sha256=rgb, pixels_sha256=pixel_sha, rank_active_before=active, recovered_rank_update=not active)
        if mechanics and step <= 17:
            assert diagnostic(row) == diagnostic(mechanics['steps'][step - 1])
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        torch.cuda.synchronize()
        row['seconds'] = time.perf_counter() - tick
        print(json.dumps(row), flush=True)
        return row

    rows = []
    checkpoint_sha = None
    if args.mechanics:
        images, _ = pair.augmented_images(control.dataset_root, arm['rows'], tuple(batches[0][:2].tolist()), 1)
        calibration = pair.pixels(state['processor'], images, 'large').cuda()

        def cosine(model, head):
            with torch.no_grad():
                a = model(pixel_values=calibration).pooler_output.float()
                b = driver.previous.training.fp16(model, calibration)
                values = {'pooled': F.cosine_similarity(a, b).tolist(), 'compact': F.cosine_similarity(pair.smoke.compact_head_features(a, head), pair.smoke.compact_head_features(b, head)).tolist()}
                assert all(min(v) >= .999 for v in values.values())
                return values

        initial_cos = cosine(state['model'], state['head'])
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        with TemporaryDirectory(prefix='discard-full-valid-anchor-', dir=root) as temporary:
            path = Path(temporary) / 'step8.pt'
            for step in range(1, 18):
                rows.append(update(state, step))
                if step == 8:
                    saved_sha = driver.save(state, identity, path)
            expected = driver.fingerprint(driver.payload(state, identity))
            del state
            gc.collect(); torch.cuda.empty_cache()
            state = driver.fresh(control, source, proof, 'full', 'cuda')
            assert driver.identity(state, proof, 'full', args.execution_sha256, schedule_sha) == {k: v for k, v in identity.items() if k != 'intervention'}
            driver.restore(state, identity, path, saved_sha, expected_step=8)
            resumed = [update(state, step) for step in range(9, 18)]
            assert all(diagnostic(a) == diagnostic(b) for a, b in zip(resumed, rows[8:], strict=True))
            assert driver.fingerprint(driver.payload(state, identity)) == expected
            saved = Path(temporary) / 'step17.pt'
            driver.save(state, identity, saved)
            del state['optimizer']
            state['model'].eval(); state['head'].eval()
            disk = torch.load(saved, map_location='cpu', weights_only=True, mmap=True)
            with torch.random.fork_rng(devices=[0]):
                model = type(state['model'])(copy.deepcopy(state['model'].config)).float().eval()
                model.load_state_dict(disk['vision'], strict=True)
                head = nn.Linear(1024, 128).eval()
                head.load_state_dict(disk['head'], strict=True)
                assert len(disk['vision']) == 400
                assert pair.smoke.digest(driver.coverage.trained.base.whole_state(model)) == pair.smoke.digest(driver.coverage.trained.base.whole_state(state['model']))
                assert driver.coverage.frozen_digest(model, state['inventory']) == proof['frozen_prefix_sha256']
                assert pair.smoke.digest(head.state_dict()) == pair.smoke.digest(state['head'].state_dict())
                model.cuda(); head.cuda()
                with torch.no_grad():
                    a = driver.previous.training.fp16(state['model'], calibration)
                    b = driver.previous.training.fp16(model, calibration)
                    assert torch.equal(a, b)
                    va = F.normalize(pair.smoke.compact_head_features(a, state['head']), dim=1)
                    vb = F.normalize(pair.smoke.compact_head_features(b, head), dim=1)
                    assert torch.equal(va, vb)
                    driver.previous.training.packed_equal(va, vb)
                terminal_cos = cosine(model, head)
        assert sum(r['recovered_rank_update'] for r in rows) == 3
        median = statistics.median(r['seconds'] for r in rows[2:])
        median_control = statistics.median(r['seconds'] for r in baseline[100]['steps'][2:17])
        admission = 100 * max(median, sum(r['seconds'] for r in rows) / 17) + 30
        assert median / median_control <= 1.10 and admission <= 269
        facts = {'native_17_equals_serialized8_plus9_exact': True, 'strict400_reload_whole_head_packed_exact': True, 'terminal_state_fingerprint': expected, 'resumed_steps': resumed, 'initial_cosines': initial_cos, 'terminal_cosines': terminal_cos, 'training_state_discarded': True, 'median_ratio_vs_archived_control': median / median_control, 'chunk100_admission_seconds': admission, 'updates': 17}
    else:
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        tick = time.perf_counter()
        rows = [update(state, step) for step in range(start + 1, args.chunk_end + 1)]
        training_wall = time.perf_counter() - tick
        args.output.mkdir(exist_ok=False)
        checkpoint_sha = driver.save(state, identity, args.output / 'resume.pt')
        exposure = np.bincount(target[batches[:args.chunk_end].ravel()], minlength=3997)
        assert exposure.min() >= 1 and state['counter'] == args.chunk_end
        ratio = training_wall / baseline[args.chunk_end]['training_wall_seconds']
        assert ratio <= 1.10
        facts = {'training_state_discarded': False, 'chunk_start': start, 'completed_step': args.chunk_end, 'updates': 100, 'total_updates': 2000, 'training_wall_seconds': training_wall, 'images_per_second': 6400 / training_wall, 'cost_ratio_vs_archived_control': ratio, 'minimum_id_exposure': int(exposure.min()), 'maximum_id_exposure': int(exposure.max()), 'previous_receipt_sha256': args.previous_sha256, 'previous_checkpoint_sha256': previous['checkpoint_sha256'] if previous else None, 'checkpoint_sha256': checkpoint_sha}
    assert driver.coverage.trained.base.runtime_identity(state['model']) == identity['runtime']
    assert json.loads(json.dumps(driver.coverage.trained.native.environment(state['model'], state['processor']))) == source['environment']
    assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
    assert driver.fingerprint(dict(state['model'].named_buffers())) == identity['buffers_sha256']
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert driver.coverage.teacher.qualified.numerical_flags() == flags and driver.coverage.terms is original_terms
    assert all(pair.sha(root / n) == h for n, h in code.items()) and pair.sha(driver.coverage.teacher.TEACHER) == driver.coverage.teacher.TEACHER_SHA
    assert torch.cuda.max_memory_allocated() < 10_000_000_000
    if args.mechanics:
        args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'intervention': identity['intervention'], 'arm': 'full', 'seed': pair.SEED, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': CPU_SHA, 'mechanics_sha256': args.mechanics_sha256, 'control_training_receipt_sha256': cpu.FULL_CONTROL_SHA, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'initial_state_sha256': initial_sha, 'schedule_sha256': schedule_sha, 'steps': rows, **facts, 'recovered_rank_updates': sum(r['recovered_rank_update'] for r in rows), 'frozen_source_code_environment_rng_preserved': True, 'all_input_hashes_equal_archived_control': True, 'quality_read': False, 'claim_eligible': False, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'total_seconds': time.perf_counter() - started})
    print('PASS discarded full17 exact resume' if args.mechanics else 'PASS fixed full2000 candidate100 chunk; no quality read', flush=True)


if __name__ == '__main__':
    main()
