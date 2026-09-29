#!/usr/bin/env python3
"""One paired discarded17 mechanics or a fixed100 native TRAIN pilot arm."""
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
import qualify_pe_native_valid_anchor_cpu as cpu

driver, lane = cpu.driver, cpu.lane
CPU_CODE = '0f7da309fdee4670dad5f3304d5cfae5e1364c0228c543f554deffffcb39a43c'
CPU_ROOT = Path('/home/riomus/runs/sfora-native-valid-anchor-cpu-v2')
CPU_SHA = '8d78e6104f453dc62563e446cd0e668c9a158890b1267c02b0097d98c2c09f5d'


def startup(root, expected):
    path = root / 'native-valid-anchor-training-execution.json'
    assert driver.pair.sha(path) == expected
    code = json.loads(path.read_text())
    assert all(driver.pair.sha(root / n) == h for n, h in code.items())
    old = cpu.code_guard(root, CPU_CODE)
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    assert driver.pair.sha(CPU_ROOT / 'valid-anchor-cpu-proof.json') == CPU_SHA
    proof = json.loads((CPU_ROOT / 'valid-anchor-cpu-proof.json').read_text())
    assert proof['pass'] and proof['code'] == old and proof['execution_sha256'] == CPU_CODE
    assert proof['optimizer_updates'] == 0 and not proof['quality_read']
    assert driver.pair.sha(cpu.FULL_OFFICIAL / 'cpu-audit.json') == cpu.FULL_AUDIT
    audit = json.loads((cpu.FULL_OFFICIAL / 'cpu-audit.json').read_text())
    assert audit['pass'] and not audit['matched_pool_comparison']['survivor']
    old_helpers = cpu.qualified.confirmation.selected.helpers
    with patch.object(cpu.qualified.confirmation.selected, 'helpers', lambda r, _: old_helpers(r, code)):
        control, source, prior, native, *_ = cpu.qualified.authority(root, cpu.PREVIOUS_CODE, 'full', cpu.FULL_TRAIN)
    return control, source, prior, native, code


def run(root, output, checkpoint, control, source, proof, code, execution, arm, seed, updates, expected=None):
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(seed)
    state = driver.fresh(control, source, proof, 'half', 'cuda')
    model, head = state['model'], state['head']
    initial = driver.fingerprint({'vision': model.state_dict(), 'head': head.state_dict(), 'classifier': state['classifier'], 'bank': state['bank']})
    runtime = driver.coverage.trained.base.runtime_identity(model)
    flags = driver.coverage.teacher.qualified.numerical_flags()
    target = state['target'].cpu().numpy()
    counts = np.bincount(target)
    batches = driver.coverage.schedule(target, seed=seed)
    rank_active = [bool(np.all(counts[target[b]] > 1)) for b in batches]
    arm_inputs = proof['arms']['half']
    with patch.object(driver.pair, 'SEED', seed):
        images, _ = driver.pair.augmented_images(control.dataset_root, arm_inputs['rows'], tuple(batches[0, :2].tolist()), 1)
    calibration = driver.pair.pixels(state['processor'], images, 'large').cuda()
    with torch.no_grad():
        a = model(pixel_values=calibration).pooler_output.float()
        b = driver.previous.training.fp16(model, calibration)
        initial_cos = {'pooled': F.cosine_similarity(a, b).tolist(), 'compact': F.cosine_similarity(driver.pair.smoke.compact_head_features(a, head), driver.pair.smoke.compact_head_features(b, head)).tolist()}
        assert all(min(v) >= .999 for v in initial_cos.values())
    del a, b, images
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    rows = []
    training_started = time.perf_counter()
    for step, batch_values in enumerate(batches[:updates], 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        batch = tuple(batch_values.tolist())
        with patch.object(driver.pair, 'SEED', seed):
            images, rgb = driver.pair.augmented_images(control.dataset_root, arm_inputs['rows'], batch, step)
        pixels = driver.pair.pixels(state['processor'], images, 'large')
        pixels_sha = driver.pair.smoke.digest({'pixels': pixels})
        inactive = not rank_active[step - 1]
        valid_count = int((counts[target[batch_values]] > 1).sum())
        if arm == 'treatment':
            with patch.object(driver.coverage, 'terms', lane.terms):
                result = driver.step(state, pixels, batch, not inactive)
        else:
            result = driver.step(state, pixels, batch, not inactive)
        if expected and step <= 17:
            assert all(result[k] == expected[step - 1][k] for k in result), 'pilot first17 diagnostics differ'
            assert rgb == expected[step - 1]['rgb_sha256'] and pixels_sha == expected[step - 1]['pixels_sha256']
        assert inactive or valid_count == 64
        assert not inactive or arm == 'treatment' or result['rank'] == 0
        torch.cuda.synchronize()
        rows.append({**result, 'seconds': time.perf_counter() - tick, 'rgb_sha256': rgb, 'pixels_sha256': pixels_sha, 'rank_active_before': not inactive, 'valid_anchors': valid_count, 'rank_recovered': arm == 'treatment' and inactive})
        print(json.dumps({'arm': arm, 'seed': seed, **rows[-1]}), flush=True)
        del images, pixels
    training_wall = time.perf_counter() - training_started
    assert state['counter'] == updates and torch.equal(cpu_rng, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert driver.coverage.trained.base.runtime_identity(model) == runtime
    assert driver.coverage.frozen_digest(model, state['inventory']) == proof['frozen_prefix_sha256']
    median = statistics.median(r['seconds'] for r in rows[2:])
    training = {'arm': arm, 'seed': seed, 'updates': updates, 'steps': rows, 'initial_state_sha256': initial, 'schedule_sha256': driver.pair.smoke.digest({'batches': torch.from_numpy(batches)}), 'training_wall_seconds': training_wall, 'median_step_seconds': median, 'images_per_second': updates * 64 / training_wall, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'recovered_rank_updates': sum(r['rank_recovered'] for r in rows), 'quality_read': False}
    driver.pair.smoke.save(output / 'training.json', training)
    state['optimizer'], state['params'], state['scaler'] = None, [], None
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    state['classifier'].grad = None
    model.eval()
    head.eval()
    with torch.random.fork_rng(devices=[0]):
        saved = {'vision': model.state_dict(), 'head': head.state_dict(), 'classifier': state['classifier'].detach(), 'bank': state['bank'], 'classes': arm_inputs['classes']}
        assert len(saved['vision']) == 400
        torch.save(saved, checkpoint)
        del saved
        disk = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        loaded = type(model)(copy.deepcopy(model.config)).float().eval()
        loaded.load_state_dict(disk['vision'], strict=True)
        assert driver.pair.smoke.digest(driver.coverage.trained.base.whole_state(loaded)) == driver.pair.smoke.digest(driver.coverage.trained.base.whole_state(model))
        assert driver.coverage.frozen_digest(loaded, state['inventory']) == proof['frozen_prefix_sha256']
        loaded.cuda()
        loaded_head = nn.Linear(1024, 128).eval().cuda()
        loaded_head.load_state_dict(disk['head'], strict=True)
        with torch.no_grad():
            a = driver.previous.training.fp16(model, calibration)
            b = driver.previous.training.fp16(loaded, calibration)
            assert torch.equal(a, b)
            va = F.normalize(driver.pair.smoke.compact_head_features(a, head), dim=1)
            vb = F.normalize(driver.pair.smoke.compact_head_features(b, loaded_head), dim=1)
            assert torch.equal(va, vb)
            driver.previous.training.packed_equal(va, vb)
            full = loaded(pixel_values=calibration).pooler_output.float()
            terminal_cos = {'pooled': F.cosine_similarity(b, full).tolist(), 'compact': F.cosine_similarity(vb, F.normalize(driver.pair.smoke.compact_head_features(full, loaded_head), dim=1)).tolist()}
            assert all(min(v) >= .999 for v in terminal_cos.values())
        assert torch.equal(disk['classifier'], state['classifier'].cpu()) and torch.equal(disk['bank'], state['bank'].cpu())
        updated_whole = driver.pair.smoke.digest(driver.coverage.trained.base.whole_state(model))
        updated_head = driver.pair.smoke.digest(head.state_dict())
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert flags == driver.coverage.teacher.qualified.numerical_flags()
    assert json.loads(json.dumps(driver.coverage.trained.native.environment(model, state['processor']))) == source['environment']
    assert all(driver.pair.sha(root / n) == h for n, h in code.items())
    assert driver.pair.sha(driver.coverage.teacher.TEACHER) == driver.coverage.teacher.TEACHER_SHA
    assert driver.pair.sha(driver.previous.training.CPU / 'half.npz') == arm_inputs['initializers_sha256']
    assert torch.cuda.max_memory_allocated() < 10_000_000_000
    receipt = {**training, 'pass': True, 'execution_sha256': execution, 'cpu_authority_sha256': CPU_SHA, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'checkpoint_sha256': driver.pair.sha(checkpoint), 'updated_whole_sha256': updated_whole, 'updated_head_sha256': updated_head, 'strict400_reload_whole_head_packed_exact': True, 'initial_cosines': initial_cos, 'terminal_cosines': terminal_cos, 'source_state_rng_environment_code_preserved': True, 'training_state_discarded': updates == 17, 'total_seconds': time.perf_counter() - started, 'claim_eligible': False}
    driver.pair.smoke.save(output / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, choices=(17, 100), required=True)
    parser.add_argument('--arm', choices=('control', 'treatment'))
    parser.add_argument('--seed', type=int, choices=(179041, 179042))
    parser.add_argument('--mechanics', type=Path)
    parser.add_argument('--mechanics-sha256')
    parser.add_argument('--startup-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    control, source, prior, proof, code = startup(root, args.execution_sha256)
    assert not args.output.exists()
    if args.startup_only:
        assert not torch.cuda.is_available() and args.updates == 17
        real_sha = driver.pair.sha
        with patch.object(driver.pair, 'sha', lambda p: '0' * 64 if Path(p).resolve() == Path(__file__).resolve() else real_sha(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed GPU driver accepted')
        driver.pair.smoke.save(args.output, {'pass': True, 'execution_sha256': args.execution_sha256, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False})
        return
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert driver.coverage.teacher.qualified.numerical_flags() == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    if args.updates == 100:
        assert args.arm and args.seed and args.mechanics and args.mechanics_sha256
        assert driver.pair.sha(args.mechanics / 'receipt.json') == args.mechanics_sha256
        mechanics = json.loads((args.mechanics / 'receipt.json').read_text())
        assert mechanics['pass'] and mechanics['execution_sha256'] == args.execution_sha256 and mechanics['training_state_discarded']
        assert mechanics['cpu_authority_sha256'] == CPU_SHA and mechanics['chunk100_admission_seconds'] <= 269
        expected = mechanics['arms'][args.arm]['steps'] if args.seed == 179041 else None
        run(root, args.output, args.output / 'native.pt', control, source, proof, code, args.execution_sha256, args.arm, args.seed, 100, expected)
        return
    assert not args.arm and not args.seed and not args.mechanics and not args.mechanics_sha256
    args.output.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    with TemporaryDirectory(prefix='discard-native-valid-anchor-', dir=root) as tmp:
        receipts = {}
        for arm in ('control', 'treatment'):
            receipts[arm] = run(root, args.output / arm, Path(tmp) / (arm + '.pt'), control, source, proof, code, args.execution_sha256, arm, 179041, 17)
            gc.collect()
            torch.cuda.empty_cache()
        a, b = receipts['control'], receipts['treatment']
        assert a['initial_state_sha256'] == b['initial_state_sha256'] and a['schedule_sha256'] == b['schedule_sha256']
        assert all(x['rgb_sha256'] == y['rgb_sha256'] and x['pixels_sha256'] == y['pixels_sha256'] for x, y in zip(a['steps'], b['steps'], strict=True))
        assert b['recovered_rank_updates'] == sum(not r['rank_active_before'] for r in a['steps']) > 0
        assert b['updated_whole_sha256'] != a['updated_whole_sha256'] and b['updated_head_sha256'] != a['updated_head_sha256']
        assert b['median_step_seconds'] / a['median_step_seconds'] <= 1.10
        admission = max(r['total_seconds'] + 83 * r['median_step_seconds'] for r in receipts.values())
        assert admission <= 269
    assert not list(args.output.rglob('*.pt'))
    driver.pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': CPU_SHA, 'arms': receipts, 'updates': 17, 'chunk100_admission_seconds': admission, 'training_state_discarded': True, 'quality_read': False, 'claim_eligible': False})


if __name__ == '__main__':
    main()
