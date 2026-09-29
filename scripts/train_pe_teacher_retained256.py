#!/usr/bin/env python3
"""One fixed corrected F5 width pilot; all native work belongs on DGX."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

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
import train_pe_full_valid_anchor as previous
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features

driver, lane, pair = previous.driver, previous.lane, previous.pair
OLD = '4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375'
TRAIN_CODE = '0da63376f73c9bc55c4c5f6e8c0f3f80fab7efe211ef6234a72af2ce4ed3ac36'
METHOD = 'teacher-preserving-corrected-width-v1'
UPDATES = 1000


def expand(state):
    old, proxy, bank = state['head'], state['classifier'], state['bank']
    assert old.in_features == 1024 and old.out_features == 128 and proxy.shape[1] == bank.shape[1] == 128
    assert not state['optimizer'].state and torch.isfinite(proxy).all() and (proxy.norm(dim=1) > 0).all()
    device = old.weight.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        head = nn.Linear(1024, 256).to(device)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
        head.weight[:128].copy_(old.weight); head.bias[:128].copy_(old.bias)
    signs = np.random.default_rng(179032).choice([-1., 1.], size=(len(proxy), 128)).astype(np.float32)
    rows = F.normalize(torch.from_numpy(signs), dim=1)
    tail_sha = pair.smoke.digest({'R': rows})
    tail = .001 * proxy.detach().norm(dim=1, keepdim=True) * rows.to(device)
    state['head'], state['classifier'] = head, nn.Parameter(torch.cat((proxy.detach(), tail), dim=1))
    state['bank'] = F.pad(bank, (0, 128))
    groups = [{**group, 'params': params} for group, params in zip(state['optimizer'].param_groups,
              ([p for p in state['model'].parameters() if p.requires_grad], list(head.parameters()), [state['classifier']]), strict=True)]
    state['optimizer'] = torch.optim.AdamW(groups)
    state['params'] = driver.coverage.parameters(state['model'], head, state['classifier'])
    assert [id(p) for g in state['optimizer'].param_groups for p in g['params']] == [id(p) for _, p in state['params']]
    return tail_sha


def terms(source, head, classifier, bank, target, positive, index, rank_active):
    width = head.out_features
    assert width in (128, 256) and classifier.shape[1] == bank.shape[1] == width
    raw = compact_head_features(source, head, output_dim=width)
    ce = pair.smoke.sharded_mask_arcface_loss(raw, classifier, target[index],
         torch.arange(width, device=source.device).unsqueeze(0), margin=.3, scale=64)
    rank = lane.valid_rank(raw, bank, head, positive[index], index)
    return ce, rank, raw


def authority(root, expected):
    path = root / 'teacher-retained256-execution.json'
    assert pair.sha(path) == expected
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'teacher width source differs'
    original = root / 'full-valid-anchor-checkpoint-execution.json'
    assert pair.sha(original) == OLD
    old = json.loads(original.read_text())
    assert all(code[n] == h for n, h in old.items())
    assert set(code) - set(old) == {'train_pe_teacher_retained256.py', 'test_pe_teacher_retained256.py', 'inshop_teacher_retained256_gate_2026-09-29.md'}
    helpers = previous.cpu.previous.qualified.confirmation.selected.helpers
    with patch.object(previous.cpu.previous.qualified.confirmation.selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, _, _ = previous.startup(root, TRAIN_CODE)
    assert pair.SEED == 179032 and driver.TOTAL_UPDATES == 2000
    assert len(proof['arms']['half']['rows']) == 13283 and len(proof['arms']['half']['classes']) == 2004
    assert pair.sha(driver.coverage.teacher.TEACHER) == driver.coverage.teacher.TEACHER_SHA
    return control, source, prior, proof, code


def fresh(control, source, proof, width, device):
    state = driver.fresh(control, source, proof, 'half', device)
    tail = expand(state) if width == 256 else None
    return state, tail


def cpu(root, args, control, source, proof, code):
    assert not torch.cuda.is_available() and args.width == 256
    state, tail = fresh(control, source, proof, 256, 'cpu')
    before = driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    eligible = (state['positive'] >= 0).any(dim=1)
    index = torch.tensor([int(eligible.nonzero()[0]), int((~eligible).nonzero()[0])])
    images, rgb = pair.augmented_images(control.dataset_root, proof['arms']['half']['rows'], tuple(index.tolist()), 1)
    pixels = pair.pixels(state['processor'], images, 'large')
    rng = torch.random.get_rng_state().clone()
    pooled = state['model'](pixel_values=pixels).pooler_output.float()
    ce, rank, raw = terms(pooled, state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index, False)
    original = F.linear(F.normalize(pooled, dim=1), state['head'].weight[:128], state['head'].bias[:128])
    a, b = (pack_int8_unit_embeddings(F.normalize(v.detach(), dim=1)) for v in (original, raw))
    assert torch.equal(a.codes, b.codes[:, :128]) and b.codes[:, 128:].count_nonzero() == 0 and torch.equal(a.inverse_norms, b.inverse_norms)
    tail_gradient = torch.autograd.grad(ce, state['head'].weight, retain_graph=True)[0][128:]
    assert torch.isfinite(tail_gradient).all() and tail_gradient.norm() > 0
    individual = lane.valid_rank(raw[:1], state['bank'], state['head'], state['positive'][index[:1]], index[:1])
    assert torch.equal(rank, individual * .5)
    raw.retain_grad(); rank.backward(retain_graph=True)
    assert raw.grad[0].norm() > 0 and raw.grad[1].count_nonzero() == 0
    state['optimizer'].zero_grad(set_to_none=True)
    (ce + 8 * rank).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in state['params'])
    assert all(p.grad is None for p in state['model'].parameters() if not p.requires_grad)
    assert before == driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
    assert torch.equal(rng, torch.random.get_rng_state())
    real_sha = pair.sha
    with patch.object(pair, 'sha', lambda p: 'changed' if Path(p).resolve() == Path(__file__).resolve() else real_sha(p)):
        try:
            authority(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'teacher width source differs'
        else:
            raise AssertionError('changed driver accepted')
    pair.smoke.save(args.output, {'pass': True, 'code': code, 'execution_sha256': args.execution_sha256, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA,
        'tail_sha256': tail, 'fit_rows': 13283, 'classes': 2004, 'width': 256, 'eligible_row': int(index[0]), 'singleton_row': int(index[1]),
        'rgb_sha256': rgb, 'pixels_sha256': pair.smoke.digest({'pixels': pixels}), 'new_head_CE_gradient_norm': float(tail_gradient.norm()),
        'padded_teacher_packed_exact': True, 'valid_anchor_denominator_and_singleton_gradient_exact': True,
        'frozen_state_and_training_RNG_preserved': True, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False, 'GPU_training_qualified': False})
    print('PASS actual native teacher256 CPU gradient/geometry/denominator/source; no training or quality')


def normal_log(root, name):
    log = (root / (name + '-v1.log')).read_text()
    assert all(v in log for v in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))


def run(root, args, control, source, prior, proof, code):
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert args.cpu_proof and pair.sha(args.cpu_proof) == args.cpu_sha256
    cpu_proof = json.loads(args.cpu_proof.read_text())
    assert cpu_proof['pass'] and cpu_proof['code'] == code and cpu_proof['new_head_CE_gradient_norm'] > 0 and cpu_proof['changed_driver_rejected']
    normal_log(root, 'cpu')
    control_mechanics = None
    if args.width == 256 or args.phase == 'train':
        assert args.control_mechanics and pair.sha(args.control_mechanics / 'receipt.json') == args.control_sha256
        control_mechanics = json.loads((args.control_mechanics / 'receipt.json').read_text())
        assert control_mechanics['pass'] and control_mechanics['width'] == 128 and control_mechanics['execution_sha256'] == args.execution_sha256
        assert control_mechanics['training_state_discarded'] and control_mechanics['native_17_equals_serialized8_plus9_exact'] and control_mechanics['chunk100_admission_seconds'] <= 269
        normal_log(root, 'mechanics-128')
    candidate_mechanics = None
    if args.phase == 'train':
        assert args.candidate_mechanics and pair.sha(args.candidate_mechanics / 'receipt.json') == args.candidate_sha256
        candidate_mechanics = json.loads((args.candidate_mechanics / 'receipt.json').read_text())
        assert candidate_mechanics['pass'] and candidate_mechanics['width'] == 256 and candidate_mechanics['execution_sha256'] == args.execution_sha256
        assert candidate_mechanics['training_state_discarded'] and candidate_mechanics['native_17_equals_serialized8_plus9_exact'] and candidate_mechanics['strict400_reload_whole_head_packed_exact']
        assert candidate_mechanics['median_ratio_vs_control'] <= 1.10 and candidate_mechanics['chunk100_admission_seconds'] <= 269
        assert candidate_mechanics['new_head_and_bank_tails_active'] and candidate_mechanics['classifier_tail_moments_active']
        normal_log(root, 'mechanics-256')
    flags = driver.coverage.teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state, tail = fresh(control, source, proof, args.width, 'cuda')
    assert tail == (cpu_proof['tail_sha256'] if args.width == 256 else None)
    initial = driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    target = state['target'].cpu().numpy(); counts = np.bincount(target)
    with patch.object(driver, 'TOTAL_UPDATES', UPDATES):
        batches = driver.schedule(target)
        schedule_sha = pair.smoke.digest({'batches': torch.from_numpy(batches)})
        identity = {**driver.identity(state, proof, 'half', args.execution_sha256, schedule_sha), 'intervention': METHOD, 'width': args.width, 'tail_sha256': tail}
        start = 0 if args.phase == 'mechanics' else args.chunk_end - 100
        last = None
        if start:
            assert args.previous and pair.sha(args.previous / 'receipt.json') == args.previous_sha256
            last = json.loads((args.previous / 'receipt.json').read_text())
            assert last['pass'] and last['completed_step'] == start and last['width'] == args.width
            assert last['execution_sha256'] == args.execution_sha256 and last['initial_state_sha256'] == initial and last['schedule_sha256'] == schedule_sha
            driver.restore(state, identity, args.previous / 'resume.pt', last['checkpoint_sha256'], start)
        reference = None
        if args.phase == 'train' and args.width == 256:
            assert args.control_chunk and pair.sha(args.control_chunk / 'receipt.json') == args.control_chunk_sha256
            reference = json.loads((args.control_chunk / 'receipt.json').read_text())
            assert reference['pass'] and reference['width'] == 128 and reference['completed_step'] == args.chunk_end and reference['execution_sha256'] == args.execution_sha256
            assert reference['schedule_sha256'] == schedule_sha
        def update(s, step):
            torch.cuda.synchronize(); tick = time.perf_counter()
            batch = tuple(batches[step - 1].tolist())
            images, rgb = pair.augmented_images(control.dataset_root, proof['arms']['half']['rows'], batch, step)
            pixels = pair.pixels(s['processor'], images, 'large'); pixel_sha = pair.smoke.digest({'pixels': pixels})
            active = bool((counts[target[list(batch)]] > 1).all())
            if reference is not None or control_mechanics is not None and args.width == 256 and step <= 17:
                base = reference['steps'][step - start - 1] if reference else control_mechanics['steps'][step - 1]
                assert base['step'] == step and base['rgb_sha256'] == rgb and base['pixels_sha256'] == pixel_sha
            with patch.object(driver.coverage, 'terms', terms):
                row = driver.step(s, pixels, batch, active)
            row.update(rgb_sha256=rgb, pixels_sha256=pixel_sha, rank_active_before=active)
            assert s['counter'] == step and torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize(); row['seconds'] = time.perf_counter() - tick
            print(json.dumps(row), flush=True)
            return row
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        rows, facts = [], {}
        if args.phase == 'mechanics':
            with TemporaryDirectory(prefix='discard-teacher-width-', dir=root) as temporary:
                path = Path(temporary) / 'step8.pt'
                for step in range(1, 18):
                    rows.append(update(state, step))
                    if step == 8:
                        saved_sha = driver.save(state, identity, path)
                expected = driver.fingerprint(driver.payload(state, identity))
                del state; gc.collect(); torch.cuda.empty_cache()
                state, restored_tail = fresh(control, source, proof, args.width, 'cuda')
                assert restored_tail == tail
                driver.restore(state, identity, path, saved_sha, 8)
                resumed = [update(state, step) for step in range(9, 18)]
                assert all(previous.diagnostic(a) == previous.diagnostic(b) for a, b in zip(resumed, rows[8:], strict=True))
                assert driver.fingerprint(driver.payload(state, identity)) == expected
                if args.width == 256:
                    assert state['head'].weight[128:].norm() > 0 and state['bank'][:, 128:].count_nonzero() > 0
                    assert state['optimizer'].state[state['classifier']]['exp_avg'][:, 128:].count_nonzero() > 0
                saved = Path(temporary) / 'step17.pt'; driver.save(state, identity, saved)
                del state['optimizer']; gc.collect(); torch.cuda.empty_cache()
                disk = torch.load(saved, map_location='cpu', weights_only=True, mmap=True)
                state['model'].eval(); state['head'].eval()
                images, _ = pair.augmented_images(control.dataset_root, proof['arms']['half']['rows'], tuple(batches[0, :2].tolist()), 1)
                calibration = pair.pixels(state['processor'], images, 'large').cuda()
                with torch.random.fork_rng(devices=[0]):
                    model = type(state['model'])(copy.deepcopy(state['model'].config)).float().eval()
                    model.load_state_dict(disk['vision'], strict=True)
                    head = nn.Linear(1024, args.width).eval(); head.load_state_dict(disk['head'], strict=True)
                    assert len(disk['vision']) == 400 and pair.smoke.digest(head.state_dict()) == pair.smoke.digest(state['head'].state_dict())
                    assert pair.smoke.digest(driver.coverage.trained.base.whole_state(model)) == pair.smoke.digest(driver.coverage.trained.base.whole_state(state['model']))
                    model.cuda(); head.cuda()
                    with torch.no_grad():
                        a, b = (driver.previous.training.fp16(m, calibration) for m in (state['model'], model))
                        assert torch.equal(a, b)
                        va, vb = (F.normalize(compact_head_features(v, h, output_dim=args.width), dim=1) for v, h in ((a, state['head']), (b, head)))
                        assert torch.equal(va, vb); driver.previous.training.packed_equal(va, vb)
            median = statistics.median(r['seconds'] for r in rows[2:])
            admission = 100 * max(median, statistics.mean(r['seconds'] for r in rows)) + 30
            ratio = median / statistics.median(r['seconds'] for r in control_mechanics['steps'][2:]) if args.width == 256 else 1.
            assert admission <= 269 and ratio <= 1.10
            facts = {'training_state_discarded': True, 'native_17_equals_serialized8_plus9_exact': True, 'strict400_reload_whole_head_packed_exact': True,
                'new_head_and_bank_tails_active': args.width == 256, 'classifier_tail_moments_active': args.width == 256,
                'terminal_state_fingerprint': expected, 'resumed_steps': resumed, 'median_ratio_vs_control': ratio, 'chunk100_admission_seconds': admission}
        else:
            tick = time.perf_counter(); rows = [update(state, step) for step in range(start + 1, args.chunk_end + 1)]
            wall = time.perf_counter() - tick
            args.output.mkdir(exist_ok=False)
            checkpoint_sha = driver.save(state, identity, args.output / 'resume.pt')
            facts = {'training_state_discarded': False, 'chunk_start': start, 'completed_step': args.chunk_end, 'total_updates': UPDATES,
                'checkpoint_sha256': checkpoint_sha, 'previous_receipt_sha256': args.previous_sha256,
                'previous_checkpoint_sha256': last['checkpoint_sha256'] if last else None, 'training_wall_seconds': wall, 'images_per_second': 6400 / wall,
                'control_chunk_sha256': args.control_chunk_sha256, 'cost_ratio_vs_control': wall / reference['training_wall_seconds'] if reference else None}
        assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
        assert driver.fingerprint(dict(state['model'].named_buffers())) == identity['buffers_sha256']
        assert driver.coverage.trained.base.runtime_identity(state['model']) == identity['runtime']
        assert json.loads(json.dumps(driver.coverage.trained.native.environment(state['model'], state['processor']))) == source['environment']
        assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
        assert driver.coverage.teacher.qualified.numerical_flags() == flags and all(pair.sha(root / n) == h for n, h in code.items())
        if args.phase == 'mechanics':
            args.output.mkdir(exist_ok=False)
        pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'intervention': METHOD, 'width': args.width, 'seed': pair.SEED,
            'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': args.cpu_sha256, 'control_mechanics_sha256': args.control_sha256,
            'candidate_mechanics_sha256': args.candidate_sha256, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA,
            'initial_state_sha256': initial, 'tail_sha256': tail, 'schedule_sha256': schedule_sha, 'steps': rows, **facts,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'total_seconds': time.perf_counter() - started, 'quality_read': False, 'claim_eligible': False})
    print('PASS discarded width17 mechanics' if args.phase == 'mechanics' else 'PASS fixed width TRAIN100 chunk; no quality read')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True); p.add_argument('--phase', choices=('cpu', 'mechanics', 'train'), required=True)
    p.add_argument('--width', type=int, choices=(128, 256), required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-proof', type=Path); p.add_argument('--cpu-sha256')
    p.add_argument('--control-mechanics', type=Path); p.add_argument('--control-sha256')
    p.add_argument('--candidate-mechanics', type=Path); p.add_argument('--candidate-sha256')
    p.add_argument('--chunk-end', type=int); p.add_argument('--previous', type=Path); p.add_argument('--previous-sha256')
    p.add_argument('--control-chunk', type=Path); p.add_argument('--control-chunk-sha256')
    args = p.parse_args(); assert not args.output.exists()
    if args.phase == 'train':
        assert args.chunk_end is not None and 100 <= args.chunk_end <= UPDATES and args.chunk_end % 100 == 0
        assert bool(args.chunk_end - 100) == bool(args.previous) == bool(args.previous_sha256)
    torch.set_num_threads(8); torch.manual_seed(179032)
    root = Path(__file__).resolve().parent
    control, source, prior, proof, code = authority(root, args.execution_sha256)
    if args.phase == 'cpu':
        cpu(root, args, control, source, proof, code)
    else:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        run(root, args, control, source, prior, proof, code)


if __name__ == '__main__':
    main()
