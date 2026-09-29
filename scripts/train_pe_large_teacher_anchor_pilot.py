#!/usr/bin/env python3
"""One fresh100-update teacher-anchor TRAIN pilot after discarded mechanics."""
import argparse
import copy
import json
import os
import time
from contextlib import closing
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import qualify_pe_large_teacher_anchor_gpu as gpu
from pe_large_teacher_anchor import anchor
from unittest.mock import patch
import train_pe_large_teacher_anchor as mechanics
from pe_l14_prefetch import prefetch
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

native, base, pair = gpu.native, gpu.native.base, gpu.native.base.pair
GPU_SHA = '7ca9a4445be9c2342684ef2ba46e3177bee0d1ab5367dd22b8d57f489400ac87'


MECHANICS_SHA = 'eea37e87f6b271e32af62d35aa5b17b15e231b8c02f732d61b6420607303cd16'
MECHANICS_CODE_SHA = '3d0b60123b0aaa4a041bb97acf23a0b356304f3c9cb4a869ace7cabf283b7f0c'
MECHANICS_DIR = Path('/home/riomus/runs/sfora-large-teacher-anchor-mechanics-v1')


def startup(root, execution_sha):
    manifest = root / 'teacher-anchor-pilot-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'pilot execution code differs'
    control, frozen, prior, cpu, initial, inputs, trained_code, teacher = mechanics.startup(root, MECHANICS_CODE_SHA)
    assert all(code[n] == h for n, h in trained_code.items())
    receipt_path = MECHANICS_DIR / 'receipt.json'
    assert pair.sha(receipt_path) == MECHANICS_SHA and not (MECHANICS_DIR / 'pe.pt').exists(), 'discarded mechanics authority differs'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['advance'] and receipt['updates'] == 17 and receipt['discard_training_state'] and not receipt['quality_read']
    assert receipt['updated_gpu_strict_reload_exact'] and receipt['updated_whole_native_B64_calibration_exact'] and receipt['fit_only_suffix_head_packed_exact']
    assert receipt['median_step_3_17_seconds'] <= 0.71769696 and receipt['execution_sha256'] == MECHANICS_CODE_SHA and receipt['preflight_sha256'] == GPU_SHA
    assert pair.sha(MECHANICS_DIR / 'training.json') == receipt['training_sha256']
    return control, frozen, prior, cpu, initial, inputs, code, teacher


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-startup-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, prior, cpu, initial, inputs, code, teacher = startup(root, args.execution_sha256)
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        original_sha = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == root / 'train_pe_large_teacher_anchor_pilot.py' else original_sha(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'pilot execution code differs'
            else:
                raise AssertionError('changed anchor pilot code accepted')
        print('PASS actual anchor pilot startup/discarded17 authority and changed-driver rejection; no CUDA/images/quality')
        return
    assert torch.cuda.is_available() and not args.output.exists()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert gpu.numerical_flags() == initial['numerical_flags']
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, processor = pair.smoke.load_arm(control, 'large')
    assert native.freeze(model) == {k: tuple(v) for k, v in cpu['inventory'].items()}
    assert pair.smoke.digest(base.whole_state(model)) == cpu['whole_original_source_sha256']
    assert json.loads(json.dumps(native.environment(model, processor))) == initial['environment']
    teacher_sha = pair.smoke.digest({'teacher': teacher})
    teacher = teacher.cuda()
    model.cuda().train()
    head = nn.Linear(1024, 128)
    with np.load(base.INIT / 'initializers.npz', allow_pickle=False) as init:
        head.load_state_dict({k: torch.from_numpy(init['large.head.' + k]) for k in ('weight', 'bias')})
        classifier = nn.Parameter(torch.from_numpy(init['large.classifier'].copy()).cuda())
        bank = torch.from_numpy(init['large.bank'].copy()).cuda()
    head.cuda().train()
    active = [p for p in model.parameters() if p.requires_grad]
    members = active + list(head.parameters()) + [classifier]
    optimizer = torch.optim.AdamW([{'params': active, 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-4}, {'params': [classifier], 'lr': 1e-4}], weight_decay=0.05)
    assert len(active) == 32 and len({id(p) for p in members}) == len(members) == 35 and not optimizer.state
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in members}
    _, scaler = pair.smoke.training_precision('fp16', device='cuda')
    assert scaler.get_scale() == 128
    target = torch.tensor(frozen['target'], device='cuda')
    positives = pair.smoke.member_bank_positive_ordinals(np.asarray(frozen['target'], dtype=np.int64), allow_singletons=True).cuda()
    images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1, 2, 3), None)
    calibration_pixels = pair.pixels(processor, images, 'large').cuda()
    with torch.no_grad():
        full_precision = model(pixel_values=calibration_pixels).pooler_output.float()
        half_precision = gpu.fp16(model, calibration_pixels)
        cached = torch.from_numpy(np.load(control.cache / 'large.fit.npy', mmap_mode='r')[:4].copy()).cuda()
        calibration = {'fp16_fp32': F.cosine_similarity(half_precision, full_precision).tolist(), 'cached_fresh_fp16': F.cosine_similarity(cached, half_precision).tolist()}
        assert all(min(v) >= 0.999 for v in calibration.values())
    pair.smoke.save(args.output / 'calibration.json', calibration)
    del images, calibration_pixels, full_precision, half_precision, cached
    frozen_sha = pair.smoke.digest(native.frozen_state(model))
    assert frozen_sha == cpu['frozen_complement_sha256']
    def groups():
        return {'native_penultimate_block': dict(model.encoder.layers[-2].named_parameters()), 'native_final_block': dict(model.encoder.layers[-1].named_parameters()), 'compact_head': dict(head.named_parameters()), 'classifier': {'classifier': classifier}}
    initial_groups = {n: pair.smoke.digest(v) for n, v in groups().items()}
    runtime = base.runtime_identity(model)
    seconds, losses, scales, norms, rgb_hashes, worker_seconds, diagnostics, components = [], [], [], [], [], [], [], []

    def prepare(index):
        tick = time.perf_counter()
        images, rgb = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], frozen['batches'][index], index + 1)
        pixels = pair.pixels(processor, images, 'large')
        assert rgb == prior['rgb_sha256'][index]
        if index < 17:
            assert pair.smoke.digest({'pixels': pixels}) == inputs['pixels_sha256'][index]
        assert pixels.device.type == 'cpu' and not pixels.requires_grad and pixels.is_contiguous()
        return pixels, rgb, time.perf_counter() - tick

    def block_input(module, values):
        assert module is model.encoder.layers[-2] and len(values) == 2 and values[1] is None
        base.assert_frozen_tokens((values[0],))

    def second_block_input(module, values):
        assert module is model.encoder.layers[-1] and len(values) == 2 and values[1] is None
        assert values[0].requires_grad and values[0].grad_fn is not None and values[0].dtype == torch.float32

    def pool_input(module, values):
        assert module is model.head and len(values) == 1
        assert values[0].requires_grad and values[0].grad_fn is not None and values[0].dtype == torch.float32

    training_rng = torch.random.get_rng_state().clone()
    training_started = time.perf_counter()
    with closing(prefetch(prepare, 100)) as prepared:
        for step, batch in enumerate(frozen['batches'][:100], 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            x, rgb, worker_time = next(prepared)
            x = x.cuda()
            index = torch.tensor(batch, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            hooks = [model.encoder.layers[-2].register_forward_pre_hook(block_input), model.encoder.layers[-1].register_forward_pre_hook(second_block_input), model.head.register_forward_pre_hook(pool_input)]
            try:
                source = gpu.fp16(model, x)
            finally:
                for hook in hooks:
                    hook.remove()
            if step == 1:
                backends = gpu.attention_nodes(source.grad_fn)
                assert backends == initial['backward_nodes']
            raw = pair.smoke.compact_head_features(source, head)
            ce = pair.smoke.sharded_mask_arcface_loss(raw, classifier, target[index], torch.arange(128, device='cuda').unsqueeze(0), margin=0.3, scale=64)
            rank = pair.smoke.member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False) if frozen['rank_active'][step - 1] else ce.new_zeros(())
            geometry = anchor(raw, teacher[index])
            loss = ce + 8 * rank + geometry
            assert torch.isfinite(source).all() and torch.isfinite(raw).all() and torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all((p.grad is None) == (not p.requires_grad) and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in members)
            if step in (1, 100):
                gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad}
                extra = [float(p.grad.norm()) for p in [*head.parameters(), classifier]]
                diagnostics.append({'step': step, 'native_gradient_norms': gradients, 'head_proxy_gradient_norms': extra})
                pair.smoke.save(args.output / f'gradient-step-{step}.json', diagnostics[-1])
                assert all(np.isfinite(v) and v > 0 for v in (*gradients.values(), *extra)), 'native FP16 data gradient invalid/zero'
            norm = torch.nn.utils.clip_grad_norm_(members, 1, error_if_nonfinite=True)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() >= scale_before, 'optimizer update skipped'
            rows, positions = pair.smoke.member_bank_refresh_rows(tuple(batch))
            bank[torch.tensor(rows, device='cuda')] = pair.smoke.member_bank_refresh_values(source, raw, torch.tensor(positions, device='cuda'), live_head=False)
            assert all(torch.isfinite(p).all() for p in members) and torch.isfinite(bank).all()
            assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
            losses.append(float(loss.detach()))
            components.append({'ce': float(ce.detach()), 'rank': float(rank.detach()), 'anchor': float(geometry.detach())})
            scales.append(scaler.get_scale())
            norms.append(float(norm))
            rgb_hashes.append(rgb)
            worker_seconds.append(worker_time)
            print(json.dumps({'step': step, 'seconds': seconds[-1], 'loss': losses[-1]}), flush=True)
    training_wall = time.perf_counter() - training_started
    assert torch.equal(training_rng, torch.random.get_rng_state())
    assert base.runtime_identity(model) == runtime and pair.smoke.digest(native.frozen_state(model)) == frozen_sha
    assert pair.smoke.digest({'teacher': teacher}) == teacher_sha and not teacher.requires_grad and teacher.grad is None
    final_groups = {n: pair.smoke.digest(v) for n, v in groups().items()}
    assert all(final_groups[n] != h for n, h in initial_groups.items())
    previous = json.loads((MECHANICS_DIR / 'training.json').read_text())
    assert losses[:17] == previous['losses'] and scales[:17] == previous['scales'] and components[:17] == previous['objective_components']
    median = float(np.median(seconds[2:]))
    pair.smoke.save(args.output / 'training.json', {'updates': 100, 'median_step_3_100_seconds': median, 'step_seconds': seconds, 'training_wall_seconds_including_fill_drain': training_wall, 'worker_input_seconds': worker_seconds, 'one_pending_cpu_batch': True, 'caller_cpu_rng_unchanged': True, 'objective_components': components, 'teacher_export_sha256': gpu.anchor_cpu.cache.RECEIPT_SHA, 'teacher_constant_content_preserved': True, 'losses': losses, 'scales': scales, 'preclip_gradient_norms': norms, 'rgb_sha256': rgb_hashes, 'diagnostics': diagnostics, 'backward_nodes': backends, 'initial_group_sha256': initial_groups, 'final_group_sha256': final_groups, 'frozen_sha256': frozen_sha, 'execution_sha256': args.execution_sha256, 'preflight_sha256': GPU_SHA, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'quality_read': False, 'training_state_discarded': False})
    receipt = {'advance': False, 'updates': 100, 'quality_read': False, 'discard_training_state': False, 'preflight_sha256': GPU_SHA, 'execution_sha256': args.execution_sha256, 'training_sha256': pair.sha(args.output / 'training.json'), 'median_step_3_100_seconds': median}
    if median > 0.71769696:
        pair.smoke.save(args.output / 'receipt.json', {**receipt, 'reason': 'fixed training cost gate failed'})
        print('KILL cost; no held read/checkpoint persisted', flush=True)
        return
    del optimizer, members, source, raw, ce, rank, loss, geometry
    model.zero_grad(set_to_none=True)
    model.eval()
    head.eval()
    with torch.no_grad():
        updated = gpu.fp16(model, x)
    checkpoint = args.output / 'pe.pt'
    torch.save({'vision': model.state_dict(), 'head': head.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    loaded = type(model)(copy.deepcopy(model.config)).float().eval()
    loaded.load_state_dict(saved['vision'], strict=True)
    native.freeze(loaded)
    loaded.cuda()
    loaded_head = nn.Linear(1024, 128).eval()
    loaded_head.load_state_dict(saved['head'], strict=True)
    loaded_head.cuda()
    with torch.no_grad():
        assert torch.equal(gpu.fp16(loaded, x), updated), 'actual updated whole-native B64 calibration differs'
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    with torch.no_grad(), native.verify_export(loaded, model) as (encode, scopes):
        a, b = encode(x)
        va = F.normalize(pair.smoke.compact_head_features(a, loaded_head).float(), dim=1)
        vb = F.normalize(pair.smoke.compact_head_features(b, head).float(), dim=1)
        assert torch.equal(a, updated) and torch.isfinite(va).all() and torch.equal(va, vb)
    pa, pb = pack_int8_unit_embeddings(va.cpu()), pack_int8_unit_embeddings(vb.cpu())
    assert np.array_equal(pa.codes, pb.codes) and np.array_equal(pa.inverse_norms, pb.inverse_norms)
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert pair.smoke.digest(native.frozen_state(loaded)) == frozen_sha
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert json.loads(json.dumps(native.environment(model, processor))) == initial['environment'] and gpu.numerical_flags() == initial['numerical_flags']
    assert pair.smoke.digest(native.frozen_state(model)) == frozen_sha and torch.cuda.max_memory_allocated() < 10_000_000_000
    live_chunks, loaded_chunks = [], []
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    with torch.no_grad(), native.verify_export(loaded, model) as (encode, held_scopes):
        for offset in range(0, len(frozen['held_manifest']), 32):
            rows = frozen['held_manifest'][offset:offset + 32]
            images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
            pixels = pair.pixels(processor, images, 'large').cuda()
            a, b = encode(pixels)
            va = F.normalize(pair.smoke.compact_head_features(a, loaded_head).float(), dim=1)
            vb = F.normalize(pair.smoke.compact_head_features(b, head).float(), dim=1)
            assert torch.isfinite(va).all() and torch.equal(va, vb)
            loaded_chunks.append(va.cpu().numpy())
            live_chunks.append(vb.cpu().numpy())
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            print(json.dumps({'held_images_verified': offset + len(rows)}), flush=True)
    # No vector persistence or scores before successful whole-session authentication.
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert pair.smoke.digest(native.frozen_state(loaded)) == pair.smoke.digest(native.frozen_state(model)) == frozen_sha
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert json.loads(json.dumps(native.environment(model, processor))) == initial['environment'] and gpu.numerical_flags() == initial['numerical_flags']
    loaded_vectors, live_vectors = np.concatenate(loaded_chunks), np.concatenate(live_chunks)
    assert loaded_vectors.shape == live_vectors.shape == (12599, 128) and np.array_equal(loaded_vectors, live_vectors)
    pa, pb = pack_int8_unit_embeddings(torch.from_numpy(loaded_vectors)), pack_int8_unit_embeddings(torch.from_numpy(live_vectors))
    assert np.array_equal(pa.codes, pb.codes) and np.array_equal(pa.inverse_norms, pb.inverse_norms)
    np.save(args.output / 'pe.held.npy', loaded_vectors, allow_pickle=False)
    np.save(args.output / 'pe.live-held.npy', live_vectors, allow_pickle=False)
    labels = tuple(r['product'] for r in frozen['held_manifest'])
    quality = pair.packed_quality(loaded_vectors, labels, frozen['query'], frozen['gallery'])
    live_quality = pair.packed_quality(live_vectors, labels, frozen['query'], frozen['gallery'])
    assert all(np.array_equal(np.asarray(quality[k]), np.asarray(live_quality[k])) for k in ('per_query_r1', 'per_query_ap'))
    dense_prior = json.loads((base.INIT / 'receipt.json').read_text())['arms']['pe']
    products = np.asarray(labels)[frozen['query']]
    intervals = {}
    for name in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(quality[name]) - np.asarray(dense_prior['quality'][name])
        intervals[name] = {'mean_delta': float(np.mean(delta)), 'product_lower95': pair.bootstrap_lower(delta, products), 'product_upper95': -pair.bootstrap_lower(-delta, products), 'query_lower95': pair.bootstrap_lower(delta, np.arange(len(delta))), 'query_upper95': -pair.bootstrap_lower(-delta, np.arange(len(delta)))}
    advance = bool(quality['recall_at_1'] >= 0.951720176 and quality['map_at_r'] >= 0.776237120 and all(v['product_lower95'] > 0 for v in intervals.values()))
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert json.loads(json.dumps(native.environment(model, processor))) == initial['environment'] and gpu.numerical_flags() == initial['numerical_flags']
    assert pair.smoke.digest(base.whole_state(loaded)) == pair.smoke.digest(base.whole_state(model)) and pair.smoke.digest(native.frozen_state(model)) == frozen_sha
    assert torch.cuda.max_memory_allocated() < 10_000_000_000
    pair.smoke.save(args.output / 'receipt.json', {**receipt, 'advance': advance, 'reason': 'fixed TRAIN quality gates', 'quality_read': 'In-Shop TRAIN-held only', 'quality': quality, 'paired_dense_pe_intervals': intervals, 'updated_gpu_strict_reload_exact': True, 'updated_whole_native_B64_calibration_exact': True, 'full_held_strict_loaded_whole_and_independent_live_suffix_exact': True, 'full_held_independent_whole_encoder_exact': False, 'packed_per_query_parity': True, 'full_held_head_reload_exact': True, 'frozen_complement_unchanged': True, 'full_live_loaded_native_state_exact': True, 'held_images': len(loaded_vectors), 'scope_observations': scopes, 'held_scope_count': len(held_scopes), 'checkpoint_sha256': pair.sha(checkpoint), 'held_sha256': pair.sha(args.output / 'pe.held.npy'), 'live_held_sha256': pair.sha(args.output / 'pe.live-held.npy'), 'mechanics_receipt_sha256': MECHANICS_SHA, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter() - started, 'official_read': False, 'claim_eligible': False})
    print('GO TRAIN feasibility' if advance else 'KILL fixed TRAIN quality gate', flush=True)


if __name__ == '__main__':
    main()
