#!/usr/bin/env python3
"""Frozen 17-update native final-block mechanics; discard state, never read held."""
import argparse
import copy
import json
import os
import time
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import qualify_pe_large_final_block_gpu as gpu
from pe_l14_prefetch import prefetch
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

native, base, pair = gpu.native, gpu.native.base, gpu.native.base.pair
GPU_SHA = '15dbd47b20cc17e31ec7a10d353caecbe073cb0ab8294600999f13e926188489'
QUAL_CODE_SHA = '138b2282343d6ca51450edc0d066ad4faa1abc783b034c2a87a532e86316db94'
INPUT = Path('/home/riomus/runs/sfora-large-native-pool-prefetch-cpu-v1/preflight.json')
INPUT_SHA = 'f85a525d7a366476d861af70cac1137b6fed8a5fa3b9696a69927849d344cca0'


def startup(root, execution_sha):
    manifest = root / 'large-final-block-training-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'training execution code differs'
    control, frozen, prior, cpu, qualified = gpu.startup(root, QUAL_CODE_SHA)
    assert all(code[n] == h for n, h in qualified.items())
    receipt = Path('/home/riomus/runs/sfora-large-final-block-gpu-v1/preflight.json')
    assert pair.sha(receipt) == GPU_SHA
    initial = json.loads(receipt.read_text())
    assert initial['code'] == qualified and initial['environment'] == cpu['environment']
    assert initial['optimizer_updates'] == 0 and initial['held_images'] == 0 and not initial['quality_read']
    assert initial['updated_strict400_reload_whole_calibration_suffix_head_packed_exact'] and initial['original_source_restored']
    assert pair.sha(INPUT) == INPUT_SHA
    inputs = json.loads(INPUT.read_text())
    assert inputs['serial_worker_pixels_exact'] and inputs['caller_cpu_rng_unchanged']
    assert pair.sha(base.INIT / 'initializers.npz') == 'd0118caf6a87d779e892e70a3e3ad5f1026bfa783f661d0e3a53c5147ce880a6'
    return control, frozen, prior, cpu, initial, inputs, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, prior, cpu, initial, inputs, code = startup(root, args.execution_sha256)
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
    assert len(active) == 16 and len({id(p) for p in members}) == len(members) == 19 and not optimizer.state
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
        return {'native_final_block': dict(model.encoder.layers[-1].named_parameters()), 'compact_head': dict(head.named_parameters()), 'classifier': {'classifier': classifier}}
    initial_groups = {n: pair.smoke.digest(v) for n, v in groups().items()}
    runtime = base.runtime_identity(model)
    seconds, losses, scales, norms, rgb_hashes, worker_seconds, diagnostics = [], [], [], [], [], [], []

    def prepare(index):
        tick = time.perf_counter()
        images, rgb = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], frozen['batches'][index], index + 1)
        pixels = pair.pixels(processor, images, 'large')
        assert rgb == prior['rgb_sha256'][index] and pair.smoke.digest({'pixels': pixels}) == inputs['pixels_sha256'][index]
        assert pixels.device.type == 'cpu' and not pixels.requires_grad and pixels.is_contiguous()
        return pixels, rgb, time.perf_counter() - tick

    def block_input(module, values):
        assert module is model.encoder.layers[-1] and len(values) == 2 and values[1] is None
        base.assert_frozen_tokens((values[0],))

    def pool_input(module, values):
        assert module is model.head and len(values) == 1
        assert values[0].requires_grad and values[0].grad_fn is not None and values[0].dtype == torch.float32

    training_rng = torch.random.get_rng_state().clone()
    training_started = time.perf_counter()
    with closing(prefetch(prepare, 17)) as prepared:
        for step, batch in enumerate(frozen['batches'][:17], 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            x, rgb, worker_time = next(prepared)
            x = x.cuda()
            index = torch.tensor(batch, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            hooks = [model.encoder.layers[-1].register_forward_pre_hook(block_input), model.head.register_forward_pre_hook(pool_input)]
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
            loss = ce + 8 * rank
            assert torch.isfinite(source).all() and torch.isfinite(raw).all() and torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all((p.grad is None) == (not p.requires_grad) and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in members)
            if step in (1, 17):
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
            scales.append(scaler.get_scale())
            norms.append(float(norm))
            rgb_hashes.append(rgb)
            worker_seconds.append(worker_time)
            print(json.dumps({'step': step, 'seconds': seconds[-1], 'loss': losses[-1]}), flush=True)
    training_wall = time.perf_counter() - training_started
    assert torch.equal(training_rng, torch.random.get_rng_state())
    assert base.runtime_identity(model) == runtime and pair.smoke.digest(native.frozen_state(model)) == frozen_sha
    final_groups = {n: pair.smoke.digest(v) for n, v in groups().items()}
    assert all(final_groups[n] != h for n, h in initial_groups.items())
    median = float(np.median(seconds[2:]))
    pair.smoke.save(args.output / 'training.json', {'updates': 17, 'median_step_3_17_seconds': median, 'step_seconds': seconds, 'training_wall_seconds_including_fill_drain': training_wall, 'worker_input_seconds': worker_seconds, 'one_pending_cpu_batch': True, 'caller_cpu_rng_unchanged': True, 'losses': losses, 'scales': scales, 'preclip_gradient_norms': norms, 'rgb_sha256': rgb_hashes, 'diagnostics': diagnostics, 'backward_nodes': backends, 'initial_group_sha256': initial_groups, 'final_group_sha256': final_groups, 'frozen_sha256': frozen_sha, 'execution_sha256': args.execution_sha256, 'preflight_sha256': GPU_SHA, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'quality_read': False, 'training_state_discarded': True})
    receipt = {'advance': False, 'updates': 17, 'quality_read': False, 'discard_training_state': True, 'preflight_sha256': GPU_SHA, 'execution_sha256': args.execution_sha256, 'training_sha256': pair.sha(args.output / 'training.json'), 'median_step_3_17_seconds': median}
    if median > 0.71769696:
        pair.smoke.save(args.output / 'receipt.json', {**receipt, 'reason': 'fixed training cost gate failed'})
        print('KILL cost; no held read/checkpoint persisted', flush=True)
        return
    del optimizer, members, source, raw, ce, rank, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    head.eval()
    with torch.no_grad():
        updated = gpu.fp16(model, x)
    with TemporaryDirectory(dir=root, prefix='discard-final-block-mechanics-') as tmp:
        checkpoint = Path(tmp) / 'native.pt'
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
        with native.verify_export(loaded, model) as (encode, scopes):
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
    pair.smoke.save(args.output / 'receipt.json', {**receipt, 'advance': True, 'updated_gpu_strict_reload_exact': True, 'updated_whole_native_B64_calibration_exact': True, 'fit_only_suffix_head_packed_exact': True, 'scope_observations': scopes, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter() - started})
    print('PASS native final-block 17 mechanics; training state discarded; no held/quality', flush=True)


if __name__ == '__main__':
    main()
