#!/usr/bin/env python3
"""Fresh strong-control TRAIN-fit targets; CPU proof then one bounded GPU export."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import qualify_pe_large_final_two_gpu as qualified
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

native, base, pair = qualified.native, qualified.native.base, qualified.native.base.pair
QUAL_SHA = 'ae5b792bead8343f79418cef72f76bbe7747ec89e9e37466b6ca9fb8c7e176be'
TEACHER = base.INIT / 'large.pt'
TEACHER_SHA = 'f5fbf0e8eb3492274b6febeb6fd06a4b0d8b0fce4c7170069f3fa0d7b0a8df4d'
CPU = Path('/home/riomus/runs/sfora-large-teacher-fit-cpu-v1/preflight.json')


def startup(root, execution_sha):
    path = root / 'teacher-fit-execution.json'
    assert pair.sha(path) == execution_sha
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'teacher execution code differs'
    assert TEACHER.stat().st_size == 1272298083 and pair.sha(TEACHER) == TEACHER_SHA, 'teacher checkpoint authority differs'
    control, frozen, _, _, qualified_code = qualified.startup(root, QUAL_SHA)
    assert all(code[n] == h for n, h in qualified_code.items())
    receipt = json.loads((base.INIT / 'receipt.json').read_text())
    assert receipt['arms']['large']['checkpoint_sha256'] == TEACHER_SHA
    assert len(frozen['fit_manifest']) == 13283 and len(set(frozen['target'])) == 2004
    return control, frozen, code


def teacher_pair(control, device):
    live, processor = pair.smoke.load_arm(control, 'large')
    saved = torch.load(TEACHER, map_location='cpu', weights_only=True, mmap=True)
    assert len(saved['vision']) == 400
    live.load_state_dict(saved['vision'], strict=True)
    loaded = type(live)(copy.deepcopy(live.config)).float().eval()
    loaded.load_state_dict(saved['vision'], strict=True)
    heads = []
    for model in (loaded, live):
        # Read-only helper role mask; the teacher was trained on12 blocks, not2.
        native.freeze(model)
        model.eval().to(device)
        head = nn.Linear(1024, 128).eval().requires_grad_(False)
        head.load_state_dict(saved['head'], strict=True)
        heads.append(head.to(device))
    return loaded, live, heads, processor


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-cpu', action='store_true')
    parser.add_argument('--cpu-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, code = startup(root, args.execution_sha256)
    assert not args.output.exists()
    device = torch.device('cpu' if args.qualify_cpu else 'cuda')
    assert torch.cuda.is_available() != args.qualify_cpu
    if args.qualify_cpu:
        authority = None
    else:
        assert args.cpu_sha256 and pair.sha(CPU) == args.cpu_sha256
        authority = json.loads(CPU.read_text())
        assert authority['code'] == code and authority['teacher_checkpoint_sha256'] == TEACHER_SHA
        assert authority['read_only'] and authority['prefix_data_mutation_rejected_at_exit']
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    loaded, live, heads, processor = teacher_pair(control, device)
    environment = json.loads(json.dumps(native.environment(live, processor)))
    whole = pair.smoke.digest(base.whole_state(live))
    prefix = pair.smoke.digest(native.frozen_state(live))
    head_sha = pair.smoke.digest(heads[0].state_dict())
    assert whole == pair.smoke.digest(base.whole_state(loaded))
    assert head_sha == pair.smoke.digest(heads[1].state_dict())
    if authority is not None:
        assert whole == authority['teacher_whole_sha256'] and prefix == authority['teacher_verification_prefix_sha256']
        assert head_sha == authority['teacher_head_sha256'] and environment == authority['environment']
    images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1), None)
    pixels = pair.pixels(processor, images, 'large')
    pixels_sha = pair.smoke.digest({'pixels': pixels})
    pixels = pixels.to(device)
    with torch.no_grad():
        direct = live(pixel_values=pixels).pooler_output.float() if args.qualify_cpu else qualified.fp16(live, pixels)
        direct_loaded = loaded(pixel_values=pixels).pooler_output.float() if args.qualify_cpu else qualified.fp16(loaded, pixels)
        assert torch.equal(direct, direct_loaded)
        calibration = None
        if not args.qualify_cpu:
            assert pixels_sha == authority['first_two_fit_pixels_sha256']
            full_precision = live(pixel_values=pixels).pooler_output.float()
            calibration = {'pooled': F.cosine_similarity(direct, full_precision).tolist(), 'compact': F.cosine_similarity(F.normalize(pair.smoke.compact_head_features(direct, heads[1]), dim=1), F.normalize(pair.smoke.compact_head_features(full_precision, heads[1]), dim=1)).tolist()}
            assert all(min(v) >= 0.999 for v in calibration.values())
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if not args.qualify_cpu else None
    loaded_chunks, live_chunks = [], []
    numerical = qualified.numerical_flags() if not args.qualify_cpu else None
    with torch.no_grad(), native.verify_export(loaded, live) as (encode, scopes):
        rows = frozen['fit_manifest'][:2] if args.qualify_cpu else frozen['fit_manifest']
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
            x = pair.pixels(processor, images, 'large').to(device)
            a, b = encode(x)
            if args.qualify_cpu:
                assert torch.equal(a, direct)
            va = F.normalize(pair.smoke.compact_head_features(a, heads[0]).float(), dim=1)
            vb = F.normalize(pair.smoke.compact_head_features(b, heads[1]).float(), dim=1)
            assert torch.isfinite(va).all() and torch.equal(va, vb)
            loaded_chunks.append(va.cpu().numpy())
            live_chunks.append(vb.cpu().numpy())
            if not args.qualify_cpu:
                assert torch.cuda.max_memory_allocated() < 10_000_000_000
                print(json.dumps({'fit_images_verified': start + len(batch)}), flush=True)
    assert len(scopes) == (1 if args.qualify_cpu else (len(frozen['fit_manifest']) + 31) // 32)
    assert all(not fact['between_encoder_amp_enabled'] and all(fact[k] == (not args.qualify_cpu) for k in ('loaded_amp_inside', 'captured_block_amp_inside', 'captured_final_block_amp_inside', 'live_amp_inside')) for fact in scopes)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    prefix_negative = False
    if args.qualify_cpu:
        parameter = live.encoder.layers[12].mlp.fc2.bias
        before, version = parameter.detach().clone(), parameter._version
        try:
            with native.verify_export(loaded, live) as (encode, _):
                parameter.data[0].add_(0.01)
                assert parameter._version == version
                encode(pixels)
        except AssertionError as error:
            assert str(error) == 'native pair state changed'
            prefix_negative = True
        else:
            raise AssertionError('trained teacher prefix mutation accepted')
        finally:
            with torch.no_grad():
                parameter.copy_(before)
    if args.qualify_cpu:
        with torch.no_grad():
            assert torch.equal(live(pixel_values=pixels).pooler_output.float(), direct)
    assert all(pair.smoke.digest(base.whole_state(m)) == whole and pair.smoke.digest(native.frozen_state(m)) == prefix for m in (loaded, live))
    assert all(pair.smoke.digest(h.state_dict()) == head_sha for h in heads)
    assert all(p.grad is None for m in (loaded, live, *heads) for p in m.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for model in (loaded, live) for m in model.modules())
    assert json.loads(json.dumps(native.environment(live, processor))) == environment
    assert pair.sha(TEACHER) == TEACHER_SHA and all(pair.sha(root / n) == h for n, h in code.items())
    if not args.qualify_cpu:
        assert qualified.numerical_flags() == numerical and torch.cuda.max_memory_allocated() < 10_000_000_000
    values, reference = np.concatenate(loaded_chunks), np.concatenate(live_chunks)
    assert values.shape == ((2, 128) if args.qualify_cpu else (13283, 128)) and np.array_equal(values, reference)
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    a, b = pack_int8_unit_embeddings(torch.from_numpy(values)), pack_int8_unit_embeddings(torch.from_numpy(reference))
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.inverse_norms, b.inverse_norms)
    facts = {'code': code, 'teacher_checkpoint_sha256': TEACHER_SHA, 'teacher_whole_sha256': whole, 'teacher_verification_prefix_sha256': prefix, 'teacher_head_sha256': head_sha, 'environment': environment, 'first_two_fit_pixels_sha256': pixels_sha, 'fit_manifest': frozen['fit_manifest'], 'target_products': frozen['target'], 'read_only': True, 'optimizer_updates': 0, 'held_images': 0, 'quality_read': False, 'strict400_native_head_reload_and_direct_whole_calibration_exact': True, 'independent_two_block_suffix_head_packed_exact': True, 'cpu_cuda_rng_unchanged': True, 'prefix_data_mutation_rejected_at_exit': prefix_negative if args.qualify_cpu else authority['prefix_data_mutation_rejected_at_exit'], 'scope_observations': scopes, 'numerical_flags': numerical, 'fp16_fp32_calibration_cosines': calibration, 'cuda': not args.qualify_cpu}
    args.output.mkdir(exist_ok=False)
    if not args.qualify_cpu:
        np.save(args.output / 'teacher.fit.npy', values, allow_pickle=False)
        np.save(args.output / 'teacher.reference-fit.npy', reference, allow_pickle=False)
        facts.update({'cpu_authority_sha256': args.cpu_sha256, 'fit_images': len(values), 'fit_sha256': pair.sha(args.output / 'teacher.fit.npy'), 'reference_fit_sha256': pair.sha(args.output / 'teacher.reference-fit.npy'), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    pair.smoke.save(args.output / ('preflight.json' if args.qualify_cpu else 'receipt.json'), facts)
    print('PASS read-only teacher native/head/packed authority and complete fit targets; no training/held/quality' if not args.qualify_cpu else 'PASS actual teacher CPU strict native/head/calibration/suffix, trained-prefix content rejection and read-only state/RNG authority')


if __name__ == '__main__':
    main()
