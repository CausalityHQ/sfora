#!/usr/bin/env python3
"""One fixed100-pair public F32-autocast/F16-native latency pilot; no p99 claim."""
import argparse
import json
import inspect
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import qualify_pe_large_public_fp16 as fp16
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, trained, teacher, b1, serving = fp16.pair, fp16.trained, fp16.teacher, fp16.b1, fp16.serving
FP16_CODE_SHA = '75974763d263ac1435911af8c3b743cff6c16acb76260a2b2386761b72d790f2'
B32 = Path('/home/riomus/runs/sfora-large-public-fp16-b32-v1')
B32_SHA = 'dda25d9e34da8ce7486fecd0f96bd1d03dd244043742345de28a0b2fa71f4d36'
B32_AUDIT_SHA = '913c914813e9eede2e5e67c6c23243a2d90b7a38c426f203a1eed3a6efbfb4cf'


def startup(root, execution_sha, b1_path, b1_sha, audit_sha):
    manifest = root / 'precision-pair-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'precision pilot code differs'
    control, frozen, cpu, prior, old = fp16.startup(root, FP16_CODE_SHA)
    assert all(code[n] == h for n, h in old.items())
    assert pair.sha(B32 / 'receipt.json') == B32_SHA and pair.sha(B32 / 'cpu-audit.json') == B32_AUDIT_SHA
    assert pair.sha(b1_path / 'receipt.json') == b1_sha and pair.sha(b1_path / 'cpu-audit.json') == audit_sha
    receipt = json.loads((b1_path / 'receipt.json').read_text())
    audit = json.loads((b1_path / 'cpu-audit.json').read_text())
    assert receipt['advance'] and audit['advance'] and audit['pass'] and audit['receipt_sha256'] == b1_sha
    assert receipt['code'] == old and receipt['b32_receipt_sha256'] == B32_SHA and receipt['b32_audit_sha256'] == B32_AUDIT_SHA
    assert receipt['all_B1_actual_public_native_r1_exact'] and receipt['B1_original_processor_native_packed_exact_queries'] == 32
    proof = root / 'fp16-cpu-proof.json'
    assert pair.sha(proof) == receipt['cpu_authority_sha256']
    authority = json.loads(proof.read_text())
    assert authority['pass'] and authority['code'] == old and authority['fp16_whole_sha256'] == receipt['fp16_whole_sha256']
    return control, frozen, cpu, prior, code, authority


def equal(actual, expected):
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)), 'repeated public native ordinals/score bits differ'


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--b1', type=Path, required=True)
    parser.add_argument('--b1-sha256', required=True)
    parser.add_argument('--b1-audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-startup-only', action='store_true')
    parser.add_argument('--cpu-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, cpu, prior, code, authority = startup(root, args.execution_sha256, args.b1, args.b1_sha256, args.b1_audit_sha256)
    assert not args.output.exists()
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256, args.b1, args.b1_sha256, args.b1_audit_sha256)
            except AssertionError as error:
                assert str(error) == 'precision pilot code differs'
            else:
                raise AssertionError('changed precision pilot accepted')
        pair.smoke.save(args.output, {'pass': True, 'code': code, 'execution_sha256': args.execution_sha256, 'b1_sha256': args.b1_sha256, 'b1_audit_sha256': args.b1_audit_sha256, 'changed_driver_rejected': True, 'optimizer_updates': 0})
        print('PASS matched precision pilot startup and changed-code rejection')
        return
    assert torch.cuda.is_available() and args.cpu_sha256
    proof = root / 'precision-pair-cpu.json'
    assert pair.sha(proof) == args.cpu_sha256
    proof_data = json.loads(proof.read_text())
    assert proof_data['pass'] and proof_data['code'] == code and proof_data['b1_sha256'] == args.b1_sha256 and proof_data['b1_audit_sha256'] == args.b1_audit_sha256
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
    assert Path(inspect.getfile(serving.CutilePackedInt8Gallery)).resolve() == root / 'src/sfora/cutile_int8.py'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    old = pack_int8_unit_embeddings(torch.from_numpy(np.load(serving.HELD / 'large.held.npy', allow_pickle=False)))
    new = fp16.load_packed(B32, json.loads((B32 / 'receipt.json').read_text()), 'held')
    paths = [control.dataset_root / frozen['held_manifest'][i]['relative_path'] for i in frozen['query'][:32]]
    assert all(pair.sha(p) == frozen['held_manifest'][i]['image_sha256'] for p, i in zip(paths, frozen['query'][:32], strict=True))
    def decode(size):
        result = []
        for path in paths[:size]:
            with Image.open(path) as image:
                result.append(image.convert('RGB'))
        return result
    encoders, indexes, galleries, results = {}, {}, {}, {}
    with ExitStack() as stack:
        for mode, values in (('fp32_autocast', old), ('fp16_native', new)):
            encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=teacher.TEACHER, expected_checkpoint_sha256=teacher.TEACHER_SHA, model_file_sha256=pair.smoke.MODEL_HASHES, precision=mode, device=torch.device('cuda'))
            expected = cpu['teacher_whole_sha256'] if mode == 'fp32_autocast' else authority['fp16_whole_sha256']
            assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == expected
            assert pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
            gallery = PackedInt8Embeddings(values.codes[frozen['gallery']].contiguous(), values.inverse_norms[frozen['gallery']].contiguous())
            galleries[mode] = gallery
            indexes[mode] = stack.enter_context(Siglip2CompactIndex(encoder, serving.CutilePackedInt8Gallery.open_packed(serving.LIBRARY, gallery)))
            encoders[mode] = encoder
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        rng = np.random.default_rng(pair.SEED)
        for size in (1, 32):
            expected = {}
            for mode, index in indexes.items():
                query = index.encoder.encode_images(decode(size))
                gallery = galleries[mode]
                scores = (query.codes.float() @ gallery.codes.float().T) * query.inverse_norms.float()[:, None] * gallery.inverse_norms.float()[None, :]
                order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
                reference = (order.numpy(), scores.gather(1, order).numpy())
                expected[mode] = index.search_images(decode(size))
                equal(expected[mode], reference)
                for _ in range(5):
                    equal(index.search_images(decode(size)), expected[mode])
            pairs = []
            for i in range(100):
                modes = list(indexes) if rng.integers(2) == 0 else list(reversed(indexes))
                sample = {'order': modes}
                for mode in modes:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = indexes[mode].search_images(decode(size))
                    torch.cuda.synchronize()
                    sample[mode + '_ms'] = 1000 * (time.perf_counter() - start)
                    equal(actual, expected[mode])
                pairs.append(sample)
                assert torch.cuda.max_memory_allocated() < 10_000_000_000
            control_ms = np.asarray([s['fp32_autocast_ms'] for s in pairs])
            candidate_ms = np.asarray([s['fp16_native_ms'] for s in pairs])
            differences = candidate_ms - control_ms
            bootstrap = np.random.default_rng(pair.SEED).choice(differences, size=(5000, len(pairs)), replace=True).mean(axis=1)
            results[str(size)] = {'calls_per_mode': 100, 'pairs': pairs, 'fp32_autocast_p50_ms': float(np.quantile(control_ms, .5)), 'fp32_autocast_p95_ms': float(np.quantile(control_ms, .95)), 'fp16_native_p50_ms': float(np.quantile(candidate_ms, .5)), 'fp16_native_p95_ms': float(np.quantile(candidate_ms, .95)), 'paired_mean_candidate_minus_control_ms': float(differences.mean()), 'paired_mean_delta_lower95_ms': float(np.quantile(bootstrap, .025)), 'paired_mean_delta_upper95_ms': float(np.quantile(bootstrap, .975)), 'pilot_mean_faster': bool(np.quantile(bootstrap, .975) < 0), 'p99_certified': False}
            print(json.dumps({k:v for k,v in results[str(size)].items() if k != 'pairs'}), flush=True)
    for mode, encoder in encoders.items():
        expected = cpu['teacher_whole_sha256'] if mode == 'fp32_autocast' else authority['fp16_whole_sha256']
        assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == expected and pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
        assert json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor))) == cpu['environment']
        assert all(p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert teacher.qualified.numerical_flags() == flags and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA and pair.sha(serving.LIBRARY) == serving.LIBRARY_SHA
    assert all(pair.sha(root / n) == h for n,h in code.items()) and torch.cuda.max_memory_allocated() < 10_000_000_000
    pair.smoke.save(args.output, {'code': code, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': args.cpu_sha256, 'checkpoint_sha256': teacher.TEACHER_SHA, 'b32_receipt_sha256': B32_SHA, 'b1_receipt_sha256': args.b1_sha256, 'b1_audit_sha256': args.b1_audit_sha256, 'gallery_images': 6245, 'threads': 8, 'warmups': 5, 'timing': results, 'pilot_go': all(v['pilot_mean_faster'] for v in results.values()), 'source_head_rng_environment_code_library_preserved': True, 'all_repeated_native_ordinals_score_bits_exact': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0, 'official_read': False, 'claim_eligible': False, 'p99_certified': False})
    print('PASS fixed100-pair public latency pilot; no certified p99 or product claim')


if __name__ == '__main__':
    main()
