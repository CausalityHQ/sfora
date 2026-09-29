#!/usr/bin/env python3
"""Fresh read-only TRAIN-held qualification of the trained Large vision/head."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import export_pe_large_teacher_fit as teacher
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

native, base, pair = teacher.native, teacher.base, teacher.pair
TEACHER_CODE_SHA = '39736900038d368b8153748041da23dbe3b732246ddb09242c399674288e0a45'
CPU_SHA = '4731d7a4a4a3fc645e22dd67cd58ebbb39479df550ed031983d159fd7afcccb8'
FIT = Path('/home/riomus/runs/sfora-large-teacher-fit-targets-v1/receipt.json')
FIT_SHA = '277f9b774470355f76c16675a9c7808abe602c83612d83f4d90e4d37cb5507fe'


def startup(root, execution_sha):
    manifest = root / 'trained-candidate-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'trained candidate code differs'
    control, frozen, old = teacher.startup(root, TEACHER_CODE_SHA)
    assert all(code[n] == h for n, h in old.items())
    assert pair.sha(teacher.CPU) == CPU_SHA and pair.sha(FIT) == FIT_SHA
    cpu, fit = json.loads(teacher.CPU.read_text()), json.loads(FIT.read_text())
    assert cpu['code'] == fit['code'] == old and fit['cpu_authority_sha256'] == CPU_SHA
    for key in ('teacher_checkpoint_sha256', 'teacher_whole_sha256', 'teacher_verification_prefix_sha256', 'teacher_head_sha256', 'environment', 'first_two_fit_pixels_sha256'):
        assert cpu[key] == fit[key]
    assert cpu['teacher_checkpoint_sha256'] == teacher.TEACHER_SHA
    assert fit['read_only'] and fit['optimizer_updates'] == fit['held_images'] == 0 and not fit['quality_read']
    assert fit['cpu_cuda_rng_unchanged'] and fit['prefix_data_mutation_rejected_at_exit']
    assert len(frozen['held_manifest']) == 12599 and len(frozen['query']) == 6354 and len(frozen['gallery']) == 6245
    assert sorted(frozen['query'] + frozen['gallery']) == list(range(12599))
    products = {r['product'] for r in frozen['held_manifest']}
    assert len(products) == 1993 and products.isdisjoint(r['product'] for r in frozen['fit_manifest'])
    return control, frozen, cpu, fit, code


def score(values, frozen, device):
    labels = tuple(r['product'] for r in frozen['held_manifest'])
    quality = pair.packed_quality(values, labels, frozen['query'], frozen['gallery'], device=device)
    dense = json.loads((base.INIT / 'receipt.json').read_text())['arms']['pe']['quality']
    products = np.asarray(labels)[frozen['query']]
    intervals = {}
    for name in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(quality[name]) - np.asarray(dense[name])
        intervals[name] = {'mean_delta': float(np.mean(delta))}
        for kind, groups in (('product', products), ('query', np.arange(len(delta)))):
            intervals[name][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            intervals[name][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    advance = bool(quality['recall_at_1'] >= .951720176 and quality['map_at_r'] >= .776237120 and all(v['product_lower95'] > 0 for v in intervals.values()))
    return quality, intervals, advance


def audit(output, receipt_sha, frozen, code, execution_sha):
    assert not torch.cuda.is_available() and not (output / 'cpu-audit.json').exists()
    path = output / 'receipt.json'
    assert pair.sha(path) == receipt_sha
    receipt = json.loads(path.read_text())
    assert receipt['execution_sha256'] == execution_sha and receipt['code'] == code
    assert receipt['teacher_checkpoint_sha256'] == teacher.TEACHER_SHA
    assert receipt['read_only'] and receipt['optimizer_updates'] == 0 and receipt['official_read'] is False and receipt['claim_eligible'] is False
    assert receipt['held_manifest'] == frozen['held_manifest'] and receipt['query'] == frozen['query'] and receipt['gallery'] == frozen['gallery']
    assert receipt['full_held_independent_whole_encoder_exact'] is False
    assert all(receipt[k] for k in ('strict400_whole_calibration_exact', 'independent_two_block_suffix_head_packed_exact', 'source_head_rng_environment_unchanged'))
    assert receipt['held_images'] == 12599 and len(receipt['scope_observations']) == 394 and receipt['peak_cuda_allocated_bytes'] < 10_000_000_000
    values = []
    for name, key in (('large.held.npy', 'held_sha256'), ('large.reference-held.npy', 'reference_held_sha256')):
        assert pair.sha(output / name) == receipt[key]
        array = np.load(output / name, allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (12599, 128) and np.isfinite(array).all()
        assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5, rtol=0)
        values.append(array)
    assert np.array_equal(*values)
    quality, intervals, advance = score(values[0], frozen, torch.device('cpu'))
    assert all(np.max(np.abs(np.asarray(v) - np.asarray(receipt['quality'][k]))) < 1e-6 for k, v in quality.items())
    assert all(abs(v - receipt['paired_dense_pe_intervals'][k][n]) < 1e-6 for k, row in intervals.items() for n, v in row.items())
    assert advance == receipt['advance'] and pair.sha(path) == receipt_sha
    assert pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA
    assert all(pair.sha(Path(__file__).resolve().parent / n) == h for n, h in code.items())
    pair.smoke.save(output / 'cpu-audit.json', {'pass': True, 'advance': advance, 'receipt_sha256': receipt_sha, 'execution_sha256': execution_sha, 'quality': quality, 'paired_dense_pe_intervals': intervals, 'official_read': False, 'claim_eligible': False})
    print('PASS complete trained Large CPU packed per-query/interval/decision replay')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-startup-only', action='store_true')
    parser.add_argument('--audit-cpu', action='store_true')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    assert not (args.check_startup_only and args.audit_cpu)
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, cpu, fit, code = startup(root, args.execution_sha256)
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'trained candidate code differs'
            else:
                raise AssertionError('changed trained candidate driver accepted')
        print('PASS trained candidate source/CPU/fit/held-row startup and changed-driver rejection; no CUDA/images/quality')
        return
    if args.audit_cpu:
        assert args.receipt_sha256
        audit(args.output, args.receipt_sha256, frozen, code, args.execution_sha256)
        return
    assert torch.cuda.is_available() and not args.output.exists()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    numerical = teacher.qualified.numerical_flags()
    assert numerical == fit['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    loaded, live, heads, processor = teacher.teacher_pair(control, torch.device('cuda'))
    environment = json.loads(json.dumps(native.environment(live, processor)))
    assert environment == cpu['environment']

    def unchanged():
        assert all(pair.smoke.digest(base.whole_state(m)) == cpu['teacher_whole_sha256'] and pair.smoke.digest(native.frozen_state(m)) == cpu['teacher_verification_prefix_sha256'] for m in (loaded, live))
        assert all(pair.smoke.digest(h.state_dict()) == cpu['teacher_head_sha256'] for h in heads)
        assert all(p.grad is None for m in (loaded, live, *heads) for p in m.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for model in (loaded, live) for m in model.modules())
        assert json.loads(json.dumps(native.environment(live, processor))) == environment
        assert teacher.qualified.numerical_flags() == numerical and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA
        assert all(pair.sha(root / n) == h for n, h in code.items())
        assert torch.cuda.max_memory_allocated() < 10_000_000_000

    unchanged()
    images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1), None)
    pixels = pair.pixels(processor, images, 'large')
    assert pair.smoke.digest({'pixels': pixels}) == cpu['first_two_fit_pixels_sha256']
    pixels = pixels.cuda()
    with torch.no_grad():
        direct = teacher.qualified.fp16(live, pixels)
        assert torch.equal(direct, teacher.qualified.fp16(loaded, pixels))
        full_precision = live(pixel_values=pixels).pooler_output.float()
        calibration = {'pooled': F.cosine_similarity(direct, full_precision).tolist(), 'compact': F.cosine_similarity(F.normalize(pair.smoke.compact_head_features(direct, heads[1]), dim=1), F.normalize(pair.smoke.compact_head_features(full_precision, heads[1]), dim=1)).tolist()}
        assert all(min(v) >= .999 for v in calibration.values())
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    chunks, reference_chunks = [], []
    with torch.no_grad(), native.verify_export(loaded, live) as (encode, scopes):
        for start in range(0, len(frozen['held_manifest']), 32):
            batch = frozen['held_manifest'][start:start + 32]
            images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
            x = pair.pixels(processor, images, 'large').cuda()
            a, b = encode(x)
            va = F.normalize(pair.smoke.compact_head_features(a, heads[0]).float(), dim=1)
            vb = F.normalize(pair.smoke.compact_head_features(b, heads[1]).float(), dim=1)
            assert torch.isfinite(va).all() and torch.equal(va, vb)
            chunks.append(va.cpu().numpy())
            reference_chunks.append(vb.cpu().numpy())
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            print(json.dumps({'held_images_verified': start + len(batch)}), flush=True)
    assert len(scopes) == 394 and all(not r['between_encoder_amp_enabled'] and all(r[k] for k in ('loaded_amp_inside', 'captured_block_amp_inside', 'captured_final_block_amp_inside', 'live_amp_inside')) for r in scopes)
    unchanged()
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    values, reference = np.concatenate(chunks), np.concatenate(reference_chunks)
    assert values.shape == (12599, 128) and np.array_equal(values, reference)
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    a, b = pack_int8_unit_embeddings(torch.from_numpy(values)), pack_int8_unit_embeddings(torch.from_numpy(reference))
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.inverse_norms, b.inverse_norms)
    quality, intervals, advance = score(values, frozen, torch.device('cuda'))
    labels = tuple(r['product'] for r in frozen['held_manifest'])
    reference_quality = pair.packed_quality(reference, labels, frozen['query'], frozen['gallery'])
    assert all(np.array_equal(np.asarray(quality[k]), np.asarray(reference_quality[k])) for k in ('per_query_r1', 'per_query_ap'))
    unchanged()
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    args.output.mkdir(exist_ok=False)
    np.save(args.output / 'large.held.npy', values, allow_pickle=False)
    np.save(args.output / 'large.reference-held.npy', reference, allow_pickle=False)
    pair.smoke.save(args.output / 'receipt.json', {'advance': advance, 'code': code, 'execution_sha256': args.execution_sha256, 'teacher_checkpoint_sha256': teacher.TEACHER_SHA, 'teacher_whole_sha256': cpu['teacher_whole_sha256'], 'teacher_verification_prefix_sha256': cpu['teacher_verification_prefix_sha256'], 'teacher_head_sha256': cpu['teacher_head_sha256'], 'cpu_authority_sha256': CPU_SHA, 'fit_authority_sha256': FIT_SHA, 'environment': environment, 'numerical_flags': numerical, 'fp16_fp32_calibration': calibration, 'held_manifest': frozen['held_manifest'], 'query': frozen['query'], 'gallery': frozen['gallery'], 'held_images': 12599, 'scope_observations': scopes, 'held_sha256': pair.sha(args.output / 'large.held.npy'), 'reference_held_sha256': pair.sha(args.output / 'large.reference-held.npy'), 'quality': quality, 'paired_dense_pe_intervals': intervals, 'strict400_whole_calibration_exact': True, 'independent_two_block_suffix_head_packed_exact': True, 'full_held_independent_whole_encoder_exact': False, 'source_head_rng_environment_unchanged': True, 'read_only': True, 'optimizer_updates': 0, 'quality_read': 'In-Shop TRAIN-held only', 'official_read': False, 'claim_eligible': False, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    print('GO trained Large native TRAIN quality; serving/confirmation remain' if advance else 'KILL trained Large fresh native TRAIN quality gate')


if __name__ == '__main__':
    main()
