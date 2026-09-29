#!/usr/bin/env python3
"""Fixed terminal private TRAIN width qualification, not public256 serving."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

import argparse
import copy
import json
import math
import os
import statistics
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import train_pe_teacher_retained256 as training
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features

driver, pair = training.driver, training.pair
SOURCE = 'a85dd55c516f054c4c63341b31e0c7e4e77fb6925fd9ea29adabf097b1156bcb'
CONTROLLER = Path('/home/riomus/runs/sfora-teacher-retained256-controller-v1')
CONTROLLER_SHA = 'b1f1e3117372732a3c54aa530444e388b77a2b4f99bbefdfde26e28c2c443f7f'


def survives(comparison, costs):
    values = [comparison[m][k] for m in ('per_query_r1', 'per_query_ap') for k in ('mean_difference', 'product_lower95')]
    return bool(all(math.isfinite(v) for v in values + list(costs.values()))
        and comparison['per_query_r1']['mean_difference'] >= .005
        and comparison['per_query_ap']['mean_difference'] >= .01
        and all(comparison[m]['product_lower95'] > 0 for m in ('per_query_r1', 'per_query_ap'))
        and all(0 < costs[k] <= 1.10 for k in ('training_wall_ratio', 'median_step_ratio')))


def run_path(width, end=1000):
    return Path(f'/home/riomus/runs/sfora-teacher-retained256-{width}-{end}-v1')


def weights_path(width):
    return Path(f'/home/riomus/runs/sfora-teacher-retained256-weights-{width}-v1')


def wire_path(width):
    return Path(f'/home/riomus/runs/sfora-teacher-retained256-held-{width}-v1')


def authority(root, expected):
    manifest = root / 'teacher-retained256-qualification-execution.json'
    assert pair.sha(manifest) == expected
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'width qualification source differs'
    old = json.loads((root / 'teacher-retained256-execution.json').read_text())
    assert pair.sha(root / 'teacher-retained256-execution.json') == SOURCE
    assert all(code[n] == h for n, h in old.items())
    assert set(code) - set(old) == {'qualify_pe_teacher_retained256.py', 'test_pe_teacher_retained256_qualification.py'}
    selected = training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, _ = training.authority(root, SOURCE)
        _, frozen, _ = driver.coverage.trained.teacher.startup(root, driver.coverage.trained.TEACHER_CODE_SHA)
    assert frozen['fit_manifest'] == proof['arms']['half']['rows']
    assert len(frozen['query']) == 6354 and len(frozen['gallery']) == 6245
    assert sorted(frozen['query'] + frozen['gallery']) == list(range(12599))
    assert len({r['product'] for r in frozen['held_manifest']}) == 1993
    assert {r['product'] for r in frozen['held_manifest']}.isdisjoint(r['product'] for r in frozen['fit_manifest'])
    assert pair.sha(CONTROLLER / 'run_pe_teacher_retained256_pilot.py') == CONTROLLER_SHA
    state = json.loads((CONTROLLER / 'state.json').read_text())
    assert state['status'] == 'complete' and state['current_end'] == 1000 and state['source_sha256'] == SOURCE and not state['quality_read']
    assert len(state['completed']) == 20
    log = (CONTROLLER / 'controller.log').read_text()
    assert all(v in log for v in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    receipts = {128: [], 256: []}
    for width in (128, 256):
        previous_sha = None
        for end in range(100, 1001, 100):
            path = run_path(width, end) / 'receipt.json'; value = json.loads(path.read_text())
            assert value['pass'] and value['width'] == width and value['seed'] == 179032 and value['execution_sha256'] == SOURCE
            assert value['intervention'] == training.METHOD and value['total_updates'] == 1000 and value['chunk_start'] == end - 100 and value['completed_step'] == end
            assert value['previous_receipt_sha256'] == previous_sha and value['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA
            assert not value['quality_read'] and not value['training_state_discarded'] and value['peak_cuda_allocated_bytes'] < 10_000_000_000
            assert len(value['steps']) == 100 and [r['step'] for r in value['steps']] == list(range(end - 99, end + 1))
            assert value['schedule_sha256'] == 'a38d1be83d261856153dd75f59039250137f033fe3880245c13c658b0582d1aa'
            phase_log = (root / f'sfora-teacher-retained256-{width}-{end}-v1.log').read_text()
            assert all(v in phase_log for v in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
            previous_sha = pair.sha(path)
            assert any(r['width'] == width and r['end'] == end and r['receipt_sha256'] == previous_sha for r in state['completed'])
            receipts[width].append(value)
    for a, b in zip(receipts[128], receipts[256], strict=True):
        assert b['control_chunk_sha256'] == pair.sha(run_path(128, a['completed_step']) / 'receipt.json')
        assert all((x['step'], x['rgb_sha256'], x['pixels_sha256']) == (y['step'], y['rgb_sha256'], y['pixels_sha256']) for x, y in zip(a['steps'], b['steps'], strict=True))
    return control, source, prior, proof, frozen, code, receipts


def native(control, saved, width, device):
    model, processor = pair.smoke.load_arm(control, 'large')
    model.load_state_dict(saved['vision'], strict=True)
    head = nn.Linear(1024, width); head.load_state_dict(saved['head'], strict=True)
    assert len(saved['vision']) == 400 and saved['head']['weight'].shape == (width, 1024)
    if 'buffers' in saved:
        buffers = dict(model.named_buffers())
        assert buffers.keys() == saved['buffers'].keys()
        assert all(torch.equal(v, saved['buffers'][n]) for n, v in buffers.items())
    model.eval().requires_grad_(False); head.eval().requires_grad_(False)
    model.to(device); head.to(device)
    return model, head, processor


def export_cpu(root, args, control, source, proof, frozen, code, receipts):
    assert not torch.cuda.is_available() and args.output == weights_path(args.width) and not args.output.exists()
    final = receipts[args.width][-1]; checkpoint = run_path(args.width) / 'resume.pt'
    assert pair.sha(checkpoint) == final['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    identity = saved['identity']
    assert identity['width'] == args.width and identity['global_step'] == identity['total_updates'] == 1000 and identity['execution_sha256'] == SOURCE
    assert identity['arm'] == 'half' and identity['intervention'] == training.METHOD and identity['seed'] == 179032
    assert identity['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA and identity['schedule_sha256'] == final['schedule_sha256']
    assert saved['bank'].shape == (13283, args.width) and saved['classifier'].shape == (2004, args.width)
    assert identity['tail_sha256'] == final['tail_sha256'] and driver.fingerprint(saved['buffers']) == identity['buffers_sha256']
    assert all(torch.isfinite(v).all() for role in ('vision', 'head', 'buffers') for v in saved[role].values())
    with torch.random.fork_rng():
        model, head, processor = native(control, saved, args.width, 'cpu')
        pair.smoke.freeze_prefix(model, 'large')
        assert driver.coverage.frozen_digest(model, proof['inventory']) == proof['frozen_prefix_sha256']
        images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1), None)
        pixels = pair.pixels(processor, images, 'large')
        with torch.inference_mode():
            pooled = model(pixel_values=pixels).pooler_output
            expected = pack_int8_unit_embeddings(F.normalize(compact_head_features(pooled, head, output_dim=args.width), dim=1))
        clone = type(model)(copy.deepcopy(model.config)).float().eval()
        clone.load_state_dict(saved['vision'], strict=True)
        cloned_head = nn.Linear(1024, args.width).eval(); cloned_head.load_state_dict(saved['head'], strict=True)
        assert pair.smoke.digest(driver.coverage.trained.base.whole_state(clone)) == pair.smoke.digest(driver.coverage.trained.base.whole_state(model))
        with torch.inference_mode():
            actual = pack_int8_unit_embeddings(F.normalize(compact_head_features(clone(pixel_values=pixels).pooler_output, cloned_head, output_dim=args.width), dim=1))
        assert torch.equal(expected.codes, actual.codes) and torch.equal(expected.inverse_norms, actual.inverse_norms)
        whole = pair.smoke.digest(driver.coverage.trained.base.whole_state(model))
        f16_whole = pair.smoke.digest(driver.coverage.trained.base.whole_state(model.half()))
    args.output.mkdir(exist_ok=False)
    torch.save({'vision': saved['vision'], 'head': saved['head']}, args.output / 'native.pt')
    real_sha = pair.sha
    with patch.object(pair, 'sha', lambda p: 'altered' if Path(p).resolve() == Path(__file__).resolve() else real_sha(p)):
        try:
            authority(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'width qualification source differs'
        else:
            raise AssertionError('changed qualifier accepted')
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'width': args.width, 'code': code, 'execution_sha256': args.execution_sha256,
        'training_receipt_sha256': pair.sha(run_path(args.width) / 'receipt.json'), 'source_resume_sha256': final['checkpoint_sha256'],
        'checkpoint_sha256': pair.sha(args.output / 'native.pt'), 'whole_sha256': whole, 'f16_whole_sha256': f16_whole,
        'head_sha256': pair.smoke.digest(saved['head']), 'actual_updated_CPU_B2_packed_reload_exact': True, 'frozen_prefix_exact': True,
        'changed_driver_rejected': True, 'quality_read': False, 'training_state_resumable': False, 'public256_qualified': False})
    print('PASS actual updated private width CPU strict400/head/frozen/packed/F16cast/source export')


def metrics(root, q, g, labels, device):
    assert q.codes.shape[1] == g.codes.shape[1] and q.codes.shape[1] in (128, 256)
    for packed, names in zip((q, g), labels, strict=True):
        assert packed.codes.dtype == torch.int8 and packed.codes.shape[0] == len(names)
        assert packed.inverse_norms.dtype == torch.float16 and packed.inverse_norms.shape == (len(names),)
        norms = torch.linalg.vector_norm(packed.codes.float(), dim=1)
        assert packed.codes.min() >= -127 and (norms > 0).all() and torch.equal(norms.reciprocal().half(), packed.inverse_norms)
    official = training.previous.cpu.previous.qualified.confirmation.selected.helpers(root, json.loads((root / 'teacher-retained256-qualification-execution.json').read_text()))
    return official.score_asymmetric(q.codes.float().to(device), g.codes.float().to(device), *labels,
        query_inverse=q.inverse_norms.float().to(device), gallery_inverse=g.inverse_norms.float().to(device))


def wires(root, args, control, source, prior, frozen, code, receipts):
    exported = weights_path(args.width)
    assert pair.sha(exported / 'receipt.json') == args.cpu_sha256
    value = json.loads((exported / 'receipt.json').read_text())
    assert value['pass'] and value['width'] == args.width and value['code'] == code and value['execution_sha256'] == args.execution_sha256
    assert value['training_receipt_sha256'] == pair.sha(run_path(args.width) / 'receipt.json')
    assert value['actual_updated_CPU_B2_packed_reload_exact'] and value['changed_driver_rejected']
    assert pair.sha(exported / 'native.pt') == value['checkpoint_sha256']
    rows = {role: [frozen['held_manifest'][i] for i in frozen[role]] for role in ('query', 'gallery')}
    labels = tuple(tuple(r['product'] for r in rows[role]) for role in ('query', 'gallery'))
    if args.phase == 'audit':
        assert not torch.cuda.is_available() and args.receipt_sha256 and not (args.output / 'cpu-audit.json').exists()
        assert pair.sha(args.output / 'receipt.json') == args.receipt_sha256
        received = json.loads((args.output / 'receipt.json').read_text())
        assert received['code'] == code and received['checkpoint_sha256'] == value['checkpoint_sha256'] and received['width'] == args.width
        assert received['query_rows'] == rows['query'] and received['gallery_rows'] == rows['gallery']
        parts = {}
        for role in rows:
            arrays = []
            for field in ('codes', 'inverse'):
                path = args.output / (role + '.' + field + '.npy')
                assert pair.sha(path) == received[role + '_' + field + '_sha256']
                arrays.append(torch.from_numpy(np.load(path, allow_pickle=False)))
            parts[role] = PackedInt8Embeddings(*arrays)
            path = args.output / (role + '.float.npy')
            assert pair.sha(path) == received[role + '_float_sha256']
            floats = torch.from_numpy(np.load(path, allow_pickle=False))
            assert floats.dtype == torch.float32 and floats.shape == (len(rows[role]), args.width) and torch.isfinite(floats).all()
            replayed = pack_int8_unit_embeddings(floats)
            assert torch.equal(replayed.codes, parts[role].codes) and torch.equal(replayed.inverse_norms, parts[role].inverse_norms)
        quality = metrics(root, parts['query'], parts['gallery'], labels, 'cpu')
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(received['quality'][k]))) < 1e-6 for k, v in quality.items())
        pair.smoke.save(args.output / 'cpu-audit.json', {'pass': True, 'width': args.width, 'receipt_sha256': args.receipt_sha256, 'quality': quality, 'public256_qualified': False})
        print('PASS complete saved private TRAIN packed wires independently CPU replayed'); return
    assert torch.cuda.is_available() and not args.output.exists() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert driver.coverage.teacher.qualified.numerical_flags() == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    saved = torch.load(exported / 'native.pt', map_location='cpu', weights_only=True, mmap=True)
    with torch.random.fork_rng(devices=[0]):
        model, head, processor = native(control, saved, args.width, 'cuda'); model.half()
        reference, projection, original_processor = native(control, saved, args.width, 'cuda'); reference.half()
    assert processor is not original_processor
    rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    arrays, floats = {}, {}
    with torch.inference_mode():
        for role in rows:
            chunks = []
            for start in range(0, len(rows[role]), 32):
                images, _ = pair.augmented_images(control.dataset_root, rows[role][start:start + 32], tuple(range(min(32, len(rows[role]) - start))), None)
                pixels = pair.pixels(processor, images, 'large').half().cuda()
                features = F.normalize(compact_head_features(model(pixel_values=pixels).pooler_output, head, output_dim=args.width), dim=1).cpu()
                if start == 0:
                    original_pixels = pair.pixels(original_processor, images, 'large').half().cuda()
                    original = F.normalize(compact_head_features(reference(pixel_values=original_pixels).pooler_output, projection, output_dim=args.width), dim=1).cpu()
                    assert torch.equal(features, original); driver.previous.training.packed_equal(features, original)
                assert torch.isfinite(features).all() and torch.cuda.max_memory_allocated() < 10_000_000_000
                chunks.append(features)
            floats[role] = torch.cat(chunks); arrays[role] = pack_int8_unit_embeddings(floats[role])
    quality = metrics(root, arrays['query'], arrays['gallery'], labels, 'cuda')
    for m, h in ((model, head), (reference, projection)):
        assert pair.smoke.digest(driver.coverage.trained.base.whole_state(m)) == value['f16_whole_sha256']
        assert pair.smoke.digest(h.state_dict()) == value['head_sha256'] and not m.training and not h.training
        assert all(p.grad is None for p in m.parameters()) and all(p.grad is None for p in h.parameters())
    assert torch.equal(rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert all(pair.sha(root / n) == h for n, h in code.items()) and pair.sha(exported / 'native.pt') == value['checkpoint_sha256']
    args.output.mkdir(exist_ok=False); hashes = {}
    for role in rows:
        for field, tensor in (('codes', arrays[role].codes), ('inverse', arrays[role].inverse_norms), ('float', floats[role])):
            path = args.output / (role + '.' + field + '.npy'); np.save(path, tensor.numpy(), allow_pickle=False)
            hashes[role + '_' + field + '_sha256'] = pair.sha(path)
    pair.smoke.save(args.output / 'receipt.json', {**hashes, 'width': args.width, 'code': code, 'execution_sha256': args.execution_sha256,
        'checkpoint_sha256': value['checkpoint_sha256'], 'cpu_sha256': args.cpu_sha256, 'query_rows': rows['query'], 'gallery_rows': rows['gallery'],
        'quality': quality, 'independent_original_processor_first32_each_role_exact': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'batch': 32, 'precision': 'private_native_fp16', 'public256_qualified': False, 'public_latency_measured': False, 'official_read': False})
    print('PASS private nativeFP16 terminal TRAIN wires; public256/native kernel/speed remain unqualified')


def decision(root, args, code, receipts):
    assert not torch.cuda.is_available() and not args.output.exists()
    audits, values, hashes = {}, {}, {}
    for width in (128, 256):
        path = wire_path(width)
        value = json.loads((path / 'receipt.json').read_text()); audit = json.loads((path / 'cpu-audit.json').read_text())
        assert value['width'] == width and value['code'] == code and value['execution_sha256'] == args.execution_sha256
        assert value['checkpoint_sha256'] == json.loads((weights_path(width) / 'receipt.json').read_text())['checkpoint_sha256']
        assert audit['pass'] and audit['receipt_sha256'] == pair.sha(path / 'receipt.json') and audit['width'] == width
        assert not value['public256_qualified'] and not value['official_read'] and value['independent_original_processor_first32_each_role_exact']
        for role in ('query', 'gallery'):
            for field in ('codes', 'inverse', 'float'):
                assert pair.sha(path / (role + '.' + field + '.npy')) == value[role + '_' + field + '_sha256']
        for phase in ('cpu', 'private', 'audit'):
            training.normal_log(root, f'width-{width}-{phase}')
        values[width], audits[width] = value, audit
        hashes[width] = {'receipt_sha256': pair.sha(path / 'receipt.json'), 'audit_sha256': pair.sha(path / 'cpu-audit.json')}
    assert values[128]['query_rows'] == values[256]['query_rows'] and values[128]['gallery_rows'] == values[256]['gallery_rows']
    labels = np.asarray([r['product'] for r in values[128]['query_rows']]); comparison = {}
    for metric in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(audits[256]['quality'][metric]) - np.asarray(audits[128]['quality'][metric])
        assert delta.shape == (6354,) and np.isfinite(delta).all()
        comparison[metric] = {'mean_difference': float(delta.mean())}
        for kind, groups in (('product', labels), ('query', np.arange(len(delta)))):
            comparison[metric][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            comparison[metric][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    costs = {width: {'training_wall_seconds': sum(r['training_wall_seconds'] for r in receipts[width]),
        'median_step_seconds': statistics.median(s['seconds'] for r in receipts[width] for s in r['steps']),
        'unit_walls_excluded_from_images_per_second': True} for width in (128, 256)}
    ratios = {'training_wall_ratio': costs[256]['training_wall_seconds'] / costs[128]['training_wall_seconds'],
              'median_step_ratio': costs[256]['median_step_seconds'] / costs[128]['median_step_seconds']}
    for width in costs:
        costs[width]['images_per_second'] = 64000 / costs[width]['training_wall_seconds']
    outcome = survives(comparison, ratios)
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'decision.json', {'pass': True, 'decision': 'GO' if outcome else 'KILL', 'procedure_only': True,
        'execution_sha256': args.execution_sha256, 'code': code, 'matched_receipts': hashes,
        'control_quality': audits[128]['quality'], 'candidate_quality': audits[256]['quality'], 'comparison': comparison,
        'training_costs': costs, 'cost_ratios': ratios, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019,
        'dataset_split': 'previously observed In-Shop TRAIN-held6354q/6245g/1993held vs13283fit/2004ids',
        'uncertainty_scope': 'fixed F5 and one matched schedule/augmentation seed, not teacher/training seed population',
        'public256_qualified': False, 'public_latency_measured': False, 'official_read': False, 'full_production_goal_achieved': False})
    print(('GO' if outcome else 'KILL') + ' fixed corrected width TRAIN procedure; global product goal remains active')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True); p.add_argument('--phase', choices=('cpu', 'private', 'audit', 'decision'), required=True)
    p.add_argument('--width', choices=(128, 256), type=int, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-sha256'); p.add_argument('--receipt-sha256')
    args = p.parse_args(); root = Path(__file__).resolve().parent
    torch.set_num_threads(8); torch.manual_seed(179032)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    control, source, prior, proof, frozen, code, receipts = authority(root, args.execution_sha256)
    if args.phase == 'cpu':
        export_cpu(root, args, control, source, proof, frozen, code, receipts)
    elif args.phase == 'decision':
        decision(root, args, code, receipts)
    else:
        wires(root, args, control, source, prior, frozen, code, receipts)


if __name__ == '__main__':
    main()
