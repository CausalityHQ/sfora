#!/usr/bin/env python3
"""One frozen native trajectory average; truthful TRAIN public qualification."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
import qualify_pe_large_optimization_checkpoint as previous
from evaluate_unicom_checkpoint_soup import average_model_states
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

qualified = previous.confirmation
pair, trained, teacher = qualified.pair, qualified.trained, qualified.teacher
OLD_CODE = '8c6771b3fc63de4967120c6ae1a388b8e5311a454bfd060716488859fac6aa0c'
TRAIN = '40d99e5f240c076ce71006cd670de41cbf814c51d36344c25473ebde9f1d560e'
ENDPOINT = Path('/home/riomus/runs/sfora-large-optimization-half-weights-v1')
ENDPOINT_SHA = 'ae9b2e929b7dc792af12f6d641d0dee2e934e1579f4cf91bdc7ba11ebbd43af7'
AVERAGE = Path('/home/riomus/runs/sfora-native-trajectory-average-weights-v1')
STEPS = tuple(range(1100, 2001, 100))


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def average_native(states):
    assert tuple(s['identity']['global_step'] for s in states) == STEPS
    roles = states[-1]['identity']['model_roles']
    assert all(s['identity']['model_roles'] == roles for s in states)
    assert tuple(n for n, _ in roles) == tuple(states[-1]['vision'])
    for s in states:
        assert s['vision'].keys() == states[-1]['vision'].keys() and s['head'].keys() == states[-1]['head'].keys()
        assert s['buffers'].keys() == states[-1]['buffers'].keys()
        assert all(torch.isfinite(v).all() for role in ('vision', 'head', 'buffers') for v in s[role].values())
        assert all(same_bytes(v, states[-1]['buffers'][n]) for n, v in s['buffers'].items())
        assert all(train or same_bytes(s['vision'][n], states[-1]['vision'][n]) for n, train in roles)
    trainable = tuple({**{'vision.' + n: s['vision'][n] for n, train in roles if train},
                       **{'head.' + n: v for n, v in s['head'].items()}} for s in states)
    mean = average_model_states(trainable)
    result = {'vision': {n: mean['vision.' + n] if train else states[-1]['vision'][n] for n, train in roles},
              'head': {n: mean['head.' + n] for n in states[-1]['head']}}
    assert all(torch.isfinite(v).all() for role in result.values() for v in role.values())
    return result


def authority(root, execution):
    path = root / 'native-trajectory-average-execution.json'
    assert pair.sha(path) == execution
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'trajectory source differs'
    old_path = root / 'large-optimization-checkpoint-execution.json'
    assert pair.sha(old_path) == OLD_CODE
    old = json.loads(old_path.read_text())
    assert all(code[n] == h == pair.sha(root / n) for n, h in old.items())
    assert set(code) - set(old) == {'qualify_pe_native_trajectory_average.py', 'evaluate_unicom_checkpoint_soup.py', 'test_pe_native_trajectory_average.py', 'fixed-trajectory-average-plan.md'}
    helpers = qualified.selected.helpers
    # Authenticate unchanged legacy sources before extending their import inventory.
    with patch.object(qualified.selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, run, terminal, _, official, prior_code = previous.authority(root, OLD_CODE, 'half', TRAIN)
        _, frozen, _ = trained.teacher.startup(root, trained.TEACHER_CODE_SHA)
    assert prior_code == old and frozen['fit_manifest'] == proof['arms']['half']['rows']
    assert len(frozen['fit_manifest']) == 13283 and len(frozen['held_manifest']) == 12599
    assert len(frozen['query']) == 6354 and len(frozen['gallery']) == 6245
    assert sorted(frozen['query'] + frozen['gallery']) == list(range(12599))
    assert len({r['product'] for r in frozen['held_manifest']}) == 1993
    assert {r['product'] for r in frozen['held_manifest']}.isdisjoint(r['product'] for r in frozen['fit_manifest'])
    assert pair.sha(ENDPOINT / 'native.pt') == ENDPOINT_SHA
    endpoint = json.loads((ENDPOINT / 'receipt.json').read_text())
    assert endpoint['pass'] and endpoint['training_receipt_sha256'] == TRAIN and endpoint['source_resume_sha256'] == terminal['checkpoint_sha256']
    assert endpoint['checkpoint_sha256'] == ENDPOINT_SHA and endpoint['execution_sha256'] == OLD_CODE
    return control, source, prior, proof, run, terminal, frozen, official, code, endpoint


def export(root, execution, output):
    assert not torch.cuda.is_available() and output == AVERAGE and not output.exists()
    start = time.perf_counter()
    _, _, _, proof, _, terminal, _, _, code, _ = authority(root, execution)
    states, inputs = [], []
    base_identity = None
    for step in STEPS:
        run = Path(f'/home/riomus/runs/sfora-large-optimization-half-{step}-v1')
        receipt = json.loads((run / 'receipt.json').read_text())
        assert receipt['completed_step'] == step and receipt['execution_sha256'] == previous.TRAIN_CODE_SHA
        checkpoint = run / 'resume.pt'
        assert pair.sha(checkpoint) == receipt['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        identity = state['identity']
        assert identity['global_step'] == step and identity['total_updates'] == 2000 and identity['arm'] == 'half'
        assert identity['seed'] == pair.SEED and identity['execution_sha256'] == previous.TRAIN_CODE_SHA
        assert identity['source_checkpoint_sha256'] == teacher.TEACHER_SHA and identity['schedule_sha256'] == terminal['schedule_sha256']
        assert identity['frozen_prefix_sha256'] == proof['frozen_prefix_sha256'] and len(state['vision']) == 400
        assert previous.driver.fingerprint(state['buffers']) == identity['buffers_sha256']
        current = {k: v for k, v in identity.items() if k != 'global_step'}
        base_identity = current if base_identity is None else base_identity
        assert current == base_identity
        assert identity['parameter_names'] == [n for n, train in identity['model_roles'] if train] + ['compact_head.' + n for n in state['head']] + ['classifier']
        states.append(state)
        inputs.append({'step': step, 'receipt_sha256': pair.sha(run / 'receipt.json'), 'resume_sha256': receipt['checkpoint_sha256']})
    serving = average_native(tuple(states))
    assert any(not same_bytes(serving['vision'][n], states[-1]['vision'][n]) for n, train in base_identity['model_roles'] if train)
    assert any(not same_bytes(v, states[-1]['head'][n]) for n, v in serving['head'].items())
    output.mkdir(exist_ok=False)
    torch.save(serving, output / 'native.pt')
    whole = {**serving['vision'], **{'runtime.' + n: v for n, v in states[-1]['buffers'].items()}}
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(output / 'receipt.json', {'pass': True, 'intervention': 'fixed-native-trajectory-average-v1', 'arm': 'half', 'completed_step': 2000,
        'execution_sha256': execution, 'training_receipt_sha256': TRAIN, 'source_checkpoint_sha256': teacher.TEACHER_SHA,
        'inputs': inputs, 'steps': STEPS, 'checkpoint_sha256': pair.sha(output / 'native.pt'), 'updated_whole_sha256': pair.smoke.digest(whole),
        'updated_head_sha256': pair.smoke.digest(serving['head']), 'endpoint_checkpoint_sha256': ENDPOINT_SHA,
        'frozen_runtime_buffers_exact': True, 'trainable_vision_and_head_fp64_uniform_mean': True,
        'optimizer_updates': 0, 'quality_read': False, 'export_wall_seconds': time.perf_counter() - start,
        'aggregate_archived_training_wall_seconds': terminal['aggregate_training_wall_seconds'], 'training_state_resumable': False})
    print('PASS fixed ten-checkpoint serving-only average; no quality read')


def loaded_authority(root, execution, variant):
    result = authority(root, execution)
    run = ENDPOINT if variant == 'endpoint' else AVERAGE
    value = result[-1] if variant == 'endpoint' else json.loads((run / 'receipt.json').read_text())
    assert value['pass'] and value['training_receipt_sha256'] == TRAIN
    assert pair.sha(run / 'native.pt') == value['checkpoint_sha256']
    if variant == 'average':
        assert value['execution_sha256'] == execution and value['steps'] == list(STEPS)
        assert value['frozen_runtime_buffers_exact'] and value['trainable_vision_and_head_fp64_uniform_mean'] and not value['quality_read']
    return result, run, value


def cpu(root, args):
    result, run, value = loaded_authority(root, args.execution_sha256, args.variant)
    control, source, prior, proof, _, _, _, official, code, _ = result
    def startup(r, execution, arm, training):
        assert r == root and execution == args.execution_sha256 and arm == 'half' and training == TRAIN
        assert all(pair.sha(root / n) == h for n, h in code.items()), 'coverage checkpoint code differs'
        assert pair.sha(run / 'native.pt') == value['checkpoint_sha256']
        return control, source, prior, proof, run, value, {}, official, code
    argv = ['average-native-cpu', '--execution-sha256', args.execution_sha256, '--arm', 'half', '--training-sha256', TRAIN, '--output', str(args.output), '--qualify-cpu']
    with patch.object(qualified, 'startup', startup), patch.object(sys, 'argv', argv):
        qualified.main()


def wires(root, args, audit):
    result, run, value = loaded_authority(root, args.execution_sha256, args.variant)
    control, source, prior, proof, _, _, frozen, official, code, _ = result
    assert args.cpu_proof and args.cpu_sha256 and pair.sha(args.cpu_proof) == args.cpu_sha256
    cpu_proof = json.loads(args.cpu_proof.read_text())
    assert cpu_proof['pass'] and cpu_proof['code'] == code and cpu_proof['checkpoint_sha256'] == value['checkpoint_sha256']
    assert cpu_proof['strict_updated_native_CPU_B2_whole_head_packed_exact'] and cpu_proof['frozen_prefix_matches_original'] and cpu_proof['changed_driver_rejected']
    rows = {role: [frozen['held_manifest'][i] for i in frozen[role]] for role in ('query', 'gallery')}
    qlabels, glabels = (tuple(r['product'] for r in rows[role]) for role in ('query', 'gallery'))
    if audit:
        assert not torch.cuda.is_available() and args.receipt_sha256 and not (args.output / 'cpu-audit.json').exists()
        assert pair.sha(args.output / 'receipt.json') == args.receipt_sha256
        measured = json.loads((args.output / 'receipt.json').read_text())
        assert measured['code'] == code and measured['checkpoint_sha256'] == value['checkpoint_sha256'] and measured['variant'] == args.variant
        assert measured['cpu_authority_sha256'] == args.cpu_sha256 and measured['query_rows'] == rows['query'] and measured['gallery_rows'] == rows['gallery']
        assert measured['native_all_query_top10_ordinal_score_bits_exact'] and measured['source_state_rng_environment_code_library_preserved']
        q, g = (qualified.selected.fp16.load_packed(args.output, measured, role) for role in ('query', 'gallery'))
        quality, _, _ = qualified.selected.metrics(official, q, g, qlabels, glabels, torch.device('cpu'))
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(measured['quality'][k]))) < 1e-6 for k, v in quality.items())
        pair.smoke.save(args.output / 'cpu-audit.json', {'pass': True, 'variant': args.variant, 'receipt_sha256': args.receipt_sha256, 'quality': quality, 'official_read': False, 'claim_eligible': False})
        print('PASS complete saved TRAIN wires independently CPU audited')
        return
    assert torch.cuda.is_available() and not args.output.exists() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=run / 'native.pt', expected_checkpoint_sha256=value['checkpoint_sha256'], model_file_sha256=pair.smoke.MODEL_HASHES, precision='fp16_native', device=torch.device('cuda'))
    reference, head, processor = qualified.updated(control, source, proof, run, value)
    reference.half().cuda(); head.cuda()
    qualified.same_runtime(reference, encoder.vision)
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    def preserved():
        for model in (reference, encoder.vision):
            assert pair.smoke.digest(trained.base.whole_state(model)) == cpu_proof['updated_f16_whole_sha256']
            assert all(p.grad is None for p in model.parameters()) and not model.training
            assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert all(pair.smoke.digest(h.state_dict()) == cpu_proof['updated_head_sha256'] and all(p.grad is None for p in h.parameters()) for h in (head, encoder.head))
        assert all(json.loads(json.dumps(trained.native.environment(m, p))) == cpu_proof['environment'] for m, p in ((reference, processor), (encoder.vision, encoder.processor)))
        assert pair.sha(run / 'native.pt') == value['checkpoint_sha256'] and all(pair.sha(root / n) == h for n, h in code.items())
        assert pair.sha(qualified.selected.serving.LIBRARY) == qualified.selected.serving.LIBRARY_SHA
        assert teacher.qualified.numerical_flags() == flags and torch.cuda.max_memory_allocated() < 10_000_000_000
    preserved()
    arrays = {}
    for role in ('query', 'gallery'):
        chunks = []
        for start in range(0, len(rows[role]), 32):
            batch = rows[role][start:start + 32]
            images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
            actual = encoder.encode_images(images)
            if start == 0:
                with torch.inference_mode():
                    pixels = pair.pixels(processor, images, 'large').cuda().half()
                    pooled = reference(pixel_values=pixels).pooler_output
                    expected = pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled, head), dim=1).cpu())
                qualified.selected.fp16.same(actual, expected)
            chunks.append(actual)
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            if (start // 32 + 1) % 64 == 0 or start + 32 >= len(rows[role]):
                print(json.dumps({role + '_public_B32_images': min(start + 32, len(rows[role]))}), flush=True)
        arrays[role] = PackedInt8Embeddings(torch.cat([v.codes for v in chunks]), torch.cat([v.inverse_norms for v in chunks]))
    q, g = arrays['query'], arrays['gallery']
    with qualified.selected.serving.CutilePackedInt8Gallery.open_packed(qualified.selected.serving.LIBRARY, g) as native:
        for start in range(0, len(q.codes), 32):
            block = PackedInt8Embeddings(q.codes[start:start + 32].contiguous(), q.inverse_norms[start:start + 32].contiguous())
            scores = (block.codes.float().cuda() @ g.codes.float().cuda().T) * block.inverse_norms.float().cuda()[:, None] * g.inverse_norms.float().cuda()[None, :]
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
            qualified.selected.pilot.equal(native.search_packed(block), (order.cpu().numpy(), scores.gather(1, order).cpu().numpy()))
    quality, _, _ = qualified.selected.metrics(official, q, g, qlabels, glabels, torch.device('cuda'))
    preserved()
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    args.output.mkdir(exist_ok=False)
    facts = {}
    for role, packed in arrays.items():
        for field, tensor in (('codes', packed.codes), ('inverse', packed.inverse_norms)):
            path = args.output / (role + '.' + field + '.npy')
            np.save(path, tensor.numpy(), allow_pickle=False)
            facts[role + '_' + field + '_sha256'] = pair.sha(path)
    pair.smoke.save(args.output / 'receipt.json', {**facts, 'variant': args.variant, 'code': code, 'execution_sha256': args.execution_sha256,
        'checkpoint_sha256': value['checkpoint_sha256'], 'cpu_authority_sha256': args.cpu_sha256, 'training_receipt_sha256': TRAIN,
        'query_rows': rows['query'], 'gallery_rows': rows['gallery'], 'query_images': 6354, 'gallery_images': 6245, 'quality': quality,
        'native_all_query_top10_ordinal_score_bits_exact': True, 'independent_original_processor_first32_each_role_packed_exact': True,
        'source_state_rng_environment_code_library_preserved': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'optimizer_updates': 0, 'official_read': False, 'claim_eligible': False, 'batch': 32, 'precision': 'fp16_native', 'public_latency_measured': False})
    print('PASS actual public TRAIN B32/native all-query bits; no official or latency claim')


def survives(comparison):
    return bool(comparison['per_query_r1']['product_lower95'] > 0 and comparison['per_query_ap']['mean_difference'] >= 0)


def decision(root, args):
    assert not torch.cuda.is_available() and not args.output.exists()
    result, _, exported = loaded_authority(root, args.execution_sha256, 'average')
    _, _, _, _, _, terminal, frozen, _, code, endpoint = result
    receipts, audits, hashes = [], [], []
    for variant, receipt_sha, audit_sha, weights in (
        ('endpoint', args.endpoint_sha256, args.endpoint_audit_sha256, endpoint),
        ('average', args.average_sha256, args.average_audit_sha256, exported),
    ):
        run = Path('/home/riomus/runs/sfora-native-trajectory-average-' + variant + '-held-v1')
        assert receipt_sha and audit_sha and pair.sha(run / 'receipt.json') == receipt_sha and pair.sha(run / 'cpu-audit.json') == audit_sha
        value, audit = (json.loads((run / n).read_text()) for n in ('receipt.json', 'cpu-audit.json'))
        assert value['code'] == code and value['execution_sha256'] == args.execution_sha256 and value['variant'] == variant
        assert value['checkpoint_sha256'] == weights['checkpoint_sha256'] and value['training_receipt_sha256'] == TRAIN
        assert audit['pass'] and audit['receipt_sha256'] == receipt_sha and audit['variant'] == variant
        assert value['query_images'] == 6354 and value['gallery_images'] == 6245 and value['batch'] == 32 and value['precision'] == 'fp16_native'
        assert value['native_all_query_top10_ordinal_score_bits_exact'] and value['independent_original_processor_first32_each_role_packed_exact'] and value['source_state_rng_environment_code_library_preserved']
        assert not value['official_read'] and not value['claim_eligible'] and not value['public_latency_measured'] and value['optimizer_updates'] == 0
        assert value['peak_cuda_allocated_bytes'] < 10_000_000_000
        for role in ('query', 'gallery'):
            assert value[role + '_rows'] == [frozen['held_manifest'][i] for i in frozen[role]]
            for field in ('codes', 'inverse'):
                assert pair.sha(run / (role + '.' + field + '.npy')) == value[role + '_' + field + '_sha256']
        for phase in ('cpu', 'public', 'audit'):
            log = (root / (variant + '-' + phase + '-v1.log')).read_text()
            assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(value['quality'][k]))) < 1e-6 for k, v in audit['quality'].items())
        receipts.append(value); audits.append(audit); hashes.append({'variant': variant, 'receipt_sha256': receipt_sha, 'audit_sha256': audit_sha})
    log = (root / 'average-export-v1.log').read_text()
    assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
    labels = np.asarray([r['product'] for r in receipts[0]['query_rows']])
    comparison = {}
    for metric in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(audits[1]['quality'][metric]) - np.asarray(audits[0]['quality'][metric])
        assert delta.shape == (6354,) and np.isfinite(delta).all()
        comparison[metric] = {'mean_difference': float(delta.mean())}
        for kind, groups in (('product', labels), ('query', np.arange(len(delta)))):
            comparison[metric][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            comparison[metric][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    outcome = survives(comparison)
    assert all(pair.sha(root / n) == h for n, h in code.items())
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'decision.json', {'pass': True, 'decision': 'GO' if outcome else 'KILL', 'procedure_only': True,
        'integrity_resource_pass': True, 'execution_sha256': args.execution_sha256, 'code': code,
        'dataset_split': 'previously observed DeepFashion In-Shop TRAIN-held', 'query_images': 6354, 'gallery_images': 6245,
        'fit_images': 13283, 'fit_identities': 2004, 'held_identities': 1993, 'matched_receipts': hashes,
        'endpoint_quality': audits[0]['quality'], 'average_quality': audits[1]['quality'], 'comparison': comparison,
        'archived_training_wall_seconds': terminal['aggregate_training_wall_seconds'], 'averaging_wall_seconds': exported['export_wall_seconds'],
        'additional_training_updates': 0, 'public_latency_measured': False, 'official_read': False, 'claim_eligible': False,
        'bootstrap_draws': 5000, 'bootstrap_seed': 179019, 'uncertainty_scope': 'fixed single archived trajectory; not training seed population',
        'full_production_quality_speed_goal_achieved': False})
    print(('GO' if outcome else 'KILL') + ' fixed averaging TRAIN procedure; overall production goal remains open')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--phase', choices=('export', 'cpu', 'public', 'audit', 'decision'), required=True)
    p.add_argument('--variant', choices=('endpoint', 'average'), default='average')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-proof', type=Path)
    p.add_argument('--cpu-sha256')
    p.add_argument('--receipt-sha256')
    p.add_argument('--endpoint-sha256')
    p.add_argument('--endpoint-audit-sha256')
    p.add_argument('--average-sha256')
    p.add_argument('--average-audit-sha256')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    if args.phase == 'export':
        export(root, args.execution_sha256, args.output)
    elif args.phase == 'cpu':
        cpu(root, args)
    elif args.phase == 'decision':
        decision(root, args)
    else:
        wires(root, args, args.phase == 'audit')


if __name__ == '__main__':
    main()
