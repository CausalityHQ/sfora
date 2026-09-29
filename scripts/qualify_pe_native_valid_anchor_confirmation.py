#!/usr/bin/env python3
"""Fixed first-seed paired confirmation on the previously observed official protocol."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
import qualify_pe_native_valid_anchor_held as held
import qualify_pe_native_valid_anchor_serving as serving
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, trained = held.driver.pair, held.trained
selected = held.cpu.qualified.confirmation.selected
SERVING_CODE = 'b9e0a65a302ff8e71f93beb330e26eb6fd12ad47f0c35981ea02665e93e00413'
PROTOCOL = held.cpu.qualified.confirmation.PROTOCOL
PROTOCOL_SHA = held.cpu.qualified.confirmation.PROTOCOL_SHA


def guard(root, code):
    assert all(pair.sha(root / n) == h for n, h in code.items()), "confirmation source differs"


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--arm', choices=('control', 'treatment'), required=True)
    p.add_argument('--serving-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualify-cpu', action='store_true')
    p.add_argument('--cpu-sha256')
    p.add_argument('--role', choices=('gallery', 'query'))
    p.add_argument('--gallery', type=Path)
    p.add_argument('--gallery-sha256')
    p.add_argument('--audit-cpu', action='store_true')
    p.add_argument('--receipt-sha256')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / 'native-valid-anchor-confirmation-execution.json'
    assert pair.sha(manifest) == args.execution_sha256
    code = json.loads(manifest.read_text())
    old = json.loads((root / 'native-valid-anchor-serving-execution.json').read_text())
    assert pair.sha(root / 'native-valid-anchor-serving-execution.json') == SERVING_CODE
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    guard(root, code)
    assert pair.sha(serving.DECISION) == serving.DECISION_SHA
    decision = json.loads(serving.DECISION.read_text())
    assert decision['decision'] == 'GO'
    entry = next(v for v in decision['inputs'] if v['seed'] == 179041 and v['arm'] == args.arm)
    source_path = Path('/home/riomus/runs/sfora-native-valid-anchor-held-v3') / f'source-179041-{args.arm}-proof.json'
    assert pair.sha(source_path) == entry['source_sha256']
    source = json.loads(source_path.read_text())
    helpers = selected.helpers
    with patch.object(selected, 'helpers', lambda r, _: helpers(r, code)):
        control, frozen, prior, run, terminal, _ = held.authority(root, serving.HELD_CODE, 179041, args.arm, source['native_training_receipt_sha256'])
    public_path = Path('/home/riomus/runs/sfora-native-valid-anchor-serving-v2') / f'{args.arm}-serving.json'
    assert pair.sha(public_path) == args.serving_sha256
    public = json.loads(public_path.read_text())
    assert public['code'] == old and public['checkpoint_sha256'] == terminal['checkpoint_sha256']
    assert public['complete_public_B32_held_packed_exact'] and public['source_head_rng_code_native_file_unchanged']
    assert public['native_top10_ordinal_score_bits_exact_queries'] == 6354 and public['public_B1_direct_native_packed_top10_exact_queries'] == 32
    assert public['precision'] == 'fp32_autocast'
    assert pair.sha(PROTOCOL) == PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['pass'] and protocol['train_query_gallery_ids_disjoint'] and protocol['prior_official_benchmark_exposure']
    assert pair.sha(control.dataset_root / 'Eval/list_eval_partition.txt') == protocol['partition_sha256'] == selected.PARTITION_SHA
    official = helpers(root, code)
    records = official.parse_inshop_partition(control.dataset_root)
    assert {r.label for r in records if r.split == 'train'} == {r['product'] for r in frozen['fit_manifest'] + frozen['held_manifest']}
    for role, count in (('query', 14218), ('gallery', 12612)):
        rows = [r for r in records if r.split == role]
        assert len(rows) == count
        assert [(str(r.image_path.relative_to(control.dataset_root)), r.label) for r in rows] == [(r['relative_path'], r['product']) for r in protocol['protocol'][role]]
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    binding = {'arm': args.arm, 'seed': 179041, 'checkpoint_sha256': terminal['checkpoint_sha256'], 'training_receipt_sha256': source['native_training_receipt_sha256'], 'source_cpu_sha256': entry['source_sha256'], 'serving_sha256': args.serving_sha256, 'protocol_sha256': PROTOCOL_SHA, 'code': code, 'execution_sha256': args.execution_sha256, 'precision': 'fp32_autocast'}
    if args.qualify_cpu:
        assert not torch.cuda.is_available() and not args.output.exists() and not args.role and not args.audit_cpu
        # Reuse the qualified actual wire scorer against this updated TRAIN source.
        held_dir = Path(f'/home/riomus/runs/sfora-native-valid-anchor-held-179041-{args.arm}-v1')
        assert pair.sha(held_dir / 'large.held.npy') == json.loads((held_dir / 'receipt.json').read_text())['held_sha256']
        packed = pack_int8_unit_embeddings(torch.from_numpy(np.load(held_dir / 'large.held.npy', allow_pickle=False)))
        labels = tuple(r['product'] for r in frozen['held_manifest'])
        q, g = (PackedInt8Embeddings(packed.codes[frozen[k]], packed.inverse_norms[frozen[k]]) for k in ('query', 'gallery'))
        quality = official.score_asymmetric(q.codes.float(), g.codes.float(), tuple(labels[i] for i in frozen['query']), tuple(labels[i] for i in frozen['gallery']), query_inverse=q.inverse_norms.float(), gallery_inverse=g.inverse_norms.float())
        assert pair.sha(held_dir / 'cpu-audit.json') == entry['audit_sha256']
        audited = json.loads((held_dir / 'cpu-audit.json').read_text())['quality']
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(audited[k]))) < 1e-6 for k, v in quality.items())
        original_sha = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original_sha(p)):
            try:
                guard(root, code)
            except AssertionError as error:
                assert str(error) == 'confirmation source differs'
            else:
                raise AssertionError('changed confirmation driver accepted')
        guard(root, code)
        pair.smoke.save(args.output, {**binding, 'pass': True, 'actual_updated_TRAIN_wire_scorer_exact': True, 'changed_driver_rejected': True, 'official_quality_read': False, 'optimizer_updates': 0})
        print('PASS fixed source/serving/protocol and actual updated TRAIN wire scorer; no official pixels decoded')
        return
    proof = root / f'{args.arm}-confirmation-cpu.json'
    assert args.cpu_sha256 and pair.sha(proof) == args.cpu_sha256
    qualified = json.loads(proof.read_text())
    assert qualified['pass'] and all(qualified[k] == v for k, v in binding.items())
    assert args.role and args.gallery if args.audit_cpu else args.role
    labels = {r: tuple(v['product'] for v in protocol['protocol'][r]) for r in ('query', 'gallery')}
    if args.audit_cpu:
        assert not torch.cuda.is_available() and args.role == 'query' and args.receipt_sha256
        assert pair.sha(args.output / 'receipt.json') == args.receipt_sha256
        measured = json.loads((args.output / 'receipt.json').read_text())
        assert all(measured[k] == v for k, v in binding.items()) and measured['native_all_query_top10_ordinal_score_bits_exact']
        assert measured['source_state_rng_environment_code_library_preserved'] and not (args.output / 'cpu-audit.json').exists()
        assert pair.sha(args.gallery / 'receipt.json') == args.gallery_sha256 == measured['gallery_receipt_sha256']
        gallery = json.loads((args.gallery / 'receipt.json').read_text())
        assert all(gallery[k] == v for k, v in binding.items()) and gallery['source_state_rng_environment_code_library_preserved']
        q = selected.fp16.load_packed(args.output, measured, 'query')
        g = selected.fp16.load_packed(args.gallery, gallery, 'gallery')
        quality, intervals, advance = selected.metrics(official, q, g, labels['query'], labels['gallery'], torch.device('cpu'))
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(measured['quality'][k]))) < 1e-6 for k, v in quality.items())
        assert all(abs(v - measured['quality_intervals'][k][n]) < 1e-6 for k, row in intervals.items() for n, v in row.items())
        assert advance == measured['advance_minimum_dated_reference_screen']
        assert all(pair.sha(root / n) == h for n, h in code.items())
        pair.smoke.save(args.output / 'cpu-audit.json', {**binding, 'pass': True, 'receipt_sha256': args.receipt_sha256, 'quality': quality, 'quality_intervals': intervals, 'advance_minimum_dated_reference_screen': advance, 'claim_eligible': False, 'prior_official_benchmark_exposure': True})
        return
    assert torch.cuda.is_available() and not args.output.exists()
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = trained.teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=run / 'native.pt', expected_checkpoint_sha256=terminal['checkpoint_sha256'], model_file_sha256=pair.smoke.MODEL_HASHES, precision='fp32_autocast', device=torch.device('cuda'))

    def preserved():
        assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == source['teacher_whole_sha256']
        assert pair.smoke.digest(encoder.head.state_dict()) == source['teacher_head_sha256']
        assert json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor))) == source['environment']
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
        assert all(p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())

    preserved()
    native, processor = pair.smoke.load_arm(control, 'large')
    disk = torch.load(run / 'native.pt', map_location='cpu', weights_only=True, mmap=True)
    native.load_state_dict(disk['vision'], strict=True)
    head = nn.Linear(1024, 128)
    head.load_state_dict(disk['head'], strict=True)
    assert pair.smoke.digest(trained.base.whole_state(native)) == source['teacher_whole_sha256'] and pair.smoke.digest(head.state_dict()) == source['teacher_head_sha256']
    native.requires_grad_(False).eval().cuda(); head.requires_grad_(False).eval().cuda()
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    chunks = []
    rows = protocol['protocol'][args.role]
    for start in range(0, len(rows), 32):
        images = []
        for row in rows[start:start + 32]:
            path = control.dataset_root / row['relative_path']
            assert pair.sha(path) == row['image_sha256']
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        actual = encoder.encode_images(images)
        if start == 0:
            with torch.inference_mode():
                pixels = pair.pixels(processor, images, 'large').cuda()
                pooled = trained.teacher.qualified.fp16(native, pixels)
                expected = pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled, head), dim=1).cpu())
            selected.fp16.same(actual, expected)
        chunks.append(actual)
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        if (start // 32 + 1) % 64 == 0 or start + 32 >= len(rows):
            print(args.role, min(start + 32, len(rows)), flush=True)
    packed = PackedInt8Embeddings(torch.cat([v.codes for v in chunks]), torch.cat([v.inverse_norms for v in chunks]))
    result = {}
    if args.role == 'query':
        assert args.gallery and pair.sha(args.gallery / 'receipt.json') == args.gallery_sha256
        gallery = json.loads((args.gallery / 'receipt.json').read_text())
        assert all(gallery[k] == v for k, v in binding.items()) and gallery['source_state_rng_environment_code_library_preserved']
        g = selected.fp16.load_packed(args.gallery, gallery, 'gallery')
        with serving.public.CutilePackedInt8Gallery.open_packed(serving.public.LIBRARY, g) as index:
            for start in range(0, len(packed.codes), 32):
                block = PackedInt8Embeddings(packed.codes[start:start + 32].contiguous(), packed.inverse_norms[start:start + 32].contiguous())
                scores = (block.codes.float().cuda() @ g.codes.float().cuda().T) * block.inverse_norms.float().cuda()[:, None] * g.inverse_norms.float().cuda()[None, :]
                order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
                selected.pilot.equal(index.search_packed(block), (order.cpu().numpy(), scores.gather(1, order).cpu().numpy()))
        quality, intervals, advance = selected.metrics(official, packed, g, labels['query'], labels['gallery'], torch.device('cuda'))
        result = {'quality': quality, 'quality_intervals': intervals, 'advance_minimum_dated_reference_screen': advance, 'gallery_receipt_sha256': args.gallery_sha256, 'native_all_query_top10_ordinal_score_bits_exact': True}
    preserved()
    assert pair.smoke.digest(trained.base.whole_state(native)) == source['teacher_whole_sha256'] and pair.smoke.digest(head.state_dict()) == source['teacher_head_sha256']
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert trained.teacher.qualified.numerical_flags() == flags and pair.sha(run / 'native.pt') == terminal['checkpoint_sha256']
    assert pair.sha(PROTOCOL) == PROTOCOL_SHA and pair.sha(serving.public.LIBRARY) == serving.public.LIBRARY_SHA
    assert all(pair.sha(root / n) == h for n, h in code.items())
    args.output.mkdir(exist_ok=False)
    for field, values in (('codes', packed.codes), ('inverse', packed.inverse_norms)):
        path = args.output / (args.role + '.' + field + '.npy')
        np.save(path, values.numpy(), allow_pickle=False)
        result[args.role + '_' + field + '_sha256'] = pair.sha(path)
    pair.smoke.save(args.output / 'receipt.json', {**binding, **result, 'role': args.role, 'images': len(rows), 'source_state_rng_environment_code_library_preserved': True, 'independent_native_original_preprocessing_first32_packed_exact': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'batch': 32, 'optimizer_updates': 0, 'prior_official_benchmark_exposure': True, 'claim_eligible': False})


if __name__ == '__main__':
    main()
