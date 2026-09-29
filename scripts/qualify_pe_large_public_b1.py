#!/usr/bin/env python3
"""Complete actual B1 public quality and scoped latency-stage measurements."""
import argparse
import json
import inspect
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import qualify_pe_large_public_serving as serving
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, trained, teacher = serving.pair, serving.trained, serving.teacher
SERVING_CODE_SHA = '409430306640bd814485dc75acc82037abb4c329d6485b4213ebe7681b75b1be'
PUBLIC = Path('/home/riomus/runs/sfora-large-trained-serving-v2/serving-qualification.json')
PUBLIC_SHA = 'f7ead4f358b3ef71fb5867fdb555e4b5b53e3db4d4db082b7a58bd74a771a289'


def startup(root, execution_sha):
    manifest = root / 'public-b1-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'B1 execution code differs'
    control, frozen, cpu, receipt, old = serving.startup(root, SERVING_CODE_SHA)
    assert all(code[n] == h for n, h in old.items()) and pair.sha(PUBLIC) == PUBLIC_SHA
    public = json.loads(PUBLIC.read_text())
    assert public['complete_public_B32_held_packed_exact'] and public['source_head_rng_code_native_file_unchanged']
    assert public['native_top10_ordinal_score_bits_exact_queries'] == 6354
    return control, frozen, cpu, receipt, code


def score(packed, frozen, device):
    # Consume actual wire descriptors. Requantizing restored unit floats changes fixed127 codes.
    assert packed.codes.dtype == torch.int8 and packed.codes.shape == (12599, 128)
    assert packed.inverse_norms.dtype == torch.float16 and packed.inverse_norms.shape == (12599,)
    assert packed.codes.min() >= -127 and torch.isfinite(packed.inverse_norms).all()
    norms = torch.linalg.vector_norm(packed.codes.float(), dim=1)
    assert (norms > 0).all() and torch.equal(norms.reciprocal().half(), packed.inverse_norms)
    code, inverse = packed.codes.float().to(device), packed.inverse_norms.float().to(device)
    labels = tuple(r['product'] for r in frozen['held_manifest'])
    classes = {name: index for index, name in enumerate(sorted(set(labels)))}
    ids = torch.tensor([classes[label] for label in labels], device=device)
    query, gallery = frozen['query'], frozen['gallery']
    relevant = torch.bincount(ids[gallery])[ids[query]]
    assert relevant.min() >= 1
    ranks = torch.arange(1, int(relevant.max()) + 1, device=device)
    hits, aps = [], []
    for start in range(0, len(query), 128):
        rows = query[start:start + 128]
        scores = (code[rows] @ code[gallery].T) * inverse[rows, None] * inverse[None, gallery]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :len(ranks)]
        matches = ids[gallery][order] == ids[rows, None]
        counts = relevant[start:start + len(rows)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= counts[:, None])).sum(dim=1) / counts
        hits.extend(map(int, matches[:, 0].cpu().tolist()))
        aps.extend(map(float, ap.cpu().tolist()))
    quality = {'recall_at_1': float(np.mean(hits)), 'map_at_r': float(np.mean(aps)), 'per_query_r1': hits, 'per_query_ap': aps}
    dense = json.loads((trained.base.INIT / 'receipt.json').read_text())['arms']['pe']['quality']
    products, intervals = np.asarray(labels)[query], {}
    for name in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(quality[name]) - np.asarray(dense[name])
        intervals[name] = {'mean_delta': float(np.mean(delta))}
        for kind, groups in (('product', products), ('query', np.arange(len(delta)))):
            intervals[name][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            intervals[name][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    advance = bool(quality['recall_at_1'] >= .951720176 and quality['map_at_r'] >= .776237120 and all(v['product_lower95'] > 0 for v in intervals.values()))
    return quality, intervals, advance


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-cpu', action='store_true')
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--audit-cpu', action='store_true')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, cpu, prior, code = startup(root, args.execution_sha256)
    baseline = pack_int8_unit_embeddings(torch.from_numpy(np.load(serving.HELD / 'large.held.npy', allow_pickle=False)))
    if args.qualify_cpu:
        assert not torch.cuda.is_available() and not args.output.exists() and not args.audit_cpu
        quality, intervals, advance = score(baseline, frozen, torch.device('cpu'))
        assert advance and all(np.max(np.abs(np.asarray(v) - np.asarray(prior['quality'][k]))) < 1e-6 for k, v in quality.items())
        assert all(abs(v - prior['paired_dense_pe_intervals'][k][n]) < 1e-6 for k, row in intervals.items() for n, v in row.items())
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'B1 execution code differs'
            else:
                raise AssertionError('changed B1 driver accepted')
        assert all(pair.sha(root / n) == h for n, h in code.items())
        pair.smoke.save(args.output, {'pass': True, 'execution_sha256': args.execution_sha256, 'code': code, 'actual_wire_baseline_per_query_intervals_decision_exact': True, 'changed_driver_rejected': True, 'public_authority_sha256': PUBLIC_SHA, 'quality_read': 'saved TRAIN-held CPU replay only', 'optimizer_updates': 0})
        print('PASS actual-wire score/per-query/CI/decision CPU baseline and changed-driver rejection')
        return
    if args.audit_cpu:
        assert not torch.cuda.is_available() and args.receipt_sha256
        path = args.output / 'receipt.json'
        assert pair.sha(path) == args.receipt_sha256 and not (args.output / 'cpu-audit.json').exists()
        receipt = json.loads(path.read_text())
        assert receipt['code'] == code and receipt['execution_sha256'] == args.execution_sha256
        assert receipt['query_images'] == 6354 and receipt['gallery_images'] == 6245 and receipt['native_per_query_r1_exact']
        assert receipt['actual_public_B1_all_queries'] and receipt['source_head_rng_code_library_preserved']
        assert receipt['public_B32_authority_sha256'] == PUBLIC_SHA and receipt['checkpoint_sha256'] == teacher.TEACHER_SHA
        assert receipt['optimizer_updates'] == 0 and not receipt['official_read']
        packed = PackedInt8Embeddings(baseline.codes.clone(), baseline.inverse_norms.clone())
        for name, key, target in (('query.codes.npy', 'query_codes_sha256', packed.codes), ('query.inverse.npy', 'query_inverse_sha256', packed.inverse_norms)):
            assert pair.sha(args.output / name) == receipt[key]
            target[frozen['query']] = torch.from_numpy(np.load(args.output / name, allow_pickle=False))
        quality, intervals, advance = score(packed, frozen, torch.device('cpu'))
        assert all(np.max(np.abs(np.asarray(v) - np.asarray(receipt['quality'][k]))) < 1e-6 for k, v in quality.items())
        assert all(abs(v - receipt['paired_dense_pe_intervals'][k][n]) < 1e-6 for k, row in intervals.items() for n, v in row.items())
        assert advance == receipt['advance'] and all(pair.sha(root / n) == h for n, h in code.items())
        pair.smoke.save(args.output / 'cpu-audit.json', {'pass': True, 'advance': advance, 'receipt_sha256': args.receipt_sha256, 'quality': quality, 'paired_dense_pe_intervals': intervals, 'claim_eligible': False})
        print('PASS complete actual B1 wire CPU quality/interval/decision replay')
        return
    assert torch.cuda.is_available() and not args.output.exists() and args.cpu_sha256
    proof = root / 'cpu-proof.json'
    assert pair.sha(proof) == args.cpu_sha256
    qualified = json.loads(proof.read_text())
    assert qualified['code'] == code and qualified['pass'] and qualified['actual_wire_baseline_per_query_intervals_decision_exact']
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
    assert Path(inspect.getfile(serving.CutilePackedInt8Gallery)).resolve() == root / 'src/sfora/cutile_int8.py'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=teacher.TEACHER, expected_checkpoint_sha256=teacher.TEACHER_SHA, model_file_sha256=pair.smoke.MODEL_HASHES, precision='fp32_autocast', device=torch.device('cuda'))
    assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == cpu['teacher_whole_sha256'] and pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
    environment = json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor)))
    assert environment == cpu['environment']
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    packed = PackedInt8Embeddings(baseline.codes.clone(), baseline.inverse_norms.clone())
    gallery = PackedInt8Embeddings(baseline.codes[frozen['gallery']].contiguous(), baseline.inverse_norms[frozen['gallery']].contiguous())
    captured, native_hits = [], []
    labels = tuple(r['product'] for r in frozen['held_manifest'])

    def decode(path):
        with Image.open(path) as image:
            return image.convert('RGB')

    with Siglip2CompactIndex(encoder, serving.CutilePackedInt8Gallery.open_packed(serving.LIBRARY, gallery)) as index:
        actual_search = index.gallery.search_packed

        def capture(query):
            assert query.codes.shape == (1, 128)
            captured.append(query)
            return actual_search(query)

        with patch.object(index.gallery, 'search_packed', capture):
            for i, row in enumerate(frozen['query']):
                item = frozen['held_manifest'][row]
                path = control.dataset_root / item['relative_path']
                assert pair.sha(path) == item['image_sha256']
                ordinals, _ = index.search_images([decode(path)])
                native_hits.append(int(labels[row] == labels[frozen['gallery'][int(ordinals[0, 0])]]))
                assert torch.cuda.max_memory_allocated() < 10_000_000_000
                if (i + 1) % 128 == 0 or i + 1 == 6354:
                    print(json.dumps({'public_B1_queries_verified': i + 1}), flush=True)
        query_codes = torch.cat([p.codes for p in captured])
        query_inverse = torch.cat([p.inverse_norms for p in captured])
        assert query_codes.shape == (6354, 128) and query_inverse.shape == (6354,)
        packed.codes[frozen['query']] = query_codes
        packed.inverse_norms[frozen['query']] = query_inverse
        quality, intervals, advance = score(packed, frozen, torch.device('cuda'))
        assert native_hits == quality['per_query_r1']
        profile = {}
        for size in (1, 32):
            paths = [control.dataset_root / frozen['held_manifest'][i]['relative_path'] for i in frozen['query'][:size]]
            for _ in range(5):
                index.search_images([decode(p) for p in paths])
            events, samples = [], []
            def begin(module, inputs):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                events.append((start, stop))
            def end(module, inputs, result):
                events[-1][1].record()
            hooks = [encoder.vision.register_forward_pre_hook(begin), encoder.vision.register_forward_hook(end)]
            try:
                for _ in range(10):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    images = [decode(p) for p in paths]
                    decoded = time.perf_counter()
                    index.search_images(images)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - tick
                    samples.append({'decode_ms': 1000 * (decoded - tick), 'vision_cuda_ms': events[-1][0].elapsed_time(events[-1][1]), 'full_wall_ms': 1000 * wall})
            finally:
                for hook in hooks:
                    hook.remove()
            profile[str(size)] = samples
    assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == cpu['teacher_whole_sha256'] and pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
    assert all(p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor))) == environment
    assert teacher.qualified.numerical_flags() == flags and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA and pair.sha(serving.LIBRARY) == serving.LIBRARY_SHA
    assert all(pair.sha(root / n) == h for n, h in code.items()) and torch.cuda.max_memory_allocated() < 10_000_000_000
    args.output.mkdir(exist_ok=False)
    np.save(args.output / 'query.codes.npy', query_codes.numpy(), allow_pickle=False)
    np.save(args.output / 'query.inverse.npy', query_inverse.numpy(), allow_pickle=False)
    pair.smoke.save(args.output / 'receipt.json', {'advance': advance, 'code': code, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': args.cpu_sha256, 'public_B32_authority_sha256': PUBLIC_SHA, 'checkpoint_sha256': teacher.TEACHER_SHA, 'query_images': 6354, 'gallery_images': 6245, 'query_codes_sha256': pair.sha(args.output / 'query.codes.npy'), 'query_inverse_sha256': pair.sha(args.output / 'query.inverse.npy'), 'quality': quality, 'paired_dense_pe_intervals': intervals, 'native_per_query_r1_exact': True, 'actual_public_B1_all_queries': True, 'source_head_rng_code_library_preserved': True, 'profile_hook_samples': profile, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0, 'official_read': False, 'claim_eligible': False, 'paired_speed_win': False, 'p99_certified': False})
    print('GO complete public B1 TRAIN quality' if advance else 'KILL fixed public B1 TRAIN quality')


if __name__ == '__main__':
    main()
