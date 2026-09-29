#!/usr/bin/env python3
"""Pinned trained-source FP16 cast, complete public B32/B1 quality and CPU replay."""
import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import qualify_pe_large_public_b1 as b1
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, trained, teacher, serving = b1.pair, b1.trained, b1.teacher, b1.serving
B1_CODE_SHA = '646f7a587aa06a7cfbb3af9f002d6322085cc09a1482115c731731fdc75571ef'
B1 = Path('/home/riomus/runs/sfora-large-public-b1-v1')
B1_SHA = 'f71be3e7af9e5878e25040ac7b9e024f7dfed4f99651e8b3fc8e3cfbaec0517c'
B1_AUDIT_SHA = '1d80faee99fab8cf69a33b48949b4ffdcea429cdf77cf676dbbb6178cfed5705'


def startup(root, execution_sha):
    path = root / 'public-fp16-execution.json'
    assert pair.sha(path) == execution_sha
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'FP16 execution code differs'
    control, frozen, cpu, prior, old = b1.startup(root, B1_CODE_SHA)
    assert all(code[n] == h for n, h in old.items())
    assert pair.sha(B1 / 'receipt.json') == B1_SHA and pair.sha(B1 / 'cpu-audit.json') == B1_AUDIT_SHA
    audit = json.loads((B1 / 'cpu-audit.json').read_text())
    assert audit['pass'] and audit['advance'] and audit['receipt_sha256'] == B1_SHA
    return control, frozen, cpu, prior, code


def same(a, b):
    assert torch.equal(a.codes, b.codes) and torch.equal(a.inverse_norms, b.inverse_norms), 'public/native FP16 wire differs'


def load_packed(path, receipt, name):
    arrays = []
    for field in ('codes', 'inverse'):
        file = path / (name + '.' + field + '.npy')
        assert pair.sha(file) == receipt[name + '_' + field + '_sha256']
        arrays.append(torch.from_numpy(np.load(file, allow_pickle=False)))
    return PackedInt8Embeddings(*arrays)


def exact_replay(quality, intervals, advance, receipt):
    assert all(np.max(np.abs(np.asarray(v) - np.asarray(receipt['quality'][k]))) < 1e-6 for k, v in quality.items())
    assert all(abs(v - receipt['paired_dense_pe_intervals'][k][n]) < 1e-6 for k, row in intervals.items() for n, v in row.items())
    assert advance == receipt['advance']


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-cpu', action='store_true')
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--b32', type=Path)
    parser.add_argument('--b32-sha256')
    parser.add_argument('--b32-audit-sha256')
    parser.add_argument('--audit-cpu', action='store_true')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, cpu, prior, code = startup(root, args.execution_sha256)
    if args.qualify_cpu:
        assert not torch.cuda.is_available() and not args.output.exists() and not args.audit_cpu and not args.b32
        loaded, live, heads, processor = teacher.teacher_pair(control, torch.device('cpu'))
        assert all(pair.smoke.digest(trained.base.whole_state(m)) == cpu['teacher_whole_sha256'] for m in (loaded, live))
        assert all(pair.smoke.digest(h.state_dict()) == cpu['teacher_head_sha256'] for h in heads)
        assert json.loads(json.dumps(trained.native.environment(live, processor))) == cpu['environment']
        saved = torch.load(teacher.TEACHER, map_location='cpu', weights_only=True, mmap=True)
        for model in (loaded, live):
            model.half()
            assert len(model.state_dict()) == len(saved['vision']) == 400
            assert all(torch.equal(v, saved['vision'][n].half() if v.is_floating_point() else saved['vision'][n]) for n, v in model.state_dict().items())
            assert all(p.dtype == torch.float16 and p.grad is None for p in model.parameters())
        cast = pair.smoke.digest(trained.base.whole_state(live))
        assert cast == pair.smoke.digest(trained.base.whole_state(loaded)) and cast != cpu['teacher_whole_sha256']
        parameter = live.encoder.layers[12].mlp.fc2.bias
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter[0].add_(.125)
            assert pair.smoke.digest(trained.base.whole_state(live)) != cast
            parameter.copy_(before)
        assert pair.smoke.digest(trained.base.whole_state(live)) == cast
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'FP16 execution code differs'
            else:
                raise AssertionError('altered FP16 driver accepted')
        assert all(pair.sha(root / n) == h for n, h in code.items()) and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA
        pair.smoke.save(args.output, {'pass': True, 'code': code, 'execution_sha256': args.execution_sha256, 'checkpoint_sha256': teacher.TEACHER_SHA, 'fp32_whole_sha256': cpu['teacher_whole_sha256'], 'fp16_whole_sha256': cast, 'head_sha256': cpu['teacher_head_sha256'], 'strict400_loaded_pair_exact_cast': True, 'cast_mutation_rejected': True, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False})
        print('PASS actual trained native CPU pair exact F16 cast and mutation/driver rejection')
        return
    if args.audit_cpu:
        assert not torch.cuda.is_available() and args.receipt_sha256
        path = args.output / 'receipt.json'
        assert pair.sha(path) == args.receipt_sha256 and not (args.output / 'cpu-audit.json').exists()
        receipt = json.loads(path.read_text())
        assert receipt['code'] == code and receipt['execution_sha256'] == args.execution_sha256
        assert receipt['checkpoint_sha256'] == teacher.TEACHER_SHA and receipt['source_rng_environment_code_library_preserved']
        assert receipt['optimizer_updates'] == 0 and not receipt['official_read']
        proof = root / 'fp16-cpu-proof.json'
        assert pair.sha(proof) == receipt['cpu_authority_sha256']
        authority = json.loads(proof.read_text())
        assert authority['pass'] and authority['code'] == code and authority['fp16_whole_sha256'] == receipt['fp16_whole_sha256']
        if receipt['batch'] == 32:
            packed = load_packed(args.output, receipt, 'held')
            assert receipt['all_B32_public_native_whole_head_packed_exact'] and receipt['held_images'] == 12599
        else:
            assert args.b32 and pair.sha(args.b32 / 'receipt.json') == args.b32_sha256 == receipt['b32_receipt_sha256']
            packed = load_packed(args.b32, json.loads((args.b32 / 'receipt.json').read_text()), 'held')
            query = load_packed(args.output, receipt, 'query')
            packed.codes[frozen['query']], packed.inverse_norms[frozen['query']] = query.codes, query.inverse_norms
            assert receipt['all_B1_actual_public_native_r1_exact'] and receipt['query_images'] == 6354
        quality, intervals, advance = b1.score(packed, frozen, torch.device('cpu'))
        exact_replay(quality, intervals, advance, receipt)
        assert all(pair.sha(root / n) == h for n, h in code.items())
        pair.smoke.save(args.output / 'cpu-audit.json', {'pass': True, 'advance': advance, 'receipt_sha256': args.receipt_sha256, 'quality': quality, 'paired_dense_pe_intervals': intervals, 'claim_eligible': False})
        print('PASS complete FP16 public wire CPU per-query/interval/decision replay')
        return
    assert torch.cuda.is_available() and not args.output.exists() and args.cpu_sha256
    proof = root / 'fp16-cpu-proof.json'
    assert pair.sha(proof) == args.cpu_sha256
    authority = json.loads(proof.read_text())
    assert authority['pass'] and authority['code'] == code and authority['strict400_loaded_pair_exact_cast'] and authority['cast_mutation_rejected']
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
    assert Path(inspect.getfile(serving.CutilePackedInt8Gallery)).resolve() == root / 'src/sfora/cutile_int8.py'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=teacher.TEACHER, expected_checkpoint_sha256=teacher.TEACHER_SHA, model_file_sha256=pair.smoke.MODEL_HASHES, precision='fp16_native', device=torch.device('cuda'))
    def preserved():
        assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == authority['fp16_whole_sha256']
        assert pair.smoke.digest(encoder.head.state_dict()) == authority['head_sha256']
        assert json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor))) == cpu['environment']
        assert all(p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
    preserved()
    def images(rows):
        result = []
        for row in rows:
            item = frozen['held_manifest'][row]
            path = control.dataset_root / item['relative_path']
            assert pair.sha(path) == item['image_sha256']
            with Image.open(path) as image:
                result.append(image.convert('RGB'))
        return result
    if not args.b32:
        loaded, live, heads, processor = teacher.teacher_pair(control, torch.device('cpu'))
        assert all(pair.smoke.digest(trained.base.whole_state(m)) == cpu['teacher_whole_sha256'] for m in (loaded, live))
        loaded.half().cuda().eval()
        head = heads[0].cuda()
        assert pair.smoke.digest(trained.base.whole_state(loaded)) == authority['fp16_whole_sha256']
        fit_images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1), None)
        pixels = pair.pixels(processor, fit_images, 'large')
        assert pair.smoke.digest({'pixels': pixels}) == cpu['first_two_fit_pixels_sha256']
        with torch.inference_mode():
            low = loaded(pixel_values=pixels.cuda().half()).pooler_output
            full = live.cuda()(pixel_values=pixels.cuda()).pooler_output
            calibration = {'pooled': F.cosine_similarity(low.float(), full.float()).tolist(), 'compact': F.cosine_similarity(pair.smoke.compact_head_features(low, head), pair.smoke.compact_head_features(full, head)).tolist()}
            assert all(min(v) >= .999 for v in calibration.values())
        del live, heads, full, low, pixels, fit_images
        torch.cuda.empty_cache()
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        chunks = []
        with torch.inference_mode():
            for start in range(0, 12599, 32):
                batch = images(range(start, min(start + 32, 12599)))
                actual = encoder.encode_images(batch)
                pixels = pair.pixels(processor, batch, 'large').cuda().half()
                pooled = loaded(pixel_values=pixels).pooler_output
                reference = pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled, head), dim=1).cpu())
                same(actual, reference)
                chunks.append(actual)
                assert torch.cuda.max_memory_allocated() < 10_000_000_000
                if (start // 32 + 1) % 16 == 0 or start + 32 >= 12599:
                    print(json.dumps({'B32_public_native_images_verified': min(start + 32, 12599)}), flush=True)
        packed = PackedInt8Embeddings(torch.cat([c.codes for c in chunks]), torch.cat([c.inverse_norms for c in chunks]))
        quality, intervals, advance = b1.score(packed, frozen, torch.device('cuda'))
        assert pair.smoke.digest(trained.base.whole_state(loaded)) == authority['fp16_whole_sha256'] and pair.smoke.digest(head.state_dict()) == authority['head_sha256']
        facts = {'batch': 32, 'held_images': 12599, 'all_B32_public_native_whole_head_packed_exact': True, 'fp16_fp32_first_two_fit_cosines': calibration}
        arrays = {'held': packed}
    else:
        assert pair.sha(args.b32 / 'receipt.json') == args.b32_sha256 and pair.sha(args.b32 / 'cpu-audit.json') == args.b32_audit_sha256
        receipt = json.loads((args.b32 / 'receipt.json').read_text())
        audit = json.loads((args.b32 / 'cpu-audit.json').read_text())
        assert receipt['advance'] and receipt['code'] == code and audit['pass'] and audit['advance'] and audit['receipt_sha256'] == args.b32_sha256
        packed = load_packed(args.b32, receipt, 'held')
        gallery = PackedInt8Embeddings(packed.codes[frozen['gallery']].contiguous(), packed.inverse_norms[frozen['gallery']].contiguous())
        labels = [r['product'] for r in frozen['held_manifest']]
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        with torch.inference_mode():
            for row in frozen['query'][:32]:
                batch = images([row])
                actual = encoder.encode_images(batch)
                pixels = pair.pixels(encoder.processor, batch, 'large').cuda().half()
                pooled = encoder.vision(pixel_values=pixels).pooler_output
                reference = pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled, encoder.head), dim=1).cpu())
                same(actual, reference)
        captured, hits = [], []
        with Siglip2CompactIndex(encoder, serving.CutilePackedInt8Gallery.open_packed(serving.LIBRARY, gallery)) as index:
            actual_search = index.gallery.search_packed
            def capture(query):
                assert query.codes.shape == (1, 128)
                captured.append(query)
                return actual_search(query)
            with patch.object(index.gallery, 'search_packed', capture):
                for i, row in enumerate(frozen['query']):
                    ordinals, _ = index.search_images(images([row]))
                    hits.append(int(labels[row] == labels[frozen['gallery'][int(ordinals[0, 0])]]))
                    assert torch.cuda.max_memory_allocated() < 10_000_000_000
                    if (i + 1) % 128 == 0 or i + 1 == 6354:
                        print(json.dumps({'B1_public_native_queries_verified': i + 1}), flush=True)
        query = PackedInt8Embeddings(torch.cat([p.codes for p in captured]), torch.cat([p.inverse_norms for p in captured]))
        packed.codes[frozen['query']], packed.inverse_norms[frozen['query']] = query.codes, query.inverse_norms
        quality, intervals, advance = b1.score(packed, frozen, torch.device('cuda'))
        assert hits == quality['per_query_r1']
        facts = {'batch': 1, 'query_images': 6354, 'gallery_images': 6245, 'b32_receipt_sha256': args.b32_sha256, 'b32_audit_sha256': args.b32_audit_sha256, 'all_B1_actual_public_native_r1_exact': True, 'B1_original_processor_native_packed_exact_queries': 32}
        arrays = {'query': query}
    preserved()
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert teacher.qualified.numerical_flags() == flags and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA and pair.sha(serving.LIBRARY) == serving.LIBRARY_SHA
    assert all(pair.sha(root / n) == h for n, h in code.items()) and torch.cuda.max_memory_allocated() < 10_000_000_000
    args.output.mkdir(exist_ok=False)
    for name, values in arrays.items():
        for field, value in (('codes', values.codes), ('inverse', values.inverse_norms)):
            path = args.output / (name + '.' + field + '.npy')
            np.save(path, value.numpy(), allow_pickle=False)
            facts[name + '_' + field + '_sha256'] = pair.sha(path)
    facts.update({'advance': advance, 'quality': quality, 'paired_dense_pe_intervals': intervals, 'code': code, 'execution_sha256': args.execution_sha256, 'cpu_authority_sha256': args.cpu_sha256, 'checkpoint_sha256': teacher.TEACHER_SHA, 'fp16_whole_sha256': authority['fp16_whole_sha256'], 'source_rng_environment_code_library_preserved': True, 'numerical_flags': flags, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0, 'official_read': False, 'claim_eligible': False, 'paired_speed_win': False})
    pair.smoke.save(args.output / 'receipt.json', facts)
    print('GO complete public FP16 TRAIN quality' if advance else 'KILL fixed public FP16 TRAIN quality')


if __name__ == '__main__':
    main()
