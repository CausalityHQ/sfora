#!/usr/bin/env python3
"""Actual trained Large public encoder/native index parity and bounded latency pilot."""
import argparse
import inspect
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import qualify_pe_large_trained_candidate as trained
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, teacher = trained.pair, trained.teacher
HELD = Path('/home/riomus/runs/sfora-large-trained-held-v1')
HELD_SHA = '6e26885473653696bdb47d515ffa96d7191bec1ec702be2bea3b8134b6c76b26'
AUDIT_SHA = '619bc8534df73bd39df53ab7540b93d3de98840c66172e11ad814ef021943b88'
LIBRARY = Path('/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so')
LIBRARY_SHA = '39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c'
TRAINED_CODE_SHA = '61729205d81cedd7bac5d80bdc150a46921ff7d9a177e8164e272c0256687d62'


def startup(root, execution_sha):
    manifest = root / 'trained-serving-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'public serving code differs'
    control, frozen, cpu, _, old = trained.startup(root, TRAINED_CODE_SHA)
    assert all(code[n] == h for n, h in old.items())
    assert pair.sha(HELD / 'receipt.json') == HELD_SHA and pair.sha(HELD / 'cpu-audit.json') == AUDIT_SHA
    receipt, audit = (json.loads((HELD / n).read_text()) for n in ('receipt.json', 'cpu-audit.json'))
    assert receipt['advance'] and audit['pass'] and audit['advance'] and audit['receipt_sha256'] == HELD_SHA
    assert receipt['code'] == old and receipt['held_manifest'] == frozen['held_manifest']
    assert pair.sha(HELD / 'large.held.npy') == receipt['held_sha256'] and pair.sha(LIBRARY) == LIBRARY_SHA
    return control, frozen, cpu, receipt, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-startup-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, cpu, receipt, code = startup(root, args.execution_sha256)
    # Authenticate the original closed module set before importing the new serving layer.
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
    assert Path(inspect.getfile(CutilePackedInt8Gallery)).resolve() == root / 'src/sfora/cutile_int8.py'
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        original = pair.sha
        with patch.object(pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else original(p)):
            try:
                startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'public serving code differs'
            else:
                raise AssertionError('changed serving driver accepted')
        print('PASS actual public serving startup/quality/source/native authority and changed-driver rejection; no CUDA/images')
        return
    assert torch.cuda.is_available() and not args.output.exists()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == receipt['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot, checkpoint=teacher.TEACHER, expected_checkpoint_sha256=teacher.TEACHER_SHA, model_file_sha256=pair.smoke.MODEL_HASHES, precision='fp32_autocast', device=torch.device('cuda'))
    assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == cpu['teacher_whole_sha256']
    assert pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
    assert json.loads(json.dumps(trained.native.environment(encoder.vision, encoder.processor))) == cpu['environment']
    values = np.load(HELD / 'large.held.npy', allow_pickle=False)
    packed = pack_int8_unit_embeddings(torch.from_numpy(values))
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    for start in range(0, 12599, 32):
        rows = frozen['held_manifest'][start:start + 32]
        images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
        actual = encoder.encode_images(images)
        assert torch.equal(actual.codes, packed.codes[start:start + len(rows)]) and torch.equal(actual.inverse_norms, packed.inverse_norms[start:start + len(rows)]), 'complete public encoder packed parity differs'
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        print(json.dumps({'public_held_images_verified': start + len(rows)}), flush=True)
    gallery_rows = frozen['gallery']
    gallery = PackedInt8Embeddings(packed.codes[gallery_rows].contiguous(), packed.inverse_norms[gallery_rows].contiguous())
    gallery_codes = gallery.codes.float()
    gallery_inverse = gallery.inverse_norms.float()

    def reference(query):
        scores = (query.codes.float() @ gallery_codes.T) * query.inverse_norms.float()[:, None] * gallery_inverse[None, :]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        return order.numpy(), scores.gather(1, order).numpy()

    def equal(actual, expected):
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)), 'native/public top10 ordinal or score bits differ'

    timing = {}
    with Siglip2CompactIndex(encoder, CutilePackedInt8Gallery.open_packed(LIBRARY, gallery)) as index:
        for start in range(0, len(frozen['query']), 32):
            selected = frozen['query'][start:start + 32]
            query = PackedInt8Embeddings(packed.codes[selected].contiguous(), packed.inverse_norms[selected].contiguous())
            equal(index.gallery.search_packed(query), reference(query))
        selected = frozen['query'][:32]
        paths = [control.dataset_root / frozen['held_manifest'][i]['relative_path'] for i in selected]
        assert all(pair.sha(p) == frozen['held_manifest'][i]['image_sha256'] for p, i in zip(paths, selected, strict=True))

        def decode(paths):
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            return images

        b1_same_as_export = []
        for row, path in zip(selected, paths, strict=True):
            images = decode([path])
            actual = encoder.encode_images(images)
            pixels = pair.pixels(encoder.processor, images, 'large').cuda()
            with torch.inference_mode():
                pooled = teacher.qualified.fp16(encoder.vision, pixels)
                raw = pair.smoke.compact_head_features(pooled, encoder.head).float()
                expected = pack_int8_unit_embeddings(torch.nn.functional.normalize(raw, dim=1).cpu())
            assert torch.equal(actual.codes, expected.codes) and torch.equal(actual.inverse_norms, expected.inverse_norms), 'public B1 direct-preprocessing/native packed parity differs'
            equal(index.search_images(images), reference(expected))
            b1_same_as_export.append(bool(torch.equal(actual.codes, packed.codes[row:row + 1]) and torch.equal(actual.inverse_norms, packed.inverse_norms[row:row + 1])))
        for size in (1, 32):
            batch_paths = paths[:size]
            expected = index.search_images(decode(batch_paths))
            for _ in range(5):
                equal(index.search_images(decode(batch_paths)), expected)
            samples = []
            for _ in range(20):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = index.search_images(decode(batch_paths))
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - tick)
                equal(result, expected)
            timing[str(size)] = {'calls': len(samples), 'warm_cache_decode_to_top10_seconds': samples, 'p50_ms': float(1000 * np.quantile(samples, .5)), 'p95_ms': float(1000 * np.quantile(samples, .95)), 'p99_ms_uncertified': float(1000 * np.quantile(samples, .99)), 'images_per_second_from_median': float(size / np.median(samples))}
    assert pair.smoke.digest(trained.base.whole_state(encoder.vision)) == cpu['teacher_whole_sha256'] and pair.smoke.digest(encoder.head.state_dict()) == cpu['teacher_head_sha256']
    assert all(p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())
    assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert teacher.qualified.numerical_flags() == flags and pair.sha(LIBRARY) == LIBRARY_SHA and pair.sha(teacher.TEACHER) == teacher.TEACHER_SHA
    assert all(pair.sha(root / n) == h for n, h in code.items()) and torch.cuda.max_memory_allocated() < 10_000_000_000
    pair.smoke.save(args.output, {'execution_sha256': args.execution_sha256, 'code': code, 'checkpoint_sha256': teacher.TEACHER_SHA, 'trained_quality_receipt_sha256': HELD_SHA, 'trained_quality_cpu_audit_sha256': AUDIT_SHA, 'native_library_sha256': LIBRARY_SHA, 'precision': 'fp32_autocast', 'complete_public_B32_held_packed_exact': True, 'held_images': 12599, 'native_top10_ordinal_score_bits_exact_queries': 6354, 'public_B1_direct_native_packed_top10_exact_queries': 32, 'B1_sentinel_matches_B32_export': b1_same_as_export, 'gallery_images': 6245, 'quality': {k: v for k, v in receipt['quality'].items() if not k.startswith('per_query')}, 'timing_pilot': timing, 'source_head_rng_code_native_file_unchanged': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0, 'official_read': False, 'claim_eligible': False, 'paired_public_speed_win': False, 'p99_certified': False})
    print('PASS actual public complete B32 packed/native top10/B1 reference parity and unpaired20-call latency pilot; no certified speed/official claim')


if __name__ == '__main__':
    main()
