#!/usr/bin/env python3
"""Unpaired100-call B1/B32 warm-cache decode-to-top10 diagnostic; p99 uncertified."""
if not __debug__:
    raise SystemExit('Benchmark requires Python assertions; optimized mode is forbidden')

import argparse
import hashlib
import sys
import time
from pathlib import Path

import qualify_full_valid_anchor_serving as serving

CALLS = 100
WARMUPS = 10


def sequence(size, calls=CALLS):
    assert size in (1, 32) and calls in (CALLS, WARMUPS)
    return [[(call * size + i) % serving.COUNTS['query'] for i in range(size)] for call in range(calls)]


def result_sha(result):
    digest = hashlib.sha256()
    for array in result:
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--gpu-proof', type=Path, required=True)
    parser.add_argument('--gpu-sha256', required=True)
    parser.add_argument('--gpu-log', type=Path, required=True)
    parser.add_argument('--gpu-log-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    serving.new_output(args.output)
    root = Path(__file__).resolve().parent
    path = args.authority or root / 'full-valid-anchor-serving-authority.json'
    authority = serving.read(path, args.authority_sha256)
    state = serving.resource_start('gpu', authority)
    data = serving.startup(root, path, args.authority_sha256)
    authority, code = data[:2]
    binding = serving.bindings(authority, args.authority_sha256, code)
    pins = (args.gpu_proof, args.gpu_sha256, args.gpu_log, args.gpu_log_sha256)
    admitted = serving.proof_authority(*pins, 'gpu', binding)
    # Admission completes before model construction, image loading or native open.
    full, control, _, prior, proof = serving.runtime_startup(root, data)
    import numpy as np
    import torch
    from sfora.siglip2_compact_serving import Siglip2CompactIndex
    from sfora.cutile_int8 import CutilePackedInt8Gallery
    assert admitted['model_snapshot'] == str(control.large_snapshot)
    assert admitted['model_file_sha256'] == full.pair.smoke.MODEL_HASHES
    assert admitted['dataset_root'] == str(control.dataset_root)
    assert admitted['frozen_prefix_sha256'] == proof['frozen_prefix_sha256'] and admitted['numerical_flags'] == prior['numerical_flags']
    compiler_version = serving.configure(full, authority, True, prior)
    before_rng = serving.rng_fingerprint(full, True)
    encoder = serving.public_encoder(full, control)
    before = serving.model_facts(full, encoder.vision, encoder.head, encoder.processor, serving.F16)
    assert before == admitted['models']['public_f16']
    gallery = serving.load_wires(authority)['gallery']
    measurements = {}
    with Siglip2CompactIndex(encoder, CutilePackedInt8Gallery.open_packed(serving.artifact(authority, 'native_library'), gallery)) as index:
        for size in (1, 32):
            ordered = sequence(size)
            warmup_order = sequence(size, WARMUPS)
            paths = [serving.image_paths(control, data[-1], 'query', indices) for indices in ordered]
            expected = []
            # Same-size query arithmetic, including B1. No saved B32 substitution.
            # Keep only CPU top10 results; every timed call decodes its files afresh.
            for batch in paths:
                packed = encoder.encode_images(serving.decode(batch))
                reference = serving.cpu_reference(packed, gallery)
                expected.append(reference)
                del packed
                serving.usage(state)
                assert torch.cuda.max_memory_allocated() < serving.CUDA_CAP
            for call in range(WARMUPS):
                actual = index.search_images(serving.decode(paths[call]))
                torch.cuda.synchronize()
                serving.same_results(actual, expected[call])
            raw_seconds, result_hashes = [], []
            wall_start = time.perf_counter()
            for batch, reference in zip(paths, expected, strict=True):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                actual = index.search_images(serving.decode(batch))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                # All validation, hashing and resource observation lie outside timing.
                serving.same_results(actual, reference)
                assert elapsed > 0
                raw_seconds.append(elapsed)
                result_hashes.append(result_sha(actual))
                serving.usage(state)
                assert torch.cuda.max_memory_allocated() < serving.CUDA_CAP
            timing_wall = time.perf_counter() - wall_start
            assert len(raw_seconds) == CALLS
            measurements[str(size)] = {'calls': CALLS, 'warmups': WARMUPS,
                'ordered_query_indices': ordered, 'warmup_query_indices': warmup_order,
                'raw_seconds': raw_seconds, 'p50_ms': float(1000 * np.quantile(raw_seconds, .5)),
                'p95_ms': float(1000 * np.quantile(raw_seconds, .95)),
                'p99_ms_UNCERTIFIED': float(1000 * np.quantile(raw_seconds, .99)),
                'images_per_second_from_total_timed_seconds': size * CALLS / sum(raw_seconds),
                'timing_wall_seconds': timing_wall, 'sum_timed_seconds': sum(raw_seconds),
                'result_sha256': result_hashes, 'expected_result_sha256': [result_sha(x) for x in expected],
                'whole_decode_search_synchronize_timed': True}
            print(f'PASS unpaired B{size}100 complete timed calls', flush=True)
    assert serving.model_facts(full, encoder.vision, encoder.head, encoder.processor, serving.F16) == before
    assert serving.rng_fingerprint(full, True) == before_rng
    assert full.qualified.teacher.qualified.numerical_flags() == prior['numerical_flags']
    assert serving.startup(root, path, args.authority_sha256) == data
    serving.loaded_code_guard(root, code)
    serving.proof_authority(*pins, 'gpu', binding)
    peak = torch.cuda.max_memory_allocated()
    assert peak < serving.CUDA_CAP
    executable = Path(sys.executable).resolve()
    serving.save(args.output, {'schema': 'full-valid-anchor-serving-diagnostic-v1', 'pass': True,
        'bindings': binding, 'gpu_proof_sha256': args.gpu_sha256, 'gpu_log_sha256': args.gpu_log_sha256,
        'gpu_proof': str(args.gpu_proof), 'gpu_log': str(args.gpu_log), 'precision': 'fp16_native',
        'gallery_images': serving.COUNTS['gallery'], 'compiler_version': compiler_version,
        'invocation': {'argv': sys.argv, 'python': str(executable), 'python_sha256': serving.sha(executable),
                       'version': sys.version, 'optimize': sys.flags.optimize,
                       'driver': str(Path(__file__).resolve()), 'driver_sha256': serving.sha(__file__)},
        'models': {'public_f16': before}, 'rng_sha256': before_rng, 'timing': measurements,
        'query_order_protocol': 'dataset order from0, modulo14218; B1 then B32;100 calls and10 warmups each',
        'whole_decode_search_synchronize_timed': True, 'unpaired_diagnostic_only': True,
        'decoded_previously_observed_official_sentinels': True, 'official_quality_scored': False,
        'quality_read': False, 'optimizer_updates': 0, 'python_assertions_enabled': __debug__,
        'source_head_buffers_processor_runtime_rng_preserved': True,
        'loaded_entrypoint_code_verified': True,
        'rng_components': ['torch_CPU', 'all_torch_CUDA', 'Python_random', 'NumPy_global'],
        'p99_certified': False, 'paired_public_speed_win': False, 'sustained_QPS_claim': False,
        'claim_eligible': False, 'production_joint_goal_met': False,
        'paired10000_CI_gate': 'requires separately frozen matched protocol',
        'peak_cuda_allocated_bytes': peak, 'resources': serving.usage(state)})
    print('PASS unpaired full2000 diagnostic; p99 uncertified, no matched speed/QPS/quality claim', flush=True)


if __name__ == '__main__':
    main()
