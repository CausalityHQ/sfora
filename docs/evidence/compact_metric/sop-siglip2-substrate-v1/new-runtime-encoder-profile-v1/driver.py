"""Engineering-only new-runtime encoder profile; no TRAIN, quality, or state reuse."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path('/home/riomus/runs/sfora-native256-source-cpu-v4')
OUTPUT = Path('/home/riomus/runs/sfora-new-runtime-encoder-profile-v1')
SOURCE_E = '3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b'
PINS = {'qualify_siglip2_substrate_cpu.py': 'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38',
        'extract_siglip2_vision_source.py': 'a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d',
        'test_siglip2_substrate_cpu.py': 'aa45d7db558194ce294a70cc465a8250c871374d99d95c6adcb016dc60e7fc04'}

def main():
    started = time.perf_counter()
    if sys.flags.optimize or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('unoptimized single GPU invocation required')
    if hashlib.sha256((ROOT / 'execution.json').read_bytes()).hexdigest() != SOURCE_E:
        raise ValueError('original source closure differs')
    if json.loads((ROOT / 'execution.json').read_text()) != PINS:
        raise ValueError('exact source inventory differs')
    for name, expected in PINS.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('source member differs')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('_new_runtime_source', ROOT / 'qualify_siglip2_substrate_cpu.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    args = SimpleNamespace(execution_sha256=SOURCE_E, sources=ROOT / 'sources.json',
        sources_sha256='8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b',
        fit_manifest=ROOT / 'fit.json', fit_manifest_sha256='d32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251',
        arm='so400', output=OUTPUT)
    context = q.authority(args)
    q.cgroup_memory()
    scope_path = Path('/home/riomus/runs/sfora-identity-diversity-metadata-v1/scope.json')
    q.bound_file(context['extract'], context['guards'], scope_path,
        '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726')
    scope = json.loads(scope_path.read_text())
    for row in context['fit']['rows'][:2]:
        q.require(any(all(row[k] == r[k] for k in ('relative_path', 'product', 'image_sha256')) and
                      row['train_row'] == r['original_train_row'] for r in scope['control']['rows']),
                  'witness must be frozen control TRAIN row')
    context['packages'] = q.package_origins(context)
    OUTPUT.mkdir()
    print(json.dumps({'phase': 'strict_source_factory', 'seconds': time.perf_counter()-started}), flush=True)
    model, processor, roles = q.fresh_source(context)
    pixels, cpu_raw, sample = q.pixels_and_raw(context, model, processor)
    import torch
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.half().to('cuda').eval()
    gpu_pixels = pixels.repeat(16, 1, 1, 1).half().to('cuda')
    torch.cuda.synchronize()
    print(json.dumps({'phase': 'gpu_profile_begin', 'seconds': time.perf_counter()-started,
        'batch': 32, 'witness_images': 2, 'repeated_train_witnesses': True}), flush=True)
    samples = []
    with torch.inference_mode():
        for _ in range(3):
            result = model(pixel_values=gpu_pixels).pooler_output
        torch.cuda.synchronize()
        q.require(torch.isfinite(result).all().item(), 'nonfinite GPU source output')
        relative = float(torch.linalg.vector_norm(result[:2].float().cpu()-cpu_raw) /
                         torch.linalg.vector_norm(cpu_raw))
        q.require(relative <= .02, 'FP16 GPU/FP32 CPU engineering drift exceeds 2%')
        profile_started = time.perf_counter()
        while time.perf_counter() - profile_started < 30:
            tick = time.perf_counter()
            result = model(pixel_values=gpu_pixels).pooler_output
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-tick)
            q.require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'CUDA peak cap')
    origins = q.imported_origins(context['extract'], context['packages'])
    for path, expected in origins['files'].items():
        q.require(context['guards'].setdefault(path, expected) == expected, 'current origin conflict')
    peak = torch.cuda.max_memory_allocated()
    del model, processor, pixels, gpu_pixels, result, cpu_raw
    gc.collect()
    torch.cuda.empty_cache()
    q.rehash(context)
    memory = q.cgroup_memory()
    q.require(time.perf_counter()-started < 300, 'whole engineering envelope exceeded')
    record = {'schema': 'new-runtime-encoder-profile-v1', 'pass': True,
        'qualification_eligible': False, 'state_reuse_eligible': False, 'quality_read': False,
        'training_updates': 0, 'source_execution_sha256': SOURCE_E, 'origins': origins,
        'input_guards': context['guards'], 'train_sample': sample,
        'batch': 32, 'unique_train_witness_images': 2, 'repeated_witnesses': True,
        'measurement': 'preloaded FP16 encoder forward plus synchronization only',
        'cpu_gpu_relative_l2': relative, 'timed_forwards': len(samples),
        'forward_seconds': samples, 'median_forward_seconds': statistics.median(samples),
        'throughput_images_per_second': 32*len(samples)/sum(samples),
        'peak_cuda_allocated_bytes': peak, 'cgroup_after': memory,
        'whole_seconds': time.perf_counter()-started, 'invocation_id': os.environ['INVOCATION_ID']}
    with (OUTPUT / 'receipt.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: record[k] for k in ('pass', 'timed_forwards', 'throughput_images_per_second', 'whole_seconds')}), flush=True)

if __name__ == '__main__':
    main()
