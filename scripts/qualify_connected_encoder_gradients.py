#!/usr/bin/env python3
"""ONE engineering-only original So400 last-MLP gradient circuit, never a fit.

CLI: python -B qualify_connected_encoder_gradients.py --authority FILE
     --authority-sha256 SHA --output NEWDIR
FILE has exactly schema=connected-encoder-gradients-authority-v1, files (this
driver, its test, and HELPERS), python={path,sha256}, output, resource_policy
(POLICY), both_locks_held=true. The parent pins the prospective driver/test and
interpreter, holds BOTH lifetime locks, and supplies the original enclosing
300s/8GiB/noSwap/GPU<10GB unit. No peak reset. A receipt alone cannot attest
outer normal exit, timeout or locks; the parent must collect that terminal.

Reuses the original q source-v4 authority/factory and the profile's two CONTROL
TRAIN witnesses, repeated to B16. No additional images, gallery, optimizer,
serialization/reload, learned state, training objective or quality claim.
Synthetic head/A/C/means exist only to test the derivative circuit. All 448
encoder tensors remain byte-identical, including the four with gradients.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace

ROOT = Path('/home/riomus/runs/sfora-native256-source-cpu-v4')
SOURCE_E = '3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b'
SOURCE_PINS = {
    'qualify_siglip2_substrate_cpu.py': 'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38',
    'extract_siglip2_vision_source.py': 'a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d',
    'test_siglip2_substrate_cpu.py': 'aa45d7db558194ce294a70cc465a8250c871374d99d95c6adcb016dc60e7fc04'}
HELPERS = {
    'train_siglip2_cached_readout.py': 'a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c',
    'connected_residual_readout.py': 'f0782630a462667ba52aa72b84d724f82c752a0f7f1ef66ac7346eba5e9800e1',
    'prototype_residual_readout.py': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68',
    'quadratic_readout.py': '12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6'}
FILES = {*HELPERS, 'qualify_connected_encoder_gradients.py', 'test_connected_encoder_gradients.py'}
POLICY = {'seconds': 300, 'host_bytes': 8589934592, 'swap_bytes': 0,
          'cuda_allocated_bytes_exclusive': 10000000000}
MLP = tuple(f'encoder.layers.26.mlp.{layer}.{field}'
            for layer in ('fc1', 'fc2') for field in ('weight', 'bias'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            'canonical regular file required: ' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate authority key')
        result[key] = value
    return result


def authenticate(path, expected, output):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
            'authority SHA256 required')
    require(sha(path) == expected, 'authority SHA256 differs')
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024 and hashlib.sha256(raw).hexdigest() == expected,
            'authority changed or oversized')
    a = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(a, dict) and a.keys() ==
            {'schema', 'files', 'python', 'output', 'resource_policy', 'both_locks_held'} and
            a['schema'] == 'connected-encoder-gradients-authority-v1' and
            a['both_locks_held'] is True and a['resource_policy'] == POLICY and
            all(type(v) is int for v in a['resource_policy'].values()) and
            a['output'] == str(output) and isinstance(a['files'], dict) and
            a['files'].keys() == FILES, 'authority schema/policy/files/output differ')
    require(output.is_absolute() and output.resolve() == output and
            output.parent.is_dir() and not output.exists(), 'canonical exclusive new output required')
    root = Path(__file__).absolute().parent
    require(root.resolve() == root, 'canonical source directory required')
    guards = {str(path): expected}
    for name, digest in a['files'].items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None and
                (name not in HELPERS or HELPERS[name] == digest) and sha(root / name) == digest,
                'source SHA256 differs: ' + name)
        guards[str(root / name)] = digest
    python = a['python']
    require(isinstance(python, dict) and python.keys() == {'path', 'sha256'} and
            python['path'] == str(Path(sys.executable).resolve()) and
            sha(python['path']) == python['sha256'], 'interpreter authority differs')
    guards[python['path']] = python['sha256']
    return guards


def source_context(output, guards):
    require(sha(ROOT / 'execution.json') == SOURCE_E and
            json.loads((ROOT / 'execution.json').read_text(), object_pairs_hook=pairs) == SOURCE_PINS,
            'original source execution differs')
    for name, digest in SOURCE_PINS.items():
        require(sha(ROOT / name) == digest, 'original source member differs')
    sys.path.insert(0, str(ROOT))
    q = SimpleNamespace(**runpy.run_path(str(ROOT / 'qualify_siglip2_substrate_cpu.py')))
    args = SimpleNamespace(execution_sha256=SOURCE_E, sources=ROOT / 'sources.json',
        sources_sha256='8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b',
        fit_manifest=ROOT / 'fit.json',
        fit_manifest_sha256='d32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251',
        arm='so400', output=output)
    context = q.authority(args)
    for path, digest in guards.items():
        q.bound_file(context['extract'], context['guards'], path, digest)
    scope = q.read_json(context['extract'], context['guards'],
        Path('/home/riomus/runs/sfora-identity-diversity-metadata-v1/scope.json'),
        '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726')
    for row in context['fit']['rows'][:2]:
        require(any(all(row[k] == r[k] for k in ('relative_path', 'product', 'image_sha256')) and
                    row['train_row'] == r['original_train_row'] for r in scope['control']['rows']),
                'witness must be frozen control TRAIN row')
    return q, context


def select_mlp(model, expected):
    parameters = dict(model.named_parameters())
    require(len(parameters) == 448 and parameters.keys() == expected.keys() == model.state_dict().keys()
            and set(MLP) <= parameters.keys(), 'strict 448 encoder inventory required')
    for name, p in parameters.items():
        require(p.grad is None and str(p.dtype) == 'torch.float32' and p.device.type == 'cpu' and
                list(p.shape) == expected[name], 'source parameter differs: ' + name)
        p.requires_grad_(name in MLP)
    require(sum(p.requires_grad for p in parameters.values()) == 4, 'exact four MLP gradients required')


def gradient_fact(torch, value, label):
    grad = value.grad
    require(grad is not None and torch.isfinite(grad).all().item(), 'missing/nonfinite gradient: ' + label)
    nonzero, norm = int(torch.count_nonzero(grad).item()), float(grad.norm())
    require(nonzero > 0 and math.isfinite(norm) and norm > 0, 'zero/nonfinite gradient norm: ' + label)
    return {'norm': norm, 'nonzero': nonzero}


def probe(q, model, pixels):
    import torch
    from torch.nn import functional as F
    root = Path(__file__).absolute().parent
    modules = {name: SimpleNamespace(**runpy.run_path(str(root / name))) for name in HELPERS}
    primitive, readout = modules['quadratic_readout.py'], modules['prototype_residual_readout.py']
    shapes = {'primary.weight': (128, 1152), 'primary.bias': (128,), 'down.weight': (32, 1152),
              'up.weight': (128, 32), 'center': (1152,), 'preactivation_std': ()}
    tensors = {name: torch.randn(shape) * .03 for name, shape in shapes.items()}
    tensors['preactivation_std'].fill_(1)
    head = modules['train_siglip2_cached_readout.py'].head_from('control', tensors=tensors)
    head.requires_grad_(False).eval()
    head_before = {name: q.tensor_fact(t) for name, t in head.state_dict().items()}
    del tensors
    head.to('cuda')
    A = torch.nn.Parameter(torch.randn(128, 160, device='cuda') * .01)
    C = torch.nn.Parameter(torch.randn(128, 1152, device='cuda') * .01)
    means = {'linear': torch.randn(32, device='cuda') * .02,
             'concat': torch.randn(160, device='cuda') * .02}
    mu = torch.randn(1152, device='cuda') * .02
    fixed = {'A': A, 'C': C, 'mu': mu, **means}
    before = {name: q.tensor_fact(t.detach().cpu()) for name, t in fixed.items()}
    gpu_pixels = pixels.repeat(8, 1, 1, 1).to('cuda')
    with torch.autocast('cuda', enabled=False):
        pooled = model(pixel_values=gpu_pixels).pooler_output
        require(pooled.shape == (16, 1152) and pooled.dtype == torch.float32 and
                pooled.requires_grad and torch.isfinite(pooled).all().item(), 'connected FP32 B16 source required')
        # Original query order: normalize pooled FP32, head/readout, normalize raw.
        features = F.normalize(pooled.float(), dim=1)
        features.retain_grad()
        raw = modules['connected_residual_readout.py'].raw_features(
            features, head, A, means, C, mu, primitive, readout)
        require(torch.all(raw.norm(dim=1) > 0).item(), 'nonzero query raw required')
        unit = F.normalize(raw, dim=1)
        cotangent = torch.randn_like(unit)
        (unit * cotangent).sum().backward()
        gradients = {name: gradient_fact(torch, p, name)
                     for name, p in model.named_parameters() if name in MLP}
        gradients['features'] = gradient_fact(torch, features, 'features')
        require(all(p.grad is None for name, p in model.named_parameters() if name not in MLP),
                'frozen encoder gradient appeared')
        model.zero_grad(set_to_none=True)
        A.grad = C.grad = None
        # Original detached source, same expression/order/values, new readout graph.
        mutant = readout.raw_features(features.detach(), head, A, means, 'concat', primitive)
        mutant = mutant + F.linear(features.detach() - mu, C)
        require(torch.equal(raw.detach(), mutant.detach()), 'original readout forward bytes differ')
        (F.normalize(mutant, dim=1) * cotangent).sum().backward()
        require(all(p.grad is None for p in model.parameters()), 'detached mutant reached encoder')
        gradient_fact(torch, A, 'mutant A')
        gradient_fact(torch, C, 'mutant C')
    require(all(not p.requires_grad and p.grad is None for p in head.parameters()) and
            head_before == {name: q.tensor_fact(t.detach().cpu()) for name, t in head.state_dict().items()},
            'frozen head bytes/roles/gradients changed')
    require(before == {name: q.tensor_fact(t.detach().cpu()) for name, t in fixed.items()},
            'fixed synthetic readout changed')
    torch.cuda.synchronize()
    return {'gradients': gradients, 'detached_source_encoder_gradients_absent': True,
            'original_readout_forward_bitwise': True, 'head_bytes_unchanged': True,
            'fixed_synthetic_readout_unchanged': True}


def qualify(args):
    started = time.perf_counter()
    require(not sys.flags.optimize and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and os.environ.get('INVOCATION_ID'),
            'unoptimized single GPU enclosing unit and deterministic workspace required')
    guards = authenticate(args.authority, args.authority_sha256, args.output)
    q, context = source_context(args.output, guards)
    memory_before = q.cgroup_memory()
    context['packages'] = q.package_origins(context)
    args.output.mkdir()
    import torch
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), 'grad mode required')
    torch.manual_seed(179061)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, processor, roles = q.fresh_source(context)
    pixels, cpu_raw, sample = q.pixels_and_raw(context, model, processor)
    select_mlp(model, context['expected'])
    model.to('cuda').eval()
    result = probe(q, model, pixels)
    require(all((p.requires_grad is (name in MLP)) and p.grad is None
                for name, p in model.named_parameters()), 'encoder roles/gradients changed')
    require(model.state_dict().keys() == context['expected'].keys(), 'exit encoder inventory changed')
    for name, tensor in model.state_dict().items():
        require(q.tensor_fact(tensor.detach().cpu()) == {'dtype': 'torch.float32',
                'shape': context['mapping'][name]['shape'], 'sha256': context['mapping'][name]['sha256']},
                'encoder bytes changed: ' + name)
    del tensor, model, processor, pixels, cpu_raw, roles
    gc.collect()
    torch.cuda.empty_cache()
    origins = q.imported_origins(context['extract'], context['packages'])
    for path, digest in origins['files'].items():
        require(context['guards'].setdefault(path, digest) == digest, 'current origin conflict')
    q.rehash(context)
    memory_after = q.cgroup_memory()
    require(memory_after['path'] == memory_before['path'], 'enclosing cgroup changed')
    peak = torch.cuda.max_memory_allocated()
    elapsed = time.perf_counter() - started
    require(peak < POLICY['cuda_allocated_bytes_exclusive'] and elapsed < POLICY['seconds'],
            'whole engineering envelope exceeded')
    record = {'schema': 'connected-encoder-gradients-v1', 'pass': True,
        'engineering_only': True, 'model_fit_qualified': False, 'state_reuse': False,
        'state_reuse_eligible': False, 'quality_read': False, 'training_updates': 0,
        'batch': 16, 'unique_train_witness_images': 2, 'repeated_witnesses': True,
        'synthetic_readout': True, 'train_sample': sample, 'gradient_parameters': list(MLP),
        'frozen_encoder_tensors': 444, 'unchanged_encoder_tensors': 448,
        'source_execution_sha256': SOURCE_E, 'authority_sha256': args.authority_sha256,
        'origins': origins, 'input_guards': context['guards'], 'resource_policy': POLICY,
        'peak_cuda_allocated_bytes': peak, 'whole_seconds': elapsed,
        'cgroup_before': memory_before, 'cgroup_after': memory_after,
        'invocation_id': os.environ['INVOCATION_ID'],
        'terminal_exit_and_both_locks_require_parent_receipt': True, **result}
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'pass': True, 'engineering_only': True, 'whole_seconds': elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    qualify(parser.parse_args())


if __name__ == '__main__':
    main()
