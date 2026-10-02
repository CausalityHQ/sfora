#!/usr/bin/env python3
"""Prospective deterministic TRAIN prototype residual ridge; native execution UNRUN.

Freeze FILES in execution.json. CLI, in order: --execution-sha256 SHA
--authority FILE --authority-sha256 SHA --phase cpu|fit --arm linear|quadratic
--output NEW_ABSOLUTE_DIRECTORY. CPU/linear qualifies BOTH full TRAIN fits.
Fit independently reconstructs complete warm source and inputs twice, then
saves/reloads the new complete payload. No seeds, updates, optimizer or search.
All native CPU FP32 runs require CUDA hidden, both parent lifetime locks,
8GiB/noSwap/events0 and normal original terminal/uncached exit authentication.
The parent supplies exact prospective authority/CPU UNIT pins after review.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import ast
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
from types import FunctionType, ModuleType, SimpleNamespace

SCHEMA = 'siglip2-prototype-residual-ridge-v1'
AUTHORITY_SCHEMA = 'siglip2-prototype-residual-launch-v1'
FILES = {'fit_siglip2_prototype_residual.py', 'test_siglip2_prototype_residual.py'}
ARMS = ('linear', 'quadratic')
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
ORIGINAL_REFERENCE = {'root': '/home/riomus/runs/sfora-so400-quadratic-readout-source-v7',
                      'execution_sha256': '84414302a6749842cae20e2608e932066334916218df68f91f10fd3c589ca382'}
ORIGINAL_CODE = {
    'quadratic_encoder_frames.py': '1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db',
    'quadratic_readout.py': '12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6',
    'test_siglip2_quadratic_readout.py': '1410060ac38065f39aa8e3f0a43fe47fb42321331bb00c9d43264381c4a23ba8',
    'train_siglip2_quadratic_readout.py': '17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f'}
ORIGINAL_CPU = {
    'authority': {'path': ORIGINAL_REFERENCE['root'] + '/authority-cpu-v7.json',
                  'sha256': 'd703e4c4b6a7be0fd287bb721cc2769dd4b958f196938a8182c1762cadcd4eb2'},
    'terminal': {
        'receipt': {'path': '/home/riomus/runs/sfora-so400-quadratic-readout-cpu-v7/receipt.json',
                    'sha256': '7e8566c24cae41e71b67f42bb68eda0717b8a7351a7ae9e5c145a35ab0bfa78f'},
        'log': {'path': ORIGINAL_REFERENCE['root'] + '/cpu-v7.log',
                'sha256': '3bf689314d9d7ad4b111223417ae9ca41785fa75b2cef1cf0642a072d452c373'},
        'unit': 'sfora-so400-quadratic-readout-cpu-v7', 'invocation_id': '13291ef2bdd946e7a76f82f862248234',
        'service_seconds': 94.629, 'native_peak_rss_kib': 1146616, 'both_locks_held': True}}
SOLVER_AST_SHA = '34644c570383257de53370db0c2bacadeeb8ac1232fd460986a82dbdd4b8fc9a'
SOLVER_SHA = '854e7d92b5cea2ef78dd49cf5deaded13f0af651a6dfae73368ec78acb939e27'
PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'
CANONICAL_SHA = '55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8'
RECIPE = {
    'rows': 6355, 'classes': 1008, 'width': 1152, 'output_dim': 128, 'rank': 32, 'coefficients': 4096,
    'input': 'raw canonical TRAIN; original CPU normalization then unchanged control head',
    'targets': 'image-weighted raw class means including each member minus raw source head',
    'basis': {'linear': 'Z', 'quadratic': 'Z.square()'},
    'means': 'both fixed full TRAIN FP32 means', 'regularization': 0.1,
    'ridge_scale': 'trace(Phi.T@Phi)/32', 'correction_multiplier': 1, 'intercept': False,
    'stationarity': '||M@A.T-B||_F/(||M||_F*||A.T||_F+||B||_F); M=Phi.T@Phi+lambda*I; B=Phi.T@Y',
    'stationarity_max': 1e-5, 'fit_passes': 2, 'arithmetic': 'CPU FP32; autocast disabled'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'original_reference', 'original_cpu',
               'ridge_solver', 'warm_start', 'partition', 'recipe', 'resource_policy', 'both_locks_held', 'selected_cpu'}
FIT_KEYS = {'A', 'means', 'source_mean', 'target_mean', 'prototypes', 'counts', 'fit_witness'}
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'encoder', 'config', 'buffers', 'head', 'classifier',
                'warm_payload', 'partition', 'original_rows', 'target', 'features', *FIT_KEYS,
                'output_witness', 'cpu_rng', 'numerical_flags'}
FROZEN_KEYS = ('encoder', 'config', 'buffers', 'head', 'classifier', 'warm_payload', 'partition',
               'original_rows', 'target', 'features')
FITTED_KEYS = tuple(sorted(FIT_KEYS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'SHA256 required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'current file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(fact, guards):
    require(isinstance(fact, dict) and fact.keys() == {'path', 'sha256'}, 'exact FILE required')
    raw = bound_file(guards, fact['path'], fact['sha256']).read_bytes()
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == fact['sha256'], 'JSON size/hash differs')
    return strict_json(raw)


def closure(root, sha, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': sha}, guards)
    require(code.keys() == set(names), 'exact code closure required')
    for name, digest in code.items():
        require(Path(name).name == name, 'bare closure member required')
        bound_file(guards, root / name, digest)
    return code


def load_authenticated(name, path, sha, guards):
    require(name not in sys.modules, 'fresh helper namespace required')
    path = bound_file(guards, path, sha)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'helper changed before execution')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'helper origin required')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    return module


def policy(phase):
    require(phase in ('cpu', 'fit'), 'new fixed phase required')
    return {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def check_unit(unit):
    require(isinstance(unit, dict) and unit.keys() == {'receipt', 'log', 'unit', 'invocation_id',
            'service_seconds', 'native_peak_rss_kib', 'both_locks_held'} and unit['both_locks_held'] is True,
            'complete original UNIT required')
    require(isinstance(unit['unit'], str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and
            isinstance(unit['invocation_id'], str) and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            all(type(unit[k]) in (int, float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')), 'actual UNIT resources required')
    for k in ('receipt', 'log'):
        f = unit[k]
        require(isinstance(f, dict) and f.keys() == {'path', 'sha256'} and Path(f['path']).is_absolute() and
                re.fullmatch('[0-9a-f]{64}', f['sha256']), 'UNIT FILE differs')


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and args.arm in ARMS and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'new prototype launch differs')
    require(launch['original_reference'] == ORIGINAL_REFERENCE and launch['original_cpu'] == ORIGINAL_CPU,
            'complete original source-v7/CPU pins required')
    require(launch['ridge_solver'].keys() == {'path', 'sha256'} and
            launch['ridge_solver']['sha256'] == SOLVER_SHA and
            launch['partition'].keys() == {'path', 'sha256'} and
            launch['partition']['sha256'] == PARTITION_SHA, 'solver/partition differs')
    require((args.phase != 'cpu' or args.arm == 'linear') and
            (launch['selected_cpu'] is None) == (args.phase == 'cpu'), 'new CPU qualification prerequisite differs')
    if launch['selected_cpu'] is not None:
        check_unit(launch['selected_cpu'])
        require(launch['selected_cpu'] != ORIGINAL_CPU['terminal'], 'old CPU cannot qualify new fits')


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'original_reference', 'original_cpu', 'ridge_solver',
                                 'warm_start', 'partition', 'recipe')}


def cli(root, authority, sha, execution, phase, arm, output):
    return [str(Path(root) / 'fit_siglip2_prototype_residual.py'), '--execution-sha256', execution,
            '--authority', str(authority), '--authority-sha256', sha, '--phase', phase, '--arm', arm,
            '--output', str(output)]


def authority(args):
    """Authenticate exact original source/CPU before calling any original helper."""
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded new admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    output, oldroot = args.output, Path(ORIGINAL_REFERENCE['root'])
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink() and not output.is_relative_to(root) and not output.is_relative_to(oldroot) and
            not root.is_relative_to(oldroot) and not oldroot.is_relative_to(root), 'exclusive separate output/source closure required')
    require(closure(oldroot, ORIGINAL_REFERENCE['execution_sha256'], ORIGINAL_CODE, guards) == ORIGINAL_CODE,
            'complete original source-v7 exact4 differs')
    oldlaunch = read_json(ORIGINAL_CPU['authority'], guards)
    oldrecord = read_json(ORIGINAL_CPU['terminal']['receipt'], guards)
    bound_file(guards, **{'path': ORIGINAL_CPU['terminal']['log']['path'],
                         'expected': ORIGINAL_CPU['terminal']['log']['sha256']})
    path = oldroot / 'train_siglip2_quadratic_readout.py'
    old = load_authenticated('_prototype_original', path, ORIGINAL_CODE[path.name], guards)
    oldargs = SimpleNamespace(execution_sha256=ORIGINAL_REFERENCE['execution_sha256'],
        authority=Path(ORIGINAL_CPU['authority']['path']), authority_sha256=ORIGINAL_CPU['authority']['sha256'],
        phase='cpu', arm='control', seed=179061, output=output)
    old.check_launch(oldlaunch, oldargs)
    old.check_terminal_record(oldrecord, oldlaunch, 'cpu', 'control')
    require(oldrecord['code'] == ORIGINAL_CODE and oldrecord['authority'] == ORIGINAL_CPU['authority'] and
            oldrecord['authority_sha256'] == ORIGINAL_CPU['authority']['sha256'] and
            oldrecord['execution_sha256'] == ORIGINAL_REFERENCE['execution_sha256'] and
            oldrecord['launch'] == oldlaunch and launch['warm_start'] == oldlaunch['warm_start'] and
            launch['partition'] == oldlaunch['partition'], 'original CPU/warm/partition context differs')
    legacy = old.authority(oldargs)
    require(oldrecord['source'] == legacy['source'] and
            oldrecord['numerical_flags'] == legacy['selected']['source_cpu']['numerical_flags'] and
            oldrecord['invocation']['argv'] == old.cli(oldroot, oldargs.authority, oldargs.authority_sha256,
                oldargs.execution_sha256, 'cpu', 'control', 179061, Path(ORIGINAL_CPU['terminal']['receipt']['path']).parent) and
            all(oldrecord['invocation'][k] == legacy['selected']['source_cpu']['invocation'][k]
                for k in ('python', 'python_sha256', 'python_version')),
            'original successful CPU source/invocation differs')
    original_required = dict(legacy['guards'])
    require(all(oldrecord['input_guards'].get(p) == h for p, h in original_required.items()),
            'original CPU complete original guards differ')
    for path, digest in guards.items():
        require(legacy['guards'].setdefault(path, digest) == digest, 'original/new guard conflict')
    for path, digest in legacy['guards'].items():
        require(guards.setdefault(path, digest) == digest, 'complete original guard conflict')
    original_final = old.admit_unit(legacy, oldrecord, ORIGINAL_CPU['terminal'], 'cpu')
    bound_file(guards, launch['ridge_solver']['path'], SOLVER_SHA)
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'guards': guards,
               'old': old, 'legacy': legacy, 'source': legacy['source'], 'terminals': {},
               'terminal_cgroups': {'original_cpu': original_final}, 'phase_seconds': {},
               'required_guards': {p: h for p, h in guards.items() if p != str(args.authority)}}
    if args.phase == 'fit':
        admit_terminal(context, launch['selected_cpu'], 'cpu', 'linear')
    return context


def extract_solver(path, sha, torch):
    """Compile only the exact authenticated solver/model AST in a fresh namespace."""
    raw = bound_file({}, path, sha).read_bytes()
    require(sha == SOLVER_SHA and hashlib.sha256(raw).hexdigest() == sha, 'ridge solver provenance differs')
    tree = ast.parse(raw, filename=str(path))
    names = {'RidgeStitchModel', 'fit_ridge_stitch'}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require(len(selected) == 2 and {node.name for node in selected} == names, 'exact solver definitions required')
    # No eager foundation_adapter import and no mutation of any source globals.
    require(hashlib.sha256(ast.dump(ast.Module(body=selected, type_ignores=[]),
            include_attributes=False).encode()).hexdigest() == SOLVER_AST_SHA, 'exact solver AST differs')
    name = '_prototype_ridge_solver_' + str(len(sys.modules))
    require(name not in sys.modules, 'fresh solver namespace required')
    module = ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(torch=torch, dataclass=dataclass, isfinite=math.isfinite)
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[])),
                 str(path), 'exec'), vars(module))
    return module.fit_ridge_stitch


def prepare_original(context):
    """Repeat fresh source preparation; retain only admitted immutable helpers."""
    old, legacy = context['old'], context['legacy']
    path = Path(old.__file__)
    raw = bound_file(context['guards'], path, ORIGINAL_CODE[path.name]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_CODE[path.name], 'original preparation changed')
    # Byte-authenticated extraction retains every original data predicate.
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == 'prepare_native')
    retained, removed = [], []
    for statement in node.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
                isinstance(statement.targets[0], ast.Subscript) and
                ast.dump(statement.targets[0].value) == ast.dump(ast.Name(id='context', ctx=ast.Load())) and
                isinstance(statement.targets[0].slice, ast.Constant) and
                statement.targets[0].slice.value in ('ref', 'packing')):
            removed.append(statement.targets[0].slice.value)
        else:
            retained.append(statement)
    require(removed == ['ref', 'packing'], 'exact original helper initialization required')
    node.body = retained
    retained_ast = ast.dump(ast.Module(body=[node], type_ignores=[]), include_attributes=False)
    cached = context.get('original_preparation')
    first = cached is None
    if first:
        require('ref' not in legacy and 'packing' not in legacy, 'partial original preparation is inadmissible')
        old.prepare_native(legacy)
        namespace = dict(vars(old))  # Never rebind any original module global.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        original, ref = legacy['original'], legacy['ref']
        math_root = legacy['selected']['math_context']['root']
        pack = legacy['selected']['launch']['helpers']['packing']
        helpers = [('_siglip2_pinned_adaptation_rank', sys.modules['_siglip2_pinned_adaptation_rank'],
                    Path(math_root) / 'deployed_code_rank.py', original.RANK_SHA256),
                   ('_quadratic_packing', legacy['packing'], Path(pack['path']), pack['sha256'])]
        helpers = [(*h, h[1].__spec__, h[1].__spec__.loader) for h in helpers]
        require(ref.smooth_ap_bank_loss is helpers[0][1].smooth_ap_bank_loss,
                'original reference/rank function differs')
        functions = [(owner, name, fn, fn.__code__) for owner in
                     (old, original, legacy['genuine'], ref, *(h[1] for h in helpers))
                     for name, fn in vars(owner).items() if isinstance(fn, FunctionType)]
        globals_by_id = {id(fn.__globals__): fn.__globals__ for _, _, fn, _ in functions}
        globals_by_id[id(namespace)] = namespace
        cached = {'prepare': namespace['prepare_native'], 'helpers': helpers, 'ref': ref,
                  'prepare_code': namespace['prepare_native'].__code__,
                  'prepare_globals': namespace,
                  'reference_members': dict(vars(ref)), 'functions': functions,
                  'globals': [(values, dict(values)) for values in globals_by_id.values()],
                  'references': [(Path(math_root) / name, dict(pin)) for name, pin in original.REFERENCES.items()],
                  'ast': retained_ast}
        context['original_preparation'] = cached
    require(cached['ast'] == retained_ast and
            cached['prepare'] is cached['prepare_globals']['prepare_native'] and
            cached['prepare'].__code__ is cached['prepare_code'] and
            cached['prepare'].__globals__ is cached['prepare_globals'],
            'original preparation AST/code differs')
    for name, module, origin, sha, spec, loader in cached['helpers']:
        require(sys.modules.get(name) is module and
                module.__spec__ is spec and spec.loader is loader and loader is not None and
                Path(module.__file__) == Path(spec.origin) == origin,
                'original helper object/origin differs: ' + name)
        bound_file(context['guards'], origin, sha)
    require(legacy['ref'] is cached['ref'] and legacy['packing'] is cached['helpers'][1][1] and
            vars(legacy['ref']).keys() == cached['reference_members'].keys() and
            all(vars(legacy['ref'])[k] is v for k, v in cached['reference_members'].items()),
            'original reference metadata differs')
    for owner, name, fn, code in cached['functions']:
        require(getattr(owner, name, None) is fn and fn.__code__ is code,
                'original helper function/code differs: ' + name)
    for values, members in cached['globals']:
        require(values.keys() == members.keys() and all(values[k] is v for k, v in members.items()),
                'original helper globals differ')
    for source, pin in cached['references']:
        bound_file(context['guards'], source, pin['source'])
        legacy['original'].selected_ast(source, pin)  # Genuine full-file AND AST admission.
    if not first:
        cached['prepare'](legacy)


def prepare_native(context):
    """Fresh original warm payload/complete immutable encoder, no new optimizer."""
    old, legacy = context['old'], context['legacy']
    old.require_no_model(legacy)
    # These are the original bytes consumed by reconstruction; all image/code
    # guards remain owned by admission and receive the full uncached exit audit.
    for fact in (old.WARM_CHECKPOINT, old.owned_encoder(legacy)['checkpoint'],
                 legacy['selected']['source']['caches']['canonical']):
        bound_file(context['guards'], fact['path'], fact['sha256'])
    prepare_original(context)
    import torch
    require(not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hidden CPU required')
    fact = old.WARM_CHECKPOINT
    path = bound_file(context['guards'], fact['path'], fact['sha256'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = legacy['original'].CheckpointPages(stream)
        legacy['genuine'].check_payload(disk, disk['identity'], 1000)
        require(legacy['original'].fingerprint(disk, consumed=pages.consume) == old.WARM_STATE_SHA,
                'complete original warm typed state differs')
        context['warm_payload'] = old.clone_tree(disk)
    del disk
    gc.collect()
    context['flags'] = legacy['flags']


def fit_prototype_residual(raw_train, targets, base, arm):
    """Complete canonical TRAIN -> fixed prototype targets -> centered FP32 ridge.

    base must come from fresh(context, arm), carrying the authenticated primitive
    and AST-extracted solver. Only weight.T/source_mean are used for inference;
    target_mean is retained as a witness and NEVER added to the correction.
    """
    import torch
    from torch.nn import functional as F
    require(type(arm) is str and arm in ARMS, 'fixed prototype arm required')
    primitive = getattr(base, '_prototype_primitive', None)
    solver = getattr(base, '_prototype_solver', None)
    require(primitive is not None and solver is not None, 'admitted source and solver required')
    primitive._check_features(raw_train, 'cpu', train=True)
    source = primitive._check_base(base, 'cpu')
    primitive._check_tensor(targets, (6355,), 'cpu', frozen=True, dtype='torch.int64')
    require(targets.min().item() == 0 and targets.max().item() == 1007 and
            torch.unique(targets).numel() == 1008, 'complete TRAIN1008 labels required')
    with torch.autocast('cpu', enabled=False), torch.no_grad():
        primitive._finite(torch, [raw_train, *source])
        require((raw_train.norm(dim=1) > 0).all().item(), 'nonzero raw TRAIN rows required')
        x = F.normalize(raw_train.detach().float(), dim=1)
        H = base(x).detach()
        Z = base.down(F.normalize(x, dim=1) - base.center)
        primitive._check_tensor(H, (6355, 128), 'cpu', frozen=True)
        primitive._check_tensor(Z, (6355, 32), 'cpu', frozen=True)
        primitive._finite(torch, [H, Z])
        counts = torch.bincount(targets, minlength=1008)
        require((counts > 0).all().item(), 'missing TRAIN class')
        prototypes = torch.stack([H[targets == c].mean(dim=0) for c in range(1008)])
        E = prototypes[targets] - H
        means = primitive.fit_means(raw_train, base)
        require(torch.equal(means['linear'], Z.mean(dim=0)) and
                torch.equal(means['quadratic'], Z.square().mean(dim=0)), 'both original fixed TRAIN means differ')
        V = Z if arm == 'linear' else Z.square()
        model = solver(V, E, torch.arange(6355, dtype=torch.int64), regularization=0.1)
        A = model.weight.T.contiguous().detach()
        source_mean = model.source_mean.detach().reshape(32)
        target_mean = model.target_mean.detach().reshape(128)
        require(torch.equal(source_mean, means[arm]), 'solver/source fixed mean differs')
        Phi, Y = V - source_mean, E - target_mean
        G = Phi.T @ Phi
        feature_energy = torch.trace(G).item()
        regularization = 0.1 * torch.trace(G) / 32
        M = G + regularization * torch.eye(32, dtype=torch.float32)
        B = Phi.T @ Y
        numerator = torch.linalg.vector_norm(M @ A.T - B).item()
        denominator = (torch.linalg.vector_norm(M).item() * torch.linalg.vector_norm(A.T).item()
                       + torch.linalg.vector_norm(B).item())
        require(math.isfinite(denominator) and denominator > 0, 'stationarity denominator must be finite positive')
        stationarity = numerator / denominator
        witness = {'rows': 6355, 'classes': 1008, 'coefficients': 4096, 'arm': arm,
                   'feature_energy': feature_energy, 'lambda': regularization.item(),
                   'stationarity_numerator': numerator, 'stationarity_denominator': denominator,
                   'normalized_stationarity': stationarity, 'A_nonzero': bool(A.count_nonzero().item()),
                   'target': 'raw member-inclusive prototypes', 'intercept': False}
        check_fit_witness(witness, arm)
        result = {'A': A, 'means': means, 'source_mean': source_mean, 'target_mean': target_mean,
                  'prototypes': prototypes.detach(), 'counts': counts.detach(), 'fit_witness': witness}
        primitive._finite(torch, [A, source_mean, target_mean, prototypes, *means.values()])
        primitive._check_tensor(A, (128, 32), 'cpu', frozen=True)
    return result


def check_fit_witness(witness, arm):
    require(witness.keys() == {'rows', 'classes', 'coefficients', 'arm', 'feature_energy', 'lambda',
            'stationarity_numerator', 'stationarity_denominator', 'normalized_stationarity', 'A_nonzero',
            'target', 'intercept'} and
            (witness['rows'], witness['classes'], witness['coefficients'], witness['arm']) == (6355, 1008, 4096, arm) and
            witness['A_nonzero'] is True and witness['intercept'] is False and
            witness['target'] == 'raw member-inclusive prototypes', 'complete new fit witness differs')
    require(all(type(witness[k]) in (int, float) and math.isfinite(witness[k]) and witness[k] > 0
                for k in ('feature_energy', 'lambda', 'stationarity_denominator')) and
            type(witness['stationarity_numerator']) in (int, float) and
            math.isfinite(witness['stationarity_numerator']) and witness['stationarity_numerator'] >= 0 and
            type(witness['normalized_stationarity']) in (int, float) and
            witness['normalized_stationarity'] == witness['stationarity_numerator'] / witness['stationarity_denominator'] and
            0 <= witness['normalized_stationarity'] <= 1e-5, 'normalized stationarity/feature energy failed')


def raw_features(context, state, features):
    return context['legacy']['quadratic'].raw_features(features, state['head'], state['A'], state['means'],
                                                     'control' if state['arm'] == 'linear' else 'candidate')


def reconstruct(context, arm, normalize=True):
    """Fresh complete original warm/head/input ownership; NEVER fits statistics."""
    import numpy as np
    import torch
    old, legacy = context['old'], context['legacy']
    require(arm in ARMS, 'fixed reconstructed arm required')
    prepare_native(context)
    fact = legacy['selected']['source']['caches']['canonical']
    require(fact['sha256'] == CANONICAL_SHA, 'canonical TRAIN bytes required')
    path = bound_file(context['guards'], fact['path'], fact['sha256'])
    cache = np.load(path, allow_pickle=False, mmap_mode='r')
    require(cache.shape == (6355, 1152) and cache.dtype == np.float32, 'canonical fullmatrix layout differs')
    raw = torch.from_numpy(cache.copy())
    del cache
    initial = legacy['initial']
    head = legacy['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).train()
    head._prototype_primitive = legacy['quadratic']
    legacy['quadratic']._check_base(head, 'cpu')
    old.claim_model(legacy, head)
    require(legacy['original'].fingerprint(dict(head.state_dict())) ==
            legacy['original'].fingerprint(initial['head']), 'fresh learned head differs')
    config, buffers = old.encoder_metadata(legacy)
    state = {'head': head, 'classifier': old.clone_tree(initial['classifier']), 'arm': arm,
             'encoder': old.owned_encoder(legacy).materialize(), 'config': config, 'buffers': buffers,
             'warm_payload': old.clone_tree(context['warm_payload']),
             'partition': old.owned_partition(legacy).materialize(), 'original_rows': old.clone_tree(initial['original_rows']),
             'target': old.clone_tree(initial['target']), 'raw_train': raw,
             'cpu_rng': torch.random.get_rng_state().clone(), 'numerical_flags': context['flags']}
    if normalize:
        state['features'] = legacy['genuine'].normalize_nonzero(raw).detach()
        del state['raw_train']
    require(legacy['original'].fingerprint(state['warm_payload']) == old.WARM_STATE_SHA, 'complete warm payload not preserved')
    return state


def output_witness(context, state):
    return context['old'].packed_outputs(context['legacy'], raw_features(context, state, state['features']))


def zero_source_witness(context, state):
    import torch
    legacy = context['legacy']
    zero = torch.nn.Parameter(torch.zeros((128, 32), dtype=torch.float32))
    zero_raw = legacy['quadratic'].raw_features(state['features'], state['head'], zero, state['means'],
                                               'control' if state['arm'] == 'linear' else 'candidate')
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        source_raw = state['head'](state['features']).detach()
    require(torch.equal(zero_raw, source_raw), 'zero-A source parity failed')
    return legacy['original'].fingerprint(context['old'].packed_outputs(legacy, zero_raw))


def fresh(context, arm):
    """Fit phase only: reconstruct warm, reread raw inputs, fit all new statistics."""
    import torch
    state = reconstruct(context, arm, normalize=False)
    legacy = context['legacy']
    state['head']._prototype_solver = extract_solver(context['launch']['ridge_solver']['path'], SOLVER_SHA, torch)
    before = legacy['original'].fingerprint(state_tree(state, tuple(k for k in FROZEN_KEYS if k != 'features')))
    tick = time.perf_counter()
    state['features'] = legacy['genuine'].normalize_nonzero(state['raw_train']).detach()
    fitted = fit_prototype_residual(state['raw_train'], state['target'], state['head'], arm)
    core_seconds = time.perf_counter() - tick
    del state['raw_train']
    require(legacy['original'].fingerprint(state_tree(state, tuple(k for k in FROZEN_KEYS if k != 'features'))) == before,
            'fit changed frozen source bytes')
    state.update(fitted)
    state['A'] = torch.nn.Parameter(state['A'], requires_grad=True)
    state['core_seconds'] = core_seconds
    state['zero_source_sha256'] = zero_source_witness(context, state)
    state['output_witness'] = output_witness(context, state)
    return state


def payload(context, state, ident):
    return {'schema': SCHEMA, 'identity': ident, 'source': context['source'],
            **{k: dict(state['head'].state_dict()) if k == 'head' else
               state['A'].detach() if k == 'A' else state[k] for k in
               (*FROZEN_KEYS, *FITTED_KEYS, 'output_witness', 'cpu_rng', 'numerical_flags')}}


def state_tree(state, keys):
    return {k: dict(state['head'].state_dict()) if k == 'head' else
            state['A'].detach() if k == 'A' else state[k] for k in keys}


def identity(context, state):
    fingerprint = context['legacy']['original'].fingerprint
    return {'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'], 'device': 'cpu',
            'native_inventory': state['encoder']['inventory'], 'numerical_flags': context['flags'],
            'warm_payload_sha256': fingerprint(state['warm_payload']),
            'frozen_sha256': fingerprint(state_tree(state, FROZEN_KEYS)),
            'fitted_sha256': fingerprint(state_tree(state, FITTED_KEYS)),
            'output_witness_sha256': fingerprint(state['output_witness']),
            'zero_source_sha256': state['zero_source_sha256']}


def verify_digest(fingerprint, value, expected, message):
    require(fingerprint(value) == expected, message)


def check_payload(context, saved, ident):
    import torch
    old, legacy = context['old'], context['legacy']
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            saved['source'] == ident['source'] == context['source'] and ident['method'] == method(context['launch']) and
            ident['arm'] in ARMS and ident['device'] == 'cpu' and saved['numerical_flags'] == context['flags'],
            'complete new typed payload/identity differs')
    old.check_encoder(saved['encoder'])
    require(saved['encoder']['inventory'] == ident['native_inventory'] and
            old.json_form(saved['config']) == saved['encoder']['export_runtime']['config'] and
            saved['buffers'].keys() == {'embeddings.position_ids'}, 'complete encoder448/config/buffers differ')
    primitive = legacy['quadratic']
    require(saved['head'].keys() == old.HEAD_LAYOUT.keys() and saved['means'].keys() == set(ARMS), 'complete head/means required')
    for name, shape in old.HEAD_LAYOUT.items():
        primitive._check_tensor(saved['head'][name], shape, 'cpu', frozen=True)
    shapes = {'A': (128, 32), 'source_mean': (32,), 'target_mean': (128,),
              'prototypes': (1008, 128), 'classifier': (1008, 128), 'features': (6355, 1152)}
    for name, shape in shapes.items():
        primitive._check_tensor(saved[name], shape, 'cpu', frozen=True)
    for name, shape in {'counts': (1008,), 'target': (6355,), 'original_rows': (6355,)}.items():
        primitive._check_tensor(saved[name], shape, 'cpu', frozen=True, dtype='torch.int64')
    for value in saved['means'].values():
        primitive._check_tensor(value, (32,), 'cpu', frozen=True)
    primitive._check_tensor(saved['buffers']['embeddings.position_ids'], (1, 256), 'cpu', frozen=True, dtype='torch.int64')
    primitive._check_tensor(saved['cpu_rng'], tuple(saved['cpu_rng'].shape), 'cpu', frozen=True, dtype='torch.uint8')
    require(saved['cpu_rng'].ndim == 1 and saved['cpu_rng'].numel() > 0 and
            torch.equal(saved['source_mean'], saved['means'][ident['arm']]) and
            torch.equal(saved['counts'], torch.bincount(saved['target'], minlength=1008)) and
            (saved['counts'] > 0).all().item() and saved['counts'].sum().item() == 6355 and
            bool(saved['A'].count_nonzero().item()), 'fitted means/counts/nonzero coefficients differ')
    check_fit_witness(saved['fit_witness'], ident['arm'])
    legacy['genuine'].check_payload(saved['warm_payload'], saved['warm_payload']['identity'], 1000)
    fingerprint = legacy['original'].fingerprint
    verify_digest(fingerprint, saved['warm_payload'], old.WARM_STATE_SHA, 'complete original warm payload differs')
    require(saved['partition'] == legacy['partition'].materialize() and
            torch.equal(saved['original_rows'], legacy['initial']['original_rows']) and
            torch.equal(saved['target'], legacy['initial']['target']), 'canonical full TRAIN mapping/labels differ')
    old.finite_tree(saved)
    for keys, field in ((FROZEN_KEYS, 'frozen_sha256'), (FITTED_KEYS, 'fitted_sha256')):
        verify_digest(fingerprint, {k: saved[k] for k in keys}, ident[field], 'fresh new typed member fingerprint differs')
    verify_digest(fingerprint, saved['output_witness'], ident['output_witness_sha256'], 'raw/unit/packed witness changed')


def integrity(context, state, ident):
    old, legacy = context['old'], context['legacy']
    primitive = legacy['quadratic']
    primitive._check_base(state['head'], 'cpu')
    primitive._check_weight(state['A'], 'cpu')
    require(all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in state['head'].modules()) and
            all(p.grad is None for p in state['head'].parameters()) and state['A'].grad is None,
            'source modes/hooks/gradients changed')
    require(legacy['source_driver'].numerical_flags() == context['flags'], 'original numerical flags changed')
    # Typed current-byte reads at every boundary catch .data writes without version changes.
    check_payload(context, payload(context, state, ident), ident)
    for fact in (old.WARM_CHECKPOINT, state['encoder']['checkpoint'],
                 legacy['selected']['source']['caches']['canonical'], context['launch']['ridge_solver']):
        bound_file(context['guards'], fact['path'], fact['sha256'])


def release(context, state):
    context['old'].release(context['legacy'], state)


def save(context, state, ident, path):
    import torch
    integrity(context, state, ident)
    saved = payload(context, state, ident)
    digest = context['legacy']['original'].fingerprint(saved)
    with context['legacy']['extract'].exclusive(path) as stream:
        writer = context['legacy']['original'].CheckpointWriter(stream)
        torch.save(saved, writer)
        writer.flush()
    sha = context['legacy']['extract'].sha(path)
    bound_file(context['guards'], path, sha)
    return sha, digest


def reload(context, path, sha, digest, ident):
    """Reconstruct source/inputs WITHOUT fitting; restore accepted fitted stats."""
    import torch
    old, legacy = context['old'], context['legacy']
    old.require_no_model(legacy)
    path = bound_file(context['guards'], path, sha)
    state = reconstruct(context, ident['arm'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = legacy['original'].CheckpointPages(stream)
        check_payload(context, disk, ident)
        require(legacy['original'].fingerprint(disk, consumed=pages.consume) == digest, 'serialized complete typed state differs')
        verify_digest(legacy['original'].fingerprint, state_tree(state, FROZEN_KEYS), ident['frozen_sha256'],
                      'independent original source/input reconstruction differs')
        for key in FITTED_KEYS:
            state[key] = old.clone_tree(disk[key])
        state['A'] = torch.nn.Parameter(state['A'], requires_grad=True)
        state['cpu_rng'] = old.clone_tree(disk['cpu_rng'])
        state['output_witness'] = old.clone_tree(disk['output_witness'])
    del disk
    gc.collect()
    state['zero_source_sha256'] = zero_source_witness(context, state)
    require(identity(context, state) == ident, 'restored accepted source/fitted identity differs')
    integrity(context, state, ident)
    updated = output_witness(context, state)
    verify_digest(legacy['original'].fingerprint, updated, ident['output_witness_sha256'], 'reloaded raw/unit/int8/FP16 inverse wire differs')
    verify_digest(legacy['original'].fingerprint, payload(context, state, ident), digest, 'complete independent reload changed')
    return state


def tamper_witness(context, state, ident):
    import torch
    # Neither .data write increments _version; rejection must use actual bytes.
    for tensor in (state['A'], state['means'][state['arm']], state['target'], state['head'].primary.weight):
        value, version = tensor.detach().clone(), tensor._version
        tensor.data.reshape(-1)[0] = 1 if tensor.data.reshape(-1)[0].item() == 0 else 0
        require(tensor._version == version, 'tamper must bypass version counters')
        rejected = False
        try:
            integrity(context, state, ident)
        except ValueError:
            rejected = True
        finally:
            with torch.no_grad():
                tensor.data.copy_(value)
        require(rejected, 'unchanged-version current-byte tamper accepted')
        integrity(context, state, ident)
    return True


def check_identity_record(ident, launch, arm, source, flags):
    require(ident.keys() == {'method', 'source', 'arm', 'device', 'native_inventory', 'numerical_flags',
            'warm_payload_sha256', 'frozen_sha256', 'fitted_sha256', 'output_witness_sha256', 'zero_source_sha256'} and
            ident['method'] == method(launch) and ident['source'] == source and ident['arm'] == arm and
            ident['device'] == 'cpu' and ident['numerical_flags'] == flags and
            ident['warm_payload_sha256'] == 'b78bd945256438bfd24873347e26d62c970bf9d85048663301d4fe1236ac35a9' and
            len(ident['native_inventory']) == 448 and len({r['name'] for r in ident['native_inventory']}) == 448 and
            all(r.keys() == {'name', 'shape', 'dtype', 'role'} and r['role'] == 'frozen' and r['dtype'] == 'torch.float32'
                for r in ident['native_inventory']) and
            all(re.fullmatch('[0-9a-f]{64}', ident[k]) for k in
                ('frozen_sha256', 'fitted_sha256', 'output_witness_sha256', 'zero_source_sha256')),
            'complete new source/fitted identity differs')


def check_terminal_record(record, launch, phase, arm):
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            method(record['launch']) == method(launch) and record['launch']['selected_cpu'] ==
            (None if phase == 'cpu' else launch['selected_cpu']) and
            record['resource_policy'] == policy(phase) and record['partition_sha256'] == PARTITION_SHA and
            record['cuda_initialized'] is False and record['peak_cuda_allocated_bytes'] == 0 and
            record['quality_read'] is False and record['public_encoder_qualified'] is False and
            record['fit_qualified'] is (phase == 'fit') and
            all(record[k] is True for k in ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'frozen_complement_exact',
                'training_only_fit', 'zero_A_source_parity', 'bypass_version_tamper_rejected',
                'sequential_model_ownership', 'both_locks_held_in_parent_authority')),
            'new deterministic fit terminal differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, execution_sha256=launch['execution_sha256']))
    require(record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == '' and
            record['authority_sha256'] == record['authority']['sha256'] and
            record['code'].keys() == FILES and
            type(record['total_fit_core_seconds']) in (float, int) and math.isfinite(record['total_fit_core_seconds']) and
            0 < record['total_fit_core_seconds'] <= record['wall_seconds'] < policy(phase)['seconds'] and
            0 < record['process_peak_rss_kib'] <= 8 * 1024**2,
            'new actual CPU/cost/closure terminal differs')
    facts = record['arms'] if phase == 'cpu' else {arm: record}
    require(facts.keys() == set(ARMS) if phase == 'cpu' else record['independent_refit_exact'] is True,
            'complete fullTRAIN fit qualification required')
    require(all(len(f['fit_core_seconds']) == 2 and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in f['fit_core_seconds']) and f['fit_passes'] == 2 and f['independent_refit_exact'] is True
                for f in facts.values()) and
            record['total_fit_core_seconds'] == sum(sum(f['fit_core_seconds']) for f in facts.values()),
            'both measured independent full fit-core passes required')
    for a, fact in facts.items():
        check_identity_record(fact['identity'], launch, a, record['source'], record['numerical_flags'])
        check_fit_witness(fact['fit_witness'], a)
        require(fact['output_witness_sha256'] == fact['identity']['output_witness_sha256'] and
                re.fullmatch('[0-9a-f]{64}', fact['terminal_state_sha256']) and
                fact['strict_reload_exact'] is True and fact['bypass_version_tamper_rejected'] is True,
                'new fitted/reload witness differs')
    if phase == 'cpu':
        require(record['checkpoint'] is None and record['arms']['linear']['identity']['zero_source_sha256'] ==
                record['arms']['quadratic']['identity']['zero_source_sha256'] and
                record['arms']['linear']['identity']['frozen_sha256'] == record['arms']['quadratic']['identity']['frozen_sha256'],
                'matched zero-A/full source arm parity failed')
    else:
        require(record['fit_passes'] == 2 and record['checkpoint'].keys() == {'path', 'sha256'} and
                record['checkpoint']['path'] == str(Path(record['output']) / 'resume.pt') and
                record['input_guards'].get(record['checkpoint']['path']) == record['checkpoint']['sha256'],
                'new complete checkpoint binding differs')


def admit_terminal(context, unit, phase, arm):
    """New exact-context admission, including external original normal exit proof."""
    old, legacy, guards = context['old'], context['legacy'], context['guards']
    check_unit(unit)
    record = read_json(unit['receipt'], guards)
    check_terminal_record(record, context['launch'], phase, arm)
    proof_launch = read_json(record['authority'], guards)
    require(proof_launch == record['launch'] and record['code'] == context['code'] and
            record['source'] == context['source'] and record['execution_sha256'] == context['args'].execution_sha256 and
            record['numerical_flags'] == legacy['selected']['source_cpu']['numerical_flags'] and
            record['invocation']['argv'] == cli(context['root'], record['authority']['path'], record['authority']['sha256'],
                context['args'].execution_sha256, phase, arm, Path(unit['receipt']['path']).parent) and
            all(record['invocation'][k] == legacy['selected']['source_cpu']['invocation'][k]
                for k in ('python', 'python_sha256', 'python_version')),
            'new exact terminal authority/source/CLI differs')
    require(all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items()) and
            record['input_guards'].get(record['authority']['path']) == record['authority']['sha256'],
            'new terminal complete original/new input guards differ')
    final = legacy['admission'].admit_terminal(record, unit, policy(phase)['seconds'], guards)
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        old.zero_events(value)
    require(unit['invocation_id'] not in legacy['invocations'], 'duplicate new unit invocation')
    legacy['invocations'].add(unit['invocation_id'])
    context['terminals'][phase + ':' + arm] = record
    context['terminal_cgroups'][phase + ':' + arm] = final
    for path, digest in record['input_guards'].items():
        bound_file(guards, path, digest)
    return record


def fit_run(context):
    args, legacy = context['args'], context['legacy']
    total_core, facts = 0., {}
    for arm in ARMS if args.phase == 'cpu' else (args.arm,):
        cores = []
        state = fresh(context, arm)
        ident = identity(context, state)
        if args.phase == 'fit':
            require(ident == context['terminals']['cpu:linear']['arms'][arm]['identity'],
                    'fit differs from actual fullTRAIN CPU qualification')
        integrity(context, state, ident)
        cores.append(state['core_seconds'])
        digest = legacy['original'].fingerprint(payload(context, state, ident))
        tamper = tamper_witness(context, state, ident)
        release(context, state)
        # Second pass rereads warm/source/input bytes and refits before seeing
        # any first-pass fitted tensor, prototype or sufficient statistic.
        state = fresh(context, arm)
        cores.append(state['core_seconds'])
        require(identity(context, state) == ident, 'independently refitted coefficients/means/typed state differ')
        verify_digest(legacy['original'].fingerprint, payload(context, state, ident), digest,
                      'complete second independent fit state differs')
        path = args.output / ('resume.pt' if args.phase == 'fit' else arm + '-qualification.pt')
        sha, saved_digest = save(context, state, ident, path)
        require(saved_digest == digest, 'save changed complete fitted payload')
        release(context, state)
        state = reload(context, path, sha, digest, ident)
        fact = {'identity': ident, 'fit_witness': state['fit_witness'], 'terminal_state_sha256': digest,
                'output_witness_sha256': ident['output_witness_sha256'], 'strict_reload_exact': True,
                'independent_refit_exact': True, 'bypass_version_tamper_rejected': tamper, 'fit_passes': 2,
                'fit_core_seconds': cores}
        integrity(context, state, ident)
        release(context, state)
        facts[arm] = fact
        total_core += sum(cores)
        if args.phase == 'cpu':
            path.unlink()
            context['guards'].pop(str(path))
        else:
            fact['checkpoint'] = {'path': str(path), 'sha256': sha}
    return {'total_fit_core_seconds': total_core, **({'arms': facts, 'checkpoint': None} if args.phase == 'cpu' else facts[args.arm])}


def exit_rehash(context):
    context['old'].exit_rehash(context['legacy'])
    for path, digest in {**context['legacy']['guards'], **context['guards']}.items():
        bound_file({}, path, digest)
    require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'], 'new exit code changed')


def run(args):
    started = time.perf_counter()
    require(sys.argv == cli(Path(__file__).absolute().parent, args.authority, args.authority_sha256,
            args.execution_sha256, args.phase, args.arm, args.output), 'fixed new canonical CLI order required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'hidden CPU original systemd launch required')
    context = authority(args)
    old, legacy = context['old'], context['legacy']
    source, prior = legacy['source_driver'], legacy['selected']['source_cpu']['invocation']
    before = source.cgroup_memory()
    old.zero_events(before)
    unit = Path(before['path']).name.removesuffix('.service')
    legacy['admission'].init.admit_cgroup(before, unit)
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and legacy['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'] and os.environ['INVOCATION_ID'] not in legacy['invocations'],
            'original interpreter/fresh invocation differs')
    import torch
    cpu_rng = torch.random.get_rng_state().clone()
    args.output.mkdir()
    result = fit_run(context)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized(), 'fit changed RNG/CUDA')
    exit_rehash(context)
    after = source.cgroup_memory()
    old.zero_events(after)
    legacy['admission'].init.admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and
            0 < rss <= 8 * 1024**2 and int(after['values']['memory.peak']) >= int(before['values']['memory.peak']),
            'new complete unit resource cap differs')
    guards = {**legacy['guards'], **context['guards']}
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'output': str(args.output),
        'launch': context['launch'], 'execution_sha256': args.execution_sha256, 'code': context['code'],
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'authority_sha256': args.authority_sha256,
        'source': context['source'], 'partition_sha256': PARTITION_SHA, 'numerical_flags': context['flags'],
        'pass': True, 'quality_read': False, 'fit_qualified': args.phase == 'fit', 'public_encoder_qualified': False,
        'strict_reload_exact': True, 'exit_rehash_pass': True, 'frozen_complement_exact': True,
        'training_only_fit': True, 'zero_A_source_parity': True, 'bypass_version_tamper_rejected': True,
        'sequential_model_ownership': True, 'cuda_initialized': False, 'peak_cuda_allocated_bytes': 0,
        'resource_policy': policy(args.phase), 'wall_seconds': wall, 'process_peak_rss_kib': rss,
        'cgroup_before': before, 'cgroup_after': after, 'origins': legacy['origins'], 'input_guards': guards,
        'terminal_cgroups': {**legacy['terminal_cgroups'], **context['terminal_cgroups']},
        'both_locks_held_in_parent_authority': True, 'terminal_exit_and_both_locks_require_parent_receipt': True,
        **result, 'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
            'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
            'invocation_id': os.environ['INVOCATION_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    check_terminal_record(receipt, context['launch'], args.phase, args.arm)
    legacy['admission'].init.write_json(legacy['extract'], args.output / 'receipt.json', receipt)
    require(time.perf_counter() - started < policy(args.phase)['seconds'], 'new receipt included cap exceeded')
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--phase', choices=('cpu', 'fit'), required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Prototype residual rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
