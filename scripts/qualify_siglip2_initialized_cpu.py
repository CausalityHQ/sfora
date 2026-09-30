#!/usr/bin/env python3
"""Qualify one complete FRESH native256 PCA128 initialized CPU state; zero updates.

Own execution.json has exactly FILES: this driver, its test, and an unchanged
bare copy of src/sfora/joint_relational_compaction.py (PACKING_HELPER_SHA256).
Original initializer3, exporter2 and sourceCPU3 closures remain independent.
Native imports occur only after SHA-authenticated original initializer.authority
and admit_terminal have admitted actual source/FIT/PCA bytes and original exits.

Parent launch native256-initialized-cpu-launch-v1 has exactly:
 schema, execution_sha256, packing_helper_sha256, initializer_root,
 initializer_execution_sha256, initializer_authority:{path,sha256},
 selected_initializer:{receipt:{path,sha256},log:{path,sha256},unit,
 invocation_id,service_seconds,native_peak_rss_kib,both_locks_held:true},
 resource_policy:POLICY, both_locks_held:true. All hashes are actual frozen bytes.
CLI order (also required in original PCA receipts): --execution-sha256 SHA
 --authority PATH --authority-sha256 SHA --arm {large,so400} --output NEWDIR.
Output: exclusive initialized.pt and proof.json, conditional on the parent's
whole-unit normal exit, final cgroup footer, 120s/8GiB/noSwap/CUDA hidden and
BOTH external locks. No training/mechanics/quality claim, PCA rerun or rescue.

API: context=authority(args) BEFORE native imports; fresh(context,seed,
 device='cpu') verifies fresh_source and actual sourceCPU first2 pixels/runtime
BEFORE model/head.train(), returns model/head/processor/inventory/classifier/
bank/target/params/optimizer/scaler/counter/schedules/schedule_facts/witness.
Seeds are 179032/179041; both B64x100 dense FIT ordinal schedules are bound.
CPU scaler is None; a later independently admitted CUDA caller gets fresh128.
Common augmentation seed179032*100000+step1..100 is reserved, never executed.
raw_features(model,head,pixels) is width-generic FP32 head(normalize(pooled)),
outside autocast; it does not change the existing compact_head_features API.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import struct
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'siglip2-substrate-initialized-cpu-v1'
AUTHORITY_SCHEMA = 'native256-initialized-cpu-launch-v1'
FILES = {'qualify_siglip2_initialized_cpu.py', 'test_siglip2_initialized_cpu.py',
         'joint_relational_compaction.py'}
INITIALIZER_FILES = {'initialize_siglip2_substrate_fit.py', 'test_siglip2_substrate_initializer.py',
                     'representation_ceiling.py'}
PACKING_HELPER_SHA256 = '4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67'
POLICY = {'seconds': 120, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0, 'cuda_visible_devices': ''}
NATIVE_PACKAGES = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision'}
WIDTHS = {'large': 1024, 'so400': 1152}
SEEDS = (179032, 179041)
AUGMENTATION = {'seed': 179032, 'formula': '179032*100000+step', 'steps': [1, 100],
                'applied_images': 0, 'reserved_for_future_trainer': True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, 'duplicate JSON key')
            result[name] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None, 'SHA256 required')
    with path.open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected, 'bound file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(path, expected, guards):
    path = bound_file(guards, path, expected)
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected, 'JSON size/SHA256 differs')
    return strict_json(raw)


def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json(root / 'execution.json', expected, guards)
    require(isinstance(code, dict) and code.keys() == names, 'execution requires exactly declared files')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def load_bare(name, path, expected):
    path = bound_file({}, path, expected)
    require(name not in sys.modules, 'helper already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'bare helper origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'helper SHA256 changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == path and Path(module.__spec__.origin) == path, 'loaded helper origin differs')
    bound_file({}, path, expected)
    return module


def bootstrap(args):
    guards, root = {}, Path(__file__).absolute().parent
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json(args.authority, args.authority_sha256, guards)
    require(launch.keys() == {'schema', 'execution_sha256', 'packing_helper_sha256', 'initializer_root',
                             'initializer_execution_sha256', 'initializer_authority', 'selected_initializer',
                             'resource_policy', 'both_locks_held'} and
            launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
            launch['packing_helper_sha256'] == code['joint_relational_compaction.py'] == PACKING_HELPER_SHA256 and
            launch['resource_policy'] == POLICY and launch['both_locks_held'] is True,
            'parent initialized launch authority/profile/locks differ')
    pca_root = Path(launch['initializer_root'])
    pca_code = closure(pca_root, launch['initializer_execution_sha256'], INITIALIZER_FILES, guards)
    require(root != pca_root and not root.is_relative_to(pca_root) and not pca_root.is_relative_to(root) and
            args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
            not any(args.output.is_relative_to(path) for path in (root, pca_root)) and
            not args.output.exists() and not args.output.is_symlink(), 'separate exclusive canonical output required')
    name = '_siglip2_pinned_initialized_pca'
    if name in sys.modules:
        init = sys.modules[name]
        require(Path(init.__file__) == pca_root / 'initialize_siglip2_substrate_fit.py' and
                Path(init.__spec__.origin) == Path(init.__file__), 'pinned initializer origin differs')
        bound_file(guards, init.__file__, pca_code['initialize_siglip2_substrate_fit.py'])
    else:
        init = load_bare(name, pca_root / 'initialize_siglip2_substrate_fit.py',
                         pca_code['initialize_siglip2_substrate_fit.py'])
    require(init.FILES == INITIALIZER_FILES and init.POLICY == POLICY and init.WIDTHS == WIDTHS,
            'original initializer closure/profile differs')
    return init, code, guards, launch


def array_shapes(width):
    return {'mean': [width], 'components': [128, width], 'head.weight': [128, width], 'head.bias': [128],
            'classifier': [2004, 128], 'bank': [13283, 128], 'target': [13283]}


def validate_arrays(init, path, expected, width, facts, targets, guards):
    """Authenticate all NPZ member payloads before NumPy/Torch, with bounded reads."""
    path = init.bound_file(guards, path, expected)
    shapes = array_shapes(width)
    require(facts.keys() == shapes.keys(), 'exact array facts required')
    pca_parts = {}
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        require(len(files) == 7 and {entry.filename for entry in files} == {name + '.npy' for name in shapes} and
                all(entry.compress_type == zipfile.ZIP_STORED and not entry.flag_bits & 1 for entry in files),
                'exact array archive inventory/layout required')
        for name, shape in shapes.items():
            fact = facts[name]
            dtype, itemsize = ('torch.int64', 8) if name == 'target' else ('torch.float32', 4)
            require(fact.keys() == {'dtype', 'shape', 'sha256'} and fact['shape'] == shape and fact['dtype'] == dtype,
                    'array shape/dtype fact differs: ' + name)
            init.digest_string(fact['sha256'])
            count = math.prod(shape)
            entry = archive.getinfo(name + '.npy')
            require(count * itemsize < entry.file_size <= count * itemsize + 65548, 'array bounded member size differs')
            with archive.open(entry) as stream:
                magic = stream.read(8)
                require(magic in (b'\x93NUMPY\x01\x00', b'\x93NUMPY\x02\x00'), 'array NPY version differs')
                size = 2 if magic[-2] == 1 else 4
                length = int.from_bytes(stream.read(size), 'little')
                require(0 < length <= 65536, 'array NPY header bound differs')
                header = ast.literal_eval(stream.read(length).decode('latin1'))
                require(header.keys() == {'descr', 'fortran_order', 'shape'} and
                        header['descr'] == ('<i8' if name == 'target' else '<f4') and
                        header['shape'] == tuple(shape) and header['fortran_order'] is False,
                        'array shape/dtype/header differs: ' + name)
                raw = stream.read(count * itemsize + 1)
                require(len(raw) == count * itemsize and stream.read(1) == b'' and
                        hashlib.sha256(raw).hexdigest() == fact['sha256'], 'array byte size/hash differs: ' + name)
            if name == 'target':
                require([value[0] for value in struct.iter_unpack('<q', raw)] == targets, 'original FIT target array differs')
            else:
                require(all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)), 'array nonfinite: ' + name)
            if name in ('mean', 'components', 'head.weight'):
                pca_parts[name] = raw
    require(pca_parts['components'] == pca_parts['head.weight'], 'PCA/head weight array bytes differ')
    init.bound_file(guards, path, expected)
    return hashlib.sha256(pca_parts['mean'] + pca_parts['components']).hexdigest()


def admit_pca(init, pca, launch, args):
    """Match the actual original receipt, with no unqualified tensor claim."""
    record = init.descriptor_json(launch['selected_initializer']['receipt'], pca['guards'])
    expected = {'schema': init.SCHEMA, 'phase': 'pca', 'arm': args.arm, 'width': WIDTHS[args.arm],
                'output_dim': 128, 'authority_sha256': launch['initializer_authority']['sha256'],
                'execution_sha256': launch['initializer_execution_sha256'], 'code': pca['code'],
                'pca_helper_sha256': pca['launch']['pca_helper_sha256'],
                'source_binding': pca['export']['binding'],
                'source_checkpoint_metadata_only': pca['export']['source_checkpoint_metadata_only'],
                'source_roles': pca['source_context']['proof']['runtime']['roles'],
                'source_roles_sha256': pca['exporter'].object_sha(pca['source_context']['proof']['runtime']['roles']),
                'cache': pca['export']['cache'], 'cache_facts_before_native': pca['cache_facts_before_native'],
                'ordered_input_sha256': pca['export']['ordered_input_sha256'],
                'ordered_rgb_sha256': pca['export']['ordered_rgb_sha256'],
                'class_names': pca['source_context']['fit']['class_names'], 'resource_policy': POLICY,
                'numerical_flags': pca['source_context']['proof']['numerical_flags'],
                'export_final_cgroup': pca['export_final_cgroup'],
                'startup_final_cgroup': pca['startup_final_cgroup'],
                'arithmetic': {
                    'pca': 'CPU F.normalize(FP32 cache, dim=1); original centered float64 full SVD/rank/degeneracy/maxabs-positive signs; FP32 mean/components',
                    'head': 'fork_rng CPU nn.Linear(width,128); W=components; bias=-(components@mean)',
                    'proxies': 'original pca.apply(normalized): float64 projection/norm -> FP32; class-ordered sequential FP32 sums/int64 counts; F.normalize(sums,dim=1)',
                    'bank': 'no_grad/outside autocast; head(F.normalize(cache.float(),dim=1)); F.normalize(dim=1); detach'}}
    for key in ('export_root', 'export_execution_sha256', 'export_authority', 'selected_export', 'startup'):
        expected[key] = pca['launch'][key]
    require(all(record[key] == value for key, value in expected.items()), 'actual PCA/source/FIT provenance differs')
    yes = ('pass', 'prepared', 'serialized_arrays_exact', 'constructor_rng_preserved', 'exit_rehash_pass',
           'all_arrays_finite', 'bank_detached', 'preparation_cost_only', 'both_locks_held_in_parent_authority',
           'terminal_exit_and_both_locks_require_parent_receipt')
    no = ('gradients_created', 'optimizer_created', 'source_model_loaded', 'teacher_state_reused',
          'source_features_reused', 'proxies_reused', 'head_reused', 'quality_read', 'initializer_qualified',
          'training_qualified', 'quality_qualified', 'cuda_initialized')
    require(all(record[key] is True for key in yes) and all(record[key] is False for key in no) and
            record['updates'] == record['head_updates'] == 0 and record['cpu_rng_before'] == record['cpu_rng_after'] and
            0 < record['pca_seconds'] <= record['wall_seconds'], 'actual PCA profile/counters/RNG differ')
    targets = pca['source_context']['fit']['targets']
    counts = [0] * 2004
    for target in targets:
        counts[target] += 1
    require(record['class_counts'] == counts, 'actual PCA FIT class counts differ')
    shapes = array_shapes(WIDTHS[args.arm])
    require(record['initializer_roles'] == [{'name': name, 'shape': shapes[name], 'dtype': 'torch.float32',
                                           'role': 'trainable'} for name in ('head.weight', 'head.bias', 'classifier')],
            'actual PCA initializer roles differ')
    receipt = init.canonical(launch['selected_initializer']['receipt']['path'])
    require(receipt.name == 'receipt.json' and record['artifact'].keys() == {'path', 'sha256'} and
            Path(record['artifact']['path']) == receipt.parent / 'initializers.npz', 'PCA artifact path role differs')
    prior, identity = pca['startup']['invocation'], record['invocation']
    require(identity['argv'] == [str(pca['root'] / 'initialize_siglip2_substrate_fit.py'),
            '--execution-sha256', launch['initializer_execution_sha256'], '--authority', launch['initializer_authority']['path'],
            '--authority-sha256', launch['initializer_authority']['sha256'], '--arm', args.arm, '--output', str(receipt.parent)] and
            identity['cuda_visible_devices'] == '' and
            all(identity[key] == prior[key] for key in ('python', 'python_sha256', 'python_version')),
            'original PCA argv/interpreter differs')
    require(all(record['input_guards'].get(path) == digest for path, digest in pca['guards'].items()
                if path != str(receipt)), 'original PCA input guards differ')
    origins = record['origins']
    require(origins['packages'] == pca['packages'] and set(origins['native_files']) <= origins['files'].keys(),
            'PCA native packages/origins differ')
    for name, path in origins['modules'].items():
        package = name.split('.')[0]
        require(package in origins['packages'] and
                Path(path).is_relative_to(Path(origins['packages'][package]['root'])) and
                path in origins['files'], 'PCA loaded module origin differs')
    require(all(record['input_guards'].get(path) == digest for path, digest in origins['files'].items()),
            'PCA native file guards differ')
    for path, digest in record['input_guards'].items():
        init.bound_file(pca['guards'], path, digest)
    final = init.admit_terminal(record, launch['selected_initializer'], 120, pca['guards'])
    digest = validate_arrays(init, record['artifact']['path'], record['artifact']['sha256'], WIDTHS[args.arm],
                             record['arrays'], targets, pca['guards'])
    require(digest == record['pca_sha256'], 'actual complete PCA digest differs')
    return record, final


def authority(args):
    require(not any(name.split('.')[0] in NATIVE_PACKAGES or name == 'sfora' for name in sys.modules),
            'native packages must not precede authority admission')
    init, code, guards, launch = bootstrap(args)
    init.descriptor_json(launch['initializer_authority'], guards)
    pca_args = SimpleNamespace(execution_sha256=launch['initializer_execution_sha256'],
        authority=Path(launch['initializer_authority']['path']), authority_sha256=launch['initializer_authority']['sha256'],
        arm=args.arm, output=args.output)
    pca = init.authority(pca_args)
    record, final = admit_pca(init, pca, launch, args)
    for path, digest in guards.items():
        require(pca['guards'].setdefault(path, digest) == digest, 'conflicting initialized input authority')
    root = Path(__file__).absolute().parent
    external = (pca['root'], pca['source_context']['root'], pca['source_context']['own_root'])
    require(all(not root.is_relative_to(path) and not path.is_relative_to(root) and
                not args.output.is_relative_to(path) for path in external), 'all original closures must remain separate')
    source_context = pca['source_context']
    source_context['packages'] = pca['packages']
    require(not any(name.split('.')[0] in NATIVE_PACKAGES or name == 'sfora' for name in sys.modules),
            'authority imported native packages')
    return {'args': args, 'root': root, 'code': code, 'guards': pca['guards'], 'launch': launch,
            'init': init, 'pca': pca, 'record': record, 'pca_final_cgroup': final,
            'source': pca['source'], 'source_context': source_context, 'packages': pca['packages']}


def rehash(context):
    require(context['guards'] is context['pca']['guards'] is context['source_context']['guards'],
            'complete exit guard inventory must remain shared')
    context['init'].rehash(context['pca'])
    require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'],
            'exit initialized closure differs')


def schedule(target, seed):
    """Identical operations to pe_large_coverage.schedule(target, seed), first100."""
    import numpy as np
    require(seed in SEEDS, 'fixed schedule seed required')
    target = np.asarray(target, dtype=np.int64)
    names = sorted(set(target.tolist()))
    require(target.shape == (13283,) and names == list(range(2004)), 'dense FIT target inventory differs')
    rng = np.random.default_rng(seed)
    order = rng.permutation(names)
    members = {c: np.flatnonzero(target == c) for c in names}
    batches = []
    for step in range(100):
        classes = order[(step * 64 + np.arange(64)) % len(order)]
        batches.append([int(rng.choice(members[int(c)])) for c in classes])
    batches = np.asarray(batches, dtype=np.int64)
    require(batches.shape == (100, 64) and all(len(set(target[b].tolist())) == 64 for b in batches) and
            set(target[batches.ravel()].tolist()) == set(names) and 0 <= int(batches.min()) and
            int(batches.max()) < 13283, 'class-balanced dense FIT ordinal schedule differs')
    return batches


def raw_features(model, head, pixels):
    import torch
    from torch.nn import functional as F
    pooled = model(pixel_values=pixels).pooler_output
    with torch.autocast(device_type=pixels.device.type, enabled=False):
        raw = head(F.normalize(pooled.float(), dim=1))
    require(raw.dtype == torch.float32 and raw.shape == (len(pixels), 128) and torch.isfinite(raw).all().item() and
            (raw.norm(dim=1) > 0).all().item(), 'generic raw compact output differs')
    return raw


def optimizer_state(model, head, classifier):
    import torch
    params = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    params += [('compact_head.' + name, value) for name, value in head.named_parameters()] + [('classifier', classifier)]
    require(len(params) == 208 and len({id(value) for _, value in params}) == 208 and
            all(value.requires_grad and value.dtype == torch.float32 and value.grad is None for _, value in params),
            '208 FP32 trainable optimizer members required')
    optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-5},
                                   {'params': head.parameters(), 'lr': 1e-4},
                                   {'params': [classifier], 'lr': 1e-4}], weight_decay=.05)
    defaults = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and
            [id(p) for group in optimizer.param_groups for p in group['params']] == [id(p) for _, p in params],
            'original AdamW defaults/order/fresh empty state differ')
    require(all({k: v for k, v in group.items() if k != 'params'} == {**defaults, 'lr': rate}
                for group, rate in zip(optimizer.param_groups, (1e-5, 1e-4, 1e-4))), 'original AdamW groups differ')
    return params, optimizer


def load_initializers(context):
    import numpy as np
    import torch
    from torch.nn import functional as F
    record, source = context['record'], context['source']
    with np.load(record['artifact']['path'], allow_pickle=False) as arrays:
        require(set(arrays.files) == set(record['arrays']), 'complete native initializer inventory differs')
        values = {name: torch.from_numpy(arrays[name].copy()) for name in arrays.files}
    require({name: source.tensor_fact(value) for name, value in values.items()} == record['arrays'],
            'native initializer shape/dtype/hash differs')
    require(torch.equal(values['head.weight'], values['components']) and
            torch.equal(values['head.bias'], -(values['components'] @ values['mean'])) and
            values['target'].tolist() == context['source_context']['fit']['targets'] and
            all(torch.allclose(values[name].norm(dim=1), torch.ones(len(values[name])), rtol=0, atol=1e-5)
                for name in ('bank', 'classifier')), 'native affine/target/proxy/bank geometry differs')
    # Verify the DISTINCT original FP32 bank path; never reconstruct proxies with it.
    cache = np.load(context['pca']['export']['cache']['path'], allow_pickle=False)
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        computed = F.normalize(F.linear(F.normalize(torch.from_numpy(cache).float(), dim=1),
                                       values['head.weight'], values['head.bias']), dim=1)
    require(torch.equal(computed, values['bank']), 'original actual FP32 head bank path differs')
    return values


def head_from(values, width):
    import torch
    with torch.random.fork_rng(devices=[]):
        head = torch.nn.Linear(width, 128).float()
    head.load_state_dict({'weight': values['head.weight'], 'bias': values['head.bias']}, strict=True)
    return head


def fresh(context, seed, device='cpu'):
    """Reusable fresh constructor, never a trained-state loader; no update."""
    import torch
    from torch.nn import functional as F
    require(seed in SEEDS and str(device) in ('cpu', 'cuda', 'cuda:0'), 'fixed seed/device required')
    source, src = context['source'], context['source_context']
    flags, rng = source.numerical_flags(), torch.random.get_rng_state().clone()
    require(flags == src['proof']['numerical_flags'] and not torch.cuda.is_initialized(), 'source CPU defaults/CUDA differ')
    model, processor, roles = source.fresh_source(src)
    pixels, pooled, sample = source.pixels_and_raw(src, model, processor)
    runtime = source.model_facts(model, processor, roles, context['packages'])
    require(runtime == src['proof']['runtime'] and sample == src['proof']['sample'],
            'fresh source CPU actual runtime/first2 differ')
    values = load_initializers(context)
    head = head_from(values, WIDTHS[context['args'].arm])
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        raw = head(F.normalize(pooled.float(), dim=1))
    packing = load_bare('_siglip2_pinned_initialized_packing', context['root'] / 'joint_relational_compaction.py',
                        context['launch']['packing_helper_sha256']) if '_siglip2_pinned_initialized_packing' not in sys.modules else sys.modules['_siglip2_pinned_initialized_packing']
    require(Path(packing.__file__) == context['root'] / 'joint_relational_compaction.py' and
            Path(packing.__spec__.origin) == Path(packing.__file__), 'packing helper origin differs')
    context['init'].bound_file({}, packing.__file__, PACKING_HELPER_SHA256)
    packed = packing.pack_int8_unit_embeddings(F.normalize(raw, dim=1))
    schedules = {str(s): torch.from_numpy(schedule(values['target'].tolist(), s)) for s in SEEDS}
    schedule_facts = {s: source.tensor_fact(value) for s, value in schedules.items()}
    model.to(device).train()
    head.to(device).train()
    classifier = torch.nn.Parameter(values['classifier'].to(device))
    bank, target = values['bank'].to(device).detach(), values['target'].to(device)
    params, optimizer = optimizer_state(model, head, classifier)
    scaler = torch.amp.GradScaler('cuda', init_scale=128) if str(device).startswith('cuda') else None
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and
            all(p.grad is None for p in model.parameters()) and not bank.requires_grad,
            'fresh initialized constructor changed RNG/flags/grads')
    if scaler is not None:
        require(scaler.state_dict() == {'scale': 128, 'growth_factor': 2.0, 'backoff_factor': .5,
                                      'growth_interval': 2000, '_growth_tracker': 0}, 'fresh128 scaler differs')
    return {'model': model, 'head': head, 'processor': processor, 'inventory': roles, 'classifier': classifier,
            'bank': bank, 'target': target, 'params': params, 'optimizer': optimizer, 'scaler': scaler, 'counter': 0,
            'schedules': schedules, 'schedule_facts': schedule_facts, 'seed': seed,
            'witness': {'pixels': pixels, 'raw': raw, 'codes': packed.codes, 'inverse_norms': packed.inverse_norms,
                        'wire_sha256': hashlib.sha256(packed.to_bytes()).hexdigest(), 'source_sample': sample},
            'pca': {name: values[name] for name in ('mean', 'components')}}


def state_facts(context, state):
    import torch
    source, model, head = context['source'], state['model'], state['head']
    require(state['counter'] == 0 and state['scaler'] is None and model.training and head.training and
            all(module.training for _, module in model.named_modules()) and
            all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in head.parameters()) and
            state['classifier'].grad is None and not state['bank'].requires_grad,
            'initialized modes/counters/scaler/grads/detached bank differ')
    model.eval()
    try:
        runtime = source.model_facts(model, state['processor'], state['inventory'], context['packages'])
    finally:
        model.train()
    require(runtime == context['source_context']['proof']['runtime'], 'whole fresh vision/runtime/buffers changed')
    arrays = {**state['pca'], **{'head.' + n: v for n, v in head.state_dict().items()},
              'classifier': state['classifier'], 'bank': state['bank'], 'target': state['target']}
    require({name: source.tensor_fact(value) for name, value in arrays.items()} == context['record']['arrays'],
            'complete initialized finite FP32/target arrays changed')
    params, optimizer = state['params'], state['optimizer']
    actual_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    actual_params += [('compact_head.' + n, p) for n, p in head.named_parameters()] + [('classifier', state['classifier'])]
    require([n for n, _ in params] == [n for n, _ in actual_params] and
            [id(p) for _, p in params] == [id(p) for _, p in actual_params] and
            [id(p) for group in optimizer.param_groups for p in group['params']] == [id(p) for _, p in params] and
            len(params) == 208 and not optimizer.state and optimizer.state_dict()['state'] == {},
            'actual fresh optimizer membership/state differs')
    schedules = {str(seed): torch.from_numpy(schedule(state['target'].tolist(), seed)) for seed in SEEDS}
    require(all(torch.equal(state['schedules'][s], value) for s, value in schedules.items()) and
            {s: source.tensor_fact(v) for s, v in schedules.items()} == state['schedule_facts'],
            'both dense FIT seeded schedules changed')
    return {'runtime': runtime, 'training_modules': [(n, m.training) for n, m in model.named_modules()],
            'head_training': head.training, 'arrays': context['record']['arrays'], 'parameter_names': [n for n, _ in params],
            'optimizer_defaults': optimizer.defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups],
            'optimizer_member_order': [[next(n for n, v in params if v is p) for p in g['params']]
                                       for g in optimizer.param_groups],
            'optimizer_state': optimizer.state_dict(), 'schedules': state['schedule_facts'],
            'seed': state['seed'], 'counter': 0, 'augmentation': AUGMENTATION}


def payload(state, facts, rng, flags):
    return {'schema': SCHEMA, 'vision': state['model'].state_dict(), 'buffers': dict(state['model'].named_buffers()),
            'config': facts['runtime']['config'], 'head': state['head'].state_dict(),
            'classifier': state['classifier'].detach(), 'bank': state['bank'], 'target': state['target'],
            'pca': state['pca'], 'schedules': state['schedules'], 'optimizer': state['optimizer'].state_dict(),
            'optimizer_defaults': state['optimizer'].defaults.copy(), 'scaler': None, 'cpu_rng': rng,
            'numerical_flags': flags, 'counter': state['counter'], 'seed': state['seed'], 'facts': facts}


def reload_independent(context, checkpoint, expected_sha, first, rng, flags):
    import torch
    from transformers import AutoImageProcessor
    source, src = context['source'], context['source_context']
    context['init'].bound_file(context['guards'], checkpoint, expected_sha)
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    require(disk.keys() == {'schema', 'vision', 'buffers', 'config', 'head', 'classifier', 'bank', 'target', 'pca',
                           'schedules', 'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng', 'numerical_flags',
                           'counter', 'seed', 'facts'} and disk['schema'] == SCHEMA and disk['facts'] == first and
            disk['config'] == first['runtime']['config'] and disk['numerical_flags'] == flags and
            torch.equal(disk['cpu_rng'], rng) and disk['counter'] == 0 and disk['scaler'] is None and
            disk['optimizer'] == first['optimizer_state'] and disk['optimizer_defaults'] == first['optimizer_defaults'],
            'complete initialized serialization differs')
    model = source.construct(disk['config'], src)  # Independent config clone; never another pretrained source.
    model.load_state_dict(disk['vision'], strict=True)
    buffers = dict(model.named_buffers())
    require(buffers.keys() == disk['buffers'].keys(), 'complete independent buffer inventory differs')
    with torch.no_grad():
        for name, value in buffers.items():
            require(source.tensor_fact(value) == source.tensor_fact(disk['buffers'][name]),
                    'independent nonpersistent buffer differs: ' + name)
            value.copy_(disk['buffers'][name])
    roles = source.configure_roles(model, src['expected'], model.config.num_hidden_layers)
    processor = AutoImageProcessor.from_pretrained(src['entry']['input']['preprocessor']['path'],
                                                   local_files_only=True, backend='torchvision')
    # Recheck actual unaugmented FIT bytes/pixels/source raw before train mode.
    pixels, pooled, sample = source.pixels_and_raw(src, model, processor)
    require(sample == src['proof']['sample'], 'independent actual sourceCPU pixels/raw differ')
    head = head_from({'head.weight': disk['head']['weight'], 'head.bias': disk['head']['bias']}, WIDTHS[context['args'].arm])
    classifier = torch.nn.Parameter(disk['classifier'].clone())
    model.train()
    head.train()
    params, optimizer = optimizer_state(model, head, classifier)
    require(optimizer.state_dict() == disk['optimizer'], 'independent fresh optimizer groups/state differ')
    optimizer.load_state_dict(disk['optimizer'])
    state = {'model': model, 'head': head, 'processor': processor, 'inventory': roles, 'classifier': classifier,
             'bank': disk['bank'].clone().detach(), 'target': disk['target'].clone(),
             'pca': {n: v.clone() for n, v in disk['pca'].items()},
             'schedules': {n: v.clone() for n, v in disk['schedules'].items()}, 'schedule_facts': first['schedules'],
             'params': params, 'optimizer': optimizer, 'scaler': None, 'counter': 0, 'seed': disk['seed']}
    del disk, buffers, value, pooled
    gc.collect()
    require(state_facts(context, state) == first and torch.equal(torch.random.get_rng_state(), rng) and
            source.numerical_flags() == flags, 'independent complete state/runtime/RNG/defaults reload differs')
    return state, pixels


def qualify(args):
    started = time.perf_counter()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.flags.optimize == 0,
            'initialized CPU requires explicit CUDA hidden/unoptimized Python')
    identity = os.environ.get('INVOCATION_ID')
    require(re.fullmatch('[0-9a-f]{32}', identity or '') is not None, 'original systemd invocation ID required')
    context = authority(args)
    source, init = context['source'], context['init']
    # Release consumed authority-file cache before construction, including legacy PCA admission.
    # The shared source rehash verifies every byte with the frozen extractor's bounded advice.
    rehash(context)
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    init.admit_cgroup(before, unit)
    python, prior = init.canonical(Path(sys.executable).resolve()), context['pca']['startup']['invocation']
    require(str(python) == prior['python'] and init.sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'initialized interpreter differs')
    init.bound_file(context['guards'], python, prior['python_sha256'])
    args.output.mkdir()
    import torch
    from torch.nn import functional as F
    require(not torch.cuda.is_initialized(), 'initialized qualification is CPU-only')
    flags = context['source_context']['proof']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original CPU numerical flags differ')
    rng = torch.random.get_rng_state().clone()
    state = fresh(context, SEEDS[0])
    first = state_facts(context, state)
    witness = {name: value.clone() for name, value in state['witness'].items() if isinstance(value, torch.Tensor)}
    wire_sha = state['witness']['wire_sha256']
    # Compare the actual TRAIN-mode raw path too; no autograd or updates.
    with torch.no_grad():
        train_raw = raw_features(state['model'], state['head'], witness['pixels'])
    require(torch.equal(train_raw, witness['raw']), 'fresh eval/train first2 raw differs')
    del train_raw
    checkpoint = args.output / 'initialized.pt'
    saved = payload(state, first, rng, flags)
    with context['source_context']['extract'].exclusive(checkpoint) as stream:
        torch.save(saved, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checkpoint_sha = init.sha(checkpoint)
    # No first model/parameter/optimizer/payload references remain during clone construction.
    del saved, state
    gc.collect()
    independent, pixels = reload_independent(context, checkpoint, checkpoint_sha, first, rng, flags)
    with torch.no_grad():
        raw = raw_features(independent['model'], independent['head'], pixels)
    packing = sys.modules['_siglip2_pinned_initialized_packing']
    packed = packing.pack_int8_unit_embeddings(F.normalize(raw, dim=1))
    require(torch.equal(pixels, witness['pixels']) and torch.equal(raw, witness['raw']) and
            torch.equal(packed.codes, witness['codes']) and
            torch.equal(packed.inverse_norms.view(torch.int16), witness['inverse_norms'].view(torch.int16)) and
            hashlib.sha256(packed.to_bytes()).hexdigest() == wire_sha,
            'independent actual pixels/raw2x128/int8/FP16 inverse bits differ')
    raw_fact = source.tensor_fact(raw)
    packed_facts = {'codes': source.tensor_fact(packed.codes), 'inverse_norms': source.tensor_fact(packed.inverse_norms),
                    'wire_sha256': wire_sha}
    del independent, pixels, raw, packed
    gc.collect()
    origins = source.imported_origins(context['source_context']['extract'], context['packages'])
    for path, digest in origins['files'].items():
        init.bound_file(context['guards'], path, digest)
    rehash(context)
    after = source.cgroup_memory()
    init.admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and int(before['values']['memory.peak']) <= int(after['values']['memory.peak']) and
            0 < wall < 120 and 0 < rss <= 8 * 1024**2 and not torch.cuda.is_initialized() and
            torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'complete initialized duration/RSS/cgroup/RNG/flags/CUDA differs')
    proof = {'schema': SCHEMA, 'phase': 'initialized-cpu', 'pass': True, 'initializer_qualified': True,
             'source_qualified': True, 'training_qualified': False, 'quality_qualified': False, 'quality_read': False,
             'arm': args.arm, 'width': WIDTHS[args.arm], 'output_dim': 128,
             'authority_sha256': args.authority_sha256, 'execution_sha256': args.execution_sha256, 'code': context['code'],
             'initializer_authority': context['launch']['initializer_authority'],
             'selected_initializer': context['launch']['selected_initializer'], 'pca_final_cgroup': context['pca_final_cgroup'],
             'source_binding': context['record']['source_binding'], 'source_sample': context['source_context']['proof']['sample'],
             'ordered_input_sha256': context['record']['ordered_input_sha256'],
             'ordered_rgb_sha256': context['record']['ordered_rgb_sha256'], 'initializers': context['record']['artifact'],
             'state': first, 'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_sha},
             'raw_first2': raw_fact, 'packed_first2': packed_facts, 'reload_exact': True,
             'first_model_released_before_independent_clone': True, 'fresh_source': True,
             'source_cpu_runtime_and_first2_exact': True, 'cpu_rng': source.tensor_fact(rng),
             'constructor_rng_preserved': True, 'numerical_flags': flags, 'origins': origins,
             'input_guards': context['guards'], 'exit_rehash_pass': True, 'updates': 0, 'head_updates': 0,
             'optimizer_state_entries': 0, 'gradients_created': False, 'optimizer_created': True,
             'pca_rerun': False, 'teacher_state_reused': False, 'trained_state_reused': False,
             'augmentation': AUGMENTATION, 'preparation_cost_only': True, 'cuda_initialized': False,
             'resource_policy': POLICY, 'cgroup_before': before, 'cgroup_after': after,
             'process_peak_rss_kib': rss, 'wall_seconds': wall,
             'both_locks_held_in_parent_authority': True,
             'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                            'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                            'invocation_id': identity, 'cuda_visible_devices': ''},
             'terminal_exit_and_both_locks_require_parent_receipt': True}
    context['pca']['exporter'].write_json(context['source_context']['extract'], args.output / 'proof.json', proof)
    return proof


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--arm', choices=sorted(WIDTHS), required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        result = qualify(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError,
            SyntaxError, struct.error, zipfile.BadZipFile) as error:
        raise SystemExit('Initialized CPU qualification rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
