#!/usr/bin/env python3
"""Prospective same-product mixing on matched five-tensor affine So400 heads.

Authority schema siglip2-identity-mix-launch-v1 has exactly LAUNCH_KEYS.
FILE = {path: canonical absolute file, sha256: lowercase SHA256}.
cached_reference = {root: separate immutable absolute directory,
execution_sha256: CACHED_EXECUTION_SHA}, the unchanged original TWO-file v2
cached trainer. original_authority is FILE for its ORIGINAL CPU launch, pinned
by ORIGINAL_AUTHORITY_SHA. Its original five-file reference must remain v6,
REFERENCE_EXECUTION_SHA. Original collected inventories, source/FIT/PCA/CPU
terminal predicates, package origins and uncached exit readers stay unchanged.
Old PCA/heads/proxies/banks are authenticated archive dependencies ONLY; never
loaded as new initializers. partition is FILE pinned by PARTITION_SHA.

selected_cpu is null for cpu; otherwise TERMINAL for an actually accepted NEW
CPU unit. selected_mechanics is null except train, when exactly
{control: TERMINAL, candidate: TERMINAL}. TERMINAL = {receipt: FILE, log: FILE,
unit, invocation_id, service_seconds, native_peak_rss_kib, both_locks_held:true}.
Original FINAL_CGROUP/footer/normal-exit predicates apply to each complete unit.
recipe == RECIPE, resource_policy == policy(phase), both_locks_held == true.
Parent supplies future file hashes, locks, systemd units and terminal descriptors
ONLY after their real completion; no pending or fabricated admission.

Canonical CLI order, absolute paths, unoptimized interpreter:
python -B ROOT/train_siglip2_identity_mix.py --execution-sha256 SHA
 --authority FILE --authority-sha256 SHA --phase cpu|mechanics|train
 --arm control|candidate --seed 179061|179069 --output NEW_DIRECTORY
CPU: control/179061, CUDA_VISIBLE_DEVICES='', 120s; both arms qualified.
Mechanics: each arm/179061, discarded17 vs independent8+9. TRAIN:1000 fresh
updates, only after NEW CPU and BOTH mechanics. GPU CUBLAS_WORKSPACE_CONFIG
=:4096:8, one visible device. Mechanics/TRAIN300s; ALL phases8GiB/noSwap,
zero disallowed memory events/CUDA allocation<10GB/both lifetime locks, no
peak reset; all admission/save/strict reload/uncached exit included.

CPU writes initializer.pt: complete NEW untrained initial tree plus native
raw/unit/CPU-packed and disposable objective/gradient/resume fixtures. Training
resume.pt has exactly PAYLOAD_KEYS, including full partition, original_rows,
initial head/buffers/PCA/proxies/bank/targets/positives/schedules/mixing tables,
live state, AdamW, scaler, RNG and counter. Every ordinal in schedules, donors,
bank and positives is LOCAL TRAIN; original_rows maps only6355 TRAIN rows to
the authenticated13283x1152 cache. Held features are never indexed for fitting.
Both heads use phi=.5*z. Candidate changes masked inputs only; both banks use
unmixed PRE-update detached descriptors and original last-duplicate semantics.
No scoring, images, old trained reuse or selection/validation quality reads.
Native qualification is UNRUN until actual original terminal receipts exist.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import statistics
import struct
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCHEMA = 'siglip2-identity-mix-v1'
AUTHORITY_SCHEMA = 'siglip2-identity-mix-launch-v1'
FILES = {'train_siglip2_identity_mix.py', 'test_siglip2_identity_mix.py'}
CACHED_FILES = {'train_siglip2_cached_readout.py', 'test_siglip2_cached_readout.py'}
CACHED_EXECUTION_SHA = '907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598'
CACHED_TRAINER_SHA = 'a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c'
REFERENCE_EXECUTION_SHA = '236327289f110ed2bf2012ea6e0d1ea48cf822b4e4094d9cf544081a2dc476f7'
ORIGINAL_AUTHORITY_SHA = 'b54a0b4737e66469aa0568c02deacc03b3fa58a904c1a24b1ae8bc9882efe914'
PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'
FIT_SHA = 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'
MANIFEST_SHA = 'd32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251'
ROWS, CLASSES, WIDTH, DIM = 6355, 1008, 1152, 128
SEEDS, ARMS = (179061, 179069), ('control', 'candidate')
PARAMETERS = ['compact_head.primary.weight', 'compact_head.primary.bias',
              'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
SHAPES = [(128, 1152), (128,), (32, 1152), (128, 32), (1008, 128)]
RECIPE = {'width': WIDTH, 'rows': ROWS, 'classes': CLASSES, 'output_dim': DIM,
          'rank': 32, 'initialization_seed': 179074, 'steps': 1000, 'batch': 64,
          'microbatch': 16, 'seeds': list(SEEDS), 'margin': .3, 'scale': 64,
          'rank_weight': 8, 'learning_rate': 1e-4, 'weight_decay': .05, 'clip': 1,
          'initial_scaler': 128, 'control': '0.5*z', 'candidate': '0.5*z',
          'mix_probability': .5, 'mix_rng': 'PCG64',
          'stream_offsets': [1000001, 2000001, 3000001],
          'lambda': 'float64 uniform[0,1) cast FP32',
          'bank': 'unmixed pre-update detached last duplicate',
          'encoder_updates': False, 'augmentation': False}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'cached_reference',
               'original_authority', 'partition', 'selected_cpu', 'selected_mechanics',
               'resource_policy', 'both_locks_held', 'recipe'}
STATIC_KEYS = ('pca', 'target', 'positive', 'schedules', 'mixing', 'original_rows', 'initializer', 'partition')
INITIAL_KEYS = {'head', 'classifier', 'bank', 'pca', 'target', 'positive', 'schedules', 'mixing', 'original_rows'}
PAYLOAD_KEYS = {'schema', 'identity', 'head', 'classifier', 'bank', *STATIC_KEYS,
                'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng', 'cuda_rng',
                'counter', 'seed', 'numerical_flags', 'source'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(phase):
    require(phase in ('cpu', 'mechanics', 'train'), 'fixed phase required')
    return {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'SHA256 required')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def descriptor_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'exact FILE descriptor required')
    raw = bound_file(guards, value['path'], value['sha256']).read_bytes()
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)


def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = descriptor_json({'path': str(root / 'execution.json'), 'sha256': expected}, guards)
    require(code.keys() == names, 'exact execution closure required')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and launch['seed'] == args.seed and args.arm in ARMS and
            type(args.seed) is int and args.seed in SEEDS and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'launch profile differs')
    require((args.phase != 'cpu' or (args.arm, args.seed) == ('control', SEEDS[0])) and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]), 'CPU/mechanics seed/arm differs')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if args.phase == 'train':
        require(isinstance(launch['selected_mechanics'], dict) and
                launch['selected_mechanics'].keys() == set(ARMS), 'BOTH mechanics required')
    require(launch['cached_reference'].keys() == {'root', 'execution_sha256'} and
            launch['cached_reference']['execution_sha256'] == CACHED_EXECUTION_SHA and
            launch['original_authority'].keys() == launch['partition'].keys() == {'path', 'sha256'} and
            launch['original_authority']['sha256'] == ORIGINAL_AUTHORITY_SHA and
            launch['partition']['sha256'] == PARTITION_SHA, 'original/partition pins differ')


def check_partition(partition, fit):
    require(partition.keys() == {'schema', 'global_class_names', 'original_cache', 'original_fit',
                                 'panels', 'partition_seeds'} and
            partition['schema'] == 'siglip2-identity-mix-partition-v1' and
            partition['original_cache']['sha256'] == FIT_SHA and
            partition['original_fit']['sha256'] == MANIFEST_SHA and
            partition['global_class_names'] == fit['class_names'] and
            len(fit['targets']) == 13283 and partition['partition_seeds'] == [179071, 179072] and
            partition['panels'].keys() == {'train', 'selection', 'validation'}, 'partition source differs')
    rows_seen, classes_seen = set(), set()
    for name, (count, classes) in {'train': (ROWS, CLASSES), 'selection': (3449, 498),
                                  'validation': (3479, 498)}.items():
        panel = partition['panels'][name]
        rows, ids = panel['original_rows'], panel['original_class_ids']
        require(panel.keys() == ({'original_rows', 'original_class_ids'} if name == 'train' else
                {'original_rows', 'original_class_ids', 'query', 'gallery'}) and
                len(rows) == count and len(ids) == classes and
                all(type(n) is int and 0 <= n < 13283 for n in rows) and
                all(type(n) is int and 0 <= n < 2004 for n in ids) and
                rows == sorted(set(rows)) and ids == sorted(set(ids)) and
                not rows_seen.intersection(rows) and not classes_seen.intersection(ids) and
                set(fit['targets'][r] for r in rows) == set(ids), 'partition leakage/layout differs')
        if name != 'train':
            query, gallery = panel['query'], panel['gallery']
            require(all(type(n) is int and 0 <= n < count for n in query + gallery) and
                    query == sorted(set(query)) and gallery == sorted(set(gallery)) and
                    not set(query).intersection(gallery) and set(query + gallery) == set(range(count)) and
                    len(query) == (1734 if name == 'selection' else 1749) and
                    set(fit['targets'][rows[r]] for r in query) == set(ids) ==
                    set(fit['targets'][rows[r]] for r in gallery), 'partition local query/gallery differs')
        rows_seen.update(rows); classes_seen.update(ids)
    require(rows_seen == set(range(13283)) and classes_seen == set(range(2004)), 'partition coverage differs')
    ids = partition['panels']['train']['original_class_ids']
    dense = {c: i for i, c in enumerate(ids)}
    return [dense[fit['targets'][r]] for r in partition['panels']['train']['original_rows']]


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'cached_reference', 'original_authority', 'partition', 'recipe')}


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def check_steps(rows, start, count):
    keys = {'step', 'batch', 'schedule_sha256', 'feature_rows_sha256', 'mixing_sha256', 'clean_bank_sha256',
            'ce', 'rank', 'loss', 'scale', 'preclip_norm', 'gradient_norms', 'state_sha256', 'seconds'}
    require(len(rows) == count, 'complete update records required')
    for step, row in enumerate(rows, start):
        require(row.keys() == keys and row['step'] == step and len(row['batch']) == 64 and
                all(type(n) is int and 0 <= n < ROWS for n in row['batch']) and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k]) for k in
                    ('schedule_sha256', 'feature_rows_sha256', 'mixing_sha256', 'clean_bank_sha256', 'state_sha256')) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('ce', 'rank', 'loss', 'scale', 'preclip_norm', 'seconds')) and
                row['loss'] == row['ce'] + 8 * row['rank'] and row['scale'] == 128 and row['seconds'] > 0 and
                row['gradient_norms'].keys() == set(PARAMETERS) and
                all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in row['gradient_norms'].values()),
                'complete finite update record differs')


def check_terminal_record(record, launch, phase, arm):
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == SEEDS[0] and record['pass'] is True and record['quality_read'] is False and
            record['exit_rehash_pass'] is True and record['strict_reload_exact'] is True and
            record['optimizer_members'] == 5 and method(record['launch']) == method(launch) and
            record['resource_policy'] == policy(phase) and record['trained_state_reused'] is False,
            'new method terminal binding differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, seed=SEEDS[0],
                                                execution_sha256=launch['execution_sha256']))
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['initial_arm_parity'] is True and
                record['gradient_witnesses'] is True and record['cpu_resume_exact'] is True and
                record['training_only_fit'] is True and record['completed_step'] == 0,
                'CPU witnesses incomplete')
    else:
        require(record['launch']['selected_cpu'] == launch['selected_cpu'] and
                record['completed_step'] == 17 and record['checkpoint'] is None and
                record['training_state_discarded'] is True and record['replay_exact'] is True and
                len(record['steps']) == 17 and len(record['resumed_steps']) == 9 and
                all(diagnostic(a) == diagnostic(b) for a, b in
                    zip(record['steps'][8:], record['resumed_steps'], strict=True)) and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'discarded17 replay incomplete')
        check_steps(record['steps'], 1, 17); check_steps(record['resumed_steps'], 9, 9)


def authority(args):
    require(not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'safetensors',
                                       'torchvision', 'sfora'} for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = descriptor_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and
            not output.exists() and not output.is_symlink(), 'exclusive canonical output required')
    cached = launch['cached_reference']
    cached_root = Path(cached['root'])
    require(not root.is_relative_to(cached_root) and not cached_root.is_relative_to(root) and
            not output.is_relative_to(root) and not output.is_relative_to(cached_root), 'separate immutable closures required')
    cached_code = closure(cached_root, CACHED_EXECUTION_SHA, CACHED_FILES, guards)
    require(cached_code['train_siglip2_cached_readout.py'] == CACHED_TRAINER_SHA, 'original cached trainer pin differs')
    path = bound_file(guards, cached_root / 'train_siglip2_cached_readout.py', CACHED_TRAINER_SHA)
    spec = importlib.util.spec_from_file_location('_identity_mix_cached_original', path)
    old = importlib.util.module_from_spec(spec)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CACHED_TRAINER_SHA, 'cached trainer changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(old))
    original_launch = descriptor_json(launch['original_authority'], guards)
    require(original_launch['reference']['execution_sha256'] == REFERENCE_EXECUTION_SHA,
            'original five-file reference pin differs')
    old_context = old.authority(SimpleNamespace(execution_sha256=CACHED_EXECUTION_SHA,
        authority=Path(launch['original_authority']['path']), authority_sha256=ORIGINAL_AUTHORITY_SHA,
        phase='cpu', arm='control', seed=old.SEEDS[0], output=output))
    initialized = old_context['initialized']
    prior_guards = old_context['guards'].copy()
    partition = descriptor_json(launch['partition'], guards)
    target = check_partition(partition, initialized['source_context']['fit'])
    require(partition['original_cache'] == {k: old_context['source']['features'][k] for k in ('path', 'sha256')} and
            partition['original_fit'] == original_launch_manifest(initialized), 'partition original paths differ')
    for path, digest in guards.items():
        require(initialized['guards'].setdefault(path, digest) == digest, 'conflicting new authority')
    source = {k: v for k, v in old_context['source'].items() if k not in ('initializers', 'arrays')}
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'cached': old,
               'cached_context': old_context, 'original': old_context['original'],
               'old': old_context['old'], 'old_cpu': old_context['old_cpu'], 'initialized': initialized,
               'source': source, 'partition': partition, 'target': target,
               'guards': initialized['guards'], 'terminals': {}, 'terminal_cgroups': {}}
    admission = context['original'].FlatAdmission()
    admission.init = initialized['init']
    wanted = [] if args.phase == 'cpu' else [('cpu', 'control', launch['selected_cpu'])]
    if args.phase == 'train':
        wanted += [('mechanics', arm, launch['selected_mechanics'][arm]) for arm in ARMS]
    ids = {context['old_cpu']['invocation']['invocation_id'],
           initialized['source_context']['proof']['invocation']['invocation_id']}
    for phase, arm, terminal in wanted:
        record = descriptor_json(terminal['receipt'], context['guards'])
        check_terminal_record(record, launch, phase, arm)
        invocation = record['invocation']
        prior = context['old_cpu']['invocation']
        require(descriptor_json(record['authority'], context['guards']) == record['launch'] and
                record['code'] == code and record['authority']['sha256'] == record['authority_sha256'] and
                record['execution_sha256'] == args.execution_sha256 and record['source'] == source and
                record['partition_sha256'] == PARTITION_SHA and record['numerical_flags'] == context['old_cpu']['numerical_flags'] and
                invocation['argv'] == [str(root / 'train_siglip2_identity_mix.py'), '--execution-sha256',
                    args.execution_sha256, '--authority', record['authority']['path'], '--authority-sha256',
                    record['authority']['sha256'], '--phase', phase, '--arm', arm, '--seed', str(SEEDS[0]),
                    '--output', str(Path(terminal['receipt']['path']).parent)] and
                all(invocation[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
                (invocation['cuda_visible_devices'] == '' if phase == 'cpu' else
                 invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8') and
                all(record['input_guards'].get(p) == h for p, h in prior_guards.items()),
                'original new-method launch/source/input closure differs')
        if phase == 'cpu':
            checkpoint = record['checkpoint']
            require(checkpoint.keys() == {'path', 'sha256'} and
                    checkpoint['path'] == str(Path(terminal['receipt']['path']).parent / 'initializer.pt') and
                    record['input_guards'].get(checkpoint['path']) == checkpoint['sha256'], 'CPU initializer authority differs')
        require(invocation['invocation_id'] not in ids, 'duplicate original unit invocation')
        ids.add(invocation['invocation_id'])
        context['terminal_cgroups'][phase + ':' + arm] = admission.admit_terminal(
            record, terminal, policy(phase)['seconds'], context['guards'])
        for path, digest in record['input_guards'].items():
            admission.bound_file(context['guards'], path, digest)
        context['terminals'][phase + ':' + arm] = record
    return context


def original_launch_manifest(initialized):
    src = initialized['source_context']
    return {'path': str(src['args'].fit_manifest), 'sha256': src['args'].fit_manifest_sha256}


def donor_members(target):
    members = {}
    for row, label in enumerate(target):
        require(type(label) is int and label >= 0, 'integer product target required')
        members.setdefault(label, []).append(row)
    return members


def donor_at(members, target, anchor, choice):
    group = members[target[anchor]]
    if len(group) == 1:
        return anchor
    require(type(choice) is int and 0 <= choice < len(group) - 1, 'uniform donor choice outside support')
    # O(group size), at most12 original FIT images; no rejection-RNG coupling.
    position = group.index(anchor)
    return group[choice + (choice >= position)]


def check_mixing(target, anchors, donors, coefficients, masks):
    require(len(anchors) == len(donors) == len(coefficients) == len(masks), 'mixing table lengths differ')
    counts = {c: len(v) for c, v in donor_members(target).items()}
    for anchor, donor, coefficient, mask in zip(anchors, donors, coefficients, masks, strict=True):
        require(type(anchor) is type(donor) is int and 0 <= anchor < len(target) and 0 <= donor < len(target) and
                target[anchor] == target[donor] and type(mask) is bool and type(coefficient) is float and
                math.isfinite(coefficient) and 0 <= coefficient <= 1 and
                struct.unpack('<f', struct.pack('<f', coefficient))[0] == coefficient and
                ((donor == anchor and coefficient == 0 and mask is False) if counts[target[anchor]] == 1 else donor != anchor),
                'product-safe FP32 mixing table differs')


def schedule_and_mixing(target, seed):
    import numpy as np
    require(seed in SEEDS and len(target) == ROWS and sorted(set(target)) == list(range(CLASSES)),
            'dense TRAIN target/seed differs')
    members = donor_members(target)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(list(range(CLASSES)))
    anchors = np.asarray([[int(rng.choice(members[int(c)])) for c in
        order[(step * 64 + np.arange(64)) % CLASSES]] for step in range(1000)], dtype=np.int64)
    donor_rng, coefficient_rng, mask_rng = [np.random.Generator(np.random.PCG64(seed + offset))
                                           for offset in RECIPE['stream_offsets']]
    donors = np.asarray([donor_at(members, target, int(a), 0 if len(members[target[int(a)]]) == 1 else
                                 int(donor_rng.integers(len(members[target[int(a)]]) - 1)))
                         for a in anchors.flat], dtype=np.int64).reshape(1000, 64)
    coefficient = coefficient_rng.random((1000, 64)).astype(np.float32)
    mask = mask_rng.random((1000, 64)) < .5
    singleton = donors == anchors
    coefficient[singleton] = 0; mask[singleton] = False
    check_mixing(target, anchors.ravel().tolist(), donors.ravel().tolist(),
                 coefficient.ravel().tolist(), mask.ravel().tolist())
    require(all(len(set(target[a] for a in batch)) == 64 for batch in anchors.tolist()),
            'class-balanced TRAIN schedule differs')
    return anchors, {'donors': donors, 'lambda': coefficient, 'mask': mask}


def mix_inputs(features, anchors, donors, coefficients, masks, arm, normalize):
    """Shared training path; control and untreated rows preserve exact common u."""
    require(arm in ARMS, 'fixed mixing arm required')
    result = features[anchors].clone()
    if arm == 'candidate' and bool(masks.any()):
        lam = coefficients[masks].unsqueeze(1)
        result[masks] = normalize((1 - lam) * result[masks] + lam * features[donors[masks]])
    return result


def normalize_nonzero(value):
    import torch
    from torch.nn import functional as F
    require(value.dtype == torch.float32 and torch.isfinite(value).all().item() and
            (value.norm(dim=1) > 0).all().item(), 'mix must be finite nonzero FP32')
    result = F.normalize(value, dim=1)
    require(torch.isfinite(result).all().item(), 'normalized mix nonfinite')
    return result


def clean_bank_descriptors(state, index):
    import torch
    with torch.no_grad(), torch.autocast(device_type=state['features'].device.type, enabled=False):
        return state['head'](state['features'][index]).detach()


def optimizer_state(head, classifier):
    import torch
    params = [('compact_head.' + n, p) for n, p in head.named_parameters()] + [('classifier', classifier)]
    require([n for n, _ in params] == PARAMETERS and len({id(p) for _, p in params}) == 5 and
            [tuple(p.shape) for _, p in params] == SHAPES and
            all(p.requires_grad and p.dtype == torch.float32 for _, p in params), 'EXACT five optimizer members required')
    optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': 1e-4},
                                   {'params': [classifier], 'lr': 1e-4}], weight_decay=.05)
    defaults = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and
            all({k: v for k, v in g.items() if k != 'params'} == {**defaults, 'lr': 1e-4}
                for g in optimizer.param_groups), 'original AdamW defaults/groups differ')
    return params, optimizer


def training_features(context):
    import numpy as np
    import torch
    cache = np.load(context['source']['features']['path'], allow_pickle=False, mmap_mode='r')
    require(cache.shape == (13283, WIDTH) and cache.dtype == np.float32, 'authentic full cache layout differs')
    # The ONLY feature indexing at the cache boundary is the frozen TRAIN map.
    features = torch.from_numpy(cache[context['partition']['panels']['train']['original_rows']].copy())
    del cache
    require(features.shape == (ROWS, WIDTH), 'TRAIN-only feature shape differs')
    return normalize_nonzero(features)


def initializer(context, ref, features, pca=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    require(features.shape == (ROWS, WIDTH) and features.device.type == 'cpu', 'TRAIN-only CPU PCA required')
    parent = context['initialized']['pca']
    helper = context['initialized']['init']
    pca_helper = helper.load_bare('_identity_mix_pinned_pca', parent['root'] / 'representation_ceiling.py',
                                parent['launch']['pca_helper_sha256'])
    normalized = F.normalize(features, dim=1)
    if pca is None:
        pca = pca_helper.fit_centered_pca(normalized, dimensions=DIM)
    else:
        pca = pca_helper.CenteredPcaTransform(**pca)
    head = context['cached'].head_from('control', tensors={
        'primary.weight': pca.components, 'primary.bias': -(pca.components @ pca.mean),
        'down.weight': torch.zeros(32, WIDTH), 'up.weight': torch.zeros(DIM, 32),
        'center': normalized.mean(0), 'preactivation_std': torch.ones(())})
    with torch.no_grad():
        nn.init.kaiming_uniform_(head.down.weight, a=5**.5,
                                generator=torch.Generator().manual_seed(RECIPE['initialization_seed']))
        std = head.down(normalized - head.center).std(unbiased=False)
        require(torch.isfinite(std).item() and float(std) > 0, 'TRAIN preactivation std undefined')
        head.preactivation_std.copy_(std); head.down.weight.div_(std)
        projected = pca.apply(normalized)
        target = torch.tensor(context['target'], dtype=torch.int64)
        sums = torch.zeros(CLASSES, DIM)
        for row, label in enumerate(context['target']):
            sums[label] += projected[row]
        classifier = F.normalize(sums, dim=1)
        bank = F.normalize(head(features), dim=1).detach()
    schedules, mixing = {}, {}
    for seed in SEEDS:
        anchors, tables = schedule_and_mixing(context['target'], seed)
        schedules[str(seed)] = torch.from_numpy(anchors)
        mixing[str(seed)] = {n: torch.from_numpy(v) for n, v in tables.items()}
    result = {'head': {n: v.detach().clone() for n, v in head.state_dict().items()},
              'pca': {'mean': pca.mean.clone(), 'components': pca.components.clone()},
              'classifier': classifier, 'bank': bank, 'target': target,
              'positive': ref.member_bank_positive_ordinals(target.numpy(), allow_singletons=True),
              'schedules': schedules, 'mixing': mixing,
              'original_rows': torch.tensor(context['partition']['panels']['train']['original_rows'])}
    require(all(torch.isfinite(v).all().item() for v in (classifier, bank)) and
            (classifier.norm(dim=1) > 0).all().item() and (bank.norm(dim=1) > 0).all().item(), 'initial proxy/bank nonfinite/zero')
    return result


def fresh(context, ref, arm, seed, device, initial, features):
    import torch
    head = context['cached'].head_from('control', tensors=initial['head']).to(device).train()
    classifier = torch.nn.Parameter(initial['classifier'].to(device).clone())
    params, optimizer = optimizer_state(head, classifier)
    return {'head': head, 'classifier': classifier, 'bank': initial['bank'].to(device).detach().clone(),
            'target': initial['target'].to(device).clone(), 'positive': initial['positive'].to(device).clone(),
            'pca': {n: v.clone() for n, v in initial['pca'].items()}, 'schedules': initial['schedules'],
            'mixing': initial['mixing'], 'original_rows': initial['original_rows'],
            'partition': context['partition'], 'initializer': initial, 'features': features.to(device),
            'params': params, 'optimizer': optimizer, 'seed': seed, 'arm': arm, 'counter': 0,
            'scaler': torch.amp.GradScaler(device, init_scale=128)}


def identity(context, state, flags):
    original = context['original']
    return {'method': method(context['launch']), 'arm': state['arm'], 'seed': state['seed'],
            'selected_cpu': context['launch']['selected_cpu'], 'source': context['source'],
            'parameter_names': PARAMETERS, 'device': state['features'].device.type,
            'feature_version': state['features']._version,
            'feature_state_sha256': original.fingerprint(state['features']),
            'optimizer_defaults': state['optimizer'].defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
            'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
            'static_sha256': original.fingerprint({n: state[n] for n in STATIC_KEYS}),
            'buffers_sha256': original.fingerprint(dict(state['head'].named_buffers())),
            'positive_shape': tuple(state['positive'].shape),
            'schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]),
            'mixing_sha256': original.fingerprint(state['mixing'][str(state['seed'])]), 'numerical_flags': flags}


def payload(state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'head': dict(state['head'].state_dict()),
            'classifier': state['classifier'].detach(), 'bank': state['bank'],
            **{n: state[n] for n in STATIC_KEYS},
            'optimizer': state['optimizer'].state_dict(), 'optimizer_defaults': state['optimizer'].defaults.copy(),
            'scaler': state['scaler'].state_dict(), 'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if ident['device'] == 'cuda' else [],
            'counter': state['counter'], 'seed': state['seed'],
            'numerical_flags': ident['numerical_flags'], 'source': ident['source']}


def check_payload(saved, ident, step):
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            type(saved['counter']) is int and saved['counter'] == step and saved['seed'] == ident['seed'] and
            saved['source'] == ident['source'] and saved['numerical_flags'] == ident['numerical_flags'] and
            saved['optimizer_defaults'] == ident['optimizer_defaults'] and ident['parameter_names'] == PARAMETERS,
            'complete resume identity/state differs')
    optimizer = saved['optimizer']
    require(optimizer.keys() == {'state', 'param_groups'} and optimizer['param_groups'] == ident['optimizer_serial_groups'] and
            [i for g in optimizer['param_groups'] for i in g['params']] == list(range(5)) and
            all(type(i) is int for g in optimizer['param_groups'] for i in g['params']) and
            all(type(i) is int for i in optimizer['state']) and
            set(optimizer['state']) == (set(range(5)) if step else set()), 'EXACT five optimizer states/order required')
    require(all(v.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == step for v in optimizer['state'].values()) and
            saved['scaler'].keys() == {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'} and
            saved['scaler']['scale'] == 128 and saved['scaler']['growth_factor'] == 2 and
            saved['scaler']['backoff_factor'] == .5 and saved['scaler']['growth_interval'] == 2000 and
            saved['scaler']['_growth_tracker'] == step and ident['device'] in ('cpu', 'cuda') and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0), 'optimizer counters/scaler/RNG differ')
    def tensor(value, shape, dtype):
        require(tuple(value.shape) == shape and str(value.dtype) == dtype, 'complete tensor layout differs')
    head_shapes = dict(zip(('primary.weight', 'primary.bias', 'down.weight', 'up.weight', 'center', 'preactivation_std'),
                           (*SHAPES[:4], (WIDTH,), ()), strict=True))
    def static(tree):
        require(tree['pca'].keys() == {'mean', 'components'} and
                tree['schedules'].keys() == tree['mixing'].keys() == {str(s) for s in SEEDS},
                'complete PCA/schedules/mixing inventory differs')
        for value, shape in ((tree['pca']['mean'], (WIDTH,)), (tree['pca']['components'], (DIM, WIDTH))):
            tensor(value, shape, 'torch.float32')
        tensor(tree['target'], (ROWS,), 'torch.int64'); tensor(tree['positive'], ident['positive_shape'], 'torch.int64')
        tensor(tree['original_rows'], (ROWS,), 'torch.int64')
        for seed in SEEDS:
            tensor(tree['schedules'][str(seed)], (1000, 64), 'torch.int64')
            table = tree['mixing'][str(seed)]
            require(table.keys() == {'donors', 'lambda', 'mask'}, 'complete mixing tables required')
            for n, dtype in (('donors', 'torch.int64'), ('lambda', 'torch.float32'), ('mask', 'torch.bool')):
                tensor(table[n], (1000, 64), dtype)
    for tree in (saved, saved['initializer']):
        require(tree['head'].keys() == head_shapes.keys(), 'complete head/buffers required')
        for name, shape in head_shapes.items():
            tensor(tree['head'][name], shape, 'torch.float32')
        tensor(tree['classifier'], SHAPES[4], 'torch.float32'); tensor(tree['bank'], (ROWS, DIM), 'torch.float32')
        static(tree)
    require(saved['initializer'].keys() == INITIAL_KEYS, 'complete NEW initializer required')
    for value in [saved['cpu_rng'], *saved['cuda_rng']]:
        require(len(value.shape) == 1 and value.shape[0] > 0 and str(value.dtype) == 'torch.uint8', 'complete RNG layout differs')
    for i, moments in optimizer['state'].items():
        tensor(moments['step'], (), 'torch.float32')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor(moments[name], SHAPES[i], 'torch.float32')


def integrity(context, state, ident):
    import torch
    actual = [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]
    require([n for n, _ in actual] == PARAMETERS and [tuple(p.shape) for _, p in actual] == SHAPES and
            len({id(p) for _, p in actual}) == 5 and [id(p) for _, p in actual] ==
            [id(p) for _, p in state['params']] == [id(p) for g in state['optimizer'].param_groups for p in g['params']] and
            all(p.dtype == torch.float32 and p.device.type == ident['device'] and p.requires_grad and
                torch.isfinite(p).all().item() for _, p in actual), 'live five-member inventory differs')
    require(state['head'].training and all(m.training and not m._forward_hooks and not m._forward_pre_hooks and
            not m._backward_hooks for m in state['head'].modules()), 'head modes/hooks differ')
    require(state['features'].shape == (ROWS, WIDTH) and state['features'].dtype == torch.float32 and
            not state['features'].requires_grad and state['features'].grad is None and
            state['features']._version == ident['feature_version'], 'immutable TRAIN feature cache changed')
    require(context['original'].fingerprint({n: state[n] for n in STATIC_KEYS}) == ident['static_sha256'] and
            context['original'].fingerprint(dict(state['head'].named_buffers())) == ident['buffers_sha256'] and
            state['bank'].shape == (ROWS, DIM) and state['bank'].dtype == torch.float32 and
            not state['bank'].requires_grad and state['bank'].grad is None and torch.isfinite(state['bank']).all().item(),
            'complete static state/buffers/bank differ')
    require(state['optimizer'].defaults == ident['optimizer_defaults'] and
            [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups] == ident['optimizer_groups'],
            'live optimizer options differ')
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    for moments in saved['optimizer']['state'].values():
        require(all(torch.isfinite(v).all().item() for v in moments.values()), 'optimizer moments nonfinite')


def save(context, state, ident, path):
    import torch
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    with context['initialized']['source_context']['extract'].exclusive(path) as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    return context['initialized']['init'].sha(path), context['original'].fingerprint(saved)


def restore(context, ref, path, sha, digest, ident, step, features):
    import torch
    bound_file({}, path, sha)
    disk = torch.load(path, map_location='cpu', weights_only=True)
    check_payload(disk, ident, step)
    require(context['original'].fingerprint(disk) == digest, 'serialized complete state fingerprint differs')
    state = fresh(context, ref, ident['arm'], ident['seed'], ident['device'], disk['initializer'], features)
    state['head'].load_state_dict(disk['head'], strict=True)
    with torch.no_grad():
        state['classifier'].copy_(disk['classifier'].to(ident['device']))
    state['bank'] = disk['bank'].to(ident['device']).detach().clone()
    for name in STATIC_KEYS:
        state[name] = disk[name]
    state['target'] = state['target'].to(ident['device']); state['positive'] = state['positive'].to(ident['device'])
    state['optimizer'].load_state_dict(disk['optimizer']); state['scaler'].load_state_dict(disk['scaler'])
    state['counter'] = step
    require(torch.equal(state['positive'], ref.member_bank_positive_ordinals(state['target'].cpu().numpy(),
                allow_singletons=True).to(ident['device'])), 'strict positive ordinal reload differs')
    torch.random.set_rng_state(disk['cpu_rng'].clone())
    if ident['device'] == 'cuda':
        torch.cuda.set_rng_state_all([v.clone() for v in disk['cuda_rng']])
    del disk
    gc.collect()
    integrity(context, state, ident)
    require(context['original'].fingerprint(payload(state, ident)) == digest, 'independent complete resume reload differs')
    return state


def terms(context, ref, state, index, donor, coefficient, mask):
    import torch
    inputs = mix_inputs(state['features'], index, donor, coefficient, mask, state['arm'], normalize_nonzero)
    raw = state['head'](inputs)
    ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                     torch.arange(DIM, device=raw.device).unsqueeze(0), margin=.3, scale=64)
    rank = context['original'].valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
    return raw, ce, rank


def update(context, ref, state, ident, step):
    import torch
    device = ident['device']
    if device == 'cuda':
        torch.cuda.synchronize()
    tick = time.perf_counter()
    require(state['counter'] == step - 1 and 1 <= step <= 1000, 'update counter differs')
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    table = state['mixing'][str(state['seed'])]
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version, rows, ce_sum, rank_sum = state['bank']._version, [], 0., 0.
    for offset in range(0, 64, 16):
        index = torch.tensor(batch[offset:offset + 16], device=device)
        part = {n: v[step - 1, offset:offset + 16].to(device) for n, v in table.items()}
        # Clean forward is detached BEFORE any optimizer mutation, for BOTH arms.
        rows.append(clean_bank_descriptors(state, index))
        with torch.autocast(device_type=device, enabled=False):
            raw, ce, rank = terms(context, ref, state, index, part['donors'], part['lambda'], part['mask'])
            loss = (ce + 8 * rank) * .25
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item() and torch.isfinite(loss).item(),
                'nonfinite cached update')
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach()) * .25; rank_sum += float(rank.detach()) * .25
    require(state['bank']._version == version, 'bank changed during backward')
    scaler.unscale_(optimizer)
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for _, p in state['params']), 'all five finite FP32 gradients required')
    gradients = {n: float(p.grad.double().norm()) for n, p in state['params']}
    if step <= 2:
        require(gradients['compact_head.up.weight'] > 0 and
                (gradients['compact_head.down.weight'] == 0 if step == 1 else gradients['compact_head.down.weight'] > 0),
                'initial zero/subsequent nonzero residual gradient differs')
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    require(scaler.get_scale() == scale == 128, 'skipped/scaled update forbidden')
    state['counter'] += 1
    refresh, positions = ref.member_bank_refresh_rows(batch)
    clean = torch.cat(rows)
    state['bank'][torch.tensor(refresh, device=device)] = ref.member_bank_refresh_values(
        clean, clean, torch.tensor(positions, device=device), live_head=False)
    require(state['bank']._version == version + 1, 'last duplicate detached refresh differs')
    optimizer.zero_grad(set_to_none=True)
    integrity(context, state, ident)
    require(context['initialized']['source'].numerical_flags() == ident['numerical_flags'] and
            (device != 'cuda' or torch.cuda.max_memory_allocated() < 10_000_000_000), 'flags/whole-unit CUDA peak differs')
    row = {'step': step, 'batch': batch, 'schedule_sha256': ident['schedule_sha256'],
           'feature_rows_sha256': context['original'].fingerprint(state['features'][batch]),
           'mixing_sha256': context['original'].fingerprint({n: v[step - 1] for n, v in table.items()}),
           'clean_bank_sha256': context['original'].fingerprint(clean),
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients,
           'state_sha256': context['original'].fingerprint(payload(state, ident))}
    if device == 'cuda':
        torch.cuda.synchronize()
    row['seconds'] = time.perf_counter() - tick
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def calibration(context, state):
    # The old helper is count-independent and packs on CPU even for GPU states.
    return context['cached'].calibration(context['cached_context'], state)


def cpu_witnesses(context, ref, output, flags):
    import torch
    rng = torch.random.get_rng_state().clone()
    features = training_features(context)
    initial = initializer(context, ref, features)
    digest = context['original'].fingerprint(initial)
    reconstructed = initializer(context, ref, features, pca=initial['pca'])
    require(context['original'].fingerprint(reconstructed) == digest, 'independent NEW initializer geometry differs')
    del reconstructed
    witnesses, gradients, resumes = {}, {}, {}
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu', initial, features)
        ident = identity(context, state, flags)
        integrity(context, state, ident)
        witnesses[arm] = calibration(context, state)
        gradients[arm] = [update(context, ref, state, ident, 1)]
        with TemporaryDirectory(prefix='discard-cpu-', dir=output) as directory:
            checkpoint = Path(directory) / 'step1.pt'
            sha, state_digest = save(context, state, ident, checkpoint)
            gradients[arm].append(update(context, ref, state, ident, 2))
            final_digest = context['original'].fingerprint(payload(state, ident))
            final_witness = context['original'].fingerprint(calibration(context, state))
            del state
            gc.collect()
            state = restore(context, ref, checkpoint, sha, state_digest, ident, 1, features)
            row = update(context, ref, state, ident, 2)
            require(diagnostic(row) == diagnostic(gradients[arm][1]) and
                    context['original'].fingerprint(payload(state, ident)) == final_digest and
                    context['original'].fingerprint(calibration(context, state)) == final_witness,
                    'CPU full objective/gradient/resume differs')
            resumes[arm] = {'state_sha256': final_digest, 'raw_unit_packed_sha256': final_witness}
            del state
            gc.collect()
    require(context['original'].fingerprint(witnesses['control']) == context['original'].fingerprint(witnesses['candidate']),
            'initial matched arm raw/unit/packed parity differs')
    checkpoint = output / 'initializer.pt'
    saved = {'schema': SCHEMA, 'source': context['source'], 'partition': context['partition'], 'initial': initial,
             'cpu_rng': rng, 'numerical_flags': flags, 'counter': 0,
             'fixtures': {'calibration': witnesses, 'gradients': gradients, 'resumes': resumes}}
    saved_digest = context['original'].fingerprint(saved)
    with context['initialized']['source_context']['extract'].exclusive(checkpoint) as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    del saved, initial
    gc.collect()
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(context['original'].fingerprint(disk) == saved_digest, 'CPU complete initializer reload differs')
    initial = disk['initial']
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu', initial, features)
        require(context['original'].fingerprint(calibration(context, state)) ==
                context['original'].fingerprint(witnesses[arm]), 'CPU strict raw/unit/packed initializer reload differs')
        del state
    require(torch.equal(rng, torch.random.get_rng_state()) and
            context['initialized']['source'].numerical_flags() == flags and not torch.cuda.is_initialized(),
            'CPU constructor RNG/flags/CUDA differs')
    bound_file(context['guards'], checkpoint, context['initialized']['init'].sha(checkpoint))
    return {'completed_step': 0, 'checkpoint': {'path': str(checkpoint), 'sha256': context['guards'][str(checkpoint)]},
            'cuda_initialized': False, 'initial_arm_parity': True, 'gradient_witnesses': True,
            'cpu_resume_exact': True, 'training_only_fit': True, 'initial_state_sha256': digest,
            'cpu_state_sha256': saved_digest, 'cpu_witnesses_sha256': context['original'].fingerprint(disk['fixtures'])}


def gpu_run(context, ref, output, flags):
    import torch
    args = context['args']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible uninitialized CUDA device required')
    proof = context['terminals']['cpu:control']
    bound_file({}, proof['checkpoint']['path'], proof['checkpoint']['sha256'])
    disk = torch.load(proof['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(context['original'].fingerprint(disk) == proof['cpu_state_sha256'] and disk['schema'] == SCHEMA and
            disk['source'] == context['source'] and disk['partition'] == context['partition'] and
            disk['counter'] == 0 and disk['numerical_flags'] == flags and
            context['original'].fingerprint(disk['initial']) == proof['initial_state_sha256'], 'accepted NEW initializer differs')
    initial = disk['initial']
    features = training_features(context)
    torch.cuda.manual_seed_all(SEEDS[0])
    state = fresh(context, ref, args.arm, args.seed, 'cuda', initial, features)
    del initial, disk
    ident = identity(context, state, flags)
    integrity(context, state, ident)
    start_digest = context['original'].fingerprint(payload(state, ident))
    features = state['features']
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 1000
    with TemporaryDirectory(prefix='discard-mechanics-', dir=output) as directory:
        temporary = Path(directory)
        tick = time.perf_counter()
        for step in range(1, total + 1):
            row = update(context, ref, state, ident, step)
            if args.phase == 'train' and args.seed == SEEDS[0] and step <= 17:
                require(diagnostic(row) == diagnostic(context['terminals']['mechanics:' + args.arm]['steps'][step - 1]),
                        'fresh first17 mechanics replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
        training_seconds = time.perf_counter() - tick
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        witness = context['original'].fingerprint(calibration(context, state))
        del state
        gc.collect(); torch.cuda.empty_cache()
        if args.phase == 'mechanics':
            state = restore(context, ref, temporary / 'step8.pt', sha8, digest8, ident, 8, features)
            resumed = [update(context, ref, state, ident, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    context['original'].fingerprint(payload(state, ident)) == digest, '17 versus independent8+9 differs')
            del state
            gc.collect(); torch.cuda.empty_cache()
        state = restore(context, ref, checkpoint, sha, digest, ident, total, features)
        require(context['original'].fingerprint(calibration(context, state)) == witness and
                context['original'].fingerprint(features) == ident['feature_state_sha256'], 'strict final raw/unit/packed/TRAIN reload differs')
        require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)), 'CUDA RNG changed')
        del state, features
        gc.collect(); torch.cuda.empty_cache()
    return {'completed_step': total, 'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'initial_state_sha256': start_digest, 'terminal_state_sha256': digest, 'identity': ident,
            'steps': rows, 'resumed_steps': resumed, 'replay_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics', 'training_wall_seconds': training_seconds,
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def run(args):
    started = time.perf_counter()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
            os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'explicit CUDA-hidden CPU or deterministic CUDA launch required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original systemd invocation required')
    require(sys.argv == [str(Path(__file__).absolute()), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--phase', args.phase, '--arm', args.arm, '--seed', str(args.seed), '--output', str(args.output)],
            'fixed canonical CLI order required')
    context = authority(args)
    source, init = context['initialized']['source'], context['initialized']['init']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    init.admit_cgroup(before, unit)
    prior = context['old_cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and init.sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'original qualified interpreter differs')
    import torch
    require(not torch.cuda.is_initialized(), 'admission must precede CUDA')
    flags = context['old_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original numerical flags differ')
    torch.random.default_generator.manual_seed(SEEDS[0])
    cpu_rng = torch.random.get_rng_state().clone()
    cpu_origins = source.imported_origins(context['initialized']['source_context']['extract'], context['initialized']['packages'])
    require(all(context['old_cpu']['origins']['files'].get(p) == h for p, h in cpu_origins['files'].items()),
            'actual CPU imports differ from original qualified origins')
    ref = context['original'].reference_math(context['old'])
    args.output.mkdir()
    result = cpu_witnesses(context, ref, args.output, flags) if args.phase == 'cpu' else gpu_run(context, ref, args.output, flags)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'constructor/update/reload RNG/flags differ')
    origins = context['original'].exit_rehash(context['old'])
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'] and
            closure(Path(context['launch']['cached_reference']['root']), CACHED_EXECUTION_SHA, CACHED_FILES, {}) ==
            context['cached_context']['code'], 'exit new/original cached closure differs')
    after = source.cgroup_memory(); init.admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and 0 < rss <= 8 * 1024**2 and
            (args.phase != 'cpu' or not torch.cuda.is_initialized()), 'whole-unit time/RSS/CUDA caps differ')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed,
               'pass': True, 'quality_read': False, 'strict_reload_exact': True, 'optimizer_members': 5,
               'training_qualified': args.phase == 'train', 'trained_state_reused': False,
               'encoder_updates': 0, 'head_scalars': 188544, 'trainable_scalars': 317568,
               'source': context['source'], 'partition_sha256': PARTITION_SHA,
               'launch': context['launch'], 'execution_sha256': args.execution_sha256,
               'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
               'authority_sha256': args.authority_sha256, 'code': context['code'],
               'resource_policy': policy(args.phase), 'numerical_flags': flags, 'wall_seconds': wall,
               'process_peak_rss_kib': rss, 'cgroup_before': before, 'cgroup_after': after,
               'origins': origins, 'input_guards': context['guards'], 'exit_rehash_pass': True,
               'terminal_cgroups': context['terminal_cgroups'], 'both_locks_held_in_parent_authority': True,
               'terminal_exit_and_both_locks_require_parent_receipt': True, **result,
               'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                              'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                              'invocation_id': os.environ['INVOCATION_ID'],
                              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}}
    context['initialized']['pca']['exporter'].write_json(context['initialized']['source_context']['extract'],
                                                       args.output / 'receipt.json', receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('cpu', 'mechanics', 'train'), required=True)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--seed', type=int, choices=SEEDS, required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Identity mixing rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
