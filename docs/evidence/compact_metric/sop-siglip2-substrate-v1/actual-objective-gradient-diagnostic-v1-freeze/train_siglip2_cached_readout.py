#!/usr/bin/env python3
"""Fixed-view So400 rank32 cached readout; no evaluator or quality reads.

This reuses the failed Large inner-FIT residual architecture, not a novel
architecture or an established quality/runtime improvement. Whole-image
TRAIN300 and the v7/v8 snapshot candidates remain NO-GO/KILL.

Parent freezes a separate two-file execution.json, exactly FILES, and a
siglip2-cached-readout-launch-v1 JSON with exactly LAUNCH_KEYS below.
FILE is exactly {path: canonical absolute file, sha256: lowercase SHA256}.
REFERENCE is exactly {root: canonical absolute directory, execution_sha256}:
an unchanged original trainer five-file closure (trainer SHA TRAINER_SHA).
fit_inventory is a FILE containing the original
native256-fit-export-collected-inputs-v1 inventory; initialized_inventory is
a FILE containing native256-full-initialized-cpu-collected-inputs-v1. Copy the
collected inventories unchanged; their so400 entries select original source,
FIT, initializedCPU and transitive extraction/export/PCA closures and logs.
selected_cpu is null for cpu, otherwise TERMINAL. selected_mechanics is null
for cpu/mechanics, otherwise exactly {control: TERMINAL, candidate: TERMINAL}.
TERMINAL is exactly {receipt: FILE, log: FILE, unit, invocation_id,
service_seconds, native_peak_rss_kib, both_locks_held: true}; it must describe
an ORIGINAL accepted unit, with its FINAL_CGROUP footer and normal-exit log.
No future terminal hashes may be invented. resource_policy is policy(phase),
recipe is exactly RECIPE, both_locks_held is true. CPU uses control/179032 and
qualifies BOTH arms. Mechanics uses each arm/179032, discarded17 versus
independent8+9. Train starts FRESH only after BOTH mechanics pass, in parent
order control032,candidate032,candidate041,control041, exactly1000 updates.

Exact CLI order (parent supplies enclosing limits and BOTH lifetime locks):
CUDA_VISIBLE_DEVICES='' python -B ROOT/train_siglip2_cached_readout.py
 --execution-sha256 SHA --authority CPU_JSON --authority-sha256 SHA
 --phase cpu --arm control --seed 179032 --output NEW_CPU_DIRECTORY
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B ROOT/train_siglip2_cached_readout.py
 --execution-sha256 SHA --authority MECHANICS_JSON --authority-sha256 SHA
 --phase mechanics --arm control --seed 179032 --output NEW_MECHANICS_DIRECTORY
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B ROOT/train_siglip2_cached_readout.py
 --execution-sha256 SHA --authority TRAIN_JSON --authority-sha256 SHA
 --phase train --arm control --seed 179032 --output NEW_TRAIN_DIRECTORY
Repeat mechanics for candidate; train authorities pin BOTH original mechanics
terminals and the same CPU terminal. CPU120/mechanics300/train300 include ALL
admission, construction, serialization, independent reload and uncached exit.
8GiB/noSwap/zero events/CUDA<10GB, no peak reset. Failure leaves an inadmissible
output directory; no partial state advances. Parent authenticates the final
receipt.json/log/footer. Train alone retains complete resume.pt; mechanics
deletes all checkpoints. The frozen encoder is an authenticated external
dependency, absent from optimizer/live snapshots. Cached rows are normalized
FP32 native256 B32 exports; the head normalizes again. No RGB augmentation.
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
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCHEMA = 'siglip2-cached-readout-v1'
AUTHORITY_SCHEMA = 'siglip2-cached-readout-launch-v1'
FILES = {'train_siglip2_cached_readout.py', 'test_siglip2_cached_readout.py'}
TRAINER_SHA = 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'
SOURCE_SHA = '5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189'
FIT_SHA = 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'
PCA_SHA = 'dcb92082a1e5ad1f4087a3e044ae245b6d32432d035a3214178d2ad72a494c72'
MANIFEST_SHA = 'd32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251'
REVISION = 'e8708ab72d125807e45b36fb7d4e0aacbb59f379'
FIT_INVENTORY_SHA = '4694490f347949a7d814f5001655d1c7a9eb41c5ea369f4e838b231ad2ee7662'
INITIALIZED_INVENTORY_SHA = 'da925901082ff22994a90dcdb0ccfea0cd5e2144cb3f87c3950904e9928a503e'
SEEDS = (179032, 179041)
ARMS = ('control', 'candidate')
PARAMETERS = ['compact_head.primary.weight', 'compact_head.primary.bias',
              'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
SHAPES = [(128, 1152), (128,), (32, 1152), (128, 32), (2004, 128)]
RECIPE = {'width': 1152, 'rows': 13283, 'classes': 2004, 'output_dim': 128,
          'rank': 32, 'initialization_seed': 179034, 'steps': 1000, 'batch': 64,
          'microbatch': 16, 'seeds': list(SEEDS), 'views': 'fixed-normalized-cache',
          'augmentation': False, 'encoder_updates': False, 'margin': .3, 'scale': 64,
          'rank_weight': 8, 'learning_rate': 1e-4, 'weight_decay': .05, 'clip': 1,
          'initial_scaler': 128, 'control': '0.5*z', 'candidate': 'gelu-exact'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'reference',
               'fit_inventory', 'initialized_inventory', 'selected_cpu',
               'selected_mechanics', 'resource_policy', 'both_locks_held', 'recipe'}
PAYLOAD_KEYS = {'schema', 'identity', 'head', 'classifier', 'bank', 'target', 'pca',
                'positive', 'schedules', 'optimizer', 'optimizer_defaults', 'scaler',
                'cpu_rng', 'cuda_rng', 'counter', 'seed', 'numerical_flags', 'source'}


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
            args.seed in SEEDS and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'launch profile differs')
    require((args.phase != 'cpu' or (args.arm, args.seed) == ('control', SEEDS[0])) and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]), 'CPU/mechanics seed/arm differs')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if args.phase == 'train':
        require(launch['selected_mechanics'].keys() == set(ARMS), 'BOTH mechanics terminals required')
    require(launch['reference'].keys() == {'root', 'execution_sha256'}, 'exact REFERENCE descriptor required')
    require(launch['fit_inventory'].keys() == launch['initialized_inventory'].keys() == {'path', 'sha256'} and
            launch['fit_inventory']['sha256'] == FIT_INVENTORY_SHA and
            launch['initialized_inventory']['sha256'] == INITIALIZED_INVENTORY_SHA, 'original collected inventory pins differ')


def source_binding(initialized):
    src = initialized['source_context']
    result = {'checkpoint': src['proof']['checkpoint'], 'revision': src['entry']['revision'],
              'runtime': src['proof']['runtime'], 'sample': src['proof']['sample'],
              'features': initialized['pca']['export']['cache'],
              'initializers': initialized['record']['artifact'],
              'manifest_sha256': src['args'].fit_manifest_sha256,
              'arrays': initialized['record']['arrays']}
    require(result['checkpoint']['sha256'] == SOURCE_SHA and result['revision'] == REVISION and
            result['features']['sha256'] == FIT_SHA and result['initializers']['sha256'] == PCA_SHA and
            result['features']['shape'] == [13283, 1152] and result['features']['dtype'] == 'float32' and
            result['manifest_sha256'] == MANIFEST_SHA and len(result['runtime']['vision']) == 448,
            'authenticated So400 source/FIT/PCA binding differs')
    return result


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'reference', 'fit_inventory',
                                  'initialized_inventory', 'recipe')}


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def check_steps(rows, start, count):
    keys = {'step', 'batch', 'schedule_sha256', 'feature_rows_sha256', 'ce', 'rank', 'loss',
            'scale', 'preclip_norm', 'gradient_norms', 'state_sha256', 'seconds'}
    require(len(rows) == count, 'complete update records required')
    for step, row in enumerate(rows, start):
        require(row.keys() == keys and row['step'] == step and len(row['batch']) == 64 and
                all(type(n) is int and 0 <= n < 13283 for n in row['batch']) and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k])
                    for k in ('schedule_sha256', 'feature_rows_sha256', 'state_sha256')) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k])
                    for k in ('ce', 'rank', 'loss', 'scale', 'preclip_norm', 'seconds')) and
                row['loss'] == row['ce'] + 8 * row['rank'] and row['scale'] == 128 and row['seconds'] > 0 and
                row['gradient_norms'].keys() == set(PARAMETERS) and
                all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in row['gradient_norms'].values()),
                'complete finite update record differs')


def check_terminal_record(record, launch, phase, arm):
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == SEEDS[0] and record['pass'] is True and record['quality_read'] is False and
            record['exit_rehash_pass'] is True and record['strict_reload_exact'] is True and
            record['optimizer_members'] == 5 and record['source']['checkpoint']['sha256'] == SOURCE_SHA and
            method(record['launch']) == method(launch) and record['resource_policy'] == policy(phase),
            'new method terminal binding differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, seed=SEEDS[0],
                                                execution_sha256=launch['execution_sha256']))
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['initial_arm_parity'] is True and
                record['source_reload_exact'] is True and record['gradient_witnesses'] is True and
                record['completed_step'] == 0, 'CPU witnesses incomplete')
    else:
        require(record['launch']['selected_cpu'] == launch['selected_cpu'] and
                record['completed_step'] == 17 and record['checkpoint'] is None and
                record['training_state_discarded'] is True and record['replay_exact'] is True and
                len(record['steps']) == 17 and len(record['resumed_steps']) == 9 and
                all(row['step'] == i for i, row in enumerate(record['steps'], 1)) and
                all(diagnostic(a) == diagnostic(b) for a, b in
                    zip(record['steps'][8:], record['resumed_steps'], strict=True)) and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000,
                'discarded17/independent8+9 mechanics incomplete')
        check_steps(record['steps'], 1, 17)
        check_steps(record['resumed_steps'], 9, 9)


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
    reference = launch['reference']
    reference_root = Path(reference['root'])
    names = {'train_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_adaptation.py',
             'deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py'}
    original_code = closure(reference_root, reference['execution_sha256'], names, guards)
    require(original_code['train_siglip2_substrate_adaptation.py'] == TRAINER_SHA, 'original serial trainer pin differs')
    path = reference_root / 'train_siglip2_substrate_adaptation.py'
    spec = importlib.util.spec_from_file_location('_cached_readout_original', path)
    original = importlib.util.module_from_spec(spec)
    raw = bound_file(guards, path, TRAINER_SHA).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == TRAINER_SHA, 'trainer changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(original))
    require(original_code['deployed_code_rank.py'] == original.RANK_SHA256, 'original rank helper pin differs')
    for name, pin in original.REFERENCES.items():
        require(original_code[name] == pin['source'], 'original reference source pin differs')
        original.selected_ast(reference_root / name, pin)
    fit = descriptor_json(launch['fit_inventory'], guards)
    inventory = descriptor_json(launch['initialized_inventory'], guards)
    require(fit['schema'] == 'native256-fit-export-collected-inputs-v1' and fit['quality_read'] is False and
            inventory['schema'] == 'native256-full-initialized-cpu-collected-inputs-v1' and
            inventory['quality_read'] is False and inventory['updates'] == 0 and
            inventory['paired_schedules_images_exact'] is True, 'original collected inventories differ')
    entry = inventory['records']['so400']
    cpu = entry['cpu']
    selected = {'receipt': cpu['receipt'], 'log': cpu['log'], 'unit': cpu['unit'],
                'invocation_id': cpu['invocation_id'], 'service_seconds': cpu['service_seconds'],
                'native_peak_rss_kib': cpu['native_peak_rss_kib'], 'both_locks_held': cpu['both_locks_held']}
    qualifier_root = Path(entry['qualifier_root'])
    qualifier_code = original.closure(qualifier_root, entry['qualifier_execution_sha256'], original.QUALIFIER_FILES, guards)
    q = original.load_bare('_cached_readout_qualifier', qualifier_root / 'qualify_siglip2_initialized_cpu.py',
                           qualifier_code['qualify_siglip2_initialized_cpu.py'])
    admission = original.FlatAdmission()
    initialized = admission.initialized_authority(q, SimpleNamespace(
        execution_sha256=entry['qualifier_execution_sha256'], authority=Path(entry['qualifier_authority']['path']),
        authority_sha256=entry['qualifier_authority']['sha256'], arm='so400', output=output))
    prior = initialized['guards'].copy()
    original_context = {'args': SimpleNamespace(arm='so400', seed=args.seed, output=output,
                        execution_sha256=reference['execution_sha256']), 'root': reference_root,
                        'code': original_code, 'guards': initialized['guards'], 'qualifier': q,
                        'initialized': initialized, 'source': initialized['source'],
                        'initialized_prereq_guards': prior,
                        'launch': {'qualifier_authority': entry['qualifier_authority'],
                                   'qualifier_execution_sha256': entry['qualifier_execution_sha256'],
                                   'qualifier_root': str(qualifier_root), 'selected_cpu': selected}}
    old_cpu, old_final = original.admit_cpu(original_context, admission)
    src = initialized['source_context']
    collected = fit['source_cpu']['so400']
    source_terminal = {'proof': {'path': collected['proof'], 'sha256': collected['proof_sha256']},
                       'log': {'path': collected['log'], 'sha256': collected['log_sha256']},
                       **{k: collected[k] for k in ('unit', 'invocation_id', 'service_seconds', 'native_peak_rss_kib')},
                       'both_locks_held': True}
    require(old_cpu['checkpoint'] == entry['checkpoint'] and old_cpu['width'] == entry['width'] == 1152 and
            old_cpu['state']['schedules'] == entry['schedules'] and
            all(old_cpu[k] == entry[k] for k in ('ordered_input_sha256', 'ordered_rgb_sha256')) and
            int(old_final['values']['memory.peak']) == entry['whole_cgroup_peak_bytes'] and entry['swap_bytes'] == 0 and
            src['launch']['source_cpu']['so400'] == source_terminal and
            src['launch']['sources'] == fit['sources'] and src['launch']['fit_manifest'] == fit['fit'] and
            initialized['source_context']['proof']['checkpoint'] == fit['source_cpu']['so400']['source_checkpoint'] and
            initialized['source_context']['args'].execution_sha256 == fit['source_execution_sha256'] and
            initialized['source_context']['args'].fit_manifest_sha256 == fit['fit']['sha256'],
            'collected original source/initializedCPU inventories disagree')
    for path, digest in {**guards, **old_cpu['input_guards'],
                         entry['checkpoint']['path']: entry['checkpoint']['sha256']}.items():
        admission.bound_file(initialized['guards'], path, digest)
    roots = (root, reference_root, qualifier_root, initialized['pca']['root'],
             initialized['source_context']['root'], initialized['source_context']['own_root'])
    require(all(not output.is_relative_to(p) for p in roots) and
            all(not root.is_relative_to(p) and not p.is_relative_to(root) for p in roots[1:]),
            'separate immutable closures/output required')
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'original': original,
               'old': original_context, 'initialized': initialized, 'source': source_binding(initialized),
               'guards': initialized['guards'], 'old_cpu': old_cpu, 'old_cpu_final': old_final,
               'terminals': {}, 'terminal_cgroups': {}}
    wanted = [] if args.phase == 'cpu' else [('cpu', 'control', launch['selected_cpu'])]
    if args.phase == 'train':
        wanted += [('mechanics', arm, launch['selected_mechanics'][arm]) for arm in ARMS]
    ids = {old_cpu['invocation']['invocation_id'], initialized['source_context']['proof']['invocation']['invocation_id']}
    for phase, arm, terminal in wanted:
        record = admission.descriptor_json(terminal['receipt'], context['guards'])
        check_terminal_record(record, launch, phase, arm)
        original_launch = admission.descriptor_json(record['authority'], context['guards'])
        invocation = record['invocation']
        require(original_launch == record['launch'] and record['code'] == code and
                record['authority']['sha256'] == record['authority_sha256'] and
                record['execution_sha256'] == args.execution_sha256 and
                invocation['argv'] == [str(root / 'train_siglip2_cached_readout.py'),
                    '--execution-sha256', args.execution_sha256, '--authority', record['authority']['path'],
                    '--authority-sha256', record['authority']['sha256'], '--phase', phase, '--arm', arm,
                    '--seed', str(SEEDS[0]), '--output', str(Path(terminal['receipt']['path']).parent)] and
                all(invocation[k] == old_cpu['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
                (invocation['cuda_visible_devices'] == '' if phase == 'cpu' else
                 invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8') and
                record['numerical_flags'] == old_cpu['numerical_flags'], 'original new-method launch/interpreter differs')
        require(record['source'] == context['source'] and
                all(record['input_guards'].get(p) == h for p, h in prior.items()), 'terminal source/input closure differs')
        if phase == 'cpu':
            checkpoint = record['checkpoint']
            require(checkpoint.keys() == {'path', 'sha256'} and
                    checkpoint['path'] == str(Path(terminal['receipt']['path']).parent / 'heads.pt') and
                    record['input_guards'].get(checkpoint['path']) == checkpoint['sha256'], 'CPU head checkpoint authority differs')
        require(record['invocation']['invocation_id'] not in ids, 'duplicate original unit invocation')
        ids.add(record['invocation']['invocation_id'])
        context['terminal_cgroups'][phase + ':' + arm] = admission.admit_terminal(
            record, terminal, policy(phase)['seconds'], context['guards'])
        for path, digest in record['input_guards'].items():
            admission.bound_file(context['guards'], path, digest)
        context['terminals'][phase + ':' + arm] = record
    return context


def schedule(target, seed):
    """Original class-balanced RNG arithmetic, extended prospectively to1000."""
    import numpy as np
    require(seed in SEEDS, 'fixed schedule seed required')
    target = np.asarray(target, dtype=np.int64)
    names = sorted(set(target.tolist()))
    require(target.shape == (13283,) and names == list(range(2004)), 'dense FIT target inventory differs')
    rng = np.random.default_rng(seed)
    order = rng.permutation(names)
    members = {c: np.flatnonzero(target == c) for c in names}
    batches = np.asarray([[int(rng.choice(members[int(c)])) for c in
                           order[(step * 64 + np.arange(64)) % len(order)]]
                          for step in range(1000)], dtype=np.int64)
    require(batches.shape == (1000, 64) and all(len(set(target[b].tolist())) == 64 for b in batches) and
            0 <= int(batches.min()) and int(batches.max()) < 13283, 'fixed class-balanced schedule differs')
    return batches


def head_from(arm, tensors=None, values=None, features=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    require(arm in ARMS, 'fixed head arm required')
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.primary = nn.Linear(1152, 128)
            self.down = nn.Linear(1152, 32, bias=False)
            self.up = nn.Linear(32, 128, bias=False)
            self.register_buffer('center', torch.zeros(1152))
            self.register_buffer('preactivation_std', torch.ones(()))
        def residual(self, unit):
            z = self.down(unit - self.center)
            return self.up(.5 * z if arm == 'control' else F.gelu(z, approximate='none'))
        def forward(self, source):
            unit = F.normalize(source.float(), dim=1)
            return self.primary(unit) + self.residual(unit)
    with torch.random.fork_rng(devices=[]):
        head = Residual().float()
    if tensors is not None:
        head.load_state_dict(tensors, strict=True)
    else:
        require(values is not None and features is not None and features.shape == (13283, 1152),
                'complete FIT initializer required')
        with torch.no_grad():
            head.primary.load_state_dict({'weight': values['head.weight'], 'bias': values['head.bias']}, strict=True)
            unit = F.normalize(features, dim=1)
            head.center.copy_(unit.mean(0))
            nn.init.kaiming_uniform_(head.down.weight, a=5**.5,
                                    generator=torch.Generator().manual_seed(179034))
            std = head.down(unit - head.center).std(unbiased=False)
            require(torch.isfinite(std).item() and float(std) > 0, 'FIT preactivation std undefined')
            head.preactivation_std.copy_(std)
            head.down.weight.div_(std)
            head.up.weight.zero_()
    require(sum(p.numel() for p in head.parameters()) == 188544, '188544 head scalars required')
    return head


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


def fresh(context, ref, arm, seed, device):
    import numpy as np
    import torch
    q, initialized = context['old']['qualifier'], context['initialized']
    values = q.load_initializers(initialized)
    features = torch.from_numpy(np.load(context['source']['features']['path'], allow_pickle=False).copy())
    require(features.dtype == torch.float32 and features.shape == (13283, 1152) and
            torch.isfinite(features).all().item(), 'FP32 FIT13283x1152 required')
    head = head_from(arm, values=values, features=features)
    qualified = context['terminals'].get('cpu:control')
    if qualified is not None:
        require(context['original'].fingerprint(dict(head.state_dict())) == qualified['initial_head_sha256'],
                'actual initial head differs from CPU authority')
    head.to(device).train()
    classifier = torch.nn.Parameter(values['classifier'].to(device).clone())
    params, optimizer = optimizer_state(head, classifier)
    schedules = {str(s): torch.from_numpy(schedule(values['target'].tolist(), s)) for s in SEEDS}
    for s in SEEDS:
        require(torch.equal(schedules[str(s)][:100], torch.from_numpy(q.schedule(values['target'].tolist(), s))) and
                initialized['source'].tensor_fact(schedules[str(s)][:100]) == context['old_cpu']['state']['schedules'][str(s)],
                'original first100 schedule parity differs')
    return {'head': head, 'classifier': classifier, 'bank': values['bank'].to(device).detach().clone(),
            'target': values['target'].to(device).clone(), 'pca': {n: values[n].clone() for n in ('mean', 'components')},
            'features': features.to(device), 'schedules': schedules, 'params': params, 'optimizer': optimizer,
            'positive': ref.member_bank_positive_ordinals(values['target'].numpy(), allow_singletons=True).to(device),
            'seed': seed, 'counter': 0,
            'scaler': torch.amp.GradScaler('cuda', init_scale=128) if device == 'cuda' else None}


def identity(context, state, flags, arm):
    original = context['original']
    return {'method': method(context['launch']), 'arm': arm, 'seed': state['seed'],
            'selected_cpu': context['launch']['selected_cpu'],
            'source': context['source'], 'parameter_names': PARAMETERS,
            'feature_version': state['features']._version,
            'feature_state_sha256': original.fingerprint(state['features']),
            'optimizer_defaults': state['optimizer'].defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
            'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
            'static_sha256': original.fingerprint({n: state[n] for n in ('pca', 'target', 'positive', 'schedules')}),
            'buffers_sha256': original.fingerprint(dict(state['head'].named_buffers())),
            'positive_shape': tuple(state['positive'].shape),
            'schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]), 'numerical_flags': flags}


def payload(state, ident, flags):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'head': dict(state['head'].state_dict()),
            'classifier': state['classifier'].detach(), 'bank': state['bank'], 'target': state['target'],
            'pca': state['pca'], 'positive': state['positive'], 'schedules': state['schedules'],
            'optimizer': state['optimizer'].state_dict(), 'optimizer_defaults': state['optimizer'].defaults.copy(),
            'scaler': state['scaler'].state_dict(), 'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'counter': state['counter'], 'seed': state['seed'],
            'numerical_flags': flags, 'source': ident['source']}


def check_payload(saved, ident, step):
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            type(saved['counter']) is int and saved['counter'] == step and saved['seed'] == ident['seed'] and
            saved['source'] == ident['source'] and saved['numerical_flags'] == ident['numerical_flags'] and
            saved['optimizer_defaults'] == ident['optimizer_defaults'] and ident['parameter_names'] == PARAMETERS,
            'complete resume identity/state differs')
    optimizer = saved['optimizer']
    require(optimizer.keys() == {'state', 'param_groups'} and
            optimizer['param_groups'] == ident['optimizer_serial_groups'] and
            [i for g in optimizer['param_groups'] for i in g['params']] == list(range(5)) and
            all(type(i) is int for g in optimizer['param_groups'] for i in g['params']) and
            all(type(i) is int for i in optimizer['state']) and
            set(optimizer['state']) == (set(range(5)) if step else set()), 'EXACT five optimizer states/order required')
    require(all(v.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == step
                for v in optimizer['state'].values()) and
            saved['scaler'].keys() == {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'} and
            math.isfinite(saved['scaler']['scale']) and saved['scaler']['scale'] == 128 and
            saved['scaler']['growth_factor'] == 2 and saved['scaler']['backoff_factor'] == .5 and
            saved['scaler']['growth_interval'] == 2000 and saved['scaler']['_growth_tracker'] == step and
            len(saved['cuda_rng']) == 1, 'optimizer counters/scaler/RNG differ')
    def tensor(value, shape, dtype):
        require(tuple(value.shape) == shape and str(value.dtype) == dtype, 'complete tensor layout differs')
    head_shapes = {'primary.weight': SHAPES[0], 'primary.bias': SHAPES[1],
                   'down.weight': SHAPES[2], 'up.weight': SHAPES[3],
                   'center': (1152,), 'preactivation_std': ()}
    require(saved['head'].keys() == head_shapes.keys() and saved['pca'].keys() == {'mean', 'components'} and
            saved['schedules'].keys() == {str(s) for s in SEEDS}, 'complete head/PCA/schedules inventory differs')
    for name, shape in head_shapes.items():
        tensor(saved['head'][name], shape, 'torch.float32')
    for value, shape in ((saved['classifier'], SHAPES[4]), (saved['bank'], (13283, 128)),
                         (saved['pca']['mean'], (1152,)), (saved['pca']['components'], (128, 1152))):
        tensor(value, shape, 'torch.float32')
    tensor(saved['target'], (13283,), 'torch.int64')
    tensor(saved['positive'], ident['positive_shape'], 'torch.int64')
    for value in saved['schedules'].values():
        tensor(value, (1000, 64), 'torch.int64')
    for value in [saved['cpu_rng'], *saved['cuda_rng']]:
        require(len(value.shape) == 1 and value.shape[0] > 0 and str(value.dtype) == 'torch.uint8', 'complete RNG layout differs')
    for i, moments in optimizer['state'].items():
        tensor(moments['step'], (), 'torch.float32')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor(moments[name], SHAPES[i], 'torch.float32')


def integrity(context, state, ident):
    import torch
    original = context['original']
    actual = [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]
    require([n for n, _ in actual] == PARAMETERS and [tuple(p.shape) for _, p in actual] == SHAPES and
            len({id(p) for _, p in actual}) == 5 and
            [id(p) for _, p in actual] == [id(p) for _, p in state['params']] ==
            [id(p) for g in state['optimizer'].param_groups for p in g['params']] and
            all(p.dtype == torch.float32 and p.device.type == 'cuda' and p.requires_grad and
                torch.isfinite(p).all().item() for _, p in actual), 'live five-member inventory differs')
    require(state['head'].training and all(m.training and not m._forward_hooks and not m._forward_pre_hooks and
            not m._backward_hooks for m in state['head'].modules()), 'head modes/hooks differ')
    require(state['features'].shape == (13283, 1152) and state['features'].dtype == torch.float32 and
            not state['features'].requires_grad and state['features'].grad is None and
            state['features']._version == ident['feature_version'], 'immutable in-memory FIT cache changed')
    require(original.fingerprint({n: state[n] for n in ('pca', 'target', 'positive', 'schedules')}) == ident['static_sha256'] and
            original.fingerprint(dict(state['head'].named_buffers())) == ident['buffers_sha256'] and
            state['bank'].shape == (13283, 128) and state['bank'].dtype == torch.float32 and
            not state['bank'].requires_grad and state['bank'].grad is None and torch.isfinite(state['bank']).all().item(),
            'complete static state/buffers/bank differ')
    require(state['optimizer'].defaults == ident['optimizer_defaults'] and
            [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups] == ident['optimizer_groups'],
            'live optimizer options differ')
    saved = payload(state, ident, ident['numerical_flags'])
    check_payload(saved, ident, state['counter'])
    for i, shape in enumerate(SHAPES):
        if state['counter']:
            require(all(tuple(saved['optimizer']['state'][i][n].shape) == shape and
                        saved['optimizer']['state'][i][n].dtype == torch.float32 and
                        torch.isfinite(saved['optimizer']['state'][i][n]).all().item() for n in ('exp_avg', 'exp_avg_sq')),
                    'complete finite FP32 moment inventory differs')


def save(context, state, ident, path):
    import torch
    saved = payload(state, ident, ident['numerical_flags'])
    check_payload(saved, ident, state['counter'])
    with context['initialized']['source_context']['extract'].exclusive(path) as stream:
        torch.save(saved, stream)
        stream.flush(); os.fsync(stream.fileno())
    return context['initialized']['init'].sha(path), context['original'].fingerprint(saved)


def restore(context, ref, path, sha, digest, ident, step, features):
    """No prior head/optimizer references survive. Strict complete fresh reload."""
    import torch
    original = context['original']
    bound_file({}, path, sha)
    disk = torch.load(path, map_location='cpu', weights_only=True)
    check_payload(disk, ident, step)
    require(original.fingerprint(disk) == digest, 'serialized complete state fingerprint differs')
    head = head_from(ident['arm'], tensors=disk['head']).cuda().train()
    classifier = torch.nn.Parameter(disk['classifier'].cuda().clone())
    params, optimizer = optimizer_state(head, classifier)
    state = {'head': head, 'classifier': classifier, 'params': params, 'optimizer': optimizer,
             'bank': disk['bank'].cuda().detach().clone(), 'target': disk['target'].cuda().clone(),
             'positive': disk['positive'].cuda().clone(), 'pca': {n: v.clone() for n, v in disk['pca'].items()},
             'schedules': {n: v.clone() for n, v in disk['schedules'].items()}, 'seed': disk['seed'],
             'counter': disk['counter'], 'features': features, 'scaler': torch.amp.GradScaler('cuda', init_scale=128)}
    require(torch.equal(state['positive'], ref.member_bank_positive_ordinals(state['target'].cpu().numpy(),
                allow_singletons=True).cuda()), 'strict positive ordinal reload differs')
    optimizer.load_state_dict(disk['optimizer'])
    state['scaler'].load_state_dict(disk['scaler'])
    torch.random.set_rng_state(disk['cpu_rng'].clone())
    torch.cuda.set_rng_state_all([v.clone() for v in disk['cuda_rng']])
    del disk, head, classifier, params, optimizer
    gc.collect()
    integrity(context, state, ident)
    require(original.fingerprint(payload(state, ident, ident['numerical_flags'])) == digest,
            'independent complete resume reload differs')
    return state


def terms(context, ref, state, index):
    import torch
    raw = state['head'](state['features'][index])
    ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                     torch.arange(128, device=raw.device).unsqueeze(0), margin=.3, scale=64)
    rank = context['original'].valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
    return raw, ce, rank


def update(context, ref, state, ident, step):
    import torch
    torch.cuda.synchronize(); tick = time.perf_counter()
    require(state['counter'] == step - 1 and 1 <= step <= 1000, 'update counter differs')
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version, rows, ce_sum, rank_sum = state['bank']._version, [], 0., 0.
    for offset in range(0, 64, 16):
        index = torch.tensor(batch[offset:offset + 16], device='cuda')
        with torch.autocast(device_type='cuda', enabled=False):
            raw, ce, rank = terms(context, ref, state, index)
            loss = (ce + 8 * rank) * .25
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item() and torch.isfinite(loss).item(),
                'nonfinite cached update')
        scaler.scale(loss).backward()
        rows.append(raw.detach())
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
    require(scaler.get_scale() >= scale, 'skipped update forbidden')
    state['counter'] += 1
    refresh, positions = ref.member_bank_refresh_rows(batch)
    raw = torch.cat(rows)
    state['bank'][torch.tensor(refresh, device='cuda')] = ref.member_bank_refresh_values(
        raw, raw, torch.tensor(positions, device='cuda'), live_head=False)
    require(state['bank']._version == version + 1, 'last duplicate detached refresh differs')
    optimizer.zero_grad(set_to_none=True)
    integrity(context, state, ident)
    require(context['initialized']['source'].numerical_flags() == ident['numerical_flags'] and
            torch.cuda.max_memory_allocated() < 10_000_000_000, 'flags/whole-unit CUDA peak differs')
    row = {'step': step, 'batch': batch, 'schedule_sha256': ident['schedule_sha256'],
           'feature_rows_sha256': context['original'].fingerprint(state['features'][batch]),
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients,
           'state_sha256': context['original'].fingerprint(payload(state, ident, ident['numerical_flags']))}
    torch.cuda.synchronize(); row['seconds'] = time.perf_counter() - tick
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def calibration(context, state):
    import torch
    from torch.nn import functional as F
    packing = sys.modules.get('_siglip2_pinned_initialized_packing')
    if packing is None:
        q = context['old']['qualifier']
        packing = q.load_bare('_siglip2_pinned_initialized_packing', context['initialized']['root'] /
                             'joint_relational_compaction.py', q.PACKING_HELPER_SHA256)
    with torch.no_grad(), torch.autocast(device_type=state['features'].device.type, enabled=False):
        raw = state['head'](state['features'][:64])
        unit = F.normalize(raw, dim=1)
        packed = packing.pack_int8_unit_embeddings(unit.cpu())
    return {'raw': raw.cpu(), 'unit': unit.cpu(), 'codes': packed.codes.cpu(),
            'inverse_norms': packed.inverse_norms.cpu(), 'wire': packed.to_bytes()}


def affine_reconstruction(state):
    """Archived FIT-only float64 affine reconstruction statistic, no tuning."""
    import torch
    from torch.nn import functional as F
    with torch.no_grad():
        unit = F.normalize(state['features'].cpu(), dim=1)
        head = head_from('candidate' if state['arm'] == 'candidate' else 'control',
                         tensors={n: v.cpu() for n, v in state['head'].state_dict().items()})
        design = torch.cat((unit, torch.ones(len(unit), 1)), 1).double()
        residual = head.residual(unit).double()
        fit = torch.linalg.lstsq(design, residual, driver='gelsd').solution
        energy = residual.square().sum()
        return {'fit_unexplained_energy_fraction': None if float(energy) <= 0 else
                float((residual - design @ fit).square().sum() / energy), 'residual_energy': float(energy)}


def cpu_witnesses(context, ref, output):
    import torch
    from torch.nn import functional as F
    source, src = context['initialized']['source'], context['initialized']['source_context']
    flags, rng = source.numerical_flags(), torch.random.get_rng_state().clone()
    model, processor, roles = source.fresh_source(src)
    pixels, pooled, sample = source.pixels_and_raw(src, model, processor)
    facts = source.model_facts(model, processor, roles, context['initialized']['packages'])
    require(facts == src['proof']['runtime'] and sample == src['proof']['sample'], 'actual original source witness differs')
    model.requires_grad_(False).eval()
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), 'encoder must be entirely frozen')
    static = context['original'].fingerprint({'vision': model.state_dict(), 'buffers': dict(model.named_buffers())})
    del model, processor
    gc.collect()
    path = Path(context['source']['checkpoint']['path'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(disk.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and disk['runtime'] == facts and
            disk['config'] == facts['config'] and source.tensor_fact(disk['cpu_rng']) == src['proof']['cpu_rng'],
            'complete source checkpoint differs')
    model = source.construct(disk['config'], src)
    with path.open('rb') as stream:
        pages = context['original'].CheckpointPages(stream)
        context['original'].load_vision(model, disk['vision'], pages)
        buffers = dict(model.named_buffers())
        require(buffers.keys() == disk['buffers'].keys(), 'complete source buffers differ')
        with torch.no_grad():
            for name, value in buffers.items():
                require(source.tensor_fact(value) == source.tensor_fact(disk['buffers'][name]), 'source buffer bytes differ')
                value.copy_(disk['buffers'][name]); pages.consume(disk['buffers'][name])
    del disk, buffers, value
    gc.collect()
    roles = source.configure_roles(model, src['expected'], model.config.num_hidden_layers)
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(src['entry']['input']['preprocessor']['path'],
                                                  local_files_only=True, backend='torchvision')
    pixels2, pooled2, sample2 = source.pixels_and_raw(src, model, processor)
    require(source.model_facts(model, processor, roles, context['initialized']['packages']) == facts and
            torch.equal(pixels, pixels2) and torch.equal(pooled, pooled2) and sample == sample2, 'source independent reload differs')
    model.requires_grad_(False).eval()
    require(context['original'].fingerprint({'vision': model.state_dict(), 'buffers': dict(model.named_buffers())}) == static,
            'source reload state/buffers differ')
    # Export normalizes pooled.float(), then the head normalizes that cached row.
    cached = F.normalize(pooled.float(), dim=1)
    states, witnesses, heads = {}, {}, {}
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu')
        head = state['head']
        with torch.no_grad():
            require(torch.equal(head(state['features']), head.primary(F.normalize(state['features'], dim=1))) and
                    torch.equal(F.normalize(head(state['features']), dim=1), state['bank']), 'initial PCA/bank parity differs')
            raw = head(cached)
            require(torch.equal(raw, head.primary(F.normalize(F.normalize(pooled.float(), dim=1), dim=1))),
                    'source-to-head two-stage normalization differs')
        witnesses[arm] = calibration(context, state)
        heads[arm] = {n: v.detach().clone() for n, v in head.state_dict().items()}
        states[arm] = {'head': heads[arm], 'classifier': state['classifier'].detach().clone(),
                       'bank': state['bank'].clone(), 'target': state['target'].clone(),
                       'positive': state['positive'].clone(), 'schedules': state['schedules'], 'pca': state['pca'],
                       'optimizer': state['optimizer'].state_dict(), 'optimizer_defaults': state['optimizer'].defaults.copy()}
        # Disposable full-objective witnesses; no CPU trained state is admitted.
        index = state['schedules'][str(SEEDS[0])][0]
        for step in (0, 1):
            state['optimizer'].zero_grad(set_to_none=True)
            for part in index.split(16):
                _, ce, rank = terms(context, ref, state, part)
                ((ce + 8 * rank) * .25).backward()
            require(torch.isfinite(head.up.weight.grad).all().item() and float(head.up.weight.grad.norm()) > 0,
                    'nonzero up gradient required')
            require(torch.isfinite(head.down.weight.grad).all().item() and
                    (float(head.down.weight.grad.norm()) == 0 if step == 0 else float(head.down.weight.grad.norm()) > 0),
                    'initial zero/subsequent nonzero down gradient required')
            torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
            state['optimizer'].step()
        del state, head, _, ce, rank, raw, index, part
    require(context['original'].fingerprint(heads['control']) == context['original'].fingerprint(heads['candidate']) and
            context['original'].fingerprint(witnesses['control']) == context['original'].fingerprint(witnesses['candidate']),
            'initial matched arm raw/unit/packed parity differs')
    initial_head_sha = context['original'].fingerprint(heads['control'])
    require(context['original'].fingerprint({'vision': model.state_dict(), 'buffers': dict(model.named_buffers())}) == static and
            all(not p.requires_grad and p.grad is None for p in model.parameters()), 'frozen source changed during head gradients')
    del model, processor, pooled, pooled2, pixels, pixels2, heads
    gc.collect()
    checkpoint = output / 'heads.pt'
    saved = {'schema': SCHEMA, 'source': context['source'], 'states': states, 'cpu_rng': rng, 'numerical_flags': flags}
    digest = context['original'].fingerprint(saved)
    with src['extract'].exclusive(checkpoint) as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    del states, saved
    gc.collect()
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(context['original'].fingerprint(disk) == digest and disk.keys() ==
            {'schema', 'source', 'states', 'cpu_rng', 'numerical_flags'}, 'CPU complete head reload differs')
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu')
        state['head'].load_state_dict(disk['states'][arm]['head'], strict=True)
        saved_state = disk['states'][arm]
        require(saved_state.keys() == {'head', 'classifier', 'bank', 'target', 'positive', 'schedules',
                                      'pca', 'optimizer', 'optimizer_defaults'} and
                saved_state['optimizer_defaults'] == state['optimizer'].defaults and
                saved_state['optimizer'] == state['optimizer'].state_dict(), 'complete CPU optimizer reload differs')
        # Independently recomputed static state is required, not merely copied.
        require(context['original'].fingerprint({n: state[n] for n in
                ('classifier', 'bank', 'target', 'positive', 'schedules', 'pca')}) ==
                context['original'].fingerprint({n: saved_state[n] for n in
                ('classifier', 'bank', 'target', 'positive', 'schedules', 'pca')}), 'CPU complete static reload differs')
        require(context['original'].fingerprint(calibration(context, state)) == context['original'].fingerprint(witnesses[arm]),
                'CPU raw/unit/packed independent head reload differs')
        del state
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and
            not torch.cuda.is_initialized(), 'CPU constructor RNG/flags/CUDA differs')
    bound_file(context['guards'], checkpoint, src['extract'].sha(checkpoint))
    return {'completed_step': 0, 'checkpoint': {'path': str(checkpoint), 'sha256': context['guards'][str(checkpoint)]},
            'cuda_initialized': False, 'initial_arm_parity': True, 'source_reload_exact': True,
            'gradient_witnesses': True, 'source_state_sha256': static,
            'initial_head_sha256': initial_head_sha,
            'cpu_state_sha256': digest, 'cpu_witnesses_sha256': context['original'].fingerprint(witnesses)}


def gpu_run(context, ref, output, flags):
    import torch
    args = context['args']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible uninitialized CUDA device required')
    torch.cuda.manual_seed_all(SEEDS[0])
    state = fresh(context, ref, args.arm, args.seed, 'cuda')
    ident = identity(context, state, flags, args.arm)
    integrity(context, state, ident)
    initial = context['original'].fingerprint(payload(state, ident, flags))
    features = state['features']  # Only the immutable cache survives reload boundaries.
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 1000
    with TemporaryDirectory(prefix='discard-mechanics-', dir=output) as directory:
        temporary = Path(directory)
        tick = time.perf_counter()
        for step in range(1, total + 1):
            row = update(context, ref, state, ident, step)
            if args.phase == 'train' and args.seed == SEEDS[0] and step <= 17:
                original = context['terminals']['mechanics:' + args.arm]['steps'][step - 1]
                require(diagnostic(row) == diagnostic(original), 'fresh first17 mechanics replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
        training_seconds = time.perf_counter() - tick
        state['arm'] = args.arm
        reconstruction = affine_reconstruction(state)
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        witness = context['original'].fingerprint(calibration(context, state))
        del state
        gc.collect(); torch.cuda.empty_cache()
        if args.phase == 'mechanics':
            state = restore(context, ref, temporary / 'step8.pt', sha8, digest8, ident, 8, features)
            resumed = [update(context, ref, state, ident, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    context['original'].fingerprint(payload(state, ident, flags)) == digest, '17 versus independent8+9 differs')
            del state
            gc.collect(); torch.cuda.empty_cache()
        state = restore(context, ref, checkpoint, sha, digest, ident, total, features)
        require(context['original'].fingerprint(calibration(context, state)) == witness, 'strict final raw/unit/packed reload differs')
        require(context['original'].fingerprint(features) == ident['feature_state_sha256'], 'complete in-memory FIT bytes changed')
        require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                'cached training/reload changed CUDA RNG')
        del state, features
        gc.collect(); torch.cuda.empty_cache()
    return {'completed_step': total, 'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'initial_state_sha256': initial, 'terminal_state_sha256': digest, 'identity': ident, 'steps': rows,
            'resumed_steps': resumed, 'replay_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics', 'affine_reconstruction': reconstruction,
            'training_wall_seconds': training_seconds, 'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
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
    cpu_origins = source.imported_origins(context['initialized']['source_context']['extract'],
                                         context['initialized']['packages'])
    require(all(context['old_cpu']['origins']['files'].get(p) == h for p, h in cpu_origins['files'].items()),
            'actual CPU imports differ from original qualified origins')
    ref = context['original'].reference_math(context['old'])
    args.output.mkdir()
    result = cpu_witnesses(context, ref, args.output) if args.phase == 'cpu' else gpu_run(context, ref, args.output, flags)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'constructor/update/reload RNG/flags differ')
    origins = context['original'].exit_rehash(context['old'])  # Complete UNCACHED source/archive/environment SHA pass.
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'], 'exit own closure differs')
    after = source.cgroup_memory()
    init.admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and 0 < rss <= 8 * 1024**2 and
            (args.phase != 'cpu' or not torch.cuda.is_initialized()), 'whole-unit time/RSS/CUDA caps differ')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed,
               'pass': True, 'quality_read': False, 'strict_reload_exact': True, 'optimizer_members': 5,
               'training_qualified': args.phase == 'train', 'trained_state_reused': False,
               'fixed_cached_views': True, 'image_augmentation': False, 'encoder_updates': 0,
               'head_scalars': 188544, 'trainable_scalars': 445056, 'source': context['source'],
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
        raise SystemExit('Cached readout rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
