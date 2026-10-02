#!/usr/bin/env python3
"""Native256 So400 learned-head continuation; only existing post-layernorm adapts.

Freeze exactly FILES in execution.json. CLI (fixed order):
python -B ROOT/train_siglip2_postln_adaptation.py --execution-sha256 SHA
 --authority FILE --authority-sha256 SHA --phase cpu|mechanics|train
 --arm control|candidate --seed 179061|179069 --output NEW_ABSOLUTE_DIRECTORY

Authority has exactly LAUNCH_KEYS; genuine_reference=GENUINE_REFERENCE,
warm_start={launch:FILE,terminal:UNIT,checkpoint:FILE,terminal_state_sha256:SHA}.
FILE={path:canonical absolute file,sha256:actual lowercase SHA256}.
UNIT={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
native_peak_rss_kib,both_locks_held:true}. Parent supplies actual terminals.
partition is pinned PARTITION_SHA; recipe=RECIPE, resource_policy=policy(phase).
CPU control061 qualifies BOTH arms; selected_cpu/selected_mechanics are null.
Mechanics eacharm061 requires selected_cpu UNIT. TRAIN100 requires that CPU
and selected_mechanics={control:UNIT,candidate:UNIT}. Root alone admits069
after the first selection gate. Every new run starts at local0 from the same
complete authenticated control061 endpoint; no cached update or PCA refit.

CPU120s/CUDA-hidden; mechanics and TRAIN300s;8GiB/noSwap/events0/CUDA<10GB,
both lifetime locks and original normal terminal required. Mechanics full17
versus independent8+9 is discarded; TRAIN061 reproduces its first17. FP32
weights, F16 encoder autocast on CUDA, FP32 normalized pooled/factored head.
CPU uses fresh TRAIN witnesses in FP32; GPU parity uses canonical B16.
No held pixels. Source image/cache hashing is integrity metadata only.

resume.pt is a complete delta with exact PAYLOAD_KEYS: immutable full source,
two norm tensors, config/all buffers (including nonpersistent positions),
learned head INCLUDING buffers, classifier/bank/static schedules/rows,
optimizer/scaler/RNG/counters/flags. Independent reload reconstructs all448
native tensors sequentially; version audits each step and uncached complement
bytes at every release/exit. Receipt acceptance still needs the parent's
original successful unit footer; an inner receipt never certifies its exit.
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
from pathlib import Path
import re
import resource
import statistics
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import weakref

SCHEMA = 'siglip2-postln-adaptation-v1'
AUTHORITY_SCHEMA = 'siglip2-postln-adaptation-launch-v1'
FILES = {'train_siglip2_postln_adaptation.py', 'test_siglip2_postln_adaptation.py'}
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
GENUINE_REFERENCE = {'root': '/home/riomus/runs/sfora-so400-genuine-view-train-source-v1',
                     'execution_sha256': '2003e9a6adba30c8f90fa17f6437cbba1f4b6cd2d12e75906195d16229846e08'}
GENUINE_CODE = {'train_siglip2_genuine_views.py': '788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96',
                'test_siglip2_genuine_view_training.py': '1effe6e87b9d667d29bc7b8904023634e7913f53c4b0a17e360728f83e1cd77b'}
WARM_ROOT = '/home/riomus/runs/sfora-so400-genuine-view-train-control-179061-v1'
WARM_LAUNCH = {'path': GENUINE_REFERENCE['root'] + '/authority-train-control-179061-v1.json',
               'sha256': '84ceaa5b819d9c9db89514bed13b140dab898e842ec6cbf85eb17dce123e6655'}
WARM_RECEIPT = {'path': WARM_ROOT + '/receipt.json',
                'sha256': 'a85d7ef29909d677a12510bdafce2e1edbcd8e62d768918af572bdd3cfa87a5f'}
WARM_CHECKPOINT = {'path': WARM_ROOT + '/resume.pt',
                   'sha256': '97db53ab934edbef2656c6e9509e4518421eaba02d3864bdc336eedd7787c7d2'}
WARM_STATE_SHA = 'b78bd945256438bfd24873347e26d62c970bf9d85048663301d4fe1236ac35a9'
PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'
ARMS, SEEDS = ('control', 'candidate'), (179061, 179069)
NORM = ['post_layernorm.weight', 'post_layernorm.bias']
HEAD_NAMES = ['compact_head.primary.weight', 'compact_head.primary.bias',
              'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
HEAD_SHAPES = [(128, 1152), (128,), (32, 1152), (128, 32), (1008, 128)]
HEAD_LAYOUT = dict(zip(('primary.weight', 'primary.bias', 'down.weight', 'up.weight',
                        'center', 'preactivation_std'), [*HEAD_SHAPES[:4], (1152,), ()], strict=True))
RECIPE = {'width': 1152, 'rows': 6355, 'classes': 1008, 'output_dim': 128, 'rank': 32,
          'steps': 100, 'warm_source_updates': 1000, 'local_initial_counter': 0,
          'batch': 64, 'microbatch': 16, 'seeds': list(SEEDS), 'margin': .3, 'scale': 64,
          'rank_weight': 8, 'native_learning_rate': 1e-5, 'head_learning_rate': 1e-4,
          'weight_decay': .05, 'clip': 1, 'initial_scaler': 128,
          'control_native_tensors': 0, 'candidate_native_tensors': NORM,
          'schedule': 'unchanged genuine full1000 first100; masks metadata only',
          'bank': 'shared learned canonical-B32 bank; native pre-update detached last duplicate',
          'input': 'canonical TRAIN RGB B64/micro16; no cached update inputs',
          'arithmetic': 'FP32 vision; CUDA F16 encoder autocast; FP32 normalized pooled and factored affine head'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'genuine_reference',
               'warm_start', 'partition', 'recipe', 'resource_policy', 'both_locks_held',
               'selected_cpu', 'selected_mechanics'}
STATIC_KEYS = ('pca', 'target', 'positive', 'schedules', 'masks', 'continuation',
               'original_rows', 'partition', 'warm_start')
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'norm', 'config', 'buffers', 'head', 'classifier',
                'bank', *STATIC_KEYS, 'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng',
                'cuda_rng', 'counter', 'seed', 'numerical_flags'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_form(value):
    return json.loads(json.dumps(value, allow_nan=False))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
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
    require(digest.hexdigest() == expected, 'file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'exact FILE required')
    with bound_file(guards, value['path'], value['sha256']).open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)


def closure(root, sha, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': sha}, guards)
    require(code.keys() == set(names), 'exact code closure required')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def policy(phase):
    require(phase in ('cpu', 'mechanics', 'train'), 'fixed phase required')
    return {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def check_unit(unit):
    require(isinstance(unit, dict) and unit.keys() == {'receipt', 'log', 'unit', 'invocation_id',
            'service_seconds', 'native_peak_rss_kib', 'both_locks_held'} and unit['both_locks_held'] is True,
            'complete original UNIT required')
    require(re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            all(type(unit[k]) in (int, float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')), 'actual UNIT resources required')
    for k in ('receipt', 'log'):
        require(unit[k].keys() == {'path', 'sha256'} and Path(unit[k]['path']).is_absolute() and
                re.fullmatch('[0-9a-f]{64}', unit[k]['sha256']), 'UNIT FILE differs')


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and args.arm in ARMS and launch['seed'] == args.seed and
            type(args.seed) is int and args.seed in SEEDS and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'fixed continuation launch differs')
    require(launch['genuine_reference'] == GENUINE_REFERENCE and launch['partition'].keys() == {'path', 'sha256'} and
            launch['partition']['sha256'] == PARTITION_SHA, 'immutable source/partition differs')
    warm = launch['warm_start']
    require(warm.keys() == {'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'} and
            warm['launch'] == WARM_LAUNCH and warm['checkpoint'] == WARM_CHECKPOINT and
            warm['terminal_state_sha256'] == WARM_STATE_SHA and warm['terminal']['receipt'] == WARM_RECEIPT,
            'accepted control061 warm endpoint required')
    check_unit(warm['terminal'])
    require((args.phase != 'cpu' or (args.arm, args.seed) == ('control', SEEDS[0])) and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]), 'qualification arm/seed differs')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if launch['selected_cpu'] is not None:
        check_unit(launch['selected_cpu'])
    if args.phase == 'train':
        require(launch['selected_mechanics'].keys() == set(ARMS), 'BOTH mechanics required')
        for unit in launch['selected_mechanics'].values():
            check_unit(unit)


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'genuine_reference', 'warm_start', 'partition', 'recipe')}


def cli(root, authority, sha, execution, phase, arm, seed, output):
    return [str(Path(root) / 'train_siglip2_postln_adaptation.py'), '--execution-sha256', execution,
            '--authority', str(authority), '--authority-sha256', sha, '--phase', phase, '--arm', arm,
            '--seed', str(seed), '--output', str(output)]


def zero_events(value):
    events = dict(line.split() for line in value['values']['memory.events'].splitlines())
    require(all(int(events.get(k, '-1')) == 0 for k in ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')),
            'disallowed cgroup event')


def admit_unit(context, record, unit, phase):
    check_unit(unit)
    final = context['admission'].admit_terminal(record, unit, policy(phase)['seconds'], context['guards'])
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        zero_events(value)
    require(unit['invocation_id'] not in context['invocations'], 'duplicate unit invocation')
    context['invocations'].add(unit['invocation_id'])
    for path, digest in record['input_guards'].items():
        context['admission'].bound_file(context['guards'], path, digest)
    return final


def check_warm_record(genuine, selected, record, warm):
    require(record['schema'] == genuine.SCHEMA and record['launch'] == selected['launch'] and
            record['source'] == selected['source'] and record['code'] == GENUINE_CODE and
            record['execution_sha256'] == GENUINE_REFERENCE['execution_sha256'] and
            record['authority'] == WARM_LAUNCH and record['authority_sha256'] == WARM_LAUNCH['sha256'] and
            (record['phase'], record['arm'], record['seed'], record['completed_step']) == ('train', 'control', 179061, 1000) and
            record['checkpoint'] == WARM_CHECKPOINT and record['terminal_state_sha256'] == WARM_STATE_SHA and
            record['partition_sha256'] == PARTITION_SHA and record['optimizer_members'] == 5 and
            record['encoder_updates'] == 0 and record['head_scalars'] == 188544 and record['trainable_scalars'] == 317568 and
            record['resumed_steps'] == [] and record['resource_policy'] == genuine.policy('train') and
            record['terminal_cgroups'] == selected['terminal_cgroups'] and
            record['numerical_flags'] == selected['source_cpu']['numerical_flags'], 'complete original warm receipt differs')
    require(all(record[k] is True for k in ('pass', 'training_qualified', 'strict_reload_exact', 'exit_rehash_pass')) and
            all(record[k] is False for k in ('quality_read', 'trained_state_reused', 'training_state_discarded', 'replay_exact')) and
            0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'original warm terminal not accepted')
    genuine.check_steps(record['steps'], 1, 1000)
    ident, mechanics = record['identity'], selected['terminals']['mechanics:control']
    require(ident['method'] == genuine.method(record['launch']) and ident['source'] == selected['source'] and
            (ident['arm'], ident['seed'], ident['device']) == ('control', 179061, 'cuda') and
            ident['parameter_names'] == HEAD_NAMES and ident['selected_cpu'] == selected['launch']['selected_cpu'] and
            ident['numerical_flags'] == record['numerical_flags'] and
            record['steps'][-1]['state_sha256'] == WARM_STATE_SHA and
            all(r['schedule_sha256'] == ident['schedule_sha256'] for r in record['steps']) and
            record['initial_state_sha256'] == mechanics['initial_state_sha256'] and
            [genuine.diagnostic(r) for r in record['steps'][:17]] == [genuine.diagnostic(r) for r in mechanics['steps']],
            'original warm identity/replay differs')
    expected_argv = [str(Path(GENUINE_REFERENCE['root']) / 'train_siglip2_genuine_views.py'),
        '--execution-sha256', GENUINE_REFERENCE['execution_sha256'], '--authority', WARM_LAUNCH['path'],
        '--authority-sha256', WARM_LAUNCH['sha256'], '--phase', 'train', '--arm', 'control', '--seed', '179061',
        '--output', WARM_ROOT]
    require(record['invocation']['argv'] == expected_argv and record['invocation']['cublas_workspace_config'] == ':4096:8' and
            record['invocation']['cuda_visible_devices'] not in (None, '') and
            all(record['invocation'][k] == selected['source_cpu']['invocation'][k]
                for k in ('python', 'python_sha256', 'python_version')) and
            all(record['input_guards'].get(p) == h for p, h in selected['guards'].items()) and
            record['input_guards'].get(WARM_CHECKPOINT['path']) == WARM_CHECKPOINT['sha256'] and
            record['median_update_seconds'] == statistics.median(r['seconds'] for r in record['steps'][2:]) and
            0 < sum(r['seconds'] for r in record['steps']) <= record['training_wall_seconds'] <= record['wall_seconds'],
            'original warm invocation/guards/cost differs')


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink() and not output.is_relative_to(root), 'exclusive separate output required')
    oldroot = Path(GENUINE_REFERENCE['root'])
    require(not root.is_relative_to(oldroot) and not oldroot.is_relative_to(root) and not output.is_relative_to(oldroot),
            'separate original closure required')
    require(closure(oldroot, GENUINE_REFERENCE['execution_sha256'], GENUINE_CODE, guards) == GENUINE_CODE,
            'original trainer2 code differs')
    path = oldroot / 'train_siglip2_genuine_views.py'
    spec = importlib.util.spec_from_file_location('_postln_genuine', path)
    require('_postln_genuine' not in sys.modules and spec is not None and spec.loader is not None, 'bare source origin differs')
    genuine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = genuine
    raw = bound_file(guards, path, GENUINE_CODE[path.name]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == GENUINE_CODE[path.name], 'source changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(genuine))
    selected = genuine.authority(SimpleNamespace(execution_sha256=GENUINE_REFERENCE['execution_sha256'],
        authority=Path(WARM_LAUNCH['path']), authority_sha256=WARM_LAUNCH['sha256'],
        phase='train', arm='control', seed=179061, output=output))
    require(selected['launch'] == read_json(WARM_LAUNCH, guards) and
            launch['partition'] == selected['launch']['partition'], 'original warm launch/partition differs')
    record = read_json(WARM_RECEIPT, guards)
    check_warm_record(genuine, selected, record, launch['warm_start'])
    for p, h in selected['guards'].items():
        require(guards.setdefault(p, h) == h, 'source guard conflict')
    admission = selected['original'].FlatAdmission()
    admission.init = selected['genuine']['reference']
    for path, digest in selected['guards'].items():
        admission.verified.add(str(admission.register(guards, path, digest)))
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'guards': guards,
               'genuine': genuine, 'selected': selected, 'prior': selected['genuine']['prior'],
               'source_driver': selected['source_driver'], 'original': selected['original'],
               'extract': selected['extract'], 'admission': admission, 'warm_record': record,
               'invocations': {selected['source_cpu']['invocation']['invocation_id']},
               'terminals': {}, 'terminal_cgroups': {}, 'phase_seconds': {}}
    context['source'] = {'genuine_reference': GENUINE_REFERENCE, 'warm_start': launch['warm_start'],
                         'native_source': selected['genuine']['reference'].binding(context['prior']),
                         'warm_source': selected['source'], 'partition_sha256': PARTITION_SHA}
    context['terminal_cgroups']['warm'] = admit_unit(context, record, launch['warm_start']['terminal'], 'train')
    required_guards = dict(guards)
    wanted = [] if args.phase == 'cpu' else [('cpu', 'control', launch['selected_cpu'])]
    if args.phase == 'train':
        wanted += [('mechanics', arm, launch['selected_mechanics'][arm]) for arm in ARMS]
    for phase, arm, unit in wanted:
        proof = read_json(unit['receipt'], guards)
        check_terminal_record(proof, launch, phase, arm)
        require(proof['source'] == context['source'] and proof['code'] == code and
                proof['numerical_flags'] == selected['source_cpu']['numerical_flags'] and
                proof['authority_sha256'] == proof['authority']['sha256'] and
                read_json(proof['authority'], guards) == proof['launch'] and
                all(proof['input_guards'].get(p) == h for p, h in required_guards.items()
                    if p != str(args.authority)) and
                proof['invocation']['argv'] == cli(root, proof['authority']['path'], proof['authority']['sha256'],
                    args.execution_sha256, phase, arm, SEEDS[0], Path(unit['receipt']['path']).parent) and
                all(proof['invocation'][k] == selected['source_cpu']['invocation'][k]
                    for k in ('python', 'python_sha256', 'python_version')), 'new prerequisite source/CLI/guards differ')
        context['terminal_cgroups'][phase + ':' + arm] = admit_unit(context, proof, unit, phase)
        context['terminals'][phase + ':' + arm] = proof
    if args.phase == 'train':
        c, a = (context['terminals']['mechanics:' + arm] for arm in ARMS)
        require(c['initial_raw_unit_packed_sha256'] == a['initial_raw_unit_packed_sha256'] and
                c['identity']['warm_members_sha256'] == a['identity']['warm_members_sha256'],
                'matched B16 initial arm parity failed')
    return context


def parameter_names(arm):
    require(arm in ARMS, 'fixed norm arm required')
    return (NORM if arm == 'candidate' else []) + HEAD_NAMES


def parameter_shapes(arm):
    return ([(1152,), (1152,)] if arm == 'candidate' else []) + HEAD_SHAPES


def check_membership(pairs, groups, arm):
    require([n for n, _ in pairs] == parameter_names(arm) and
            [tuple(p.shape) for _, p in pairs] == parameter_shapes(arm) and
            len({id(p) for _, p in pairs}) == len(pairs) and
            [id(p) for _, p in pairs] == [id(p) for group in groups for p in group] and
            all(p.requires_grad and str(p.dtype) == 'torch.float32' for _, p in pairs),
            'exact disjoint ordered five/seven optimizer members required')


def check_optimizer_identity(ident):
    defaults = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in ident['optimizer_defaults']:
        defaults['decoupled_weight_decay'] = True
    candidate = ident['arm'] == 'candidate'
    groups = [dict(defaults, lr=rate) for rate in ([1e-5] if candidate else []) + [1e-4, 1e-4]]
    members = [[0, 1], [2, 3, 4, 5], [6]] if candidate else [[0, 1, 2, 3], [4]]
    require(ident['parameter_names'] == parameter_names(ident['arm']) and
            ident['optimizer_defaults'] == defaults and ident['optimizer_groups'] == groups and
            ident['optimizer_serial_groups'] == [dict(g, params=p) for g, p in zip(groups, members, strict=True)],
            'fixed native/head/classifier AdamW identity differs')


def check_continuation(full, selected):
    require(len(full) == 1000 and len(selected) == 100 and selected == full[:100] and
            all(len(row) == 64 and all(type(v) is int and 0 <= v < 6355 for v in row) for row in full),
            'unchanged full1000 first100 schedule required')


def require_no_model(context):
    require(context.get('model_ref', lambda: None)() is None, 'previous complete model still owned')


def claim_model(context, model):
    require_no_model(context)
    context['model_ref'] = weakref.ref(model)


def check_complement(context, model, excluded=NORM):
    for name, tensor in model.named_parameters():
        if name not in excluded:
            require(context['source_driver'].tensor_fact(tensor.detach().cpu())['sha256'] == context['prior']['mapping'][name]['sha256'],
                    'fresh frozen complement bytes differ: ' + name)


def clone_tree(tree, device='cpu'):
    import torch
    if isinstance(tree, torch.Tensor):
        return tree.detach().to(device, copy=True)
    if isinstance(tree, dict):
        return {k: clone_tree(v, device) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(clone_tree(v, device) for v in tree)
    return tree


def finite_tree(tree):
    import torch
    if isinstance(tree, torch.Tensor):
        require(torch.isfinite(tree).all().item(), 'nonfinite complete tensor tree')
    elif isinstance(tree, dict):
        for value in tree.values():
            finite_tree(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            finite_tree(value)


def prepare_native(context):
    """After stdlib admission only: verify COMPLETE warm state, then copy members."""
    import torch
    genuine, selected = context['genuine'], context['selected']
    require(not torch.cuda.is_initialized(), 'CUDA initialized before native admission')
    flags = selected['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(context['source_driver'].numerical_flags() == flags, 'source numerical flags differ')
    context['flags'] = flags
    context['prior']['packages'] = selected['packages']
    audit_origins(context, initial=True)
    context['ref'] = context['original'].reference_math(selected['math_context'])
    pack = selected['launch']['helpers']['packing']
    context['packing'] = genuine.load_helper('_postln_packing', pack['path'], pack['sha256'], context['guards'])
    path = bound_file(context['guards'], WARM_CHECKPOINT['path'], WARM_CHECKPOINT['sha256'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = context['original'].CheckpointPages(stream)
        genuine.check_payload(disk, disk['identity'], 1000)
        require(json_form(disk['identity']) == context['warm_record']['identity'] and
                context['original'].fingerprint(disk, consumed=pages.consume) == WARM_STATE_SHA,
                'complete learned warm payload differs')
        finite_tree(disk)
        require(context['original'].fingerprint({k: disk[k] for k in genuine.STATIC_KEYS}) ==
                disk['identity']['static_sha256'] and
                context['original'].fingerprint({k: disk['head'][k] for k in ('center', 'preactivation_std')}) ==
                disk['identity']['buffers_sha256'] and disk['target'].tolist() == selected['target'] and
                disk['original_rows'].tolist() == selected['partition']['panels']['train']['original_rows'] and
                disk['partition'] == selected['partition'], 'complete original warm static state differs')
        for seed in SEEDS:
            schedule, masks = genuine.schedule_and_masks(selected['target'], seed)
            require(torch.equal(disk['schedules'][str(seed)], torch.from_numpy(schedule)) and
                    torch.equal(disk['masks'][str(seed)], torch.from_numpy(masks)), 'original full warm schedules differ')
        require(torch.equal(disk['positive'], context['ref'].member_bank_positive_ordinals(
            disk['target'].numpy(), allow_singletons=True)), 'complete warm positive table differs')
        # Only now extract learned head INCLUDING buffers, classifier and terminal bank.
        context['initial'] = {k: clone_tree(disk[k]) for k in
                              ('head', 'classifier', 'bank', 'pca', 'target', 'positive',
                               'schedules', 'masks', 'original_rows', 'partition')}
    del disk
    gc.collect()
    initial = context['initial']
    initial['continuation'] = {str(s): initial['schedules'][str(s)][:100].clone() for s in SEEDS}
    for seed in SEEDS:
        check_continuation(initial['schedules'][str(seed)].tolist(), initial['continuation'][str(seed)].tolist())
    initial['warm_start'] = {'endpoint': context['launch']['warm_start'], 'warm_source_updates': 1000,
                             'local_initial_counter': 0, 'bank_context': 'original canonical B32'}
    context['warm_members_sha256'] = context['original'].fingerprint({k: initial[k] for k in ('head', 'classifier', 'bank')})
    context['static_sha256'] = context['original'].fingerprint({k: initial[k] for k in STATIC_KEYS})
    audit_origins(context)


def audit_origins(context, initial=False):
    source, prior = context['source_driver'], context['prior']
    origins = source.imported_origins(context['extract'], context['selected']['packages'])
    expected = context['selected']['source_cpu']['origins']
    if initial:
        require(all(expected['modules'].get(n) == p for n, p in origins['modules'].items()) and
                all(expected['files'].get(p) == h for p, h in origins['files'].items()), 'original CPU import origin differs')
    for kind in ('files', 'modules'):
        known = {}
        for proof in (expected, context['warm_record']['origins']):
            for name, value in proof[kind].items():
                require(known.setdefault(name, value) == value, 'conflicting original native origin authority')
        require(all(known.get(n) == v for n, v in origins[kind].items()), 'unknown or changed native origin')
    for path, digest in origins['files'].items():
        bound_file(context['guards'], path, digest)
        require(prior['guards'].setdefault(path, digest) == digest, 'original native origin changed')
    context['origins'] = origins


def optimizer_state(model, head, classifier, arm):
    import torch
    native = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    pairs = native + [('compact_head.' + n, p) for n, p in head.named_parameters()] + [('classifier', classifier)]
    groups = ([{'params': [p for _, p in native], 'lr': 1e-5}] if native else [])
    groups += [{'params': list(head.parameters()), 'lr': 1e-4}, {'params': [classifier], 'lr': 1e-4}]
    check_membership(pairs, [g['params'] for g in groups], arm)
    optimizer = torch.optim.AdamW(groups, weight_decay=.05)
    defaults = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and
            all({k: v for k, v in g.items() if k != 'params'} == {**defaults, 'lr': rate}
                for g, rate in zip(optimizer.param_groups, ([1e-5] if native else []) + [1e-4, 1e-4], strict=True)),
            'fresh AdamW defaults/groups differ')
    return pairs, optimizer


def runtime(context, state):
    return context['original'].runtime({'source': context['source_driver'],
        'initialized': {'packages': context['selected']['packages']}}, state)


def native_inventory(model):
    return [{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype),
             'role': 'trainable' if p.requires_grad else 'frozen'} for n, p in model.named_parameters()]


def fresh(context, arm, seed, device):
    import torch
    require_no_model(context)
    require(arm in ARMS and seed in SEEDS and device in ('cpu', 'cuda'), 'fresh state profile differs')
    source, prior = context['source_driver'], context['prior']
    tick = time.perf_counter()
    model, processor, roles = source.fresh_source(prior)
    claim_model(context, model)
    require(source.model_facts(model, processor, roles, context['selected']['packages']) == prior['proof']['runtime'] and
            len(roles) == 448 and model.config.hidden_size == 1152 and
            sum(p.numel() for p in model.parameters()) == 427888064,
            'complete original448 source/config/buffers/processor/runtime differs')
    for name, value in model.named_parameters():
        value.requires_grad_(arm == 'candidate' and name in NORM)
    require([n for n, p in model.named_parameters() if p.requires_grad] == (NORM if arm == 'candidate' else []) and
            all(tuple(dict(model.named_parameters())[n].shape) == (1152,) for n in NORM), 'native0/2 norm roles differ')
    model.to(device).train()
    initial = context['initial']
    head = context['selected']['cached'].head_from('control', tensors=initial['head']).to(device).train()
    classifier = torch.nn.Parameter(initial['classifier'].to(device, copy=True))
    pairs, optimizer = optimizer_state(model, head, classifier, arm)
    state = {'model': model, 'processor': processor, 'head': head, 'classifier': classifier,
             'bank': initial['bank'].to(device, copy=True).detach(), 'arm': arm, 'seed': seed, 'device': device,
             'params': pairs, 'optimizer': optimizer, 'counter': 0,
             'scaler': torch.amp.GradScaler(device, init_scale=128),
             **{k: clone_tree(initial[k]) for k in STATIC_KEYS}}
    state['target'] = state['target'].to(device)
    state['positive'] = state['positive'].to(device)
    state['frozen_versions'] = [(n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad]
    state['frozen_cache'] = context['original'].frozen_cache(model)
    require(context['original'].fingerprint({k: payload_member(state, k) for k in ('head', 'classifier', 'bank')}) ==
            context['warm_members_sha256'], 'fresh learned members/buffers changed')
    add_seconds(context, 'construction', tick)
    return state


def payload_member(state, key):
    return dict(state['head'].state_dict()) if key == 'head' else state[key]


def identity(context, state):
    original, model = context['original'], state['model']
    return json_form({'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'],
            'seed': state['seed'], 'device': state['device'], 'parameter_names': parameter_names(state['arm']),
            'config': model.config.to_dict(), 'runtime': runtime(context, state),
            'native_inventory': native_inventory(model),
            'complement_sha256': original.fingerprint({n: p for n, p in model.named_parameters() if n not in NORM},
                                                     state['frozen_cache']),
            'buffers_sha256': original.fingerprint(dict(model.named_buffers())),
            'head_buffers_sha256': original.fingerprint(dict(state['head'].named_buffers())),
            'static_sha256': context['static_sha256'], 'warm_members_sha256': context['warm_members_sha256'],
            'optimizer_defaults': state['optimizer'].defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
            'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
            'positive_shape': tuple(state['positive'].shape),
            'schedule_sha256': original.fingerprint(state['continuation'][str(state['seed'])]),
            'full_schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]),
            'numerical_flags': context['flags']})


def payload(state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'source': ident['source'],
            'norm': {n: p.detach() for n, p in state['model'].named_parameters() if n in NORM},
            'config': state['model'].config.to_dict(), 'buffers': dict(state['model'].named_buffers()),
            'head': dict(state['head'].state_dict()), 'classifier': state['classifier'].detach(), 'bank': state['bank'],
            **{k: state[k] for k in STATIC_KEYS}, 'optimizer': state['optimizer'].state_dict(),
            'optimizer_defaults': state['optimizer'].defaults.copy(), 'scaler': state['scaler'].state_dict(),
            'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if state['device'] == 'cuda' else [],
            'counter': state['counter'], 'seed': state['seed'], 'numerical_flags': ident['numerical_flags']}


def check_payload(saved, ident, step):
    check_optimizer_identity(ident)
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            saved['source'] == ident['source'] and json_form(saved['config']) == ident['config'] and
            saved['numerical_flags'] == ident['numerical_flags'] and json_form(saved['optimizer_defaults']) == ident['optimizer_defaults'] and
            type(saved['counter']) is int and saved['counter'] == step and 0 <= step <= 100 and
            saved['seed'] == ident['seed'] and ident['parameter_names'] == parameter_names(ident['arm']),
            'complete delta payload identity differs')
    def tensor(value, shape, dtype='torch.float32'):
        require(tuple(value.shape) == tuple(shape) and str(value.dtype) == dtype, 'complete payload tensor layout differs')
    require(saved['norm'].keys() == set(NORM) and saved['head'].keys() == HEAD_LAYOUT.keys() and
            saved['buffers'].keys() == {'embeddings.position_ids'}, 'complete norm/head/buffer inventory differs')
    for value in saved['norm'].values():
        tensor(value, (1152,))
    for name, shape in HEAD_LAYOUT.items():
        tensor(saved['head'][name], shape)
    tensor(saved['buffers']['embeddings.position_ids'], (1, 256), 'torch.int64')
    tensor(saved['classifier'], (1008, 128)); tensor(saved['bank'], (6355, 128))
    tensor(saved['target'], (6355,), 'torch.int64'); tensor(saved['positive'], ident['positive_shape'], 'torch.int64')
    tensor(saved['original_rows'], (6355,), 'torch.int64')
    require(saved['pca'].keys() == {'mean', 'components'} and
            saved['schedules'].keys() == saved['masks'].keys() == saved['continuation'].keys() == {str(s) for s in SEEDS},
            'complete static schedule/PCA inventory differs')
    tensor(saved['pca']['mean'], (1152,)); tensor(saved['pca']['components'], (128, 1152))
    for seed in SEEDS:
        tensor(saved['schedules'][str(seed)], (1000, 64), 'torch.int64')
        tensor(saved['masks'][str(seed)], (1000, 64), 'torch.bool')
        tensor(saved['continuation'][str(seed)], (100, 64), 'torch.int64')
    require(saved['partition']['schema'] and saved['warm_start'] == {
        'endpoint': ident['method']['warm_start'], 'warm_source_updates': 1000, 'local_initial_counter': 0,
        'bank_context': 'original canonical B32'}, 'warm/local provenance differs')
    opt, count = saved['optimizer'], len(parameter_names(ident['arm']))
    require(opt.keys() == {'state', 'param_groups'} and json_form(opt['param_groups']) == ident['optimizer_serial_groups'] and
            [i for g in opt['param_groups'] for i in g['params']] == list(range(count)) and
            all(type(i) is int for g in opt['param_groups'] for i in g['params']) and
            all(type(i) is int for i in opt['state']) and set(opt['state']) == (set(range(count)) if step else set()),
            'exact five/seven optimizer state inventory differs')
    for i, moments in opt['state'].items():
        require(moments.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and float(moments['step']) == step,
                'optimizer successful-step counter differs')
        tensor(moments['step'], ())
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor(moments[name], parameter_shapes(ident['arm'])[i])
    require(saved['scaler'] == {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
            'growth_interval': 2000, '_growth_tracker': step} and ident['device'] in ('cpu', 'cuda') and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0), 'scaler/RNG inventory differs')
    for value in [saved['cpu_rng'], *saved['cuda_rng']]:
        require(len(value.shape) == 1 and value.shape[0] > 0 and str(value.dtype) == 'torch.uint8', 'RNG tensor layout differs')


def integrity(context, state, ident, fresh_bytes=False):
    import torch
    tick = time.perf_counter()
    model, original = state['model'], context['original']
    require(native_inventory(model) == ident['native_inventory'] and len(ident['native_inventory']) == 448 and
            json_form(model.config.to_dict()) == ident['config'] and runtime(context, state) == ident['runtime'] and
            model.config._attn_implementation == 'sdpa' and all(p.dtype == torch.float32 and p.device.type == ident['device']
                for p in model.parameters()), 'complete native448 roles/config/runtime differs')
    require(all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks and
                not getattr(m, 'gradient_checkpointing', False) for net in (model, state['head']) for m in net.modules()),
            'training module modes/hooks differ')
    actual = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    actual += [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]
    check_membership(actual, [g['params'] for g in state['optimizer'].param_groups], state['arm'])
    require([id(p) for _, p in actual] == [id(p) for _, p in state['params']] and
            all(p.device.type == ident['device'] and torch.isfinite(p).all().item() for _, p in actual) and
            [(n, p.data_ptr(), p._version) for n, p in model.named_parameters() if not p.requires_grad] == state['frozen_versions'] and
            all(p.grad is None for p in model.parameters() if not p.requires_grad), 'live members/frozen versions/grads differ')
    buffers = dict(model.named_buffers())
    require(original.fingerprint(buffers) == ident['buffers_sha256'] and
            'position_ids' in model.embeddings._non_persistent_buffers_set and
            torch.equal(buffers['embeddings.position_ids'], torch.arange(256, device=ident['device']).expand(1, -1)) and
            original.fingerprint(dict(state['head'].named_buffers())) == ident['head_buffers_sha256'] and
            original.fingerprint({k: state[k] for k in STATIC_KEYS}) == ident['static_sha256'] == context['static_sha256'],
            'complete static/head/nonpersistent buffers differ')
    require(json_form(state['optimizer'].defaults) == ident['optimizer_defaults'] and
            json_form([{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups]) == ident['optimizer_groups'] and
            not state['bank'].requires_grad and state['bank'].grad is None and
            state['bank'].device.type == ident['device'] and torch.isfinite(state['bank']).all().item(), 'optimizer/bank differs')
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    finite_tree(saved['optimizer'])
    require(context['source_driver'].numerical_flags() == ident['numerical_flags'], 'numerical flags changed')
    if fresh_bytes:
        check_complement(context, model, excluded=NORM if state['arm'] == 'candidate' else ())
        require(original.fingerprint({n: p for n, p in model.named_parameters() if n not in NORM}) ==
                ident['complement_sha256'], 'uncached446 complement changed')
    add_seconds(context, 'integrity', tick)


def add_seconds(context, name, tick):
    timings = context.setdefault('phase_seconds', {})
    delta = time.perf_counter() - tick
    timings[name] = timings.get(name, 0.) + delta
    if context['args'].phase == 'cpu':
        print(json.dumps({'event': 'POSTLN_PHASE', 'phase': name, 'delta_seconds': delta,
                          'cumulative_seconds': timings[name]}), flush=True)


def release(context, state):
    import torch
    state.clear()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    require_no_model(context)


def save(context, state, ident, path):
    import torch
    state['optimizer'].zero_grad(set_to_none=True)
    integrity(context, state, ident, fresh_bytes=True)
    tick = time.perf_counter()
    saved = payload(state, ident)
    digest = context['original'].fingerprint(saved)
    with context['extract'].exclusive(path) as stream:
        writer = context['original'].CheckpointWriter(stream)
        torch.save(saved, writer); writer.flush()
    sha = context['extract'].sha(path)
    add_seconds(context, 'serialization', tick)
    return sha, digest


def restore(context, path, sha, digest, ident, step):
    """Complete independent source reconstruction; no previous model or mmap owner."""
    import torch
    require_no_model(context)
    tick = time.perf_counter()
    path = bound_file({}, path, sha)
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = context['original'].CheckpointPages(stream)
        check_payload(disk, ident, step)
        require(context['original'].fingerprint(disk, consumed=pages.consume) == digest, 'complete serialized delta differs')
        finite_tree(disk)
        add_seconds(context, 'reload', tick)
        state = fresh(context, ident['arm'], ident['seed'], ident['device'])
        tick = time.perf_counter()
        require(identity(context, state) == ident, 'independent source/static/runtime reconstruction differs')
        with torch.no_grad():
            for name, param in state['model'].named_parameters():
                if name in NORM:
                    if state['arm'] == 'control':
                        require(torch.equal(param.cpu(), disk['norm'][name]), 'control norm changed')
                    else:
                        param.copy_(disk['norm'][name])
                    pages.consume(disk['norm'][name])
            for name, value in state['model'].named_buffers():
                require(torch.equal(value.cpu(), disk['buffers'][name]), 'immutable independent buffer differs')
                pages.consume(disk['buffers'][name])
            state['classifier'].copy_(disk['classifier'])
            pages.consume(disk['classifier'])
        state['head'].load_state_dict(disk['head'], strict=True)
        for value in disk['head'].values():
            pages.consume(value)
        state['bank'] = pages.copy(disk['bank'], ident['device']).detach()
        require(context['original'].fingerprint({k: disk[k] for k in STATIC_KEYS}) == ident['static_sha256'],
                'complete independent static reload differs')
        # Keep independently recreated static tensors, after byte equality; no mapped aliases survive.
        optimizer = clone_tree(disk['optimizer'])
        state['optimizer'].load_state_dict(optimizer)
        state['scaler'].load_state_dict(disk['scaler'])
        state['counter'] = step
        torch.random.set_rng_state(pages.copy(disk['cpu_rng']))
        if ident['device'] == 'cuda':
            torch.cuda.set_rng_state_all([pages.copy(v) for v in disk['cuda_rng']])
    del disk, optimizer, param, value
    gc.collect()
    add_seconds(context, 'reload', tick)
    integrity(context, state, ident, fresh_bytes=True)
    tick = time.perf_counter()
    require(context['original'].fingerprint(payload(state, ident)) == digest, 'complete independent restored state differs')
    add_seconds(context, 'reload', tick)
    return state


def canonical_row(context, state, ordinal):
    manifest, fit = context['selected']['genuine']['selected'], context['prior']['fit']
    require(type(ordinal) is int and 0 <= ordinal < len(manifest['original_rows']), 'TRAIN local ordinal differs')
    original = manifest['original_rows'][ordinal]
    require(type(original) is int and 0 <= original < len(fit['rows']) and
            original == int(state['original_rows'][ordinal]), 'TRAIN to original FIT ordinal differs')
    row, path = manifest['rows'][ordinal], Path(manifest['resolved_paths'][ordinal])
    root = Path(fit['dataset_root'])
    target = manifest['targets'][ordinal]
    require(row == fit['rows'][original] and path == context['prior']['all_images'][original] and
            (root / row['relative_path']).resolve() == path and path.is_relative_to(root) and
            type(row['train_row']) is int and row['train_row'] >= 0 and target == int(state['target'][ordinal]) and
            state['partition']['panels']['train']['original_class_ids'][target] == fit['targets'][original] and
            row['product'] == fit['class_names'][fit['targets'][original]], 'original TRAIN row/path/product mapping differs')
    return row, path, {'train_local': ordinal, 'fit_ordinal': original, 'train_row': row['train_row'],
                       'path': str(path), 'relative_path': row['relative_path'], 'image_sha256': row['image_sha256'],
                       'target': target, 'original_target': fit['targets'][original], 'product': row['product']}


def canonical_pixels(context, state, batch):
    import torch
    from PIL import Image
    tick = time.perf_counter()
    require(0 < len(batch) <= 64 and all(type(i) is int and 0 <= i < 6355 for i in batch), 'TRAIN-only batch required')
    images, mapping, rgb = [], [], hashlib.sha256()
    before_rng = torch.random.get_rng_state().clone()
    try:
        for ordinal in batch:
            row, path, fact = canonical_row(context, state, ordinal)
            mapping.append(fact)
            bound_file(context['guards'], path, row['image_sha256'])
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            images.append(image)
            rgb.update(str(image.size).encode()); rgb.update(image.tobytes())
        pixels = state['processor'](images=images, return_tensors='pt')['pixel_values']
    finally:
        for image in images:
            image.close()
    require(pixels.dtype == torch.float32 and tuple(pixels.shape) == (len(batch), 3, 256, 256) and
            not pixels.requires_grad and pixels.grad_fn is None and torch.isfinite(pixels).all().item() and
            torch.equal(before_rng, torch.random.get_rng_state()), 'canonical processor pixels/RNG differ')
    add_seconds(context, 'decode_preprocess', tick)
    return pixels, rgb.hexdigest(), context['original'].fingerprint(mapping)


def native_raw(state, pixels, capture=None):
    import torch
    from torch.nn import functional as F
    seen = []
    def norm_input(module, args):
        hidden = args[0]
        require(not hidden.requires_grad and hidden.grad_fn is None, 'post-layernorm input must be graph-free')
        seen.append(True)
        if capture is not None:
            capture.append(hidden.detach().clone())
    hook = state['model'].post_layernorm.register_forward_pre_hook(norm_input)
    try:
        with torch.autocast(device_type=state['device'], dtype=torch.float16, enabled=state['device'] == 'cuda'):
            pooled = state['model'](pixel_values=pixels.to(state['device'])).pooler_output
    finally:
        hook.remove()
    require(seen == [True] and torch.isfinite(pooled).all().item(), 'native forward/norm boundary differs')
    with torch.autocast(device_type=state['device'], enabled=False):
        raw = state['head'](F.normalize(pooled.float(), dim=1))
    require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item(), 'finite FP32 head output required')
    return raw


def packed_outputs(context, raw):
    import torch
    from torch.nn import functional as F
    with torch.no_grad():
        require((raw.norm(dim=1) > 0).all().item(), 'nonzero raw descriptors required')
        unit = F.normalize(raw, dim=1)
        packed = context['packing'].pack_int8_unit_embeddings(unit.cpu())
    return {'raw': raw.detach().cpu(), 'unit': unit.cpu(), 'codes': packed.codes.cpu(),
            'inverse_norms': packed.inverse_norms.cpu(), 'wire': packed.to_bytes()}


def calibration(context, state):
    import torch
    batch = state['continuation'][str(state['seed'])][0].tolist()
    if state['device'] == 'cpu':
        batch = batch[:2]
    pixels, rgb, mapping_sha = canonical_pixels(context, state, batch)
    tick = time.perf_counter()
    with torch.no_grad():
        raw = torch.cat([native_raw(state, pixels[start:start + 16]) for start in range(0, len(batch), 16)])
        result = packed_outputs(context, raw)
    add_seconds(context, 'witness_forward_pack', tick)
    return {**result, 'batch': batch, 'rgb_sha256': rgb, 'row_mapping_sha256': mapping_sha,
            'pixels_sha256': context['original'].fingerprint(pixels)}


def terms(context, state, raw, index):
    import torch
    ref = context['ref']
    ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                     torch.arange(128, device=state['device']).unsqueeze(0), margin=.3, scale=64)
    rank = context['original'].valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
    require(torch.isfinite(ce).item() and torch.isfinite(rank).item(), 'finite data loss required')
    return ce, rank


def gradient_norms(state):
    import torch
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for _, p in state['params']) and
            all(p.grad is None for p in state['model'].parameters() if not p.requires_grad), 'finite active/frozen gradients differ')
    result = {n: float(p.grad.double().norm()) for n, p in state['params']}
    require(all(math.isfinite(v) and v >= 0 for v in result.values()) and
            (state['arm'] != 'candidate' or all(result[n] > 0 for n in NORM)),
            'gamma and beta require nonzero DATA gradients through frozen pool')
    return result


def update(context, state, ident, step):
    import torch
    require(state['device'] == 'cuda' and state['counter'] == step - 1 and 1 <= step <= 100,
            'exact fresh continuation update required')
    torch.cuda.synchronize(); started = time.perf_counter()
    batch = state['continuation'][str(state['seed'])][step - 1].tolist()
    pixels, rgb, mapping_sha = canonical_pixels(context, state, batch)
    pixel_sha = context['original'].fingerprint(pixels)
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version = state['bank']._version
    ce_sum = rank_sum = 0.
    raw_rows = []
    tick = time.perf_counter()
    for start in range(0, 64, 16):
        index = torch.tensor(batch[start:start + 16], device=state['device'])
        raw = native_raw(state, pixels[start:start + 16])
        with torch.autocast(device_type='cuda', enabled=False):
            ce, rank = terms(context, state, raw, index)
            loss = (ce + 8 * rank) * .25
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach()) * .25; rank_sum += float(rank.detach()) * .25
        raw_rows.append(raw.detach())
        del raw, index, ce, rank, loss
    require(state['bank']._version == version, 'bank changed during data backward')
    scaler.unscale_(optimizer)
    gradients = gradient_norms(state)
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    require(scaler.get_scale() == scale == 128, 'optimizer skipped or scaler changed')
    state['counter'] += 1
    rows, positions = context['ref'].member_bank_refresh_rows(batch)
    raw = torch.cat(raw_rows).detach()
    clean_sha = context['original'].fingerprint(raw)
    state['bank'][torch.tensor(rows, device='cuda')] = context['ref'].member_bank_refresh_values(
        raw, raw, torch.tensor(positions, device='cuda'), live_head=False)
    require(state['bank']._version == version + 1 and not state['bank'].requires_grad, 'pre-update detached last-duplicate refresh differs')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); add_seconds(context, 'forward_backward_update', tick)
    integrity(context, state, ident)
    require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'lifetime CUDA peak exceeded')
    row = {'step': step, 'batch': batch, 'schedule_sha256': ident['schedule_sha256'], 'rgb_sha256': rgb,
           'pixels_sha256': pixel_sha, 'row_mapping_sha256': mapping_sha, 'clean_bank_sha256': clean_sha, 'ce': ce_sum, 'rank': rank_sum,
           'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(), 'preclip_norm': float(norm),
           'gradient_norms': gradients, 'graph_free_norm_input': True,
           'state_sha256': context['original'].fingerprint(payload(state, ident)),
           'seconds': time.perf_counter() - started}
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def check_steps(rows, start, count, arm):
    keys = {'step', 'batch', 'schedule_sha256', 'rgb_sha256', 'pixels_sha256', 'row_mapping_sha256', 'clean_bank_sha256',
            'ce', 'rank', 'loss', 'scale', 'preclip_norm', 'gradient_norms', 'graph_free_norm_input', 'state_sha256', 'seconds'}
    require(len(rows) == count, 'complete update records required')
    for step, row in enumerate(rows, start):
        require(row.keys() == keys and row['step'] == step and len(row['batch']) == 64 and
                all(type(v) is int and 0 <= v < 6355 for v in row['batch']) and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k]) for k in
                    ('schedule_sha256', 'rgb_sha256', 'pixels_sha256', 'row_mapping_sha256', 'clean_bank_sha256', 'state_sha256')) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('ce', 'rank', 'loss', 'scale', 'preclip_norm', 'seconds')) and row['seconds'] > 0 and
                row['loss'] == row['ce'] + 8 * row['rank'] and row['scale'] == 128 and row['graph_free_norm_input'] is True and
                row['gradient_norms'].keys() == set(parameter_names(arm)) and
                all(math.isfinite(v) and v >= 0 for v in row['gradient_norms'].values()) and
                (arm != 'candidate' or all(row['gradient_norms'][n] > 0 for n in NORM)), 'complete finite update record differs')


def cpu_data_witness(context, state):
    """Fresh TRAIN prefix once; independent gamma/beta changes use its detached tokens."""
    import torch
    from torch.nn import functional as F
    batch = state['continuation'][str(state['seed'])][0, :2].tolist()
    pixels, _, _ = canonical_pixels(context, state, batch)
    tick = time.perf_counter()
    captured = []
    raw = native_raw(state, pixels, capture=captured)
    require(len(captured) == 1 and captured[0].grad_fn is None, 'TRAIN norm input capture differs')
    index = torch.tensor(batch)
    ce, rank = terms(context, state, raw, index)
    (ce + 8 * rank).backward()
    gradients = gradient_norms(state)
    baseline = raw.detach().clone()
    changed = {}
    if state['arm'] == 'candidate':
        parameters = dict(state['model'].named_parameters())
        for name in NORM:
            param = parameters[name]
            channel = int(param.grad.abs().argmax())
            before = param.detach().clone()
            with torch.no_grad():
                param[channel].add_(.01)
                pooled = state['model'].head(state['model'].post_layernorm(captured[0]))
                perturbed = state['head'](F.normalize(pooled.float(), dim=1))
                require(torch.isfinite(perturbed).all().item() and not torch.equal(perturbed, baseline),
                        'separate observable norm perturbation failed: ' + name)
                changed[name] = {'channel': channel, 'delta': .01,
                                 'max_raw_difference': float((perturbed - baseline).abs().max())}
                param.copy_(before)
                require(torch.equal(param, before), 'norm perturbation restoration failed')
    state['optimizer'].zero_grad(set_to_none=True)
    add_seconds(context, 'cpu_forward_backward_witness', tick)
    return {'gradient_norms': gradients, 'graph_free_norm_input': True, 'perturbations': changed,
            'frozen_pool_backward': state['arm'] == 'candidate', 'train_rows': batch}


def bypass_version_witness(context, state):
    """A discarded CPU mutation proves the exit guard reads bytes, even through .data."""
    import torch
    name, param = next((n, p) for n, p in state['model'].named_parameters() if not p.requires_grad)
    version = param._version
    before = param.detach().reshape(-1)[0].clone()
    try:
        param.data.reshape(-1)[0].add_(.25)
        require(param._version == version and not torch.equal(param.detach().reshape(-1)[0], before),
                'version-bypass falsifier did not mutate without version change')
        try:
            check_complement(context, state['model'], excluded=NORM if state['arm'] == 'candidate' else ())
        except ValueError as error:
            require(str(error) == 'fresh frozen complement bytes differ: ' + name, 'unexpected tamper rejection')
        else:
            raise ValueError('uncached frozen-complement check accepted version-bypass mutation')
    finally:
        param.data.reshape(-1)[0].copy_(before)
    require(param._version == version and context['source_driver'].tensor_fact(param)['sha256'] ==
            context['prior']['mapping'][name]['sha256'], 'version-bypass falsifier restore failed')
    return True


def cpu_witnesses(context):
    original, output = context['original'], context['args'].output
    results = {}
    with TemporaryDirectory(prefix='discard-cpu-', dir=output) as directory:
        for arm in ARMS:
            state = fresh(context, arm, SEEDS[0], 'cpu')
            ident = identity(context, state)
            integrity(context, state, ident, fresh_bytes=True)
            witness = original.fingerprint(calibration(context, state))
            before = original.fingerprint(payload(state, ident))
            gradients = cpu_data_witness(context, state)
            bypass_rejected = bypass_version_witness(context, state)
            require(original.fingerprint(payload(state, ident)) == before, 'CPU gradient/perturbation changed complete state')
            check_complement(context, state['model'], excluded=())  # Candidate perturbations restored exactly.
            path = Path(directory) / (arm + '.pt')
            sha, digest = save(context, state, ident, path)
            release(context, state)
            state = restore(context, path, sha, digest, ident, 0)
            require(original.fingerprint(calibration(context, state)) == witness, 'independent CPU raw/unit/packed differs')
            integrity(context, state, ident, fresh_bytes=True)
            results[arm] = {'identity': ident, 'state_sha256': digest, 'raw_unit_packed_sha256': witness,
                            'gradient_witness': gradients, 'strict_reload_exact': True,
                            'bypass_version_tamper_rejected': bypass_rejected}
            release(context, state)
    require(results['control']['raw_unit_packed_sha256'] == results['candidate']['raw_unit_packed_sha256'],
            'matched initial CPU arm parity differs')
    return {'completed_step': 0, 'checkpoint': None, 'identity': None, 'arms': results,
            'initial_raw_unit_packed_sha256': results['control']['raw_unit_packed_sha256'],
            'raw_unit_packed_sha256': results['control']['raw_unit_packed_sha256'],
            'initial_state_sha256': None, 'terminal_state_sha256': None, 'steps': [], 'resumed_steps': [],
            'initial_arm_parity': True, 'gradient_witnesses': True, 'training_only_fit': True,
            'replay_exact': True, 'training_state_discarded': True, 'training_wall_seconds': 0.,
            'median_update_seconds': None, 'peak_cuda_allocated_bytes': 0}


def gpu_run(context):
    import torch
    args, original = context['args'], context['original']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible uninitialized CUDA device required')
    torch.cuda.manual_seed_all(SEEDS[0])
    state = fresh(context, args.arm, args.seed, 'cuda')
    ident = identity(context, state)
    cpu_ident = context['terminals']['cpu:control']['arms'][args.arm]['identity']
    require(ident == {**cpu_ident, 'device': 'cuda', 'seed': args.seed,
                     'schedule_sha256': original.fingerprint(state['continuation'][str(args.seed)]),
                     'full_schedule_sha256': original.fingerprint(state['schedules'][str(args.seed)])},
            'actual CUDA construction differs from new CPU-qualified complete identity')
    integrity(context, state, ident, fresh_bytes=True)
    start_digest = original.fingerprint(payload(state, ident))
    start_witness = original.fingerprint(calibration(context, state))
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 100
    if args.phase == 'train' and args.seed == SEEDS[0]:
        mechanics = context['terminals']['mechanics:' + args.arm]
        require(start_digest == mechanics['initial_state_sha256'] and
                start_witness == mechanics['initial_raw_unit_packed_sha256'], 'fresh TRAIN initial mechanics replay differs')
    with TemporaryDirectory(prefix='discard-mechanics-', dir=args.output) as directory:
        temporary = Path(directory)
        tick = time.perf_counter()
        for step in range(1, total + 1):
            row = update(context, state, ident, step)
            if args.phase == 'train' and args.seed == SEEDS[0] and step <= 17:
                require(diagnostic(row) == diagnostic(mechanics['steps'][step - 1]), 'fresh TRAIN first17 replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
        training_seconds = time.perf_counter() - tick
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else args.output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        witness = original.fingerprint(calibration(context, state))
        integrity(context, state, ident, fresh_bytes=True)
        release(context, state)
        if args.phase == 'mechanics':
            state = restore(context, temporary / 'step8.pt', sha8, digest8, ident, 8)
            resumed = [update(context, state, ident, step) for step in range(9, 18)]
            require([diagnostic(r) for r in resumed] == [diagnostic(r) for r in rows[8:]] and
                    original.fingerprint(payload(state, ident)) == digest and
                    original.fingerprint(calibration(context, state)) == witness, 'full17 vs independent8+9 state/output differs')
            integrity(context, state, ident, fresh_bytes=True)
            release(context, state)
        state = restore(context, checkpoint, sha, digest, ident, total)
        require(original.fingerprint(calibration(context, state)) == witness and
                all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                'final independent raw/unit/packed/RNG differs')
        integrity(context, state, ident, fresh_bytes=True)
        release(context, state)
    if args.phase == 'train':
        bound_file(context['guards'], checkpoint, sha)
    return {'completed_step': total, 'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'identity': ident, 'initial_state_sha256': start_digest, 'terminal_state_sha256': digest,
            'initial_raw_unit_packed_sha256': start_witness, 'raw_unit_packed_sha256': witness,
            'steps': rows, 'resumed_steps': resumed, 'replay_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics', 'training_wall_seconds': training_seconds,
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def check_terminal_record(record, launch, phase, arm):
    seed = record['seed']
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            (seed == SEEDS[0] or phase == 'train' and seed in SEEDS) and record['pass'] is True and
            record['quality_read'] is False and record['strict_reload_exact'] is True and record['exit_rehash_pass'] is True and
            record['frozen_complement_exact'] is True and record['sequential_model_ownership'] is True and
            record['optimizer_members'] == len(parameter_names(arm)) and record['native_trainable_tensors'] == (2 if arm == 'candidate' else 0) and
            record['warm_source_updates'] == 1000 and record['local_initial_counter'] == 0 and
            record['resource_policy'] == policy(phase) and method(record['launch']) == method(launch),
            'accepted new continuation terminal differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256=launch['execution_sha256']))
    invocation = record['invocation']
    require(invocation['optimize'] == 0 and (invocation['cuda_visible_devices'] == '' if phase == 'cpu' else
            invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8'),
            'original new CPU/CUDA invocation differs')
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['completed_step'] == 0 and record['checkpoint'] is None and
                record['initial_arm_parity'] is True and record['gradient_witnesses'] is True and record['training_only_fit'] is True and
                record['arms'].keys() == set(ARMS) and record['peak_cuda_allocated_bytes'] == 0, 'actual CPU qualification incomplete')
        for a, fact in record['arms'].items():
            ident, grad = fact['identity'], fact['gradient_witness']
            check_identity_record(ident, launch, a, SEEDS[0], 'cpu')
            require(ident['parameter_names'] == parameter_names(a) and ident['arm'] == a and ident['device'] == 'cpu' and
                    ident['source'] == record['source'] and ident['numerical_flags'] == record['numerical_flags'] and
                    fact['strict_reload_exact'] is True and fact['bypass_version_tamper_rejected'] is True and
                    grad['graph_free_norm_input'] is True and
                    grad['gradient_norms'].keys() == set(parameter_names(a)), 'CPU per-arm roles/reload differs')
            if a == 'candidate':
                require(grad['perturbations'].keys() == set(NORM) and grad['frozen_pool_backward'] is True and
                        all(grad['gradient_norms'][n] > 0 and grad['perturbations'][n]['max_raw_difference'] > 0 for n in NORM),
                        'CPU gamma/beta witness missing')
        require(record['arms']['control']['raw_unit_packed_sha256'] == record['arms']['candidate']['raw_unit_packed_sha256'],
                'CPU arm parity differs')
        require(all(record['arms']['control']['identity'][k] == record['arms']['candidate']['identity'][k] for k in
                    ('source', 'config', 'runtime', 'complement_sha256', 'buffers_sha256', 'head_buffers_sha256',
                     'static_sha256', 'warm_members_sha256', 'schedule_sha256', 'full_schedule_sha256')),
                'CPU shared complete initial state differs')
    else:
        count = 17 if phase == 'mechanics' else 100
        check_identity_record(record['identity'], launch, arm, seed, 'cuda')
        require(record['launch']['selected_cpu'] == launch['selected_cpu'] and record['completed_step'] == count and
                record['identity']['source'] == record['source'] and
                record['identity']['numerical_flags'] == record['numerical_flags'] and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000 and
                record['steps'][-1]['state_sha256'] == record['terminal_state_sha256'] and
                all(r['schedule_sha256'] == record['identity']['schedule_sha256'] for r in record['steps']) and
                record['median_update_seconds'] == statistics.median(r['seconds'] for r in record['steps'][2:]) and
                0 < sum(r['seconds'] for r in record['steps']) <= record['training_wall_seconds'] <= record['wall_seconds'],
                'new exact step/CPU/state bindings differ')
        check_steps(record['steps'], 1, count, arm)
        if phase == 'mechanics':
            require(record['checkpoint'] is None and record['training_state_discarded'] is True and record['replay_exact'] is True and
                    [diagnostic(r) for r in record['steps'][8:]] == [diagnostic(r) for r in record['resumed_steps']],
                    'discarded full17/independent8+9 replay differs')
            check_steps(record['resumed_steps'], 9, 9, arm)
        else:
            require(record['training_state_discarded'] is False and record['resumed_steps'] == [] and
                    record['checkpoint'].keys() == {'path', 'sha256'} and
                    record['checkpoint']['path'] == str(Path(record['output']) / 'resume.pt') and
                    record['input_guards'].get(record['checkpoint']['path']) == record['checkpoint']['sha256'],
                    'fresh TRAIN100 checkpoint binding differs')


def check_identity_record(ident, launch, arm, seed, device):
    check_optimizer_identity(ident)
    require(ident['method'] == method(launch) and (ident['arm'], ident['seed'], ident['device']) == (arm, seed, device) and
            len(ident['native_inventory']) == len({r['name'] for r in ident['native_inventory']}) == 448 and
            [r['name'] for r in ident['native_inventory'] if r['role'] == 'trainable'] == (NORM if arm == 'candidate' else []) and
            all(r.keys() == {'name', 'shape', 'dtype', 'role'} and r['dtype'] == 'torch.float32' and
                r['role'] in ('frozen', 'trainable') and (r['name'] not in NORM or r['shape'] == [1152])
                for r in ident['native_inventory']) and
            {r['name'] for r in ident['native_inventory'] if r['name'] in NORM} == set(NORM) and
            all(isinstance(ident[k], str) and re.fullmatch('[0-9a-f]{64}', ident[k]) for k in
                ('complement_sha256', 'buffers_sha256', 'head_buffers_sha256', 'static_sha256',
                 'warm_members_sha256', 'schedule_sha256', 'full_schedule_sha256')), 'new complete native identity differs')


def exit_rehash(context):
    tick = time.perf_counter()
    require_no_model(context)
    audit_origins(context)
    context['selected']['exporter'].rehash(context['selected']['genuine'])
    for path, digest in context['guards'].items():
        bound_file({}, path, digest)  # Deliberately uncached even if stat/version is unchanged.
    require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'], 'exit code changed')
    add_seconds(context, 'exit_rehash', tick)


def run(args):
    started = time.perf_counter()
    require(sys.argv == cli(Path(__file__).absolute().parent, args.authority, args.authority_sha256,
        args.execution_sha256, args.phase, args.arm, args.seed, args.output), 'fixed canonical CLI order required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
            os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'explicit hidden CPU or deterministic CUDA launch required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original systemd unit required')
    context = authority(args)
    add_seconds(context, 'admission', started)
    source, prior = context['source_driver'], context['selected']['source_cpu']['invocation']
    before = source.cgroup_memory(); zero_events(before)
    unit = Path(before['path']).name.removesuffix('.service')
    context['admission'].init.admit_cgroup(before, unit)
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and context['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'] and os.environ['INVOCATION_ID'] not in context['invocations'],
            'original interpreter or fresh invocation differs')
    tick = time.perf_counter()
    prepare_native(context)
    add_seconds(context, 'native_admission', tick)
    import torch
    torch.random.default_generator.manual_seed(SEEDS[0])
    cpu_rng = torch.random.get_rng_state().clone()
    args.output.mkdir()
    result = cpu_witnesses(context) if args.phase == 'cpu' else gpu_run(context)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and source.numerical_flags() == context['flags'],
            'complete run RNG/flags changed')
    exit_rehash(context)
    after = source.cgroup_memory(); zero_events(after)
    context['admission'].init.admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and 0 < rss <= 8 * 1024**2 and
            int(after['values']['memory.peak']) >= int(before['values']['memory.peak']) and
            (not torch.cuda.is_initialized() if args.phase == 'cpu' else torch.cuda.max_memory_allocated() < 10_000_000_000),
            'whole-unit time/RSS/lifetime CUDA caps differ')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed, 'output': str(args.output),
               'pass': True, 'quality_read': False, 'strict_reload_exact': True, 'exit_rehash_pass': True,
               'frozen_complement_exact': True, 'sequential_model_ownership': True,
               'optimizer_members': len(parameter_names(args.arm)), 'native_trainable_tensors': 2 if args.arm == 'candidate' else 0,
               'warm_source_updates': 1000, 'local_initial_counter': 0, 'training_qualified': args.phase == 'train',
               'source': context['source'], 'partition_sha256': PARTITION_SHA, 'launch': context['launch'],
               'execution_sha256': args.execution_sha256, 'code': context['code'],
               'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
               'authority_sha256': args.authority_sha256, 'resource_policy': policy(args.phase),
               'numerical_flags': context['flags'], 'wall_seconds': wall, 'process_peak_rss_kib': rss,
               'cgroup_before': before, 'cgroup_after': after, 'origins': context['origins'],
               'input_guards': context['guards'], 'terminal_cgroups': context['terminal_cgroups'],
               'phase_seconds': context['phase_seconds'], 'cuda_initialized': torch.cuda.is_initialized(),
               'both_locks_held_in_parent_authority': True, 'terminal_exit_and_both_locks_require_parent_receipt': True,
               **result, 'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                   'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                   'invocation_id': os.environ['INVOCATION_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                   'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    check_terminal_record(receipt, context['launch'], args.phase, args.arm)
    context['admission'].init.write_json(context['extract'], args.output / 'receipt.json', receipt)
    require(time.perf_counter() - started < policy(args.phase)['seconds'], 'receipt included whole-unit cap exceeded')
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
        raise SystemExit('Post-layernorm continuation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
