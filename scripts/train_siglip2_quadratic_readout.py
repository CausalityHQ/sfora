#!/usr/bin/env python3
"""Fixed-basis quadratic cached readout, composed with the qualified immutable encoder.

Freeze exactly FILES in execution.json (trainer, test, copied primitive).
CLI (fixed order): python -B ROOT/train_siglip2_quadratic_readout.py
 --execution-sha256 SHA --authority FILE --authority-sha256 SHA
 --phase cpu|mechanics|train --arm control|candidate --seed 179061|179069
 --output NEW_ABSOLUTE_DIRECTORY
Authority has exactly LAUNCH_KEYS. FILE={path:canonical absolute file,sha256}.
UNIT={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
 native_peak_rss_kib,both_locks_held:true}; actual original successful terminals.
CPU control061 qualifies BOTH arms; selected_cpu/selected_mechanics=null.
Mechanics each061 selects CPU; fresh TRAIN1000 selects CPU and BOTH mechanics.
Root owns069 admission and all held scoring. No quality reads here.

Source/control/candidate use the same complete learned control061 factory.
Only zero FP32 A[128,32] trains. Means use raw canonical full TRAIN6355 CPU
FP32, original CPU normalization THEN head normalization, no PCA/refit.
Updates consume original training_features-normalized canonical rows only.
All1000 original schedules and seeds061/069 survive; masks are metadata.
Original terminal bank survives until pre-update last-duplicate refresh.

PAYLOAD_KEYS bind complete immutable encoder checkpoint/inventory/config/
processor/nonpersistent buffers/accepted CPU proof/all-frozen export, original
complete warm-state proof, complete head/classifier/A/means/bank/PCA/targets/
positives/ordered rows/schedules/masks/optimizer/scaler/RNG/counters/flags.
JSON identity is presentation; typed payload fingerprints never use json_form.
Independent UPDATED step1->step2 reload qualifies the cached readout, without
constructing a new vision factory or certifying a public image endpoint.
CPU120/other300 seconds;8GiB/noSwap/events0/CUDAallocated<10GB/both locks,
whole-service normal exit and fresh uncached exit hashing. Native UNRUN until
root qualification. No precision, cap, rate or intermediate-selection rescue.
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

SCHEMA = 'siglip2-quadratic-readout-v1'
AUTHORITY_SCHEMA = 'siglip2-quadratic-readout-launch-v1'
FILES = {'train_siglip2_quadratic_readout.py', 'test_siglip2_quadratic_readout.py', 'quadratic_readout.py'}
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
HEAD_NAMES = ['compact_head.primary.weight', 'compact_head.primary.bias',
              'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
HEAD_SHAPES = [(128, 1152), (128,), (32, 1152), (128, 32), (1008, 128)]
HEAD_LAYOUT = dict(zip(('primary.weight', 'primary.bias', 'down.weight', 'up.weight',
                        'center', 'preactivation_std'), [*HEAD_SHAPES[:4], (1152,), ()], strict=True))
RECIPE = {'bank': 'original terminal bank; canonical pre-update detached last duplicate', 'batch': 64, 'candidate': 'h0+A(z.square()-mean(z.square()))', 'certificate': 'updated cached readout composed with qualified immutable encoder', 'classes': 1008, 'clip': 1, 'control': 'h0+A(z-mean(z))', 'initial_scaler': 128, 'input': 'original training_features normalized canonical TRAIN only; both control factory', 'learning_rate': 0.0001, 'local_initial_counter': 0, 'margin': 0.3, 'means': 'raw canonical TRAIN6355 CPU FP32 fullmatrix; original CPU normalize then head normalize; independent means', 'microbatch': 16, 'output_dim': 128, 'rank': 32, 'rank_weight': 8, 'rows': 6355, 'scale': 64, 'schedule': 'unchanged genuine full1000; masks metadata only', 'seeds': [179061, 179069], 'steps': 1000, 'trainable': 'A FP32[128,32], zero, sole AdamW member', 'warm_source_updates': 1000, 'weight_decay': 0.05, 'width': 1152}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'genuine_reference',
               'warm_start', 'partition', 'recipe', 'resource_policy', 'both_locks_held',
               'selected_cpu', 'selected_mechanics'}
STATIC_KEYS = ('pca', 'target', 'positive', 'schedules', 'masks', 'original_rows', 'partition', 'warm_start', 'views')
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'encoder', 'config', 'buffers', 'head', 'classifier', 'A', 'means', 'bank', *STATIC_KEYS, 'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng', 'cuda_rng', 'counter', 'seed', 'numerical_flags'}
CANONICAL_SHA = '55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8'
ENCODER_KEYS = {'checkpoint', 'source_binding', 'source_proof', 'export_binding', 'export_terminal', 'export_runtime', 'inventory', 'warm_payload_sha256', 'warm_state_sha256'}


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
    return [str(Path(root) / 'train_siglip2_quadratic_readout.py'), '--execution-sha256', execution,
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
    spec = importlib.util.spec_from_file_location('_quadratic_genuine', path)
    require('_quadratic_genuine' not in sys.modules and spec is not None and spec.loader is not None, 'bare source origin differs')
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
    for path, digest in guards.items():
        admission.verified.add(str(admission.register(guards, path, digest)))
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'guards': guards,
               'genuine': genuine, 'selected': selected, 'prior': selected['genuine']['prior'],
               'source_driver': selected['source_driver'], 'original': selected['original'],
               'extract': selected['extract'], 'admission': admission, 'warm_record': record,
               'invocations': {selected['source_cpu']['invocation']['invocation_id']},
               'terminals': {}, 'terminal_cgroups': {}, 'phase_seconds': {}}
    context['quadratic'] = genuine.load_helper('_quadratic_primitive', root / 'quadratic_readout.py', code['quadratic_readout.py'], guards)
    context['encoder'] = composition(context)
    context['source'] = {'genuine_reference': GENUINE_REFERENCE, 'warm_start': launch['warm_start'],
                         'native_source': selected['genuine']['reference'].binding(context['prior']),
                         'warm_source': selected['source'], 'partition_sha256': PARTITION_SHA, 'encoder_composition_sha256': selected['exporter'].object_sha(context['encoder'])}
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


def audit_origins(context, initial=False, admission=None):
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
        if admission is None:
            bound_file(context['guards'], path, digest)
        else:
            # imported_origins genuinely hashed these bytes in this exit, and
            # both exact module/file authority checks above have passed.
            admission.verified.add(str(admission.register(context['guards'], path, digest)))
        require(prior['guards'].setdefault(path, digest) == digest, 'original native origin changed')
    context['origins'] = origins


def add_seconds(context, name, tick):
    timings = context.setdefault('phase_seconds', {})
    delta = time.perf_counter() - tick
    timings[name] = timings.get(name, 0.) + delta
    if context['args'].phase == 'cpu':
        print(json.dumps({'event': 'QUADRATIC_PHASE', 'phase': name, 'delta_seconds': delta,
                          'cumulative_seconds': timings[name]}), flush=True)


def packed_outputs(context, raw):
    import torch
    from torch.nn import functional as F
    with torch.no_grad():
        require((raw.norm(dim=1) > 0).all().item(), 'nonzero raw descriptors required')
        unit = F.normalize(raw, dim=1)
        packed = context['packing'].pack_int8_unit_embeddings(unit.cpu())
    return {'raw': raw.detach().cpu(), 'unit': unit.cpu(), 'codes': packed.codes.cpu(),
            'inverse_norms': packed.inverse_norms.cpu(), 'wire': packed.to_bytes()}


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def exit_rehash(context):
    tick = time.perf_counter()
    require_no_model(context)
    # This reader is empty and exit-owned; startup admission is never reused.
    admission = context['original'].FlatAdmission()
    audit_origins(context, admission=admission)
    exporter, genuine = context['selected']['exporter'], context['selected']['genuine']
    prior, source = genuine['prior'], genuine['prior']['source_driver']
    require(admission.all_fit_images(prior) == prior['all_images'], 'FIT image resolution changed')
    require(source.fit_rows(prior['extract'], prior['fit']) == prior['images'], 'FIT image resolution changed')
    for path, digest in prior['guards'].items():
        admission.bound_file({}, path, digest)
    # Keep the small authenticated rereads and live extractor-origin bootstrap.
    require(source.bootstrap(prior['root'], prior['args'].execution_sha256)[1] == prior['code'],
            'exit closure differs')
    require(genuine['reference'].bootstrap(prior['own_root'], prior['export_args'].execution_sha256) == prior['own_code'],
            'exit exporter closure differs')
    for path, digest in genuine['guards'].items():
        admission.bound_file({}, path, digest)
    require(exporter.closure(genuine['root'], genuine['args'].execution_sha256, exporter.FILES, {}) == genuine['code'],
            'exit genuine closure differs')
    selected = exporter.selected_manifest(exporter.file_json(genuine['launch']['partition'], {}), prior['fit'])
    selected['resolved_paths'] = [str(prior['all_images'][r]) for r in selected['original_rows']]
    require(selected == genuine['selected'], 'exit TRAIN mapping changed')
    exporter.image_rows_node(genuine['launch']['image_rows']['path'])
    for path, digest in context['guards'].items():
        admission.bound_file({}, path, digest)
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
               'certificate': RECIPE['certificate'], 'public_encoder_qualified': False,
               'trainable_scalars': 4096, 'source_factory': 'control', 'canonical_updates_only': True,
               'optimizer_members': len(parameter_names(args.arm)), 'native_trainable_tensors': 0,
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
        raise SystemExit('Quadratic readout rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


def admitted_file(context, path, sha):
    """Reuse only successfully owned hashes from this authority call; exit is fresh."""
    return context['admission'].bound_file(context['guards'], path, sha)


def check_encoder(encoder):
    require(encoder.keys() == ENCODER_KEYS and encoder['warm_payload_sha256'] == WARM_CHECKPOINT['sha256'] and
            encoder['warm_state_sha256'] == WARM_STATE_SHA, 'complete immutable encoder composition required')
    proof, exported = encoder['source_proof'], encoder['export_runtime']
    source = proof['runtime']
    require(encoder['checkpoint'] == proof['checkpoint'] and
            encoder['source_binding'] == encoder['export_binding']['source'] and
            encoder['checkpoint']['sha256'] == '5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189' and
            all(proof[k] is True for k in ('pass', 'source_qualified', 'reload_exact', 'exit_rehash_pass')) and
            all(proof[k] is False for k in ('quality_read', 'training_qualified', 'gradients_created', 'optimizer_created')) and
            proof['updates'] == 0 and source['attn_implementation'] == 'sdpa' and
            source['config']['hidden_size'] == 1152 and source['processor']['backend'] == 'torchvision',
            'accepted immutable source proof differs')
    require(source.keys() == exported.keys() == {'attn_implementation', 'buffers', 'config', 'modules', 'processor', 'roles', 'vision'} and
            all(source[k] == exported[k] for k in source if k != 'roles') and
            len(source['roles']) == len(source['vision']) == 448 and
            len({r['name'] for r in source['roles']}) == 448 and
            sum(r['role'] == 'trainable' for r in source['roles']) == 205 and
            exported['roles'] == encoder['inventory'] == [{**r, 'role': 'frozen'} for r in source['roles']],
            'complete source448/all-frozen accepted export mapping differs')
    for row in encoder['inventory']:
        require(row.keys() == {'name', 'shape', 'dtype', 'role'} and row['role'] == 'frozen' and
                source['vision'][row['name']] == {'shape': row['shape'], 'dtype': 'torch.float32',
                    'sha256': source['vision'][row['name']]['sha256']} and row['dtype'] == 'torch.float32' and
                re.fullmatch('[0-9a-f]{64}', source['vision'][row['name']]['sha256']), 'complete encoder tensor facts differ')
    require(source['buffers'].keys() == {'embeddings.position_ids'} and
            source['buffers']['embeddings.position_ids'] == {'shape': [1, 256], 'dtype': 'torch.int64',
                'persistent': False, 'sha256': 'bbd330b12e8159e117376ef24fa106413bc9fc18032a0d43e95c5dae5e47953f'} and
            encoder['export_binding']['source']['source_cpu']['proof']['sha256'] ==
                'e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf',
            'config/processor/nonpersistent buffer/source proof binding differs')


def composition(context):
    selected, prior = context['selected'], context['prior']
    exported, proof = selected['export_record'], selected['source_cpu']
    encoder = {'checkpoint': proof['checkpoint'], 'source_binding': selected['genuine']['reference'].binding(prior),
               'source_proof': proof, 'export_binding': exported['binding'],
               'export_terminal': selected['launch']['selected_export'], 'export_runtime': exported['source_runtime'],
               'inventory': exported['source_runtime']['roles'], 'warm_payload_sha256': WARM_CHECKPOINT['sha256'],
               'warm_state_sha256': WARM_STATE_SHA}
    check_encoder(encoder)
    require(exported['source_checkpoint'] == encoder['checkpoint'] and
            exported['binding'] == selected['exporter'].binding(selected['genuine']) and
            exported['strict_independent_reload_exact'] is True and exported['constructor_and_view_rng_preserved'] is True and
            exported['caches']['canonical']['sha256'] == CANONICAL_SHA and
            selected['source']['source_checkpoint'] == encoder['checkpoint'] and
            set(prior['expected']) == set(proof['runtime']['vision']) and
            all(proof['runtime']['vision'][n] == {'shape': shape, 'dtype': 'torch.float32',
                'sha256': prior['mapping'][n]['sha256']} for n, shape in prior['expected'].items()) and
            selected['source']['caches'] == exported['caches'], 'explicit source/proof/cache composition differs')
    # These guards were freshly authenticated by genuine.authority in THIS call.
    for path, digest in {**prior['guards'], **selected['guards']}.items():
        require(context['guards'].setdefault(path, digest) == digest, 'complete composition guard conflict')
        context['admission'].verified.add(str(context['admission'].register(context['guards'], path, digest)))
    require(all(context['guards'].get(p) == h for p, h in proof['input_guards'].items()) and
            context['guards'].get(encoder['checkpoint']['path']) == encoder['checkpoint']['sha256'],
            'complete immutable source ownership missing')
    return encoder


def parameter_names(arm):
    require(type(arm) is str and arm in ARMS, 'fixed readout arm required')
    return ['A']


def parameter_shapes(arm):
    parameter_names(arm)
    return [(128, 32)]


def check_membership(pairs, groups, arm, frozen=()):
    require([n for n, _ in pairs] == parameter_names(arm) and len(pairs) == 1 and
            [tuple(p.shape) for _, p in pairs] == parameter_shapes(arm) and
            len(groups) == 1 and [id(p) for _, p in pairs] == [id(p) for group in groups for p in group] and
            all(p.requires_grad and p.is_leaf and p.grad_fn is None and str(p.dtype) == 'torch.float32'
                for _, p in pairs) and all(not p.requires_grad and p.grad is None and p.grad_fn is None
                for p in frozen) and all(id(p) != id(pairs[0][1]) for p in frozen),
            'exact sole FP32 A optimizer member and frozen complement required')


def check_optimizer_identity(ident):
    defaults = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in ident['optimizer_defaults']:
        defaults['decoupled_weight_decay'] = True
    groups = [dict(defaults, lr=1e-4)]
    require(ident['parameter_names'] == parameter_names(ident['arm']) and
            ident['optimizer_defaults'] == defaults and ident['optimizer_groups'] == groups and
            ident['optimizer_serial_groups'] == [dict(groups[0], params=[0])],
            'fixed sole A AdamW identity differs')


def check_continuation(full, selected):
    require(len(full) == len(selected) == 1000 and selected == full and
            all(len(row) == 64 and all(type(v) is int and 0 <= v < 6355 for v in row) for row in full),
            'unchanged full1000 schedule required')


def require_no_model(context):
    require(context.get('model_ref', lambda: None)() is None, 'previous complete readout still owned')


def claim_model(context, model):
    require_no_model(context)
    context['model_ref'] = weakref.ref(model)


def encoder_metadata(context):
    """Read typed config/nonpersistent buffers from qualified bytes; no vision factory."""
    import torch
    fact = context['encoder']['checkpoint']
    path = admitted_file(context, fact['path'], fact['sha256'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(disk.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
            disk['runtime'] == context['encoder']['source_proof']['runtime'] and
            json_form(disk['config']) == disk['runtime']['config'] and
            disk['vision'].keys() == disk['runtime']['vision'].keys() and
            disk['buffers'].keys() == disk['runtime']['buffers'].keys() and
            context['source_driver'].tensor_fact(disk['cpu_rng']) == context['encoder']['source_proof']['cpu_rng'],
            'complete immutable encoder checkpoint metadata differs')
    for name, value in disk['vision'].items():
        require(list(value.shape) == disk['runtime']['vision'][name]['shape'] and
                str(value.dtype) == 'torch.float32' and not value.requires_grad and value.grad is None,
                'complete serialized encoder448 layout differs')
    for name, value in disk['buffers'].items():
        require(context['source_driver'].tensor_fact(value) == {k: v for k, v in
                disk['runtime']['buffers'][name].items() if k != 'persistent'}, 'nonpersistent serialized encoder bytes differ')
    result = clone_tree(disk['config']), clone_tree(disk['buffers'])
    del disk, value
    gc.collect()
    return result


def prepare_native(context):
    """Authenticate complete original warm state before extracting learned members."""
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
    context['packing'] = genuine.load_helper('_quadratic_packing', pack['path'], pack['sha256'], context['guards'])
    path = admitted_file(context, WARM_CHECKPOINT['path'], WARM_CHECKPOINT['sha256'])
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
        context['initial'] = {k: clone_tree(disk[k]) for k in
                             ('head', 'classifier', 'bank', 'pca', 'target', 'positive',
                              'schedules', 'masks', 'original_rows', 'partition', 'views')}
    del disk
    gc.collect()
    initial = context['initial']
    for seed in SEEDS:
        check_continuation(initial['schedules'][str(seed)].tolist(), initial['schedules'][str(seed)].tolist())
    initial['warm_start'] = {'endpoint': context['launch']['warm_start'], 'warm_source_updates': 1000,
                            'local_initial_counter': 0, 'bank_context': 'original canonical B32'}
    context['warm_members_sha256'] = context['original'].fingerprint({k: initial[k] for k in ('head', 'classifier', 'bank')})
    context['static_sha256'] = context['original'].fingerprint({k: initial[k] for k in STATIC_KEYS})
    audit_origins(context)


def canonical_features(context):
    import numpy as np
    import torch
    fact = context['selected']['source']['caches']['canonical']
    require(fact['sha256'] == CANONICAL_SHA, 'pinned canonical TRAIN cache required')
    path = admitted_file(context, fact['path'], fact['sha256'])
    cache = np.load(path, allow_pickle=False, mmap_mode='r')
    require(cache.shape == (6355, 1152) and cache.dtype == np.float32, 'canonical fullmatrix layout differs')
    raw = torch.from_numpy(cache.copy())
    del cache
    features = context['genuine'].normalize_nonzero(raw)
    return raw, features


def optimizer_state(A, arm, frozen):
    import torch
    pairs = [('A', A)]
    groups = [{'params': [A], 'lr': 1e-4}]
    check_membership(pairs, [g['params'] for g in groups], arm, frozen)
    optimizer = torch.optim.AdamW(groups, weight_decay=.05)
    defaults = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and
            [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups] == [dict(defaults, lr=1e-4)],
            'fresh sole A AdamW defaults/groups differ')
    return pairs, optimizer


def frozen_members(state):
    return [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]


def frozen_tree(state):
    return {'encoder': state['encoder'], 'config': state['config'], 'buffers': state['buffers'],
            'head': dict(state['head'].state_dict()), 'classifier': state['classifier'], 'means': state['means'],
            'features': state['features'], **{k: state[k] for k in STATIC_KEYS}}


def frozen_tensors(state):
    return [*frozen_members(state), *state['head'].named_buffers(), *state['means'].items(),
            *state['buffers'].items(), ('features', state['features'])]


def fresh(context, arm, seed, device):
    import torch
    require_no_model(context)
    require(arm in ARMS and seed in SEEDS and device in ('cpu', 'cuda'), 'fresh state profile differs')
    tick = time.perf_counter()
    initial, original = context['initial'], context['original']
    raw, features = canonical_features(context)
    head = context['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).train()
    context['quadratic']._check_base(head, 'cpu')
    means = context['quadratic'].fit_means(raw, head)
    means_sha = original.fingerprint(means)
    require(context.setdefault('means_sha256', means_sha) == means_sha, 'independently fitted immutable fullmatrix means differ')
    del raw
    head.to(device)
    claim_model(context, head)
    classifier = torch.nn.Parameter(initial['classifier'].to(device, copy=True), requires_grad=False)
    A = context['quadratic'].new_weight(device)
    pairs, optimizer = optimizer_state(A, arm, [*head.parameters(), classifier])
    config, buffers = encoder_metadata(context)
    state = {'head': head, 'classifier': classifier, 'A': A, 'means': clone_tree(means, device),
             'encoder': clone_tree(context['encoder']), 'config': config, 'buffers': buffers,
             'features': features.to(device, copy=True).detach(),
             'bank': initial['bank'].to(device, copy=True).detach(), 'arm': arm, 'seed': seed, 'device': device,
             'params': pairs, 'optimizer': optimizer, 'counter': 0,
             'scaler': torch.amp.GradScaler(device, init_scale=128),
             **{k: clone_tree(initial[k]) for k in STATIC_KEYS}}
    state['target'] = state['target'].to(device)
    state['positive'] = state['positive'].to(device)
    state['frozen_versions'] = [(n, p.data_ptr(), p._version) for n, p in frozen_tensors(state)]
    require(original.fingerprint({k: payload_member(state, k) for k in ('head', 'classifier', 'bank')}) ==
            context['warm_members_sha256'] and not bool(A.detach().count_nonzero()), 'fresh learned members/zero A differ')
    add_seconds(context, 'construction', tick)
    return state


def payload_member(state, key):
    return dict(state['head'].state_dict()) if key == 'head' else state[key]


def identity(context, state):
    original = context['original']
    return json_form({'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'],
        'seed': state['seed'], 'device': state['device'], 'parameter_names': parameter_names(state['arm']),
        'config': state['config'], 'native_inventory': state['encoder']['inventory'],
        'encoder_sha256': original.fingerprint(state['encoder']),
        'complement_sha256': original.fingerprint({'head': dict(state['head'].state_dict()), 'classifier': state['classifier']}),
        'buffers_sha256': original.fingerprint(state['buffers']), 'means_sha256': original.fingerprint(state['means']),
        'head_buffers_sha256': original.fingerprint(dict(state['head'].named_buffers())),
        'frozen_sha256': original.fingerprint(frozen_tree(state)),
        'features_sha256': original.fingerprint(state['features']),
        'static_sha256': context['static_sha256'], 'warm_members_sha256': context['warm_members_sha256'],
        'optimizer_defaults': state['optimizer'].defaults.copy(),
        'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
        'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
        'positive_shape': tuple(state['positive'].shape),
        'schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]),
        'full_schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]),
        'numerical_flags': context['flags']})


def payload(state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'source': ident['source'], 'encoder': state['encoder'],
            'config': state['config'], 'buffers': state['buffers'],
            'head': dict(state['head'].state_dict()), 'classifier': state['classifier'].detach(),
            'A': state['A'].detach(), 'means': state['means'], 'bank': state['bank'],
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
            type(step) is int and type(saved['counter']) is int and saved['counter'] == step and 0 <= step <= 1000 and
            type(saved['seed']) is int and saved['seed'] == ident['seed'], 'complete typed payload identity differs')
    check_encoder(saved['encoder'])
    require(saved['encoder']['inventory'] == ident['native_inventory'] and
            json_form(saved['config']) == saved['encoder']['export_runtime']['config'], 'complete encoder/config mapping differs')
    def tensor(value, shape, dtype='torch.float32'):
        require(tuple(value.shape) == tuple(shape) and str(value.dtype) == dtype, 'complete payload tensor layout differs')
    require(saved['head'].keys() == HEAD_LAYOUT.keys() and saved['means'].keys() == {'linear', 'quadratic'} and
            saved['buffers'].keys() == {'embeddings.position_ids'}, 'complete head/means/nonpersistent buffer inventory differs')
    for name, shape in HEAD_LAYOUT.items():
        tensor(saved['head'][name], shape)
    for value in saved['means'].values():
        tensor(value, (32,))
    tensor(saved['A'], (128, 32)); tensor(saved['buffers']['embeddings.position_ids'], (1, 256), 'torch.int64')
    tensor(saved['classifier'], (1008, 128)); tensor(saved['bank'], (6355, 128))
    tensor(saved['target'], (6355,), 'torch.int64'); tensor(saved['positive'], ident['positive_shape'], 'torch.int64')
    tensor(saved['original_rows'], (6355,), 'torch.int64')
    require(saved['pca'].keys() == {'mean', 'components'} and
            saved['schedules'].keys() == saved['masks'].keys() == {str(s) for s in SEEDS}, 'complete static schedule/PCA inventory differs')
    tensor(saved['pca']['mean'], (1152,)); tensor(saved['pca']['components'], (128, 1152))
    for seed in SEEDS:
        tensor(saved['schedules'][str(seed)], (1000, 64), 'torch.int64')
        tensor(saved['masks'][str(seed)], (1000, 64), 'torch.bool')
    require(saved['partition']['schema'] and saved['warm_start'] == {
        'endpoint': ident['method']['warm_start'], 'warm_source_updates': 1000, 'local_initial_counter': 0,
        'bank_context': 'original canonical B32'}, 'warm/local provenance differs')
    require(saved['views'] == {k: ident['source']['warm_source'][k] for k in
            ('caches', 'ordered_input_sha256', 'ordered_view_sha256')}, 'complete original ordered row/cache mapping differs')
    opt = saved['optimizer']
    require(opt.keys() == {'state', 'param_groups'} and json_form(opt['param_groups']) == ident['optimizer_serial_groups'] and
            [i for g in opt['param_groups'] for i in g['params']] == [0] and
            all(type(i) is int for g in opt['param_groups'] for i in g['params']) and
            all(type(i) is int for i in opt['state']) and set(opt['state']) == ({0} if step else set()),
            'exact sole A optimizer state inventory differs')
    for moments in opt['state'].values():
        require(moments.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and float(moments['step']) == step,
                'optimizer successful-step counter differs')
        tensor(moments['step'], ())
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor(moments[name], (128, 32))
    require(saved['scaler'] == {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
            'growth_interval': 2000, '_growth_tracker': step} and ident['device'] in ('cpu', 'cuda') and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0), 'scaler/RNG inventory differs')
    for value in [saved['cpu_rng'], *saved['cuda_rng']]:
        require(len(value.shape) == 1 and value.shape[0] > 0 and str(value.dtype) == 'torch.uint8', 'RNG tensor layout differs')


def check_complement(context, state, ident):
    require(context['original'].fingerprint(frozen_tree(state)) == ident['frozen_sha256'],
            'fresh frozen complement/means/source bytes differ')


def integrity(context, state, ident, fresh_bytes=False):
    import torch
    tick = time.perf_counter()
    original = context['original']
    check_encoder(state['encoder'])
    require(state['encoder'] == context['encoder'] and state['encoder']['inventory'] == ident['native_inventory'] and
            original.fingerprint(state['encoder']) == ident['encoder_sha256'] and
            json_form(state['config']) == ident['config'], 'complete immutable encoder448/config differs')
    context['quadratic']._check_base(state['head'], state['A'].device)
    require(all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in state['head'].modules()), 'original head modes/hooks differ')
    actual = [('A', state['A'])]
    check_membership(actual, [g['params'] for g in state['optimizer'].param_groups], state['arm'],
                     [p for _, p in frozen_tensors(state)])
    require([id(p) for _, p in actual] == [id(p) for _, p in state['params']] and
            all(p.device.type == ident['device'] and torch.isfinite(p).all().item() for _, p in actual) and
            [(n, p.data_ptr(), p._version) for n, p in frozen_tensors(state)] == state['frozen_versions'],
            'live A/frozen versions/grads differ')
    require(original.fingerprint(state['buffers']) == ident['buffers_sha256'] and
            torch.equal(state['buffers']['embeddings.position_ids'], torch.arange(256).expand(1, -1)) and
            original.fingerprint(dict(state['head'].named_buffers())) == ident['head_buffers_sha256'] and
            original.fingerprint(state['means']) == ident['means_sha256'] == context['means_sha256'] and
            original.fingerprint({k: state[k] for k in STATIC_KEYS}) == ident['static_sha256'] == context['static_sha256'],
            'complete static/head/means/nonpersistent buffers differ')
    require(json_form(state['optimizer'].defaults) == ident['optimizer_defaults'] and
            json_form([{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups]) == ident['optimizer_groups'] and
            not state['bank'].requires_grad and state['bank'].grad is None and tuple(state['bank'].shape) == (6355, 128) and
            state['bank'].device.type == ident['device'] and torch.isfinite(state['bank']).all().item() and
            not state['features'].requires_grad and state['features'].grad is None and tuple(state['features'].shape) == (6355, 1152),
            'optimizer/canonical features/bank differs')
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    finite_tree(saved['optimizer'])
    require(context['source_driver'].numerical_flags() == ident['numerical_flags'], 'numerical flags changed')
    # Read actual bytes at EVERY use boundary: .data writes bypass version counters.
    check_complement(context, state, ident)
    if fresh_bytes:
        for path, digest in ((WARM_CHECKPOINT['path'], WARM_CHECKPOINT['sha256']),
                             (state['encoder']['checkpoint']['path'], state['encoder']['checkpoint']['sha256'])):
            admitted_file(context, path, digest)
    add_seconds(context, 'integrity', tick)


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
    """Fresh base/cache/means/config/buffers, then UPDATED A/moments/bank/RNG."""
    import torch
    require_no_model(context)
    tick = time.perf_counter()
    path = bound_file({}, path, sha)
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = context['original'].CheckpointPages(stream)
        check_payload(disk, ident, step)
        require(context['original'].fingerprint(disk, consumed=pages.consume) == digest, 'complete serialized typed state differs')
        finite_tree(disk)
        state = fresh(context, ident['arm'], ident['seed'], ident['device'])
        require(identity(context, state) == ident, 'independent complete source/means/static reconstruction differs')
        recreated = payload(state, ident)
        immutable = ('encoder', 'config', 'buffers', 'head', 'classifier', 'means', *STATIC_KEYS)
        require(context['original'].fingerprint({k: recreated[k] for k in immutable}) ==
                context['original'].fingerprint({k: disk[k] for k in immutable}), 'independent frozen typed state differs')
        with torch.no_grad():
            state['A'].copy_(disk['A'])
            pages.consume(disk['A'])
        state['bank'] = pages.copy(disk['bank'], ident['device']).detach()
        optimizer = clone_tree(disk['optimizer'])
        state['optimizer'].load_state_dict(optimizer)
        state['scaler'].load_state_dict(disk['scaler'])
        state['counter'] = step
        torch.random.set_rng_state(pages.copy(disk['cpu_rng']))
        if ident['device'] == 'cuda':
            torch.cuda.set_rng_state_all([pages.copy(v) for v in disk['cuda_rng']])
    del disk, optimizer, recreated
    gc.collect()
    integrity(context, state, ident, fresh_bytes=True)
    require(context['original'].fingerprint(payload(state, ident)) == digest, 'complete independent updated restored state differs')
    add_seconds(context, 'reload', tick)
    return state


def calibration(context, state):
    batch = state['schedules'][str(state['seed'])][0].tolist()
    raw = context['quadratic'].raw_features(state['features'][batch], state['head'], state['A'], state['means'], state['arm'])
    return packed_outputs(context, raw)


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
    require(state['A'].grad is not None and state['A'].grad.dtype == torch.float32 and
            torch.isfinite(state['A'].grad).all().item() and
            all(p.grad is None for _, p in frozen_tensors(state)), 'finite sole A/absent frozen gradients required')
    result = {'A': float(state['A'].grad.double().norm())}
    require(math.isfinite(result['A']) and result['A'] > 0, 'nonzero aggregate unscaled A DATA gradient required')
    return result


def update(context, state, ident, step):
    import torch
    device, original = state['device'], context['original']
    require(state['counter'] == step - 1 and 1 <= step <= 1000, 'exact fresh continuation update required')
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    integrity(context, state, ident, fresh_bytes=True)
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    masks = state['masks'][str(state['seed'])]
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version, before_A = state['bank']._version, state['A'].detach().clone()
    ce_sum = rank_sum = 0.
    raw_rows = []
    for start in range(0, 64, 16):
        index = torch.tensor(batch[start:start + 16], device=device)
        raw = context['quadratic'].raw_features(state['features'][index], state['head'], state['A'], state['means'], state['arm'])
        raw_rows.append(raw.detach())
        with torch.autocast(device_type=device, enabled=False):
            ce, rank = terms(context, state, raw, index)
            loss = (ce + 8 * rank) * .25
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach()) * .25; rank_sum += float(rank.detach()) * .25
        del raw, index, ce, rank, loss
    require(state['bank']._version == version, 'bank changed during data backward')
    scaler.unscale_(optimizer)
    gradients = gradient_norms(state)
    norm = torch.nn.utils.clip_grad_norm_([state['A']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    require(scaler.get_scale() == scale == 128 and not torch.equal(before_A, state['A']), 'optimizer skipped/scaler changed/no A update')
    state['counter'] += 1
    rows, positions = context['ref'].member_bank_refresh_rows(batch)
    raw = torch.cat(raw_rows).detach()
    clean_sha = original.fingerprint(raw)
    state['bank'][torch.tensor(rows, device=device)] = context['ref'].member_bank_refresh_values(
        raw, raw, torch.tensor(positions, device=device), live_head=False)
    require(state['bank']._version == version + 1 and not state['bank'].requires_grad,
            'pre-update detached last-duplicate refresh differs')
    optimizer.zero_grad(set_to_none=True)
    integrity(context, state, ident, fresh_bytes=True)
    require(device != 'cuda' or torch.cuda.max_memory_allocated() < 10_000_000_000, 'lifetime CUDA peak exceeded')
    row = {'step': step, 'batch': batch, 'schedule_sha256': ident['schedule_sha256'],
           'feature_rows_sha256': original.fingerprint(state['features'][batch]),
           'mask_sha256': original.fingerprint(masks[step - 1]), 'clean_bank_sha256': clean_sha,
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients, 'A_updated': True,
           'state_sha256': original.fingerprint(payload(state, ident))}
    if device == 'cuda':
        torch.cuda.synchronize()
    row['seconds'] = time.perf_counter() - started
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def check_steps(rows, start, count, arm):
    keys = {'step', 'batch', 'schedule_sha256', 'feature_rows_sha256', 'mask_sha256', 'clean_bank_sha256',
            'ce', 'rank', 'loss', 'scale', 'preclip_norm', 'gradient_norms', 'A_updated', 'state_sha256', 'seconds'}
    require(len(rows) == count, 'complete update records required')
    for step, row in enumerate(rows, start):
        require(row.keys() == keys and row['step'] == step and len(row['batch']) == 64 and
                all(type(v) is int and 0 <= v < 6355 for v in row['batch']) and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k]) for k in
                    ('schedule_sha256', 'feature_rows_sha256', 'mask_sha256', 'clean_bank_sha256', 'state_sha256')) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('ce', 'rank', 'loss', 'scale', 'preclip_norm', 'seconds')) and row['seconds'] > 0 and
                row['loss'] == row['ce'] + 8 * row['rank'] and row['scale'] == 128 and row['A_updated'] is True and
                row['gradient_norms'].keys() == set(parameter_names(arm)) and
                type(row['gradient_norms']['A']) in (int, float) and math.isfinite(row['gradient_norms']['A']) and
                row['gradient_norms']['A'] > 0, 'complete finite sole A update record differs')


def bypass_version_witness(context, state, ident):
    """Discarded .data mutations exercise fresh frozen/means byte checks."""
    import torch
    for value in (state['head'].primary.weight, state['means']['linear'], state['means']['quadratic']):
        version = value._version
        before = value.detach().reshape(-1)[0].clone()
        try:
            value.data.reshape(-1)[0].add_(.25)
            require(value._version == version and not torch.equal(value.reshape(-1)[0], before), 'version-bypass mutation failed')
            try:
                check_complement(context, state, ident)
            except ValueError as error:
                require(str(error) == 'fresh frozen complement/means/source bytes differ', 'unexpected tamper rejection')
            else:
                raise ValueError('fresh frozen byte check admitted .data tamper')
        finally:
            value.data.reshape(-1)[0].copy_(before)
    integrity(context, state, ident, fresh_bytes=True)
    return True


def cpu_witnesses(context):
    import torch
    original, output = context['original'], context['args'].output
    results = {}
    with TemporaryDirectory(prefix='discard-cpu-', dir=output) as directory:
        for arm in ARMS:
            state = fresh(context, arm, SEEDS[0], 'cpu')
            ident = identity(context, state)
            integrity(context, state, ident, fresh_bytes=True)
            witness = original.fingerprint(calibration(context, state))
            with torch.no_grad():
                batch = state['schedules'][str(SEEDS[0])][0].tolist()
                source = packed_outputs(context, state['head'](state['features'][batch]))
            require(original.fingerprint(source) == witness, 'initial learned-source/arm raw/unit/packed/inverse bits differ')
            bypass_rejected = bypass_version_witness(context, state, ident)
            initial_digest = original.fingerprint(payload(state, ident))
            step1 = update(context, state, ident, 1)
            require(step1['gradient_norms']['A'] > 0 and step1['A_updated'], 'actual first B64/micro16 DATA update missing')
            path = Path(directory) / (arm + '-step1.pt')
            sha1, digest1 = save(context, state, ident, path)
            require(digest1 != initial_digest, 'updated step1 full state must differ from step0')
            step2 = update(context, state, ident, 2)
            expected_digest = original.fingerprint(payload(state, ident))
            expected_witness = original.fingerprint(calibration(context, state))
            integrity(context, state, ident, fresh_bytes=True)
            del source
            release(context, state)
            state = restore(context, path, sha1, digest1, ident, 1)
            resumed2 = update(context, state, ident, 2)
            require(diagnostic(resumed2) == diagnostic(step2) and
                    original.fingerprint(payload(state, ident)) == expected_digest and
                    original.fingerprint(calibration(context, state)) == expected_witness,
                    'uninterrupted2 vs independent UPDATED1->2 typed/raw/unit/packed/inverse bits differ')
            integrity(context, state, ident, fresh_bytes=True)
            results[arm] = {'identity': ident, 'initial_state_sha256': initial_digest,
                'updated_step1_sha256': digest1, 'state_sha256': expected_digest,
                'raw_unit_packed_sha256': witness, 'updated_raw_unit_packed_sha256': expected_witness,
                'steps': [step1, step2], 'resumed_step2': resumed2, 'actual_batch': batch,
                'source_initial_parity': True, 'strict_reload_exact': True, 'updated_resume_exact': True,
                'bypass_version_tamper_rejected': bypass_rejected}
            release(context, state)
    require(results['control']['raw_unit_packed_sha256'] == results['candidate']['raw_unit_packed_sha256'],
            'matched initial source/control/candidate parity differs')
    return {'completed_step': 0, 'checkpoint': None, 'identity': None, 'arms': results,
            'initial_raw_unit_packed_sha256': results['control']['raw_unit_packed_sha256'],
            'raw_unit_packed_sha256': results['control']['raw_unit_packed_sha256'],
            'initial_state_sha256': None, 'terminal_state_sha256': None, 'steps': [], 'resumed_steps': [],
            'initial_arm_parity': True, 'gradient_witnesses': True, 'cpu_resume_exact': True,
            'training_only_fit': True, 'replay_exact': True, 'training_state_discarded': True,
            'training_wall_seconds': 0., 'median_update_seconds': None, 'peak_cuda_allocated_bytes': 0}


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
                     'schedule_sha256': original.fingerprint(state['schedules'][str(args.seed)]),
                     'full_schedule_sha256': original.fingerprint(state['schedules'][str(args.seed)])},
            'CUDA cached-readout composition differs from CPU-qualified identity')
    integrity(context, state, ident, fresh_bytes=True)
    start_digest = original.fingerprint(payload(state, ident))
    start_witness = original.fingerprint(calibration(context, state))
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 1000
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
                    original.fingerprint(calibration(context, state)) == witness, 'full17 vs independent8+9 typed state/output differs')
            integrity(context, state, ident, fresh_bytes=True)
            release(context, state)
        state = restore(context, checkpoint, sha, digest, ident, total)
        require(original.fingerprint(calibration(context, state)) == witness and
                all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                'final independent raw/unit/packed/inverse bits/RNG differs')
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


def check_identity_record(ident, launch, arm, seed, device):
    check_optimizer_identity(ident)
    require(ident['method'] == method(launch) and (ident['arm'], ident['seed'], ident['device']) == (arm, seed, device) and
            len(ident['native_inventory']) == len({r['name'] for r in ident['native_inventory']}) == 448 and
            all(r.keys() == {'name', 'shape', 'dtype', 'role'} and r['dtype'] == 'torch.float32' and r['role'] == 'frozen'
                for r in ident['native_inventory']) and
            all(isinstance(ident[k], str) and re.fullmatch('[0-9a-f]{64}', ident[k]) for k in
                ('encoder_sha256', 'complement_sha256', 'buffers_sha256', 'head_buffers_sha256', 'means_sha256',
                 'features_sha256', 'frozen_sha256', 'static_sha256', 'warm_members_sha256',
                 'schedule_sha256', 'full_schedule_sha256')), 'complete frozen encoder/readout identity differs')


def check_terminal_record(record, launch, phase, arm):
    seed = record['seed']
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            (seed == SEEDS[0] or phase == 'train' and seed in SEEDS) and record['pass'] is True and
            record['quality_read'] is False and record['strict_reload_exact'] is True and record['exit_rehash_pass'] is True and
            record['frozen_complement_exact'] is True and record['sequential_model_ownership'] is True and
            record['optimizer_members'] == 1 and record['native_trainable_tensors'] == 0 and record['trainable_scalars'] == 4096 and
            record['certificate'] == RECIPE['certificate'] and record['public_encoder_qualified'] is False and
            record['source_factory'] == 'control' and record['canonical_updates_only'] is True and
            record['warm_source_updates'] == 1000 and record['local_initial_counter'] == 0 and
            record['resource_policy'] == policy(phase) and method(record['launch']) == method(launch),
            'accepted new cached continuation terminal differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256=launch['execution_sha256']))
    invocation = record['invocation']
    require(invocation['optimize'] == 0 and (invocation['cuda_visible_devices'] == '' if phase == 'cpu' else
            invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8'),
            'original new CPU/CUDA invocation differs')
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['completed_step'] == 0 and record['checkpoint'] is None and
                record['initial_arm_parity'] is True and record['gradient_witnesses'] is True and record['cpu_resume_exact'] is True and
                record['training_only_fit'] is True and record['arms'].keys() == set(ARMS) and record['peak_cuda_allocated_bytes'] == 0,
                'actual CPU qualification incomplete')
        for a, fact in record['arms'].items():
            ident = fact['identity']
            check_identity_record(ident, launch, a, SEEDS[0], 'cpu')
            require(ident['source'] == record['source'] and ident['numerical_flags'] == record['numerical_flags'] and
                    all(fact[k] is True for k in ('source_initial_parity', 'strict_reload_exact', 'updated_resume_exact',
                                                 'bypass_version_tamper_rejected')) and
                    fact['updated_step1_sha256'] != fact['initial_state_sha256'] and
                    fact['state_sha256'] == fact['steps'][-1]['state_sha256'] and
                    fact['actual_batch'] == fact['steps'][0]['batch'] and len(fact['actual_batch']) == 64 and
                    diagnostic(fact['resumed_step2']) == diagnostic(fact['steps'][1]), 'CPU UPDATED1->2/source/data witness differs')
            check_steps(fact['steps'], 1, 2, a)
            check_steps([fact['resumed_step2']], 2, 1, a)
        require(record['arms']['control']['raw_unit_packed_sha256'] == record['arms']['candidate']['raw_unit_packed_sha256'] and
                record['initial_raw_unit_packed_sha256'] == record['arms']['control']['raw_unit_packed_sha256'], 'CPU source/arm parity differs')
        require(all(record['arms']['control']['identity'][k] == record['arms']['candidate']['identity'][k] for k in
                    ('source', 'config', 'native_inventory', 'encoder_sha256', 'complement_sha256', 'buffers_sha256',
                     'head_buffers_sha256', 'means_sha256', 'frozen_sha256', 'features_sha256',
                     'static_sha256', 'warm_members_sha256', 'schedule_sha256', 'full_schedule_sha256')),
                'CPU shared complete initial state differs')
    else:
        count = 17 if phase == 'mechanics' else 1000
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
                    'fresh TRAIN1000 checkpoint binding differs')


if __name__ == '__main__':
    main()
