#!/usr/bin/env python3
"""Fixed cache-only concat ranking trial; native qualification is UNRUN.

Own execution.json is exactly trainer/test. Frozen external nearest/fitter
closures retain original admission and source predicates. No quality reads.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import sys
import time
from tempfile import TemporaryDirectory
from types import FunctionType
import weakref

UNIT_STARTED = time.perf_counter()
SCHEMA = 'siglip2-compact-ranking-v1'
AUTHORITY_SCHEMA = 'siglip2-compact-ranking-launch-v1'
INFERENCE_SCHEMA = 'siglip2-compact-ranking-inference-v1'
BUNDLE_SCHEMA = 'siglip2-compact-ranking-bundle-v1'
FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}
ARMS = ('control', 'candidate')
SEEDS = (179061, 179069)
VIEWS = ('canonical', 'augmented')
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
NEAREST = {'root': '/home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5', 'execution_sha256': '0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311', 'code': {'nearest_ranking_readout.py': '862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738', 'test_siglip2_nearest_ranking.py': 'a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec', 'train_siglip2_nearest_ranking.py': '4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c'}}
FITTER = {'code': {'fit_siglip2_prototype_residual.py': '95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b', 'prototype_residual_readout.py': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68', 'test_siglip2_prototype_residual.py': 'c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4'}, 'execution_sha256': 'a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe', 'root': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2'}
ACCEPTED = {'arm': 'concat', 'checkpoint': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt', 'sha256': 'b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf'}, 'launch': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json', 'sha256': '109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630'}, 'terminal': {'both_locks_held': True, 'invocation_id': '94a84de4194f42f0842cee5b5c8f932a', 'log': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log', 'sha256': '93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197'}, 'native_peak_rss_kib': 2854356, 'receipt': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json', 'sha256': 'b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c'}, 'service_seconds': 234.821, 'unit': 'sfora-so400-signed-concat-fit-concat-v1'}, 'terminal_state_sha256': 'a118fd98cce0b8fafa51c897be70b2b6e2ec93ebb226b2382dd264721776b644'}
READOUT = {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py', 'sha256': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68'}
ADAM = {'lr': 1e-4, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05,
        'amsgrad': False, 'maximize': False, 'foreach': False, 'capturable': False,
        'differentiable': False, 'fused': False}
RECIPE = {'seeds': list(SEEDS), 'rows': 6355, 'classes': 1008, 'singletons': 12,
          'updates': 128, 'batch': 64, 'microbatch': 16, 'views': list(VIEWS),
          'trainable_names': ['A'], 'trainable_shapes': [[128, 160]], 'trainable_scalars': 20480,
          'adamw': {**ADAM, 'betas': list(ADAM['betas'])}, 'clip': 1., 'initial_scaler': 128.,
          'regression': 'both same-row views coordinate sum / (128*e0)',
          'ranking': 'candidate coefficient1; both mine; hinge sum / (2*K*.05)',
          'margin': .05, 'teacher': 'accepted canonical T; member-inclusive P; normalize(T); both-view e0',
          'mining': 'all6355; other original image positive; wrong identity negative; ascending original-row ties',
          'schedule': 'original first128 B64 per seed; warm-authenticated; masks unused',
          'readout': 'original CPU-renormalized genuine features; FP32 all; autocast disabled',
          'frozen': 'complete encoder448/config/buffers/processor/head/classifier/means',
          'core': 'cache/target preparation + both-arm mining + both-view forward/backward + optimizer'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'nearest', 'fitter', 'accepted',
               'readout', 'recipe', 'resource_policy', 'both_locks_held', 'selected_cpu',
               'selected_mechanics', 'native_authority'}
STATIC_KEYS = ('provenance', 'config', 'buffers', 'processor', 'head', 'classifier', 'means',
               'partition', 'original_rows', 'target', 'schedules', 'views', 'teachers')
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'A', *STATIC_KEYS, 'optimizer', 'scaler',
                'counter', 'cpu_rng', 'cuda_rng', 'numerical_flags'}
INFERENCE_KEYS = {'schema', 'source', 'config', 'buffers', 'processor', 'head', 'A', 'means',
                  'numerical_flags', 'vision_sha256', 'fixed_sha256'}
SERVING_FILES = {'qualify_siglip2_substrate_cpu.py', 'extract_siglip2_vision_source.py',
                 'train_siglip2_cached_readout.py', 'train_siglip2_substrate_adaptation.py',
                 'prototype_residual_readout.py', 'quadratic_readout.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def file_fact(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and Path(value['path']).is_absolute() and
            isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'exact FILE required')


def bound_file(guards, path, expected):
    file_fact({'path': str(path), 'sha256': expected})
    path = Path(path)
    require(path.resolve() == path and path.is_file(), 'canonical regular FILE required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        buffer = bytearray(1024**2)
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
            os.posix_fadvise(stream.fileno(), stream.tell() - count, count, os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'current FILE bytes differ: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    return path


def read_json(fact, guards):
    file_fact(fact)
    path = bound_file(guards, fact['path'], fact['sha256'])
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == fact['sha256'], 'JSON size/current bytes differ')
    return strict_json(raw)


def closure(root, sha, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': sha}, guards)
    require(isinstance(code, dict) and code.keys() == set(names) and
            all(Path(n).name == n for n in code), 'exact code closure required')
    for name, digest in code.items():
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
    require(phase in ('cpu', 'mechanics', 'train'), 'fixed phase required')
    return {'seconds': 500 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def check_unit(unit):
    require(isinstance(unit, dict) and unit.keys() == {'receipt', 'log', 'unit', 'invocation_id',
            'service_seconds', 'native_peak_rss_kib', 'both_locks_held'} and unit['both_locks_held'] is True,
            'complete UNIT required')
    for key in ('receipt', 'log'):
        file_fact(unit[key])
    require(isinstance(unit['unit'], str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and
            isinstance(unit['invocation_id'], str) and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            all(type(unit[k]) in (int, float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')), 'actual UNIT identity/resources required')


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and args.arm in ARMS and type(args.seed) is int and
            launch['seed'] == args.seed and args.seed in SEEDS and launch['nearest'] == NEAREST and
            launch['fitter'] == FITTER and launch['accepted'] == ACCEPTED and launch['readout'] == READOUT and
            launch['recipe'] == RECIPE and launch['resource_policy'] == policy(args.phase) and
            launch['both_locks_held'] is True, 'frozen compact-ranking launch differs')
    file_fact(launch['native_authority'])
    require((args.phase != 'cpu' or (args.arm == 'control' and args.seed == SEEDS[0])) and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]) and
            (launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if args.phase != 'cpu':
        check_unit(launch['selected_cpu'])
    if args.phase == 'train':
        require(isinstance(launch['selected_mechanics'], dict) and
                launch['selected_mechanics'].keys() == set(ARMS), 'both same-seed mechanics required')
        for unit in launch['selected_mechanics'].values():
            check_unit(unit)


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'nearest', 'fitter', 'accepted', 'readout', 'recipe', 'native_authority')}


def cli(root, authority, sha, execution, phase, arm, seed, output):
    return [str(Path(root) / 'train_siglip2_compact_ranking.py'), '--execution-sha256', execution,
            '--authority', str(authority), '--authority-sha256', sha, '--phase', phase,
            '--arm', arm, '--seed', str(seed), '--output', str(output)]


@contextmanager
def timed(context, name):
    tick = time.perf_counter()
    print(json.dumps({'event': 'COMPACT_PHASE', 'phase': name, 'boundary': 'begin',
                      'seconds': tick - context['started']}), flush=True)
    try:
        yield
    finally:
        delta = time.perf_counter() - tick
        context['phase_seconds'][name] = context['phase_seconds'].get(name, 0.) + delta
        print(json.dumps({'event': 'COMPACT_PHASE', 'phase': name, 'boundary': 'end',
                          'delta_seconds': delta, 'seconds': time.perf_counter() - context['started']}), flush=True)


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    roots = (root, Path(NEAREST['root']), Path(FITTER['root']))
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
            not args.output.exists() and not args.output.is_symlink() and
            all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in roots) and
            all(not root.is_relative_to(p) and not p.is_relative_to(root) for p in roots[1:]), 'separate exclusive output required')
    require(closure(NEAREST['root'], NEAREST['execution_sha256'], NEAREST['code'], guards) == NEAREST['code'],
            'frozen external nearest3 differs')
    nearest = load_authenticated('_compact_nearest', Path(NEAREST['root']) / 'train_siglip2_nearest_ranking.py',
                                 NEAREST['code']['train_siglip2_nearest_ranking.py'], guards)
    require(nearest.FITTER == FITTER and nearest.ACCEPTED == ACCEPTED, 'external accepted authority differs')
    require(nearest.closure(FITTER['root'], FITTER['execution_sha256'], FITTER['code'], guards) == FITTER['code'],
            'frozen external fitter3 differs')
    fitter = nearest.load_authenticated('_compact_fitter', Path(FITTER['root']) / 'fit_siglip2_prototype_residual.py',
                                       FITTER['code']['fit_siglip2_prototype_residual.py'], guards)
    from types import SimpleNamespace
    startup = nearest.startup_admission_adapter(fitter, guards)
    original = startup.authority(SimpleNamespace(execution_sha256=FITTER['execution_sha256'],
        authority=Path(ACCEPTED['launch']['path']), authority_sha256=ACCEPTED['launch']['sha256'],
        phase='fit', arm='concat', output=args.output))
    accepted = startup.admit_terminal(original, ACCEPTED['terminal'], 'fit', 'concat')
    require(accepted['checkpoint'] == ACCEPTED['checkpoint'] and
            accepted['terminal_state_sha256'] == ACCEPTED['terminal_state_sha256'], 'accepted concat endpoint differs')
    for p, h in original['guards'].items():
        require(guards.setdefault(p, h) == h, 'source guard conflict')
    context = {'args': args, 'root': root, 'guards': guards, 'code': code, 'launch': launch,
               'nearest': nearest, 'fitter': fitter, 'fit_context': original, 'legacy': original['legacy'],
               'old': original['old'], 'source': dict(original['source']), 'accepted_record': accepted,
               'phase_seconds': {}, 'terminals': {}, 'terminal_cgroups': {}, 'started': UNIT_STARTED}
    nearest.native_source_api(context)  # Exactly original supplemental admission and exit API.
    context['required_guards'] = {p: h for p, h in guards.items() if p != str(args.authority)}
    if args.phase != 'cpu':
        admit_terminal(context, launch['selected_cpu'], 'cpu', 'control', SEEDS[0])
    if args.phase == 'train':
        for arm in ARMS:
            admit_terminal(context, launch['selected_mechanics'][arm], 'mechanics', arm, SEEDS[0])
        a, b = (context['terminals'][f'mechanics:{SEEDS[0]}:{arm}'] for arm in ARMS)
        require(a['initial_A_sha256'] == b['initial_A_sha256'] and
                a['initial_raw_unit_packed_sha256'] == b['initial_raw_unit_packed_sha256'] and
                a['identity']['static_sha256'] == b['identity']['static_sha256'] and
                a['identity']['initial_cpu_rng_sha256'] == b['identity']['initial_cpu_rng_sha256'] and
                a['identity']['initial_cuda_rng_sha256'] == b['identity']['initial_cuda_rng_sha256'], 'matched mechanics differs')
    return context


def fingerprint(context, value, **kwargs):
    return context['nearest'].fingerprint(context, value, **kwargs)


def clone(context, value, device='cpu'):
    return context['old'].clone_tree(value, device)


def helper_guard(context):
    nearest = context['nearest']
    bound_file({}, nearest.__file__, NEAREST['code']['train_siglip2_nearest_ranking.py'])
    cached = context.get('nearest_objects')
    if cached is None:
        cached = (dict(vars(nearest)), [(fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)
                  for fn in vars(nearest).values() if isinstance(fn, FunctionType)])
        context['nearest_objects'] = cached
    require(sys.modules.get('_compact_nearest') is nearest and
            Path(nearest.__file__) == Path(nearest.__spec__.origin) == Path(NEAREST['root']) / 'train_siglip2_nearest_ranking.py' and
            vars(nearest).keys() == cached[0].keys() and all(vars(nearest)[k] is v for k, v in cached[0].items()) and
            all(fn.__code__ is code and fn.__defaults__ == defaults and fn.__kwdefaults__ == kw
                for fn, code, defaults, kw in cached[1]), 'external nearest live dependency differs')
    module = context['fitter'].prepare_readout(context['fit_context'])
    require(module.__file__ == READOUT['path'], 'original prototype helper origin differs')
    bound_file(context['guards'], module.__file__, READOUT['sha256'])
    prepared = context['fit_context'].get('original_preparation')
    if prepared is not None:
        for owner, name, fn, code in prepared['functions']:
            require(getattr(owner, name, None) is fn and fn.__code__ is code, 'original live helper changed: ' + name)
        for values, members in prepared['globals']:
            require(values.keys() == members.keys() and all(values[k] is v for k, v in members.items()),
                    'original helper global changed')
    return module


def prepare_native(context):
    """Full accepted typed admission, genuine caches, fixed teachers; no fit."""
    import torch
    from torch.nn import functional as F
    legacy, fitter, original = context['legacy'], context['fitter'], context['fit_context']
    with timed(context, 'shared_source_preparation'):
        fitter.prepare_original(original)
        original['flags'] = legacy['flags']
        context['flags'] = legacy['flags']
        readout = helper_guard(context)
    with timed(context, 'cache_target_preparation'):
        for view in VIEWS:
            fact = legacy['selected']['source']['caches'][view]
            require(fact['normalized'] is True and fact['raw_pooled_cache'] is False and
                    fact['shape'] == [6355, 1152] and fact['dtype'] == 'float32', 'genuine normalized cache required')
            legacy['genuine'].cache_facts(fact, context['guards'])
        views = legacy['genuine'].training_features(legacy['selected'])
        path = bound_file(context['guards'], ACCEPTED['checkpoint']['path'], ACCEPTED['checkpoint']['sha256'])
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        with path.open('rb') as stream:
            pages = legacy['original'].CheckpointPages(stream)
            ident = context['accepted_record']['identity']
            fitter.check_payload(original, disk, ident)
            require(fingerprint(context, disk, consumed=pages.consume) == ACCEPTED['terminal_state_sha256'],
                    'complete accepted typed payload differs')
            initial = {k: clone(context, disk[k]) for k in
                       ('encoder', 'config', 'buffers', 'head', 'classifier', 'A', 'means', 'partition', 'original_rows', 'target')}
            initial['processor'] = clone(context, disk['encoder']['export_runtime']['processor'])
            require(torch.equal(views['canonical'], disk['features']), 'accepted canonical CPU normalization differs')
            schedules = {}
            for seed in SEEDS:
                batches, _ = legacy['genuine'].schedule_and_masks(initial['target'].tolist(), seed)
                schedules[str(seed)] = torch.from_numpy(batches[:128].copy())
                require(torch.equal(schedules[str(seed)], disk['warm_payload']['schedules'][str(seed)][:128]),
                        'original warm first128 schedule differs')
            head = legacy['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).train()
            A = torch.nn.Parameter(initial['A'].clone())
            with torch.no_grad(), torch.autocast('cpu', enabled=False):
                T = readout.raw_features(views['canonical'], head, A, initial['means'], 'concat', legacy['quadratic'])
                accepted = readout.raw_features(disk['features'], head, A, initial['means'], 'concat', legacy['quadratic'])
                require(torch.equal(T, accepted) and
                        fingerprint(context, context['old'].packed_outputs(legacy, T)) == ident['output_witness_sha256'],
                        'accepted canonical output/packing differs')
                U = readout.raw_features(views['augmented'], head, A, initial['means'], 'concat', legacy['quadratic'])
                target = initial['target']
                counts = torch.bincount(target, minlength=1008)
                require(counts.shape == (1008,) and (counts > 0).all().item() and
                        counts.sum().item() == 6355 and (counts == 1).sum().item() == 12, 'TRAIN6355/1008/singletons differ')
                P = torch.zeros((1008, 128), dtype=torch.float32)
                P.index_add_(0, target, T)
                P /= counts[:, None]
                e0 = torch.cat(((T - P[target]).square().sum(1), (U - P[target]).square().sum(1))).mean()
                require(torch.isfinite(e0).item() and e0.item() > 0 and (T.norm(dim=1) > 0).all().item(),
                        'positive finite both-view e0/nonzero T required')
                initial['teachers'] = {'T': T.detach(), 'V': F.normalize(T, dim=1).detach(),
                                       'P': P.detach(), 'counts': counts, 'e0': e0.detach()}
            del head, A, T, U, P, e0, accepted
        del disk, pages
        gc.collect()
        initial['provenance'] = {'accepted': ACCEPTED, 'fitter': FITTER, 'nearest': NEAREST,
                                 'readout': READOUT, 'encoder': initial.pop('encoder')}
        initial.update(views=views, schedules=schedules)
        context['initial'] = initial
        context['initial_static_sha256'] = fingerprint(context, {k: initial[k] for k in STATIC_KEYS})
        context['initial_A_sha256'] = fingerprint(context, initial['A'])
    context['nearest'].native_source_api(context).audit_origins(legacy)


def require_no_training(context):
    reference = context.get('live_training')
    require(reference is None or reference() is None, 'previous training A still alive')
    context['nearest'].require_no_model(context)


def own_A(context, state, *, admit=False, advanced=False):
    """Current bytes, never a version/hash cache; sole owner authorizes updates."""
    owners = context.setdefault('A_owners', {})
    current = fingerprint(context, state['A'])
    prior = owners.get(id(state))
    if admit:
        require(prior is None or (prior[0] is state and prior[1] is state['A'] and prior[2] == 0),
                'A re-admission requires fresh independently loaded state')
    elif advanced:
        require(prior is not None and prior[0] is state and prior[1] is state['A'] and
                state['counter'] == prior[2] + 1 and current != prior[3], 'A refresh requires one genuine changed update')
    else:
        require(prior is not None and prior[0] is state and prior[1] is state['A'] and
                state['counter'] == prior[2] and current == prior[3], 'authorized current A bytes/counter differ')
        return current
    owners[id(state)] = (state, state['A'], state['counter'], current)
    return current


def fresh(context, arm, seed, device, initial=None):
    import torch
    require_no_training(context)
    require(arm in ARMS and seed in SEEDS and device in ('cpu', 'cuda'), 'fixed training roles required')
    initial = context['initial'] if initial is None else initial
    # Features stay CPU; the microbatch alone transfers. Training holds no vision.
    state = {k: clone(context, initial[k], device if k in ('teachers', 'target', 'means', 'classifier') else 'cpu')
             for k in STATIC_KEYS}
    state.update(arm=arm, seed=seed, device=device, counter=0)
    head = context['legacy']['selected']['cached'].head_from('control', tensors=state['head'])
    state['head_object'] = head.requires_grad_(False).to(device).train()
    A = torch.nn.Parameter(clone(context, initial['A'], device))
    state['A'] = A
    context['live_training'] = weakref.ref(A)
    optimizer = torch.optim.AdamW([A], **ADAM)
    defaults = dict(ADAM)
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and len(optimizer.param_groups) == 1 and
            optimizer.param_groups[0]['params'] == [A] and
            {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'} == defaults,
            'fresh sole-A AdamW defaults/groups differ')
    state['optimizer_object'] = optimizer
    state['scaler_object'] = torch.amp.GradScaler(device, init_scale=128., enabled=device == 'cuda')
    state['target_list'] = state['target'].tolist()
    state['row_list'] = state['original_rows'].tolist()
    state['count_list'] = state['teachers']['counts'].tolist()
    own_A(context, state, admit=True)
    return state


def static_tree(state):
    return {k: dict(state['head_object'].state_dict()) if k == 'head' else state[k] for k in STATIC_KEYS}


def identity(context, state):
    import torch
    optimizer = state['optimizer_object']
    return {'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'],
            'seed': state['seed'], 'device': state['device'], 'parameter_names': ['A'],
            'parameter_shapes': [[128, 160]], 'numerical_flags': context['flags'],
            'static_sha256': fingerprint(context, static_tree(state)),
            'initial_A_sha256': context['initial_A_sha256'], 'optimizer_defaults': optimizer.defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups],
            'initial_scaler': state['scaler_object'].state_dict(),
            'initial_cpu_rng_sha256': fingerprint(context, torch.random.get_rng_state()),
            'initial_cuda_rng_sha256': fingerprint(context, torch.cuda.get_rng_state_all()) if state['device'] == 'cuda' else None}


def payload(context, state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'source': context['source'], **static_tree(state),
            'A': state['A'].detach(), 'optimizer': state['optimizer_object'].state_dict(),
            'scaler': state['scaler_object'].state_dict(), 'counter': state['counter'],
            'cpu_rng': torch.random.get_rng_state().clone(),
            'cuda_rng': [v.clone() for v in torch.cuda.get_rng_state_all()] if state['device'] == 'cuda' else [],
            'numerical_flags': context['legacy']['source_driver'].numerical_flags()}


def check_optimizer(saved, ident, step):
    import torch
    opt = saved['optimizer']
    require(opt.keys() == {'state', 'param_groups'} and len(opt['param_groups']) == 1 and
            opt['param_groups'][0]['params'] == [0] and
            {k: v for k, v in opt['param_groups'][0].items() if k != 'params'} == ident['optimizer_groups'][0] and
            opt['state'].keys() == ({0} if step else set()), 'sole-A named optimizer ownership differs')
    if step:
        member = opt['state'][0]
        require(member.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and member['step'].shape == () and
                member['step'].dtype == torch.float32 and member['step'].device.type == 'cpu' and
                float(member['step']) == step, 'AdamW exact CPU step differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = member[key]
            require(value.shape == (128, 160) and value.dtype == torch.float32 and not value.requires_grad and
                    value.grad_fn is None and torch.isfinite(value).all().item(), 'finite FP32 A moment differs')
    require(saved['scaler'] == (dict(ident['initial_scaler'], _growth_tracker=step) if ident['device'] == 'cuda'
                                else ident['initial_scaler']), 'scaler128/counter differs')


def check_payload(context, saved, ident, step):
    import torch
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            saved['source'] == context['source'] == ident['source'] and ident['method'] == method(context['launch']) and
            ident['parameter_names'] == ['A'] and ident['parameter_shapes'] == [[128, 160]] and
            ident['arm'] in ARMS and ident['seed'] in SEEDS and ident['device'] in ('cpu', 'cuda') and
            type(saved['counter']) is int and saved['counter'] == step and 0 <= step <= 128 and
            saved['numerical_flags'] == ident['numerical_flags'] == context['flags'], 'complete payload identity differs')
    require(fingerprint(context, {k: saved[k] for k in STATIC_KEYS}) == ident['static_sha256'] ==
            context['initial_static_sha256'], 'frozen complete encoder/head/teachers/cache/schedules current bytes differ')
    context['old'].check_encoder(saved['provenance']['encoder'])
    require(saved['provenance'] == context['initial']['provenance'], 'original provenance differs')
    primitive = context['legacy']['quadratic']
    primitive._check_tensor(saved['A'], (128, 160), saved['A'].device, frozen=True)
    require(torch.isfinite(saved['A']).all().item() and
            ((fingerprint(context, saved['A']) == ident['initial_A_sha256']) if step == 0 else
             (fingerprint(context, saved['A']) != ident['initial_A_sha256'])), 'initial/updated A substitution differs')
    for seed in SEEDS:
        require(saved['schedules'][str(seed)].shape == (128, 64) and
                saved['schedules'][str(seed)].dtype == torch.int64, 'full fixed first128 schedule differs')
    for name, shape, dtype in [('classifier', (1008, 128), 'torch.float32'),
                               ('target', (6355,), 'torch.int64'), ('original_rows', (6355,), 'torch.int64')]:
        value = saved[name]
        primitive._check_tensor(value, shape, value.device, frozen=True, dtype=dtype)
    require(saved['teachers'].keys() == {'T', 'V', 'P', 'counts', 'e0'}, 'complete teachers required')
    for name, shape in [('T', (6355, 128)), ('V', (6355, 128)), ('P', (1008, 128)),
                        ('counts', (1008,)), ('e0', ())]:
        value = saved['teachers'][name]
        primitive._check_tensor(value, shape, value.device, frozen=True,
                                dtype='torch.int64' if name == 'counts' else 'torch.float32')
    helper_guard(context).check_means(saved['means'], saved['A'].device, primitive)
    require(saved['views'].keys() == set(VIEWS) and all(v.shape == (6355, 1152) and
            v.dtype == torch.float32 and v.device.type == 'cpu' and not v.requires_grad and v.grad_fn is None
            for v in saved['views'].values()), 'genuine CPU views differ')
    require(saved['cpu_rng'].dtype == torch.uint8 and saved['cpu_rng'].ndim == 1 and
            fingerprint(context, saved['cpu_rng']) == ident['initial_cpu_rng_sha256'] and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0) and
            all(v.dtype == torch.uint8 and v.ndim == 1 for v in saved['cuda_rng']) and
            (ident['device'] != 'cuda' or fingerprint(context, saved['cuda_rng']) == ident['initial_cuda_rng_sha256']),
            'saved complete CPU/CUDA RNG differs')
    check_optimizer(saved, ident, step)
    context['old'].finite_tree(saved)


def integrity(context, state, ident):
    import torch
    helper = helper_guard(context)
    own_A(context, state)
    A, head, optimizer = state['A'], state['head_object'], state['optimizer_object']
    helper.check_weight(A, A.device, 'concat', context['legacy']['quadratic'])
    context['legacy']['quadratic']._check_base(head, A.device)
    require(A.device.type == state['device'] and A.grad is None and
            all(p.grad is None and not p.requires_grad and p.device == A.device for p in head.parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in head.modules()) and len(optimizer.param_groups) == 1 and
            len(optimizer.param_groups[0]['params']) == 1 and optimizer.param_groups[0]['params'][0] is A and
            optimizer.defaults == ident['optimizer_defaults'], 'sole-A roles/hooks/frozen gradients/defaults differ')
    if state['counter']:
        require(all(optimizer.state[A][k].device == A.device for k in ('exp_avg', 'exp_avg_sq')),
                'active FP32 moments must follow A device')
    source = context['legacy']['source_driver']
    require(source.numerical_flags() == context['flags'], 'original numerical flags changed')
    source.cgroup_memory()
    require(time.perf_counter() - context['started'] < policy(context['args'].phase)['seconds'], 'whole-unit deadline exceeded')
    context['nearest'].require_no_model(context)
    check_payload(context, payload(context, state, ident), ident, state['counter'])
    if state['device'] == 'cuda':
        require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'whole-unit CUDA peak exceeded')


def release(context, state):
    context.get('A_owners', {}).pop(id(state), None)
    state.clear()
    gc.collect()
    require_no_training(context)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        sys.modules['torch'].cuda.empty_cache()


def save(context, state, ident, path):
    import torch
    with timed(context, 'save'):
        integrity(context, state, ident)
        saved = payload(context, state, ident)
        digest = fingerprint(context, saved)
        with context['legacy']['extract'].exclusive(path) as stream:
            writer = context['legacy']['original'].CheckpointWriter(stream)
            torch.save(saved, writer)
            writer.flush()
        sha = context['legacy']['extract'].sha(path)
        bound_file(context['guards'], path, sha)
    return sha, digest


def restore(context, path, sha, digest, ident, step):
    """Independent head/A/optimizer; restore saved teachers, never refit."""
    import torch
    require_no_training(context)
    with timed(context, 'independent_reload'):
        path = bound_file(context['guards'], path, sha)
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        with path.open('rb') as stream:
            pages = context['legacy']['original'].CheckpointPages(stream)
            check_payload(context, disk, ident, step)
            require(fingerprint(context, disk, consumed=pages.consume) == digest, 'complete serialized state differs')
            state = fresh(context, ident['arm'], ident['seed'], ident['device'], initial=disk)
            require(identity(context, state) == ident, 'independent optimizer/roles/static/RNG differs')
            optimizer = {'state': {}, 'param_groups': clone(context, disk['optimizer']['param_groups'])}
            for index, member in disk['optimizer']['state'].items():
                optimizer['state'][index] = {k: pages.copy(v, 'cpu' if k == 'step' else ident['device'])
                                             for k, v in member.items()}
            state['optimizer_object'].load_state_dict(optimizer)
            state['scaler_object'].load_state_dict(disk['scaler'])
            state['counter'] = step
            torch.random.set_rng_state(disk['cpu_rng'].clone())
            if ident['device'] == 'cuda':
                torch.cuda.set_rng_state_all([v.clone() for v in disk['cuda_rng']])
            own_A(context, state, admit=True)
        del disk, pages, optimizer
        gc.collect()
        integrity(context, state, ident)
        require(fingerprint(context, payload(context, state, ident)) == digest, 'strict independent complete replay differs')
    return state


def loss_denominators(full_valid):
    require(type(full_valid) is int and 0 <= full_valid <= 64, 'full B64 valid count required')
    return 128, 2 * full_valid * .05 if full_valid else None


def raw_features(context, state, features):
    return helper_guard(context).raw_features(features, state['head_object'], state['A'],
                                              state['means'], 'concat', context['legacy']['quadratic'])


def mine(context, state, unit, anchors):
    import torch
    with torch.no_grad(), torch.autocast(unit.device.type, enabled=False):
        scores = (unit.detach() @ state['teachers']['V'].T).cpu().tolist()
        selected = [context['nearest'].select_nearest(row, state['target_list'], state['row_list'], int(anchor))
                    for row, anchor in zip(scores, anchors, strict=True)]
    positive, negative = zip(*selected, strict=True)
    return torch.tensor(positive, device=unit.device), torch.tensor(negative, device=unit.device)


def loss_terms(context, state, raw, anchors, full_valid):
    import torch
    from torch.nn import functional as F
    rows, rank_denominator = loss_denominators(full_valid)
    with torch.autocast(raw.device.type, enabled=False):
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item(), 'finite FP32 raw required')
        index = torch.tensor(anchors, device=raw.device)
        mse = (raw - state['teachers']['P'][state['target'][index]]).square().sum() / (rows * state['teachers']['e0'])
        unit = F.normalize(raw, dim=1)
        positive, negative = mine(context, state, unit, anchors)
        valid = positive >= 0
        require(int(valid.sum()) == sum(state['count_list'][state['target_list'][i]] > 1 for i in anchors),
                'singleton/valid mining differs')
        if valid.any().item():
            require(rank_denominator is not None, 'global valid denominator required')
            hinge = F.relu(.05 + (unit[valid] * state['teachers']['V'][negative[valid]]).sum(1) -
                          (unit[valid] * state['teachers']['V'][positive[valid]]).sum(1))
            rank = hinge.sum() / rank_denominator
            active = int((hinge > 0).sum())
        else:
            rank, active = raw.sum() * 0., 0
        require(torch.isfinite(mse).item() and torch.isfinite(rank).item(), 'nonfinite objective')
    return mse, rank, {'positive': positive.tolist(), 'negative': negative.tolist(),
                       'valid': int(valid.sum()), 'active': active}


def cached_witness(context, state):
    import torch
    batch = state['schedules'][str(state['seed'])][0].tolist()
    with torch.no_grad():
        return {view: context['old'].packed_outputs(context['legacy'], raw_features(
                context, state, state['views'][view][batch].to(state['device']))) for view in VIEWS}


def cpu_gradients(context, state):
    """Fixed first genuine B64 falsifier and full128 versus eight micro16."""
    import torch
    with timed(context, 'cpu_loss_falsifier'):
        batch = state['schedules'][str(state['seed'])][0].tolist()
        K = sum(state['count_list'][state['target_list'][i]] > 1 for i in batch)
        A = state['A']
        def objective(micro):
            regression, ranking, active = [], [], 0
            for view in VIEWS:
                for offset in range(0, 64, micro):
                    anchors = batch[offset:offset + micro]
                    raw = raw_features(context, state, state['views'][view][anchors])
                    mse, rank, diagnostic = loss_terms(context, state, raw, anchors, K)
                    regression.append(mse)
                    ranking.append(rank)
                    active += diagnostic['active']
            mse, rank = sum(regression), sum(ranking)
            control = torch.autograd.grad(mse, A, retain_graph=True)[0]
            rank_grad = torch.autograd.grad(rank, A, retain_graph=True)[0]
            candidate = torch.autograd.grad(mse + rank, A)[0]
            return mse.detach(), rank.detach(), control, rank_grad, candidate, active
        full = objective(64)
        micro = objective(16)
        for expected, actual in zip(full[:5], micro[:5], strict=True):
            require(torch.allclose(expected, actual, rtol=1e-5, atol=1e-6), 'global both-view micro16 loss/gradient differs')
        mse, rank, control, rank_grad, candidate, active = full
        require(mse.item() > 0 and active == micro[5] and active > 0 and
                all(v.dtype == torch.float32 and torch.isfinite(v).all().item() and v.double().norm().item() > 0
                    for v in (control, rank_grad, candidate)) and
                torch.allclose(candidate - control, rank_grad, rtol=1e-5, atol=1e-6),
                'fixed firstB64 regression/rank/candidate gradient falsifier failed')
        require(A.grad is None, 'falsifier changed A gradient')
        return {'seed': state['seed'], 'mse': float(mse), 'rank': float(rank), 'active': active, 'K': K,
                'control_gradient_norm': float(control.double().norm()),
                'ranking_gradient_norm': float(rank_grad.double().norm()),
                'candidate_minus_control_equals_rank': True, 'micro16_global_reduction_exact': True}


def inference_members(context, state):
    return {'schema': INFERENCE_SCHEMA,
            'source': {'accepted_A_sha256': context['initial_A_sha256'],
                       'encoder_checkpoint_sha256': state['provenance']['encoder']['checkpoint']['sha256'],
                       'readout_sha256': READOUT['sha256']},
            'numerical_flags': context['flags'], 'A': clone(context, state['A'].detach()),
            **{k: clone(context, v) for k, v in static_tree(state).items()
               if k in ('config', 'buffers', 'processor', 'head', 'means')}}


def write_json(context, path, value):
    context['legacy']['selected']['genuine']['reference'].write_json(context['legacy']['extract'], path, value)


def export_bundle(context, members, directory):
    """After training release, own one regular original vision and serving files."""
    import torch
    require_no_training(context)
    require(directory.is_absolute() and directory.parent.resolve() == directory.parent and
            not directory.exists() and not directory.is_symlink(), 'exclusive portable bundle required')
    with timed(context, 'shared_bundle_preparation'):
        directory.mkdir()
        legacy = context['legacy']
        encoder = context['initial']['provenance']['encoder']
        vision_fact = encoder['checkpoint']
        source_path = bound_file(context['guards'], vision_fact['path'], vision_fact['sha256'])
        vision_path = directory / 'vision.pt'
        # Deliberately a copy, never a symlink/hardlink to the original run.
        shutil.copyfile(source_path, vision_path)
        with vision_path.open('rb') as stream:
            os.fsync(stream.fileno())
        require(vision_path.stat().st_nlink == 1, 'bundle must own regular vision bytes')
        bound_file(context['guards'], vision_path, vision_fact['sha256'])
        with vision_path.open('rb') as stream:
            disk = torch.load(vision_path, map_location='cpu', weights_only=True, mmap=True)
            pages = legacy['original'].CheckpointPages(stream)
            require(disk.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
                    disk['runtime'] == encoder['source_proof']['runtime'] and
                    disk['vision'].keys() == {r['name'] for r in encoder['inventory']} and len(disk['vision']) == 448 and
                    fingerprint(context, disk['buffers']) == fingerprint(context, members['buffers']) and
                    disk['config'] == members['config'], 'bundle complete original vision/config/buffers differs')
            members['vision_sha256'] = fingerprint(context, disk['vision'], consumed=pages.consume)
        del disk, pages
        members['fixed_sha256'] = fingerprint(context, {k: v for k, v in members.items() if k != 'fixed_sha256'})
        require(members.keys() == INFERENCE_KEYS, 'complete inference members required')
        endpoint = directory / 'endpoint.pt'
        with legacy['extract'].exclusive(endpoint) as stream:
            writer = legacy['original'].CheckpointWriter(stream)
            torch.save(members, writer)
            writer.flush()
        write_json(context, directory / 'processor.json', members['processor']['config'])
        modules = (legacy['source_driver'], legacy['extract'], legacy['selected']['cached'],
                   legacy['original'], helper_guard(context), legacy['quadratic'], legacy['packing'])
        module_paths = {Path(m.__file__).name: Path(m.__file__) for m in modules}
        require(module_paths.keys() == SERVING_FILES | {'joint_relational_compaction.py'}, 'exact original serving helper set required')
        code = {}
        for name, path in {**module_paths, **{n: context['root'] / n for n in FILES}}.items():
            digest = context['guards'][str(path)]
            bound_file({}, path, digest)
            shutil.copyfile(path, directory / name)
            bound_file(context['guards'], directory / name, digest)
            code[name] = digest
        origins = legacy['origins']
        package_roots = [Path(v['root']) for v in origins['packages'].values()]
        native_paths = set(origins['native_files']) | {p for p in context['guards']
            if Path(p).name in context['nearest'].NATIVE_MEMBERS}
        environment_files = {p: h for p, h in context['guards'].items()
                             if any(Path(p).is_relative_to(r) for r in package_roots) or p in native_paths}
        constructor = legacy['prior']['sources']['native_environment']['vision_constructor']['path']
        require(constructor in environment_files, 'authenticated serving constructor missing')
        files = {name: legacy['extract'].sha(directory / name) for name in ('vision.pt', 'endpoint.pt', 'processor.json')}
        for name, digest in files.items():
            bound_file(context['guards'], directory / name, digest)
        manifest = {'schema': BUNDLE_SCHEMA, 'code': code, 'files': files,
                    'endpoint_state_sha256': fingerprint(context, members),
                    'environment': {'packages': origins['packages'], 'files': environment_files,
                                    'native_files': {p: environment_files[p] for p in native_paths},
                                    'vision_constructor': constructor},
                    'vision_inventory': encoder['inventory']}
        write_json(context, directory / 'bundle.json', manifest)
        sha = legacy['extract'].sha(directory / 'bundle.json')
        bound_file(context['guards'], directory / 'bundle.json', sha)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {'path': str(directory / 'bundle.json'), 'sha256': sha}


def admit_bundle(directory, sha):
    """Only bundle and qualified installed package bytes; never original data."""
    directory, guards = Path(directory), {}
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), 'canonical bundle required')
    value = read_json({'path': str(directory / 'bundle.json'), 'sha256': sha}, guards)
    require(value.keys() == {'schema', 'code', 'files', 'endpoint_state_sha256', 'environment', 'vision_inventory'} and
            value['schema'] == BUNDLE_SCHEMA and
            value['code'].keys() == FILES | SERVING_FILES | {'joint_relational_compaction.py'} and
            value['files'].keys() == {'vision.pt', 'endpoint.pt', 'processor.json'} and
            isinstance(value['endpoint_state_sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['endpoint_state_sha256']),
            'portable bundle schema/owned serving closure differs')
    for name, digest in {**value['code'], **value['files']}.items():
        bound_file(guards, directory / name, digest)
        require((directory / name).stat().st_nlink == 1, 'bundle regular single-link ownership required')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == value['code']['train_siglip2_compact_ranking.py'],
            'bundle loader current code differs')
    require(value['code']['prototype_residual_readout.py'] == READOUT['sha256'], 'original inference readout differs')
    env = value['environment']
    require(env.keys() == {'packages', 'files', 'native_files', 'vision_constructor'} and env['vision_constructor'] in env['files'] and
            set(env['packages']) == {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision'},
            'qualified serving environment differs')
    roots = [Path(v['root']) for v in env['packages'].values()]
    require(all(any(Path(p).is_relative_to(r) for r in roots) or
                env['native_files'].get(p) == h for p, h in env['files'].items()), 'serving environment contains non-package data')
    for path, digest in env['files'].items():
        bound_file(guards, path, digest)
    return value, guards


def load_inference(directory, bundle_sha256, device):
    """Public portable API: no TRAIN cache, teacher, warm payload or optimizer."""
    directory = Path(directory)
    manifest, guards = admit_bundle(directory, bundle_sha256)
    require(device in ('cpu', 'cuda'), 'fixed inference device required')
    modules = {}
    prefix = '_compact_serving_' + str(time.time_ns()) + '_'
    for filename in SERVING_FILES | {'joint_relational_compaction.py'}:
        modules[filename] = load_authenticated(prefix + filename.removesuffix('.py'), directory / filename,
                                              manifest['code'][filename], guards)
    source = modules['qualify_siglip2_substrate_cpu.py']
    original = modules['train_siglip2_substrate_adaptation.py']
    extract = modules['extract_siglip2_vision_source.py']
    import torch
    env = manifest['environment']
    endpoint_path = directory / 'endpoint.pt'
    disk = torch.load(endpoint_path, map_location='cpu', weights_only=True, mmap=True)
    require(disk.keys() == INFERENCE_KEYS and disk['schema'] == INFERENCE_SCHEMA and
            original.fingerprint(disk) == manifest['endpoint_state_sha256'] and
            original.fingerprint({k: v for k, v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and
            source.numerical_flags() == disk['numerical_flags'], 'complete independent inference state/flags differs')
    construct_context = {'packages': env['packages'], 'guards': env['files'], 'extract': extract,
                         'sources': {'native_environment': {'vision_constructor': {'path': env['vision_constructor']}}}}
    model = source.construct(disk['config'], construct_context)
    vision_path = directory / 'vision.pt'
    vision = torch.load(vision_path, map_location='cpu', weights_only=True, mmap=True)
    inventory = manifest['vision_inventory']
    require(len(inventory) == 448 and all(v['role'] == 'frozen' for v in inventory) and
            vision.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
            vision['vision'].keys() == {r['name'] for r in inventory} and vision['config'] == disk['config'] and
            original.fingerprint(vision['buffers']) == original.fingerprint(disk['buffers']), 'complete original inference vision448 differs')
    with vision_path.open('rb') as stream:
        pages = original.CheckpointPages(stream)
        require(original.fingerprint(vision['vision'], consumed=pages.consume) == disk['vision_sha256'],
                'complete portable vision tensor bytes differ')
        original.load_vision(model, vision['vision'], pages)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(vision['buffers'][name])
                pages.consume(vision['buffers'][name])
    require(original.fingerprint(model.state_dict()) == disk['vision_sha256'] and
            original.fingerprint(dict(model.named_buffers())) == original.fingerprint(disk['buffers']),
            'strict complete inference vision/buffer reload differs')
    model.requires_grad_(False).eval().to(device)
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(directory / 'processor.json', local_files_only=True, backend='torchvision')
    require(json.loads(processor.to_json_string()) == disk['processor']['config'] and
            processor.backend == disk['processor']['backend'], 'owned processor configuration differs')
    head = modules['train_siglip2_cached_readout.py'].head_from('control', tensors=disk['head'])
    head.requires_grad_(False).to(device).train()
    A = torch.nn.Parameter(disk['A'].to(device, copy=True), requires_grad=True)
    means = {k: v.to(device, copy=True) for k, v in disk['means'].items()}
    endpoint = {'model': model, 'processor_object': processor, 'head_object': head, 'A': A, 'means': means,
                'device': device, 'modules': modules, 'guards': guards, 'flags': disk['numerical_flags'],
                'manifest': manifest}
    del disk, vision, pages
    gc.collect()
    return endpoint


def inference_outputs(endpoint, images):
    """Actual image -> FP16 vision on CUDA -> FP32 accepted readout -> wire."""
    import torch
    from torch.nn import functional as F
    modules, device = endpoint['modules'], endpoint['device']
    source = modules['qualify_siglip2_substrate_cpu.py']
    require(source.numerical_flags() == endpoint['flags'], 'inference numerical flags changed')
    require(0 < len(images) <= 32 and all(not m.training and not m._forward_hooks and
            not m._forward_pre_hooks and not m._backward_hooks for m in endpoint['model'].modules()),
            'inference batch/mode/hooks differ')
    rng = torch.random.get_rng_state().clone()
    pixels = endpoint['processor_object'](images=images, return_tensors='pt')['pixel_values']
    require(pixels.shape == (len(images), 3, 256, 256) and pixels.dtype == torch.float32 and
            torch.isfinite(pixels).all().item(), 'portable pixels differ')
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device, enabled=False):
            features = F.normalize(pooled.float(), dim=1)
            raw = modules['prototype_residual_readout.py'].raw_features(features, endpoint['head_object'],
                endpoint['A'], endpoint['means'], 'concat', modules['quadratic_readout.py'])
            require((raw.norm(dim=1) > 0).all().item(), 'nonzero inference raw required')
            unit = F.normalize(raw, dim=1)
            packed = modules['joint_relational_compaction.py'].pack_int8_unit_embeddings(unit.cpu())
    require(torch.equal(rng, torch.random.get_rng_state()), 'inference CPU RNG changed')
    return {'raw': raw.cpu(), 'unit': unit.cpu(), 'codes': packed.codes.cpu(),
            'inverse_norms': packed.inverse_norms.cpu(), 'wire': packed.to_bytes()}


def release_inference(endpoint):
    modules = tuple(endpoint['modules'].values())
    model_ref = weakref.ref(endpoint['model'])
    endpoint.clear()
    for module in modules:
        require(sys.modules.pop(module.__name__, None) is module, 'serving helper registry changed')
    gc.collect()
    require(model_ref() is None, 'portable model not released')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        sys.modules['torch'].cuda.empty_cache()


@contextmanager
def deny_training_dependencies(context, directory, environment):
    """Native witness rejects open() of original run/code/data dependencies."""
    # Audit hooks cannot be removed; leave this bounded hook inert afterwards.
    # Only strings are retained, never teacher tensors or the training context.
    directory = Path(directory)
    # Ownership preflight authenticates RECORDs outside the package code roots.
    # Classify only its pinned inventory (and the vendor owner's metadata),
    # never arbitrary guarded files under the installed site directory.
    authority = read_json(context['launch']['native_authority'], context['guards'])
    require(authority['proof']['sha256'] == context['nearest'].NATIVE_PROOF_PINS['proof'],
            'original installed ownership proof differs')
    proof = read_json(authority['proof'], context['guards'])
    site = Path(proof['authority']['installed_site_root'])
    require(site.is_absolute() and site.resolve() == site and
            all(Path(v['root']).parent == site for v in environment['packages'].values()),
            'runtime metadata installed site differs')
    records = proof['installed_record_ownership']['records']
    require(len(records) == len(set(records)) and all(Path(p).parent.parent == site and
            Path(p).parent.name.endswith('.dist-info') and Path(p).name == 'RECORD' for p in records),
            'runtime metadata RECORD boundary differs')
    owners = {p for paths in proof['installed_record_ownership']['owners'].values() for p in paths}
    require(owners <= set(records), 'runtime metadata owner differs')
    metadata = set(records) | {str(Path(p).with_name(n)) for p in owners for n in ('METADATA', 'WHEEL')}
    for path in metadata:
        fact = proof['input_guards'][path]
        require(context['guards'].get(path) == fact['sha256'], 'runtime metadata original guard differs')
        bound_file({}, path, fact['sha256'])
        require(Path(path).stat().st_size == fact['size_bytes'], 'runtime metadata size differs')
    denied = {p for p in context['guards'] if not Path(p).is_relative_to(directory) and
              p not in environment['files'] and p not in metadata}
    enabled = [True]
    def audit(event, args):
        if enabled[0] and event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(args[0])).absolute().resolve())
            require(path not in denied, 'portable loader attempted original training dependency: ' + path)
    sys.addaudithook(audit)
    try:
        yield
    finally:
        enabled[0] = False


def qualify_bundle(context, directory, sha, device, witness):
    """Sequential fresh loads, same-image accepted-helper oracle and role drift."""
    import torch
    from PIL import Image
    legacy, nearest = context['legacy'], context['nearest']
    batch = context['initial']['schedules'][str(witness['seed'])][0].tolist()[:2 if device == 'cpu' else 16]
    images = []
    try:
        with timed(context, 'bundle_image_loading'):
            for ordinal in batch:
                row, path, _ = nearest.canonical_row(context, context['initial'], ordinal)
                bound_file(context['guards'], path, row['image_sha256'])
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
        expected = None
        drift = {}
        for _ in range(2):
            # Execute the owned copied loader, with original file dependencies
            # denied. A prior trainer context cannot supply missing bundle data.
            with timed(context, 'bundle_loader_authentication'):
                portable_name = '_compact_portable_entry_' + str(time.time_ns())
                portable = load_authenticated(portable_name, directory / 'train_siglip2_compact_ranking.py',
                    context['code']['train_siglip2_compact_ranking.py'], {})
                bundle, _ = portable.admit_bundle(directory, sha)
            with timed(context, 'bundle_dependency_denial'):
                with deny_training_dependencies(context, directory, bundle['environment']):
                    with timed(context, 'bundle_loader'):
                        endpoint = portable.load_inference(directory, sha, device)
                    with timed(context, 'bundle_native_forward'):
                        output = portable.inference_outputs(endpoint, images)
            with timed(context, 'bundle_oracle'):
                context['live_model'] = weakref.ref(endpoint['model'])
                pixels = endpoint['processor_object'](images=images, return_tensors='pt')['pixel_values']
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
                        pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
                    from torch.nn import functional as F
                    features = F.normalize(pooled.float(), dim=1)
                    oracle_A = torch.nn.Parameter(endpoint['A'].detach().clone())
                    raw = helper_guard(context).raw_features(features, endpoint['head_object'], oracle_A,
                        endpoint['means'], 'concat', legacy['quadratic'])
                    require(fingerprint(context, context['old'].packed_outputs(legacy, raw)) == fingerprint(context, output),
                            'same-role original helper raw/unit/packed/wire differs')
                    del oracle_A, raw, features, pooled
                current = fingerprint(context, output)
                require(expected is None or current == expected, 'sequential independent inference parity differs')
                expected = current
                difference = output['raw'] - witness['cache_raw'][batch]
                drift = {'cache_native_drift_max_abs': float(difference.abs().max()),
                         'cache_native_drift_l2': float(difference.double().norm()),
                         'arithmetic_role': 'CPU FP32 native' if device == 'cpu' else 'CUDA FP16 B16 native'}
            with timed(context, 'bundle_release'):
                del pixels, output, difference
                portable.release_inference(endpoint)
                require(sys.modules.pop(portable_name, None) is portable, 'portable entry registry changed')
                nearest.require_no_model(context)
    finally:
        for image in images:
            image.close()
    return {'native_raw_unit_packed_sha256': expected, 'batch': batch,
            'inference_state_sha256': bundle['endpoint_state_sha256'],
            'bundle_original_dependencies_denied': True, **drift}


def update(context, state, ident, step):
    import torch
    require(state['device'] == 'cuda' and state['counter'] == step - 1 and 1 <= step <= 128, 'fixed CUDA update required')
    own_A(context, state)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    K = sum(state['count_list'][state['target_list'][i]] > 1 for i in batch)
    optimizer, scaler, A = state['optimizer_object'], state['scaler_object'], state['A']
    optimizer.zero_grad(set_to_none=True)
    before = fingerprint(context, A.detach())
    mse_sum = rank_sum = 0.
    active, mined = 0, []
    ranking_gradient = torch.zeros_like(A) if step == 1 else None
    for view in VIEWS:
        for offset in range(0, 64, 16):
            anchors = batch[offset:offset + 16]
            features = state['views'][view][anchors].to(state['device'])
            raw = raw_features(context, state, features)
            require(raw.requires_grad and raw.grad_fn is not None, 'sole-A graph detached')
            mse, rank, selected = loss_terms(context, state, raw, anchors, K)
            if step == 1:
                grad = torch.autograd.grad(rank, A, retain_graph=True)[0]
                require(grad.dtype == torch.float32 and torch.isfinite(grad).all().item(), 'finite FP32 rank gradient required')
                ranking_gradient.add_(grad.detach())
                del grad
            loss = mse + rank if state['arm'] == 'candidate' else mse
            scaler.scale(loss).backward()
            mse_sum += float(mse.detach())
            rank_sum += float(rank.detach())
            active += selected['active']
            mined.append({'view': view, 'batch': anchors, **selected})
            del raw, mse, rank, loss, features
    scaler.unscale_(optimizer)
    require(A.grad.dtype == torch.float32 and torch.isfinite(A.grad).all().item() and
            A.grad.double().norm().item() > 0, 'finite nonzero A gradient required')
    gradient = float(A.grad.double().norm())
    rank_gradient = float(ranking_gradient.double().norm()) if ranking_gradient is not None else None
    if step == 1:
        require(active > 0 and rank_gradient > 0, 'fixed first-update rank gradient inactive')
    del ranking_gradient
    norm = torch.nn.utils.clip_grad_norm_([A], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    require(scaler.get_scale() == scale == 128 and optimizer.state[A]['step'].item() == step,
            'skipped/nonfinite/rescaled update forbidden')
    state['counter'] = step
    after = fingerprint(context, A.detach())
    require(before != after, 'actual A update required')
    own_A(context, state, advanced=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    core = time.perf_counter() - tick
    with timed(context, 'update_integrity'):
        integrity(context, state, ident)
        digest = fingerprint(context, payload(context, state, ident))
    row = {'step': step, 'batch': batch, 'mined': mined, 'full_valid': K, 'mse': mse_sum, 'rank': rank_sum,
           'loss': mse_sum + (rank_sum if state['arm'] == 'candidate' else 0.), 'active_hinges': active,
           'gradient_norm': gradient, 'ranking_gradient_norm': rank_gradient, 'preclip_norm': float(norm),
           'A_before_sha256': before, 'A_after_sha256': after, 'scale': scaler.get_scale(),
           'state_sha256': digest, 'core_seconds': core, 'seconds': time.perf_counter() - tick}
    print(json.dumps({'event': 'COMPACT_UPDATE', **row}, sort_keys=True, allow_nan=False), flush=True)
    return row


def diagnostic(row):
    return {k: v for k, v in row.items() if k not in ('seconds', 'core_seconds')}


def check_steps(rows, start, count):
    require(isinstance(rows, list) and len(rows) == count and
            [r['step'] for r in rows] == list(range(start, start + count)), 'complete bounded steps required')
    for row in rows:
        require(len(row['batch']) == 64 and all(type(i) is int and 0 <= i < 6355 for i in row['batch']) and
                row['scale'] == 128 and row['gradient_norm'] > 0 and
                row['A_before_sha256'] != row['A_after_sha256'] and
                0 < row['core_seconds'] <= row['seconds'] and len(row['mined']) == 8 and
                [m['view'] for m in row['mined']] == ['canonical'] * 4 + ['augmented'] * 4 and
                all(m['batch'] == row['batch'][(i % 4) * 16:(i % 4 + 1) * 16] and len(m['positive']) == 16 and
                    len(m['negative']) == 16 and all(type(p) is int and -1 <= p < 6355 for p in m['positive']) and
                    all(type(n) is int and 0 <= n < 6355 for n in m['negative'])
                    for i, m in enumerate(row['mined'])), 'complete paired-view miner/update record differs')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('mse', 'rank', 'loss', 'gradient_norm', 'preclip_norm', 'core_seconds', 'seconds')),
                'nonfinite update record')
    if start == 1:
        require(rows[0]['active_hinges'] > 0 and rows[0]['ranking_gradient_norm'] > 0, 'first-step rank activity missing')


def tamper_witness(context, state, ident):
    import torch
    nearest = context['nearest']
    values = [state['A'], next(state['head_object'].parameters()), state['means']['concat'],
              state['teachers']['T'], state['teachers']['P'], state['views']['augmented']]
    for value in values:
        saved, version = value.detach().clone(), value._version
        try:
            value.data.reshape(-1)[0] += .25
            require(value._version == version, 'tamper must bypass tensor version')
            nearest.rejected(lambda: integrity(context, state, ident), 'current .data bytes tamper accepted')
        finally:
            value.data.copy_(saved)
        integrity(context, state, ident)
    state['A'].requires_grad_(False)
    try:
        nearest.rejected(lambda: integrity(context, state, ident), 'A role mutation accepted')
    finally:
        state['A'].requires_grad_(True)
    saved = payload(context, state, ident)
    for key, value in (('schema', 'wrong'), ('source', {}), ('teachers', {}), ('schedules', {}),
                       ('buffers', {}), ('optimizer', {'state': {}, 'param_groups': []}), ('counter', 1)):
        nearest.rejected(lambda k=key, v=value: check_payload(context, {**saved, k: v}, ident, 0),
                         'malformed complete state accepted')
    del saved


def inference_witness(context, state):
    import torch
    with torch.no_grad():
        raw = raw_features(context, state, state['views']['canonical'].to(state['device'])).detach().cpu()
    return {'seed': state['seed'], 'cache_raw': raw}


def cpu_witnesses(context):
    import torch
    args = context['args']
    require(not torch.cuda.is_initialized(), 'CPU CUDA hidden required')
    gradients, first_witness, first_ident, first_digest = [], None, None, None
    members, native_witness = None, None
    for seed in SEEDS:
        torch.random.default_generator.manual_seed(seed)
        matched = None
        for arm in ARMS:
            state = fresh(context, arm, seed, 'cpu')
            ident = identity(context, state)
            integrity(context, state, ident)
            witness = fingerprint(context, cached_witness(context, state))
            if arm == 'control':
                gradients.append(cpu_gradients(context, state))
                tamper_witness(context, state, ident)
                checkpoint = args.output / f'initializer-{seed}.pt'
                sha, digest = save(context, state, ident, checkpoint)
                matched = (ident, witness)
                if seed == SEEDS[0]:
                    first_witness, first_ident, first_digest = witness, ident, digest
                    members, native_witness = inference_members(context, state), inference_witness(context, state)
                release(context, state)
                state = restore(context, checkpoint, sha, digest, ident, 0)
                require(fingerprint(context, cached_witness(context, state)) == witness, 'CPU strict initial reload differs')
                release(context, state)
            else:
                require(ident['static_sha256'] == matched[0]['static_sha256'] and
                        ident['initial_cpu_rng_sha256'] == matched[0]['initial_cpu_rng_sha256'] and
                        witness == matched[1], 'independent matched CPU initialization differs')
                release(context, state)
    bundle = export_bundle(context, members, args.output / 'bundle')
    with timed(context, 'cpu_bundle_qualification'):
        native = qualify_bundle(context, args.output / 'bundle', bundle['sha256'], 'cpu', native_witness)
    require(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
    return {'completed_step': 0, 'identity': first_ident, 'terminal_state_sha256': first_digest,
            'initial_A_sha256': context['initial_A_sha256'], 'initial_raw_unit_packed_sha256': first_witness,
            'checkpoint': {'path': str(args.output / f'initializer-{SEEDS[0]}.pt'),
                           'sha256': context['guards'][str(args.output / f'initializer-{SEEDS[0]}.pt')]},
            'bundle': bundle, 'gradients': gradients, 'initial_arm_parity': True,
            'cpu_serialization_exact': True, 'bypass_version_tamper_rejected': True, 'malformed_state_rejected': True,
            'native_role_mutation_rejected': True, 'native_loss_reduction_exact': True,
            'inference_artifact_independent': True, 'forward_oracle_exact': True,
            'native_training_inference_exact': True, 'cuda_initialized': False, 'peak_cuda_allocated_bytes': 0,
            'training_state_discarded': True, 'strict_reload_exact': True, **native,
            'total_training_core_seconds': context['phase_seconds']['cache_target_preparation']}


def gpu_run(context):
    import torch
    args = context['args']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible fresh CUDA device required')
    torch.cuda.manual_seed_all(args.seed)
    state = fresh(context, args.arm, args.seed, 'cuda')
    ident = identity(context, state)
    integrity(context, state, ident)
    initial_witness = fingerprint(context, cached_witness(context, state))
    require(ident['static_sha256'] == context['terminals'][f'cpu:{SEEDS[0]}:control']['identity']['static_sha256'],
            'qualified CPU initial source/teachers differ')
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 128
    with TemporaryDirectory(prefix='discard-mechanics-', dir=args.output) as directory:
        temporary = Path(directory)
        # Snapshot8 belongs to the independently reconstructed branch. The
        # uninterrupted17 reference never calls save at step8.
        for step in range(1, total + 1):
            row = update(context, state, ident, step)
            if args.phase == 'train' and args.seed == SEEDS[0] and step <= 17:
                mechanics = context['terminals'][f'mechanics:{SEEDS[0]}:{args.arm}']
                require(diagnostic(row) == diagnostic(mechanics['steps'][step - 1]), 'fresh first17 mechanics replay differs')
            rows.append(row)
        witness = fingerprint(context, cached_witness(context, state))
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else args.output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        members, native_witness = inference_members(context, state), inference_witness(context, state)
        release(context, state)
        if args.phase == 'mechanics':
            state = fresh(context, args.arm, args.seed, 'cuda')
            require(identity(context, state) == ident, 'independently reconstructed mechanics identity differs')
            first8 = [update(context, state, ident, step) for step in range(1, 9)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[:8], first8, strict=True)),
                    'independent first8 replay differs')
            sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
            release(context, state)
            state = restore(context, temporary / 'step8.pt', sha8, digest8, ident, 8)
            resumed = [update(context, state, ident, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    fingerprint(context, payload(context, state, ident)) == digest,
                    'uninterrupted17 vs independently reconstructed serialized8+9 differs')
            release(context, state)
        state = restore(context, checkpoint, sha, digest, ident, total)
        require(fingerprint(context, cached_witness(context, state)) == witness, 'strict updated cache raw/unit/packed reload differs')
        substitution = clone(context, state['A'].detach())
        try:
            state['A'].data.copy_(context['initial']['A'].to(state['device']))
            context['nearest'].rejected(lambda: integrity(context, state, ident), 'original A substituted for updated endpoint')
        finally:
            state['A'].data.copy_(substitution)
        del substitution
        integrity(context, state, ident)
        release(context, state)
        bundle_directory = temporary / 'bundle' if args.phase == 'mechanics' else args.output / 'bundle'
        bundle = export_bundle(context, members, bundle_directory)
        with timed(context, 'gpu_bundle_qualification'):
            native = qualify_bundle(context, bundle_directory, bundle['sha256'], 'cuda', native_witness)
        with timed(context, 'post_calibration_api_authentication'):
            api = context['nearest'].native_source_api(context)
        with timed(context, 'post_calibration_origin_audit'):
            api.audit_origins(context['legacy'], require_exact=True)
        for path in list(context['guards']):
            if Path(path).is_relative_to(temporary):
                context['guards'].pop(path)  # Discard only after full reload/parity qualification.
    check_steps(rows, 1, total)
    if resumed:
        check_steps(resumed, 9, 9)
    return {'completed_step': total, 'identity': ident, 'steps': rows, 'resumed_steps': resumed,
            'terminal_state_sha256': digest, 'initial_A_sha256': ident['initial_A_sha256'],
            'initial_raw_unit_packed_sha256': initial_witness,
            'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'bundle': None if args.phase == 'mechanics' else bundle,
            'replay_exact': args.phase == 'mechanics', 'independent_first8_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics',
            'fresh_first17_exact': args.phase == 'train' and args.seed == SEEDS[0],
            'mechanics_seed': SEEDS[0],
            'source_substitution_rejected': True, 'strict_reload_exact': True,
            'inference_artifact_independent': True, 'forward_oracle_exact': True,
            'native_training_inference_exact': True, 'cuda_initialized': True,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), **native,
            'total_training_core_seconds': context['phase_seconds']['cache_target_preparation'] +
                sum(r['core_seconds'] for r in rows) + (sum(r['core_seconds'] for r in first8 + resumed) if resumed else 0.),
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:])}


def check_terminal_record(record, launch, phase, arm, seed):
    from types import SimpleNamespace
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == seed and method(record['launch']) == method(launch) and
            record['resource_policy'] == policy(phase) and record['optimizer_members'] == 1 and
            record['trainable_scalars'] == 20480 and record['frozen_vision_members'] == 448 and
            record['quality_read'] is False and all(record[k] is True for k in
                ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'sequential_model_ownership', 'forward_oracle_exact',
                 'native_training_inference_exact', 'inference_artifact_independent',
                 'bundle_original_dependencies_denied', 'both_locks_held_in_parent_authority')) and
            0 < record['total_training_core_seconds'] < record['wall_seconds'] < policy(phase)['seconds'] and
            0 < record['process_peak_rss_kib'] <= 8 * 1024**2, 'qualified compact whole-unit contract differs')
    check_launch(record['launch'], SimpleNamespace(execution_sha256=launch['execution_sha256'],
                                                  phase=phase, arm=arm, seed=seed))
    ident = record['identity']
    require(record['code'].keys() == FILES and record['authority_sha256'] == record['authority']['sha256'] and
            record['invocation']['optimize'] == 0 and ident['method'] == method(launch) and
            ident['source'] == record['source'] and ident['arm'] == arm and ident['seed'] == seed and
            ident['parameter_names'] == ['A'] and ident['parameter_shapes'] == [[128, 160]] and
            record['numerical_flags'] == ident['numerical_flags'] and
            isinstance(record['inference_state_sha256'], str) and re.fullmatch('[0-9a-f]{64}', record['inference_state_sha256']),
            'whole-unit source/code/optimizer/inference identity differs')
    if phase == 'cpu':
        require(record['completed_step'] == 0 and ident['device'] == 'cpu' and
                record['cuda_initialized'] is False and record['peak_cuda_allocated_bytes'] == 0 and
                record['invocation']['cuda_visible_devices'] == '' and
                isinstance(record['checkpoint'], dict) and isinstance(record['bundle'], dict) and
                all(record[k] is True for k in ('initial_arm_parity', 'cpu_serialization_exact',
                    'bypass_version_tamper_rejected', 'malformed_state_rejected', 'native_role_mutation_rejected',
                    'native_loss_reduction_exact')) and [g['seed'] for g in record['gradients']] == list(SEEDS) and
                all(g['mse'] > 0 and g['active'] > 0 and g['control_gradient_norm'] > 0 and
                    g['ranking_gradient_norm'] > 0 and g['candidate_minus_control_equals_rank'] is True and
                    type(g['K']) is int and 0 < g['K'] <= 64 and 0 < g['active'] <= 2 * g['K'] and
                    g['micro16_global_reduction_exact'] is True and
                    all(type(g[k]) in (int, float) and math.isfinite(g[k]) for k in
                        ('mse', 'rank', 'control_gradient_norm', 'ranking_gradient_norm'))
                    for g in record['gradients']), 'both-seed CPU qualification incomplete')
    else:
        require(ident['device'] == 'cuda' and record['cuda_initialized'] is True and
                record['exact_four_native_membership'] is True and record['source_substitution_rejected'] is True and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000 and
                record['invocation']['cuda_visible_devices'] not in (None, '') and
                record['invocation']['cublas_workspace_config'] == ':4096:8' and
                record['launch']['selected_cpu'] == launch['selected_cpu'] and record['mechanics_seed'] == SEEDS[0],
                'native CUDA qualification incomplete')
        count = 17 if phase == 'mechanics' else 128
        require(record['completed_step'] == count, 'fixed update count differs')
        check_steps(record['steps'], 1, count)
        require(all(r['mse'] > 0 and r['rank'] >= 0 and
                    r['loss'] == r['mse'] + (r['rank'] if arm == 'candidate' else 0.)
                    for r in record['steps']), 'matched objective arithmetic differs')
        if phase == 'mechanics':
            require(seed == SEEDS[0] and record['checkpoint'] is None and record['bundle'] is None and
                    record['training_state_discarded'] is True and record['replay_exact'] is True and
                    record['independent_first8_exact'] is True and all(diagnostic(a) == diagnostic(b)
                        for a, b in zip(record['steps'][8:], record['resumed_steps'], strict=True)), 'mechanics replay/discard differs')
            check_steps(record['resumed_steps'], 9, 9)
        else:
            require(record['training_state_discarded'] is False and record['resumed_steps'] == [] and
                    isinstance(record['checkpoint'], dict) and isinstance(record['bundle'], dict) and
                    record['launch']['selected_mechanics'] == launch['selected_mechanics'] and
                    record['fresh_first17_exact'] is (seed == SEEDS[0]), 'fresh TRAIN061/069 prerequisites differ')
    for key in ('checkpoint', 'bundle'):
        if record[key] is not None:
            file_fact(record[key])
            require(record['input_guards'].get(record[key]['path']) == record[key]['sha256'], 'serialized endpoint FILE binding differs')


def admit_terminal(context, unit, phase, arm, seed):
    """Use original authenticated uncached reader and ALL terminal predicates."""
    check_unit(unit)
    nearest, legacy, guards = context['nearest'], context['legacy'], context['guards']
    nearest.authenticate_startup_reader(context, context['fitter'])
    admission = legacy['admission']
    record = nearest.read_json(unit['receipt'], guards, admission=admission)
    check_terminal_record(record, context['launch'], phase, arm, seed)
    prior = legacy['selected']['source_cpu']['invocation']
    require(record['code'] == context['code'] and record['source'] == context['source'] and
            record['execution_sha256'] == context['args'].execution_sha256 and
            record['numerical_flags'] == legacy['selected']['source_cpu']['numerical_flags'] and
            nearest.read_json(record['authority'], guards, admission=admission) == record['launch'] and
            all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items()) and
            all(record['invocation'][k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
            record['invocation']['argv'] == cli(context['root'], record['authority']['path'], record['authority']['sha256'],
                context['args'].execution_sha256, phase, arm, seed, Path(unit['receipt']['path']).parent),
            'actual terminal source/CLI/full admission guards differ')
    final = context['fitter'].original_terminal_reader(context['fit_context'])(
        admission, record, unit, policy(phase)['seconds'], guards)
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['old'].zero_events(value)
    require(unit['invocation_id'] not in legacy['invocations'], 'reused original unit invocation')
    legacy['invocations'].add(unit['invocation_id'])
    context['terminals'][f'{phase}:{seed}:{arm}'] = record
    context['terminal_cgroups'][f'{phase}:{seed}:{arm}'] = final
    for p, h in record['input_guards'].items():
        admission.bound_file(guards, p, h)
    return record


def exit_rehash(context):
    require_no_training(context)
    with timed(context, 'source_exit_rehash'):
        helper_guard(context)
        api = context['nearest'].native_source_api(context)
        api.audit_origins(context['legacy'], require_exact=context['args'].phase != 'cpu')
        api.exit_rehash(context['fit_context'])
    with timed(context, 'own_exit_rehash'):
        for p, h in context['guards'].items():
            bound_file({}, p, h)
        require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'],
                'own exact2 exit closure differs')
        require(closure(NEAREST['root'], NEAREST['execution_sha256'], NEAREST['code'], {}) == NEAREST['code'],
                'external exact3 exit closure differs')
    with timed(context, 'post_exit_api_authentication'):
        api = context['nearest'].native_source_api(context)
    with timed(context, 'post_exit_origin_audit'):
        api.audit_origins(context['legacy'], require_exact=context['args'].phase != 'cpu')


def run(args):
    started = UNIT_STARTED
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
            os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'explicit hidden CPU/original CUDA numerics required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original systemd invocation required')
    require(sys.argv == cli(Path(__file__).absolute().parent, args.authority, args.authority_sha256,
                           args.execution_sha256, args.phase, args.arm, args.seed, args.output), 'fixed canonical CLI required')
    context = authority(args)
    context['phase_seconds']['authority'] = time.perf_counter() - started
    legacy, source = context['legacy'], context['legacy']['source_driver']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before, unit)
    prior = legacy['selected']['source_cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and legacy['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'qualified original interpreter required')
    import torch
    require(not torch.cuda.is_initialized(), 'admission must precede CUDA')
    flags = legacy['selected']['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original flags differ')
    torch.random.default_generator.manual_seed(args.seed)
    args.output.mkdir()
    helper_guard(context)
    prepare_native(context)
    context['fit_context']['unit_started'] = started
    result = cpu_witnesses(context) if args.phase == 'cpu' else gpu_run(context)
    require(source.numerical_flags() == flags, 'constructor/forward/reload numerical flags changed')
    with timed(context, 'post_run_api_authentication'):
        api = context['nearest'].native_source_api(context)
    with timed(context, 'post_run_origin_audit'):
        api.audit_origins(legacy, require_exact=args.phase != 'cpu')
    with timed(context, 'origin_guard_promotion'):
        for p, h in legacy['origins']['files'].items():
            bound_file(context['guards'], p, h)
    exit_rehash(context)
    after = source.cgroup_memory()
    legacy['selected']['genuine']['reference'].admit_cgroup(after, unit)
    for value in (before, after):
        context['old'].zero_events(value)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and
            0 < rss <= 8 * 1024**2 and (args.phase != 'cpu' or not torch.cuda.is_initialized()) and
            (args.phase == 'cpu' or torch.cuda.max_memory_allocated() < 10_000_000_000), 'whole-unit resource cap differs')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed, 'pass': True,
        'quality_read': False, 'exit_rehash_pass': True, 'sequential_model_ownership': True,
        'optimizer_members': 1, 'trainable_scalars': 20480, 'frozen_vision_members': 448,
        'source': context['source'], 'launch': context['launch'], 'code': context['code'],
        'execution_sha256': args.execution_sha256,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
        'authority_sha256': args.authority_sha256, 'output': str(args.output),
        'resource_policy': policy(args.phase), 'numerical_flags': flags, 'wall_seconds': wall,
        'process_peak_rss_kib': rss, 'cgroup_before': before, 'cgroup_after': after,
        'origins': legacy['origins'], 'input_guards': context['guards'], 'phase_seconds': context['phase_seconds'],
        'terminal_cgroups': context['terminal_cgroups'], 'both_locks_held_in_parent_authority': True,
        'terminal_exit_and_both_locks_require_parent_receipt': True,
        'exact_four_native_membership': args.phase != 'cpu',
        'costs_shared_separate': ['export', 'CPU qualification', 'shared_source_preparation', 'shared_bundle_preparation'],
        'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                      'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                      'invocation_id': os.environ['INVOCATION_ID'],
                      'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}, **result}
    check_terminal_record(receipt, context['launch'], args.phase, args.arm, args.seed)
    write_json(context, args.output / 'receipt.json', receipt)
    require(time.perf_counter() - started < policy(args.phase)['seconds'], 'receipt included deadline exceeded')
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
        raise SystemExit('Compact-ranking rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
