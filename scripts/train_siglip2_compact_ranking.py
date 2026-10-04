#!/usr/bin/env python3
"""Fixed cache-only concat ranking trial; native qualification is UNRUN.

Own execution.json is exactly trainer/test. Frozen external nearest/fitter
closures retain original admission and source predicates. No quality reads.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
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
SCHEMA = 'siglip2-compact-live-top1-v1'
AUTHORITY_SCHEMA = 'siglip2-compact-live-top1-launch-v1'
INFERENCE_SCHEMA = 'siglip2-compact-live-top1-inference-v1'
BUNDLE_SCHEMA = 'siglip2-compact-live-top1-bundle-v1'
FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}
ARMS = ('control', 'candidate')
SEEDS = (179061, 179069)
VIEWS = ('canonical', 'augmented')
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
NEAREST = {'root': '/home/riomus/runs/sfora-so400-nearest-ranking-train-source-v5', 'execution_sha256': '0723deae5c550776b0a0493e296ce789cec76718cb7801f612b4fddc6201c311', 'code': {'nearest_ranking_readout.py': '862d5db5a1831603d4fc3c1b645fa998c7ca66ff67c8141b20d7df74e92d9738', 'test_siglip2_nearest_ranking.py': 'a2d0a6fff65c371c4dd50531766677db5464ee35918887b3d55935dbdf489eec', 'train_siglip2_nearest_ranking.py': '4803bca125f54fce9e2f59a1ae3f31dff51aa860cc7c5059ab13e28ff795a72c'}}
FITTER = {'code': {'fit_siglip2_prototype_residual.py': '95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b', 'prototype_residual_readout.py': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68', 'test_siglip2_prototype_residual.py': 'c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4'}, 'execution_sha256': 'a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe', 'root': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2'}
ACCEPTED = {'arm': 'concat', 'checkpoint': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt', 'sha256': 'b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf'}, 'launch': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json', 'sha256': '109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630'}, 'terminal': {'both_locks_held': True, 'invocation_id': '94a84de4194f42f0842cee5b5c8f932a', 'log': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log', 'sha256': '93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197'}, 'native_peak_rss_kib': 2854356, 'receipt': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json', 'sha256': 'b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c'}, 'service_seconds': 234.821, 'unit': 'sfora-so400-signed-concat-fit-concat-v1'}, 'terminal_state_sha256': 'a118fd98cce0b8fafa51c897be70b2b6e2ec93ebb226b2382dd264721776b644'}
READOUT = {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/prototype_residual_readout.py', 'sha256': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68'}
ANCHOR_ENDPOINT = {'A_sha256': '461b23b5980f7aa917bc897760c20e3e4d040e15cbf68c58a55f1890a28ba834', 'role': 'discarded canonical-anchor displacement witness only',
 'source': {'root': '/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1',
            'execution_sha256': 'bf3efd0040a6cccefde9414839a8cf4943e6a2a52c49f701b865a771def7e272',
            'code': {'test_siglip2_compact_ranking.py': '5834df438ac32dd98244d7b24bb19e94e173169b19ea6770ba7db2eff0c0f72b',
                     'train_siglip2_compact_ranking.py': '74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679'}},
 'authority': {'path': '/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/authority-train-candidate-179061-v1.json',
               'sha256': '4e7cb6c9b09ec611d9d1f1aa1b8d00c62f0e5169d5ca5ad1f33b4ffde055bb8c'},
 'terminal': {'receipt': {'path': '/home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/receipt.json',
                          'sha256': '49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3'},
              'log': {'path': '/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train-candidate-179061-v1.log',
                      'sha256': 'c03eaa8b4d5372a31fac888a0a7446b67e1dd0d42fb36056f7b6931960998657'},
              'unit': 'sfora-so400-smooth-ap-train-candidate-179061-v1',
              'invocation_id': '1dbeda11e66747fa8ea5a1e1fa1f7bb6',
              'service_seconds': 344.836,
              'native_peak_rss_kib': 4619544,
              'both_locks_held': True},
 'checkpoint': {'path': '/home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt',
                'sha256': '4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3'},
 'bundle': {'path': '/home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/bundle/bundle.json',
            'sha256': 'a3d6460966c228d4f50cd7c16e643ba3b5496ebd3f8ee60741e4b2ddb22ea883'},
 'terminal_state_sha256': '5dcc6eadca4b365796551c9e3829c043614794be7e2786307110d295e8e1a3e6',
 'inference_state_sha256': 'ad7e819aa56cf163b66f039b417d397ca1d0c37518f60f74e5984607815feb15'}
ACTIVE_OBJECTIVE_SOURCE = {'code': {'test_siglip2_compact_ranking.py': '6b2e727d4aacc78e50c616024b37c34584d9b3193b8b8333ca148da57adf2a9b', 'train_siglip2_compact_ranking.py': 'ddbbf0bc02eb62c3bb87fc29768ecbd5ec5e9d885fb53215244029413df00bad'}, 'execution_sha256': '996ae38783d44c0cb01ef3bb8d5ba1817545d5296a0612da07b6c852b69bbf15', 'root': '/home/riomus/runs/sfora-so400-fullfeature-residual-train-source-v1'}
ADAM = {'lr': 1e-4, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05,
        'amsgrad': False, 'maximize': False, 'foreach': False, 'capturable': False,
        'differentiable': False, 'fused': False}
RECIPE = {'adamw': {'amsgrad': False,
           'betas': [0.9, 0.999],
           'capturable': False,
           'differentiable': False,
           'eps': 1e-08,
           'foreach': False,
           'fused': False,
           'lr': 0.0001,
           'maximize': False,
           'weight_decay': 0.05},
 'batch': 64,
 'classes': 1008,
 'clip': 1.0,
 'core': 'cache/target preparation + both-arm all-positive scoring + both-view forward/backward + every '
         'both-arm gallery forward/backward + both-arm fullfeature centering/transfers + candidate hinge '
         'selection/transfers/backward + optimizer',
 'frozen': 'complete encoder448/config/buffers/processor/head/classifier/means/muTRAIN',
 'gallery': 'both arms reconstruct connected gallery every micro16 at same pre-update A/C; both-role '
            'backward then release each graph; one optimizer step after eight micros; no serialized gallery',
 'hinge': 'candidate coefficient1 relu(.05+s_nearest_wrong-s_nearest_positive) sum / (.05*2*K); control '
          'connected zero; K counts full scheduled B64 valid occurrences bothviews/micro16',
 'hinge_coefficient': 1.0,
 'hinge_margin': 0.05,
 'initial_scaler': 128.0,
 'microbatch': 16,
 'mining': 'all6355 canonical rows; both arms current connected canonical readout at same pre-update A/C; '
           'exclude same original image; all same-identity positives; canonical ordinal traversal',
 'ranking': 'both arms backward coefficient1; all positives; SmoothAP sum / (2*K)',
 'readout': 'original CPU-renormalized genuine features; FP32 all; autocast disabled',
 'regression': 'both arms P[label]; both original views coordinate sum / (128*e0)',
 'residual': 'both arms original concat helper once + linear(actual normalized x - canonical TRAIN6355 '
             'mu,C); fresh accepted A0/zeroC; mean FP32 canonical actual CPU domain only',
 'rows': 6355,
 'schedule': 'original first128 B64 per seed; warm-authenticated; masks unused',
 'seeds': [179061, 179069],
 'selection': 'detached scores for indices only; filter invalid anchors before indexing; exclude '
              'original-row self; highest score then ascending original row; gather same connected '
              'scorematrix; singleton gallery negatives connected',
 'singletons': 12,
 'teacher': 'accepted canonical T; member-inclusive P; normalize(T); both-view e0',
 'temperature': 0.01,
 'trainable_names': {'candidate': ['A', 'C'], 'control': ['A', 'C']},
 'trainable_scalars': {'candidate': 167936, 'control': 167936},
 'trainable_shapes': {'candidate': [[128, 160], [128, 1152]], 'control': [[128, 160], [128, 1152]]},
 'updates': 128,
 'views': ['canonical', 'augmented']}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'nearest', 'fitter', 'accepted',
               'readout', 'recipe', 'resource_policy', 'both_locks_held', 'selected_cpu',
               'selected_mechanics', 'native_authority'}
STATIC_KEYS = ('provenance', 'config', 'buffers', 'processor', 'head', 'classifier', 'means',
               'partition', 'original_rows', 'target', 'schedules', 'views', 'teachers', 'mu_train', 'mu_train_provenance')
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'A', 'C', *STATIC_KEYS, 'optimizer', 'scaler',
                'counter', 'cpu_rng', 'cuda_rng', 'numerical_flags'}
INFERENCE_KEYS = {'schema', 'source', 'arm', 'config', 'buffers', 'processor', 'head', 'A', 'C', 'means',
                  'mu_train', 'mu_train_provenance',
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


def batch_bound_files(guards, items):
    """Fresh per occurrence; publish on the owner only after complete success."""
    items = list(items)
    for path, expected in items:
        file_fact({'path': str(path), 'sha256': expected})
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bound_file, {}, path, expected) for path, expected in items]
        paths = [future.result() for future in futures]
    staged = dict(guards)
    for path, (_, expected) in zip(paths, items):
        require(staged.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    guards.update(staged)
    return paths


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
    return {'seconds': 500 if phase == 'cpu' else 600, 'host_bytes': 8 * 1024**3,
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
            type(launch['seed']) is int and launch['seed'] == args.seed and args.seed in SEEDS and launch['nearest'] == NEAREST and
            launch['fitter'] == FITTER and launch['accepted'] == ACCEPTED and launch['readout'] == READOUT and
            launch['recipe'] == RECIPE and json_sha256(launch['recipe']) == json_sha256(RECIPE) and launch['resource_policy'] == policy(args.phase) and
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
                all(a[k] == b[k] for k in ('initial_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256')) and
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
        initial['C'] = torch.zeros((128, 1152), dtype=torch.float32)
        initial['mu_train'] = views['canonical'].mean(dim=0).detach().contiguous()
        initial['mu_train_provenance'] = {
            'domain': 'actual CPU-renormalized FP32 features', 'rows': 6355,
            'view': 'canonical', 'reduction': 'torch FP32 mean(dim=0) canonical ordinal order',
            'canonical_features_sha256': fingerprint(context, views['canonical']),
            'original_rows_sha256': fingerprint(context, initial['original_rows']),
            'target_sha256': fingerprint(context, initial['target']),
            'partition_sha256': fingerprint(context, initial['partition']),
            'accepted_checkpoint': clone(context, ACCEPTED['checkpoint']),
            'canonical_cache': clone(context, legacy['selected']['source']['caches']['canonical'])}
        check_mu_train_provenance(initial['mu_train_provenance'])
        context['initial'] = initial
        context['initial_static_sha256'] = fingerprint(context, {k: initial[k] for k in STATIC_KEYS})
        context['initial_A_sha256'] = fingerprint(context, initial['A'])
        context['initial_C_sha256'] = fingerprint(context, initial['C'])
        context['mu_train_sha256'] = fingerprint(context, initial['mu_train'])
        context['mu_train_provenance_sha256'] = fingerprint(context, initial['mu_train_provenance'])
    # Fresh per audit: imported_origins hashes; the original reader registers those bytes.
    context['nearest'].native_source_api(context).audit_origins(legacy, admission=legacy['original'].FlatAdmission())


def require_no_training(context):
    reference = context.get('live_training')
    require(reference is None or reference() is None, 'previous training A still alive')
    residual = context.get('live_residual')
    require(residual is None or residual() is None, 'previous residual/gallery graph still alive')
    context['nearest'].require_no_model(context)


def parameter_roles(arm):
    require(type(arm) is str and arm in ARMS, 'fixed residual arm required')
    return ['A', 'C'], [[128, 160], [128, 1152]], 167936


def check_mu_train_provenance(value):
    require(isinstance(value, dict) and value.keys() == {
        'domain', 'rows', 'view', 'reduction', 'canonical_features_sha256',
        'original_rows_sha256', 'target_sha256', 'partition_sha256',
        'accepted_checkpoint', 'canonical_cache'} and
        value['domain'] == 'actual CPU-renormalized FP32 features' and
        type(value['rows']) is int and value['rows'] == 6355 and value['view'] == 'canonical' and
        value['reduction'] == 'torch FP32 mean(dim=0) canonical ordinal order' and
        value['accepted_checkpoint'] == ACCEPTED['checkpoint'], 'canonical TRAIN-only mean provenance differs')
    for key in ('canonical_features_sha256', 'original_rows_sha256', 'target_sha256', 'partition_sha256'):
        require(isinstance(value[key], str) and re.fullmatch('[0-9a-f]{64}', value[key]),
                'typed mean provenance digest required')
    cache = value['canonical_cache']
    require(isinstance(cache, dict) and cache.get('normalized') is True and
            cache.get('raw_pooled_cache') is False and cache.get('shape') == [6355, 1152] and
            cache.get('dtype') == 'float32', 'canonical mean cache provenance differs')


def own_residual(context, state, *, admit=False, advanced=False):
    """Authorize C updates and catch current .data/role/mean substitutions."""
    import torch
    C, mu = state['C'], state['mu_train']
    primitive = context['legacy']['quadratic']
    primitive._check_tensor(C, (128, 1152), state['A'].device)
    primitive._check_tensor(mu, (1152,), state['A'].device, frozen=True)
    require(C.is_leaf and C.grad_fn is None and C.requires_grad is True and
            torch.isfinite(C).all().item() and torch.isfinite(mu).all().item() and
            fingerprint(context, mu) == context['mu_train_sha256'] and
            fingerprint(context, state['mu_train_provenance']) == context['mu_train_provenance_sha256'],
            'current residual roles/finite TRAIN mean bytes differ')
    current = fingerprint(context, C)
    owners = context.setdefault('C_owners', {})
    prior = owners.get(id(state))
    if admit:
        require(prior is None or (prior[0] is state and prior[1] is C and prior[2] == 0),
                'C re-admission requires independent fresh load')
    elif advanced:
        require(prior is not None and prior[0] is state and prior[1] is C and
                state['counter'] == prior[2] + 1 and
                current != prior[3],
                'C refresh requires one genuine role-aware update')
    else:
        require(prior is not None and prior[0] is state and prior[1] is C and
                state['counter'] == prior[2] and current == prior[3], 'authorized current C bytes/counter differ')
        return current
    owners[id(state)] = (state, C, state['counter'], current)
    return current


def fullfeature_raw_features(features, head, A, means, C, mu_train, arm, primitive, readout):
    """Public FP32 readout over the actual normalized input, with one concat call."""
    import torch
    from torch.nn import functional as F
    parameter_roles(arm)
    primitive._check_tensor(C, (128, 1152), features.device)
    primitive._check_tensor(mu_train, (1152,), features.device, frozen=True)
    require(torch.isfinite(C).all().item() and torch.isfinite(mu_train).all().item(),
            'finite residual/mean required')
    with torch.autocast(features.device.type, enabled=False):
        raw = readout.raw_features(features, head, A, means, 'concat', primitive)
        raw = raw + F.linear(features.detach().float() - mu_train, C)
        require(torch.isfinite(raw).all().item(), 'finite fullfeature raw required')
    return raw


def residual_facts(context, state):
    import torch
    return {'initial_C_sha256': context['initial_C_sha256'],
            'current_C_sha256': fingerprint(context, state['C'].detach()),
            'mu_train_sha256': fingerprint(context, state['mu_train']),
            'mu_train_provenance_sha256': fingerprint(context, state['mu_train_provenance']),
            'C_exact_zero': torch.count_nonzero(state['C']).item() == 0,
            'C_trainable': state['C'].requires_grad}


def own_A(context, state, *, admit=False, advanced=False):
    """Current bytes, never a version/hash cache; sole owner authorizes updates."""
    own_residual(context, state, admit=admit, advanced=advanced)
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
    state = {k: clone(context, initial[k], device if k in ('teachers', 'target', 'means', 'classifier', 'mu_train') else 'cpu')
             for k in STATIC_KEYS}
    state.update(arm=arm, seed=seed, device=device, counter=0)
    head = context['legacy']['selected']['cached'].head_from('control', tensors=state['head'])
    state['head_object'] = head.requires_grad_(False).to(device).train()
    A = torch.nn.Parameter(clone(context, initial['A'], device))
    state['A'] = A
    context['live_training'] = weakref.ref(A)
    C = torch.nn.Parameter(clone(context, initial['C'], device), requires_grad=True)
    state['C'] = C
    context['live_residual'] = weakref.ref(C)
    members = [A, C]
    optimizer = torch.optim.AdamW(members, **ADAM)
    defaults = dict(ADAM)
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and len(optimizer.param_groups) == 1 and
            len(optimizer.param_groups[0]['params']) == len(members) and
            all(a is b for a, b in zip(optimizer.param_groups[0]['params'], members, strict=True)) and
            {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'} == defaults,
            'fresh role-aware ordered AdamW defaults/groups differ')
    state['optimizer_object'] = optimizer
    state['scaler_object'] = torch.amp.GradScaler(device, init_scale=128., enabled=device == 'cuda')
    state['target_list'] = state['target'].tolist()
    state['row_list'] = state['original_rows'].tolist()
    state['count_list'] = state['teachers']['counts'].tolist()
    state['ranking_bank'] = ranking_bank(state['target_list'], state['row_list'])
    check_ranking_bank(state['ranking_bank'])
    own_A(context, state, admit=True)
    return state


def static_tree(state):
    return {k: dict(state['head_object'].state_dict()) if k == 'head' else state[k] for k in STATIC_KEYS}


def identity(context, state):
    import torch
    optimizer = state['optimizer_object']
    names, shapes, _ = parameter_roles(state['arm'])
    return {'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'],
            'ranking_bank_sha256': state['ranking_bank']['sha256'],
            'seed': state['seed'], 'device': state['device'], 'parameter_names': names,
            'parameter_shapes': shapes, 'numerical_flags': context['flags'],
            'static_sha256': fingerprint(context, static_tree(state)),
            'initial_A_sha256': context['initial_A_sha256'],
            'initial_C_sha256': context['initial_C_sha256'],
            'mu_train_sha256': context['mu_train_sha256'],
            'mu_train_provenance_sha256': context['mu_train_provenance_sha256'], 'optimizer_defaults': optimizer.defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups],
            'initial_scaler': state['scaler_object'].state_dict(),
            'initial_cpu_rng_sha256': fingerprint(context, torch.random.get_rng_state()),
            'initial_cuda_rng_sha256': fingerprint(context, torch.cuda.get_rng_state_all()) if state['device'] == 'cuda' else None}


def payload(context, state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'source': context['source'], **static_tree(state),
            'A': state['A'].detach(), 'C': state['C'].detach(), 'optimizer': state['optimizer_object'].state_dict(),
            'scaler': state['scaler_object'].state_dict(), 'counter': state['counter'],
            'cpu_rng': torch.random.get_rng_state().clone(),
            'cuda_rng': [v.clone() for v in torch.cuda.get_rng_state_all()] if state['device'] == 'cuda' else [],
            'numerical_flags': context['legacy']['source_driver'].numerical_flags()}


def check_optimizer(saved, ident, step):
    import torch
    names, shapes, _ = parameter_roles(ident['arm'])
    opt = saved['optimizer']
    require(ident['parameter_names'] == names and ident['parameter_shapes'] == shapes and
            opt.keys() == {'state', 'param_groups'} and len(opt['param_groups']) == 1 and
            opt['param_groups'][0]['params'] == list(range(len(names))) and
            {k: v for k, v in opt['param_groups'][0].items() if k != 'params'} == ident['optimizer_groups'][0] and
            opt['state'].keys() == (set(range(len(names))) if step else set()),
            'ordered role-aware optimizer ownership differs')
    if step:
        for index, shape in enumerate(shapes):
            member = opt['state'][index]
            require(member.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and member['step'].shape == () and
                    member['step'].dtype == torch.float32 and member['step'].device.type == 'cpu' and
                    float(member['step']) == step, 'AdamW exact CPU step differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = member[key]
                require(value.shape == tuple(shape) and value.dtype == torch.float32 and not value.requires_grad and
                        value.grad_fn is None and torch.isfinite(value).all().item(), 'finite FP32 named moment differs')
    require(saved['scaler'] == (dict(ident['initial_scaler'], _growth_tracker=step) if ident['device'] == 'cuda'
                                else ident['initial_scaler']), 'scaler128/counter differs')


def check_payload(context, saved, ident, step):
    import torch
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            saved['source'] == context['source'] == ident['source'] and ident['method'] == method(context['launch']) and
            (ident['parameter_names'], ident['parameter_shapes']) == parameter_roles(ident['arm'])[:2] and
            ident['arm'] in ARMS and ident['seed'] in SEEDS and ident['device'] in ('cpu', 'cuda') and
            type(saved['counter']) is int and saved['counter'] == step and 0 <= step <= 128 and
            saved['numerical_flags'] == ident['numerical_flags'] == context['flags'], 'complete payload identity differs')
    require(fingerprint(context, {k: saved[k] for k in STATIC_KEYS}) == ident['static_sha256'] ==
            context['initial_static_sha256'], 'frozen complete encoder/head/teachers/cache/schedules current bytes differ')
    require(ranking_bank(saved['target'].tolist(), saved['original_rows'].tolist())['sha256'] ==
            ident['ranking_bank_sha256'], 'complete payload ranking bank differs')
    context['old'].check_encoder(saved['provenance']['encoder'])
    require(saved['provenance'] == context['initial']['provenance'], 'original provenance differs')
    primitive = context['legacy']['quadratic']
    primitive._check_tensor(saved['A'], (128, 160), saved['A'].device, frozen=True)
    require(torch.isfinite(saved['A']).all().item() and
            ((fingerprint(context, saved['A']) == ident['initial_A_sha256']) if step == 0 else
             (fingerprint(context, saved['A']) != ident['initial_A_sha256'])), 'initial/updated A substitution differs')
    check_mu_train_provenance(saved['mu_train_provenance'])
    primitive._check_tensor(saved['mu_train'], (1152,), saved['A'].device, frozen=True)
    primitive._check_tensor(saved['C'], (128, 1152), saved['A'].device, frozen=True)
    require(ident['initial_C_sha256'] == context['initial_C_sha256'] and
            fingerprint(context, saved['mu_train']) == ident['mu_train_sha256'] == context['mu_train_sha256'] and
            fingerprint(context, saved['mu_train_provenance']) == ident['mu_train_provenance_sha256'] ==
            context['mu_train_provenance_sha256'] and torch.isfinite(saved['mu_train']).all().item() and
            torch.isfinite(saved['C']).all().item() and
            (torch.count_nonzero(saved['C']).item() == 0 if step == 0 else
             torch.count_nonzero(saved['C']).item() > 0), 'frozen TRAIN mean/initial or updated C differs')
    if step == 0:
        require(fingerprint(context, saved['C']) == ident['initial_C_sha256'], 'exactzero initial C differs')
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
    require(state['ranking_bank'] == ranking_bank(state['target'].tolist(), state['original_rows'].tolist()) and
            state['target_list'] == state['target'].tolist() and state['row_list'] == state['original_rows'].tolist(),
            'live complete ranking bank differs')
    A, head, optimizer = state['A'], state['head_object'], state['optimizer_object']
    helper.check_weight(A, A.device, 'concat', context['legacy']['quadratic'])
    context['legacy']['quadratic']._check_base(head, A.device)
    require(A.device.type == state['device'] and A.grad is None and
            all(p.grad is None and not p.requires_grad and p.device == A.device for p in head.parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in head.modules()) and len(optimizer.param_groups) == 1 and
            len(optimizer.param_groups[0]['params']) == len(ident['parameter_names']) and
            all(p is state[n] for p, n in zip(optimizer.param_groups[0]['params'], ident['parameter_names'], strict=True)) and
            state['C'].grad is None and state['C'].requires_grad is True and
            optimizer.defaults == ident['optimizer_defaults'], 'sole-A roles/hooks/frozen gradients/defaults differ')
    if state['counter']:
        require(all(optimizer.state[state[n]][k].device == A.device
                    for n in ident['parameter_names'] for k in ('exp_avg', 'exp_avg_sq')),
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
    context.get('C_owners', {}).pop(id(state), None)
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
    return 128, 2 * full_valid if full_valid else None


def raw_features(context, state, features):
    return fullfeature_raw_features(features, state['head_object'], state['A'], state['means'],
        state['C'], state['mu_train'], state['arm'], context['legacy']['quadratic'], helper_guard(context))


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ranking_bank(targets, original_rows):
    require(isinstance(targets, list) and isinstance(original_rows, list) and
            0 < len(targets) == len(original_rows) <= 6355 and
            all(type(t) is int and t >= 0 for t in targets) and
            all(type(r) is int and r >= 0 for r in original_rows), 'typed complete ranking bank required')
    bank = {'target': list(targets), 'original_rows': list(original_rows)}
    return {**bank, 'sha256': json_sha256(bank)}


def check_ranking_bank(bank):
    ranking_membership(bank, [])
    counts = [0] * 1008
    require(len(bank['target']) == 6355 and len(set(bank['original_rows'])) == 6355 and
            all(t < 1008 for t in bank['target']), 'canonical TRAIN6355 ranking bank required')
    for t in bank['target']:
        counts[t] += 1
    require(all(counts) and counts.count(1) == 12, 'ranking bank classes/singletons differ')


def ranking_membership(bank, anchors):
    require(isinstance(bank, dict) and bank.keys() == {'target', 'original_rows', 'sha256'} and
            bank == ranking_bank(bank['target'], bank['original_rows']) and isinstance(anchors, list) and
            len(anchors) <= 64 and all(type(a) is int and 0 <= a < len(bank['target']) for a in anchors),
            'bound ranking membership required')
    positives, counts, digests = [], [], []
    targets, rows = bank['target'], bank['original_rows']
    for anchor in anchors:
        eligible = [j for j, row in enumerate(rows) if row != rows[anchor]]
        positives.append([j for j in eligible if targets[j] == targets[anchor]])
        counts.append(len(eligible))
        digests.append(json_sha256(eligible))
    return {'positive': positives, 'eligible_counts': counts, 'eligible_sha256': digests,
            'valid': sum(bool(p) for p in positives)}


def smooth_ap_terms(scores, positive, eligible):
    """One anchor, all positives by frozen bank columns; no gallery square."""
    import torch
    comparisons = ((scores[eligible][None, :] - scores[positive][:, None]) / .01).sigmoid()
    columns = torch.tensor(eligible, device=scores.device)
    positives = torch.tensor(positive, device=scores.device)
    other = columns[None, :] != positives[:, None]
    positive_set = set(positive)
    positive_columns = torch.tensor([j in positive_set for j in eligible], device=scores.device)
    rp = 1 + (comparisons * other * positive_columns[None, :]).sum(1)
    rt = 1 + (comparisons * other).sum(1)
    return 1 - rp / rt


def ranking_gallery(context, state):
    """One ephemeral canonical graph per micro; update charges forward/backward."""
    import torch
    from torch.nn import functional as F
    require(state['arm'] in ARMS, 'fixed connected gallery arm required')
    with torch.autocast(state['device'], enabled=False):
        raw = raw_features(context, state, state['views']['canonical'].to(state['device']))
        require(raw.shape == state['teachers']['T'].shape and raw.dtype == torch.float32 and
                raw.requires_grad and raw.grad_fn is not None and torch.isfinite(raw).all().item() and
                (raw.norm(dim=1) > 0).all().item(), 'complete connected finite current canonical gallery required')
        return F.normalize(raw, dim=1)


def authenticate_active_objective(context):
    """Authenticate the archived connected-gallery candidate objective source only.

    The closed image-anchor endpoint/displacement witnesses remain historical;
    they are not prerequisites, initializers or witnesses of this method.
    """
    source, guards = ACTIVE_OBJECTIVE_SOURCE, context['guards']
    root = Path(source['root'])
    code = closure(root, source['execution_sha256'], FILES, guards)
    require(code == source['code'], 'original active objective exact2 differs')
    return load_authenticated('_compact_active_original', root / 'train_siglip2_compact_ranking.py',
                              code['train_siglip2_compact_ranking.py'], guards)


def live_top1_indices(scores, bank, anchors):
    """Detach only selection; return valid offsets before any positive indexing."""
    membership = ranking_membership(bank, anchors)
    values = scores.detach().cpu().tolist()
    targets, rows = bank['target'], bank['original_rows']
    require(len(set(rows)) == len(rows) and len(values) == len(anchors) and
            all(len(row) == len(rows) and all(math.isfinite(v) for v in row) for row in values),
            'finite complete live scores/unique original rows required')
    selected = []
    for offset, (anchor, positive) in enumerate(zip(anchors, membership['positive'], strict=True)):
        if not positive:
            continue
        wrong = [j for j, target in enumerate(targets) if target != targets[anchor] and rows[j] != rows[anchor]]
        require(bool(wrong), 'valid anchor requires a wrong identity')
        key = lambda j: (-values[offset][j], rows[j])
        selected.append((offset, min(positive, key=key), min(wrong, key=key)))
    return selected


def live_top1_hinge(scores, bank, anchors, full_valid):
    _, denominator = loss_denominators(full_valid)
    selected = live_top1_indices(scores, bank, anchors)
    require(len(selected) <= full_valid, 'global valid occurrence denominator required')
    terms = [(.05 + scores[offset][wrong] - scores[offset][positive]).relu()
             for offset, positive, wrong in selected]
    hinge = sum(terms) / (.05 * denominator) if terms else scores.sum() * 0.
    return hinge, sum(int(term.detach().item() > 0) for term in terms)


def loss_terms(context, state, raw, anchors, full_valid):
    import torch
    from torch.nn import functional as F
    rows, rank_denominator = loss_denominators(full_valid)
    bank = state['ranking_bank']
    membership = ranking_membership(bank, anchors)
    with torch.autocast(raw.device.type, enabled=False):
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item() and
                (raw.norm(dim=1) > 0).all().item(), 'finite nonzero FP32 raw required')
        index = torch.tensor(anchors, device=raw.device)
        target = state['teachers']['P'][state['target'][index]]
        mse = (raw - target).square().sum() / (rows * state['teachers']['e0'])
        scores = F.normalize(raw, dim=1) @ ranking_gallery(context, state).T
        terms, active = [], 0
        for offset, (anchor, positive) in enumerate(zip(anchors, membership['positive'], strict=True)):
            if positive:
                require(rank_denominator is not None, 'global valid denominator required')
                eligible = [j for j, row in enumerate(bank['original_rows']) if row != bank['original_rows'][anchor]]
                per_positive = smooth_ap_terms(scores[offset], positive, eligible)
                term = per_positive.mean()
                require(torch.isfinite(per_positive).all().item(), 'nonfinite SmoothAP positive ranks')
                terms.append(term)
                active += int(term.detach().item() > 0)
        rank = sum(terms) / rank_denominator if terms else raw.sum() * 0.
        hinge, hinge_active = (live_top1_hinge(scores, bank, anchors, full_valid)
                               if state['arm'] == 'candidate' else (raw.sum() * 0., 0))
        require(torch.isfinite(mse).item() and torch.isfinite(rank).item() and
                torch.isfinite(hinge).item(), 'nonfinite objective')
    return mse, rank, hinge, {**membership, 'active': active, 'hinge_active': hinge_active}


def cached_witness(context, state):
    import torch
    batch = state['schedules'][str(state['seed'])][0].tolist()
    with torch.no_grad():
        return {view: context['old'].packed_outputs(context['legacy'], raw_features(
                context, state, state['views'][view][batch].to(state['device']))) for view in VIEWS}


def authenticate_anchor_endpoint(context):
    """Original candidate authority and complete state; retain only a discarded A."""
    from types import SimpleNamespace
    pin, guards = ANCHOR_ENDPOINT, context['guards']
    root = Path(pin['source']['root'])
    code = closure(root, pin['source']['execution_sha256'], FILES, guards)
    require(code == pin['source']['code'], 'discarded anchor original exact2 differs')
    original = load_authenticated('_compact_anchor_original', root / 'train_siglip2_compact_ranking.py',
                                  code['train_siglip2_compact_ranking.py'], guards)
    launch = read_json(pin['authority'], guards)
    # Reuse the admitted runtime, never reconstruct a second legacy/model context.
    historical = {**context, 'root': root, 'code': code, 'launch': launch,
                  'args': SimpleNamespace(execution_sha256=pin['source']['execution_sha256']),
                  'required_guards': {p: h for p, h in context['required_guards'].items()
                                      if not Path(p).is_relative_to(context['root'])},
                  'terminals': {}, 'terminal_cgroups': {}}
    record = original.admit_terminal(historical, pin['terminal'], 'train', 'candidate', SEEDS[0])
    require(record['authority'] == pin['authority'] and
            all(record[k] == pin[k] for k in ('checkpoint', 'bundle', 'terminal_state_sha256', 'inference_state_sha256')) and
            record['initial_A_sha256'] == context['initial_A_sha256'] and
            record['identity']['static_sha256'] == context['initial_static_sha256'],
            'discarded anchor endpoint/common initialization differs')
    directory = Path(pin['bundle']['path']).parent
    manifest, bundle_guards = original.admit_bundle(directory, pin['bundle']['sha256'])
    require(manifest['endpoint_state_sha256'] == pin['inference_state_sha256'],
            'discarded anchor inference digest differs')
    for path, digest in bundle_guards.items():
        require(guards.setdefault(path, digest) == digest, 'discarded anchor bundle guard conflict')
    import torch
    path = bound_file(guards, pin['checkpoint']['path'], pin['checkpoint']['sha256'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = context['legacy']['original'].CheckpointPages(stream)
        # JSON projects native AdamW betas tuples to lists. Bind the complete
        # receipt representation; the mandatory typed state pin below preserves
        # tuple/list and all other native types independently of this projection.
        require(json_sha256(disk['identity']) == json_sha256(record['identity']),
                'discarded anchor native identity/JSON receipt differs')
        original.check_payload(historical, disk, disk['identity'], 128)
        require(fingerprint(context, disk, consumed=pages.consume) == pin['terminal_state_sha256'],
                'discarded anchor complete typed state differs')
        A = clone(context, disk['A'])
        require(fingerprint(context, A) == pin['A_sha256'], 'discarded anchor A bytes differ')
    del disk, pages
    endpoint = torch.load(directory / 'endpoint.pt', map_location='cpu', weights_only=True, mmap=True)
    require(endpoint.keys() == INFERENCE_KEYS and endpoint['schema'] == original.INFERENCE_SCHEMA and
            fingerprint(context, endpoint) == pin['inference_state_sha256'] and
            fingerprint(context, {k: v for k, v in endpoint.items() if k != 'fixed_sha256'}) == endpoint['fixed_sha256'] and
            torch.equal(endpoint['A'], A) and endpoint['numerical_flags'] == context['flags'] and
            all(fingerprint(context, endpoint[k]) == fingerprint(context, context['initial'][k])
                for k in ('config', 'buffers', 'processor', 'head', 'means')),
            'discarded anchor strict inference/checkpoint binding differs')
    del endpoint, historical
    gc.collect()
    return A, original


def anchor_displacement_witness(context, state, batch):
    """Pinned old A is a diagnostic parameter only; never a training initializer."""
    import torch
    A = torch.nn.Parameter(context['anchor_A'].clone())
    probe = {**state, 'A': A, 'arm': 'candidate'}
    raw = raw_features(context, probe, state['views']['canonical'][batch])
    loss = (raw - state['teachers']['T'][batch]).square().sum() / (128 * state['teachers']['e0'])
    gradient = torch.autograd.grad(loss, A)[0]
    displacement = A.detach() - context['initial']['A']
    dot = (gradient.double() * displacement.double()).sum()
    require(torch.isfinite(loss).item() and loss.item() > 0 and
            torch.isfinite(gradient).all().item() and gradient.double().norm().item() > 0 and
            torch.allclose(dot, 2 * loss.detach().double(), rtol=1e-5, atol=1e-6),
            'discarded canonical anchor is disconnected from displacement')
    return {'endpoint': ANCHOR_ENDPOINT, 'training_state_discarded': True,
            'canonical_mse': float(loss.detach()), 'gradient_norm': float(gradient.double().norm()),
            'dot_gradient_displacement': float(dot), 'gradient_sha256': fingerprint(context, gradient),
            'A_sha256': fingerprint(context, A.detach()), 'canonical_connected_identity': True}


def native_live_top1_witness():
    """Fixed tiny selection algebra; genuine first-B64 gradients are separate."""
    import torch
    bank = ranking_bank([0, 0, 0, 1, 2], [90, 30, 10, 70, 20])
    scores = torch.tensor([[99., .8, .8, .9, .9], [.2, .2, .2, 99., .8],
                           [99., .8, .8, .9, .9]], requires_grad=True)
    anchors = [0, 3, 0]
    require(live_top1_indices(scores, bank, anchors) == [(0, 2, 4), (2, 2, 4)],
            'unsorted original-row ties/self/mixed-invalid selection differs')
    full, active = live_top1_hinge(scores, bank, anchors, 2)
    micro = sum(live_top1_hinge(scores[i:i+1], bank, [a], 2)[0] for i, a in enumerate(anchors))
    gradient = torch.autograd.grad(full, scores, retain_graph=True)[0]
    micro_gradient = torch.autograd.grad(micro, scores)[0]
    expected = torch.tensor([[0., 0., -5., 0., 5.], [0., 0., 0., 0., 0.], [0., 0., -5., 0., 5.]])
    require(active == 2 and torch.equal(full, micro) and torch.equal(gradient, expected) and
            torch.equal(gradient, micro_gradient), 'hinge occurrence/full-micro/singleton-gallery/nonnearest algebra differs')
    invalid = scores.detach()[1:2].clone().requires_grad_(True)
    zero, count = live_top1_hinge(invalid, bank, [3], 0)
    require(zero.item() == 0 and count == 0 and zero.requires_grad and
            torch.count_nonzero(torch.autograd.grad(zero, invalid)[0]).item() == 0,
            'all-invalid connected zero differs')


def cpu_gradients(context, state):
    """Fixed first genuine B64 only; unchanged full/micro tolerances."""
    import torch
    from torch.nn import functional as F
    with timed(context, 'cpu_loss_falsifier'):
        batch = state['schedules'][str(state['seed'])][0].tolist()
        bank = state['ranking_bank']
        membership = ranking_membership(bank, batch)
        K = membership['valid']
        A = state['A']
        require(state['counter'] == 0 and torch.equal(A.detach(), context['initial']['A']),
                'CPU falsifier must start at accepted A0')
        frozen_before = fingerprint(context, static_tree(state))
        # Compare each genuine firstB64 view to an independent accepted-A0 readout,
        # including the original packing roles. No searched rows or updated endpoint.
        accepted_A = torch.nn.Parameter(context['initial']['A'].clone())
        for view in VIEWS:
            actual = raw_features(context, state, state['views'][view][batch])
            expected = helper_guard(context).raw_features(state['views'][view][batch], state['head_object'],
                accepted_A, state['means'], 'concat', context['legacy']['quadratic'])
            require(torch.equal(actual, expected) and
                    fingerprint(context, context['old'].packed_outputs(context['legacy'], actual)) ==
                    fingerprint(context, context['old'].packed_outputs(context['legacy'], expected)),
                    'accepted firstB64 raw/unit/packed initialization differs')
            if view == 'canonical':
                require(torch.allclose(actual, state['teachers']['T'][batch], rtol=1e-5, atol=1e-6),
                        'accepted canonical firstB64 raw differs')
            del actual, expected
        del accepted_A
        initial_gallery = ranking_gallery(context, {**state, 'arm': 'candidate'})
        require(torch.allclose(initial_gallery, state['teachers']['V'], rtol=1e-5, atol=1e-6),
                'initial current/accepted gallery unit scores differ')
        for view in VIEWS:
            raw = raw_features(context, state, state['views'][view][batch])
            require(torch.allclose(F.normalize(raw, dim=1) @ initial_gallery.T,
                                   F.normalize(raw, dim=1) @ state['teachers']['V'].T,
                                   rtol=1e-5, atol=1e-6), 'initial both-view current/fixed gallery scores differ')
            del raw
        del initial_gallery
        def objective(micro, arm, historical=False, split=False, detached=False, nonzero=False,
                      hinge_detached=False):
            query = {**state, 'arm': 'candidate' if historical else arm,
                     **{n: torch.nn.Parameter(state[n].detach().clone()) for n in ('A', 'C')}}
            if nonzero:
                with torch.no_grad():
                    query['C'].copy_((torch.arange(128 * 1152).reshape(128, 1152) % 7 + 1).float() * 1e-4)
            gallery = {**query, **({n: torch.nn.Parameter(query[n].detach().clone())
                                   for n in ('A', 'C')} if split else {})}
            names = ('canonical_regression', 'augmented_regression', 'regression', 'ranking', 'hinge', 'total')
            gradients = {n: {k: torch.zeros_like(query[n]) for k in names} for n in ('A', 'C')}
            gallery_gradient = {n: {k: torch.zeros_like(query[n]) for k in ('ranking', 'hinge')}
                                for n in ('A', 'C')}
            scalars = {k: [] for k in ('canonical_mse', 'augmented_mse', 'rank', 'hinge')}
            active = hinge_active = 0
            raw_api = context['active_original'].raw_features if historical else raw_features
            loss_api = context['active_original'].loss_terms if historical else loss_terms
            if detached:
                loss_api = FunctionType(loss_terms.__code__, {**loss_terms.__globals__,
                    'ranking_gallery': lambda c, st: ranking_gallery(c, st).detach()})
            query_members, gallery_members = (tuple(st[n] for n in ('A', 'C')) for st in (query, gallery))
            for view in VIEWS:
                for offset in range(0, 64, micro):
                    anchors = batch[offset:offset + micro]
                    raw = raw_api(context, query, state['views'][view][anchors])
                    if hinge_detached:
                        # Deliberately detach ONLY the hinge gallery; SmoothAP remains connected.
                        def mutant(scores, bank, anchors, full_valid):
                            return live_top1_hinge(F.normalize(raw, dim=1) @ ranking_gallery(context, gallery).detach().T,
                                                   bank, anchors, full_valid)
                        loss_api = FunctionType(loss_terms.__code__, {**loss_terms.__globals__,
                                                                     'live_top1_hinge': mutant})
                    if historical:
                        mse, rank, facts = loss_api(context, gallery, raw, anchors, K)
                        hinge = raw.sum() * 0.
                        facts = {**facts, 'hinge_active': 0}
                    else:
                        mse, rank, hinge, facts = loss_api(context, gallery, raw, anchors, K)
                    scalars[view + '_mse'].append(mse.detach())
                    scalars['rank'].append(rank.detach()); scalars['hinge'].append(hinge.detach())
                    active += facts['active']; hinge_active += facts['hinge_active']
                    for label, term in (('regression', mse), ('ranking', rank), ('hinge', hinge),
                                        ('total', mse + rank + hinge)):
                        grads = torch.autograd.grad(term, query_members, retain_graph=label != 'total')
                        for n, grad in zip(('A', 'C'), grads, strict=True):
                            gradients[n][label].add_(grad)
                            if label == 'regression': gradients[n][view + '_regression'].add_(grad)
                        if split and label in ('ranking', 'hinge'):
                            parts = torch.autograd.grad(term, gallery_members, retain_graph=True, allow_unused=True)
                            for n, part in zip(('A', 'C'), parts, strict=True):
                                if part is not None: gallery_gradient[n][label].add_(part)
                            del parts, part
                        del grads, grad, term
                    del raw, mse, rank, hinge, facts
            losses = {k: float(sum(v)) for k, v in scalars.items()}
            losses['mse'] = losses['canonical_mse'] + losses['augmented_mse']
            losses['loss'] = losses['mse'] + losses['rank'] + losses['hinge']
            # All returned tensors are detached accumulated gradients; no score graph escapes.
            del query, gallery, query_members, gallery_members, loss_api, raw_api
            return losses, gradients, gallery_gradient, active, hinge_active

        def matched(left, right, exact=False):
            losses, gradients, _, active, hinge_active = left
            other_losses, other_gradients, _, other_active, other_hinge_active = right
            require((active, hinge_active) == (other_active, other_hinge_active) and
                    all((v == other_losses[k] if exact else math.isclose(v, other_losses[k], rel_tol=1e-5, abs_tol=1e-6))
                        for k, v in losses.items()) and
                    all((torch.equal(v, other_gradients[n][k]) if exact else
                         torch.allclose(v, other_gradients[n][k], rtol=1e-5, atol=1e-6))
                        for n, terms in gradients.items() for k, v in terms.items()),
                    'complete both-view loss/A/C full64 micro16 or archived objective differs')

        def gradient_record(gradient):
            require(torch.isfinite(gradient).all().item(), 'finite witness gradient required')
            return {'norm': float(gradient.double().norm()), 'sha256': fingerprint(context, gradient)}

        values, arms = {}, {}
        for arm in ARMS:
            full = objective(64, arm)
            micro = objective(16, arm)
            matched(full, micro)
            losses, gradients, _, active, hinge_active = full
            require(0 < active <= 2*K and 0 < losses['rank'] <= 1 and
                    all(math.isfinite(v) and v >= 0 for v in losses.values()) and
                    all(gradients[n][k].double().norm().item() > 0 for n in ('A','C')
                        for k in ('regression', 'ranking', 'total')) and
                    (hinge_active > 0 and losses['hinge'] > 0 and
                     all(gradients[n]['hinge'].double().norm().item() > 0 for n in ('A','C'))
                     if arm == 'candidate' else hinge_active == 0 and losses['hinge'] == 0),
                    'fixed firstB64 active finite independent SmoothAP/hinge/A/C required')
            values[arm] = full
            arms[arm] = {**losses, 'active': active, 'hinge_active': hinge_active,
                         'micro16_global_reduction_exact': True,
                         'gradients': {n: {k: gradient_record(v) for k,v in terms.items()}
                                       for n,terms in gradients.items()}}
            del micro, full, gradients
        historical = objective(64, 'control', historical=True)
        matched(historical, values['control'], exact=True)
        del historical
        for n in ('A', 'C'):
            control_grad, candidate_grad = (values[a][1][n] for a in ARMS)
            require(all(torch.equal(control_grad[k], candidate_grad[k]) for k in
                        ('canonical_regression','augmented_regression','regression','ranking')) and
                    torch.allclose(candidate_grad['total'] - control_grad['total'], candidate_grad['hinge'],
                                   rtol=1e-5, atol=1e-6), 'candidate-control preclip A/C differs from hinge alone')
        del control_grad, candidate_grad
        roles_record, hinge_roles = {}, {}
        for nonzero in (False, True):
            for arm in ARMS:
                tied = objective(64, arm, nonzero=True) if nonzero else values[arm]
                if nonzero:
                    micro = objective(16, arm, nonzero=True)
                    matched(tied, micro)
                    del micro
                split = objective(64, arm, split=True, nonzero=nonzero)
                split_micro = objective(16, arm, split=True, nonzero=nonzero)
                matched(split, split_micro)
                mutant = objective(16, arm, detached=True, nonzero=nonzero)
                hinge_mutant = objective(16, arm, hinge_detached=True, nonzero=nonzero) if arm == 'candidate' else None
                require(tied[0] == split[0] and tied[3:] == split[3:] == split_micro[3:] == mutant[3:],
                        'split/tied objective/activity differs')
                for n in ('A', 'C'):
                    require(torch.allclose(tied[1][n]['total'], split[1][n]['total'] +
                                           split[2][n]['ranking'] + split[2][n]['hinge'], rtol=1e-5, atol=1e-6) and
                            torch.allclose(mutant[1][n]['total'], split[1][n]['total'], rtol=1e-5, atol=1e-6),
                            'complete A/C total query plus gallery decomposition differs')
                    for term in ('ranking', 'hinge') if arm == 'candidate' else ('ranking',):
                        total, query_grad, gallery_grad = tied[1][n][term], split[1][n][term], split[2][n][term]
                        mutation = hinge_mutant if term == 'hinge' else mutant
                        require(all(torch.isfinite(g).all().item() and g.double().norm().item() > 0
                                    for g in (total, query_grad, gallery_grad)) and
                                torch.allclose(gallery_grad, split_micro[2][n][term], rtol=1e-5, atol=1e-6) and
                                torch.allclose(total, query_grad + gallery_grad, rtol=1e-5, atol=1e-6) and
                                torch.allclose(mutation[1][n][term], query_grad, rtol=1e-5, atol=1e-6) and
                                not torch.allclose(mutation[1][n][term], total, rtol=1e-5, atol=1e-6),
                                'independent SmoothAP/hinge A/C query/gallery/detach falsifier failed')
                        record = {r + '_gradient_' + k:v for r,g in
                                  (('query',query_grad),('gallery',gallery_grad),('tied',total))
                                  for k,v in gradient_record(g).items()}
                        record.update(query_plus_gallery_exact=True, full64_micro16_exact=True,
                                      detached_gallery_mutant_rejected=True)
                        destination = hinge_roles if term == 'hinge' else roles_record
                        destination[('nonzero:' if nonzero else 'initial:') + arm + ':' + n] = record
                    if hinge_mutant is not None:
                        require(tied[3:] == hinge_mutant[3:] and
                                all(math.isclose(v, hinge_mutant[0][k], rel_tol=1e-5, abs_tol=1e-6)
                                    for k,v in tied[0].items()) and
                                all(torch.allclose(hinge_mutant[1][n][k], tied[1][n][k], rtol=1e-5, atol=1e-6)
                                    for k in ('regression','ranking')) and
                                torch.allclose(tied[1][n]['total'] - hinge_mutant[1][n]['total'],
                                               split[2][n]['hinge'], rtol=1e-5, atol=1e-6) and
                                not torch.allclose(tied[1][n]['total'], hinge_mutant[1][n]['total'],
                                                   rtol=1e-5, atol=1e-6),
                                'only-hinge-gallery-detached total mutant not rejected')
                del tied, split, split_micro, mutant, hinge_mutant, total, query_grad, gallery_grad, mutation
        nonzero_C = (torch.arange(128 * 1152).reshape(128, 1152) % 7 + 1).float() * 1e-4
        nonzero_state = {**state, 'C': nonzero_C}
        with torch.no_grad():
            for view in VIEWS:
                features = state['views'][view][batch]
                base = helper_guard(context).raw_features(features, state['head_object'], A,
                    state['means'], 'concat', context['legacy']['quadratic'])
                oracle = base + F.linear(features - state['mu_train'], nonzero_C)
                actual = raw_features(context, {**nonzero_state, 'arm': 'control'}, features)
                require(torch.equal(actual, raw_features(context, {**nonzero_state, 'arm': 'candidate'}, features)),
                        'both-arm nonzero C raw must match')
                wrong_mu = base + F.linear(features - (state['mu_train'] + .125), nonzero_C)
                require(torch.equal(actual, oracle) and not torch.allclose(actual, base, rtol=1e-5, atol=1e-6) and
                        not torch.allclose(actual, wrong_mu, rtol=1e-5, atol=1e-6) and
                        fingerprint(context, context['old'].packed_outputs(context['legacy'], actual)) ==
                        fingerprint(context, context['old'].packed_outputs(context['legacy'], oracle)),
                        'nonzero C oracle/omittedC/wrongmu mutant falsifier failed')
                del features, base, oracle, actual, wrong_mu
        del nonzero_C, nonzero_state
        mse, rank, active = arms['control']['mse'], arms['control']['rank'], arms['control']['active']
        control, candidate = (values[a][1]['A']['total'] for a in ARMS)
        rank_grad = values['control'][1]['A']['ranking']
        multi = sum(len(p) > 1 for p in membership['positive'])
        require(multi > 0, 'fixed firstB64 lacks multiple distinct-image positives')
        nonnearest_count, nonnearest_value = 0, 0.
        nonnearest_grad = torch.zeros_like(A)
        candidate_state = {**state, 'arm': 'candidate',
                           'C': torch.nn.Parameter(state['C'].detach().clone())}
        for view in VIEWS:
            for offset in range(0, 64, 16):
                anchors = batch[offset:offset + 16]
                raw = raw_features(context, candidate_state, state['views'][view][anchors])
                with torch.autocast('cpu', enabled=False):
                    scores = F.normalize(raw, dim=1) @ ranking_gallery(context, candidate_state).T
                    nonnearest = []
                    for local, anchor in enumerate(anchors):
                        positive = membership['positive'][offset + local]
                        if len(positive) > 1:
                            eligible = [j for j, row in enumerate(bank['original_rows']) if row != bank['original_rows'][anchor]]
                            nearest = min(positive, key=lambda j: (-float(scores[local, j].detach()), bank['original_rows'][j]))
                            terms = smooth_ap_terms(scores[local], positive, eligible)
                            beyond = [i for i, p in enumerate(positive) if p != nearest]
                            nonnearest.append(terms[beyond].sum() / len(positive) / (2 * K))
                            nonnearest_count += len(beyond)
                if nonnearest:
                    contribution = sum(nonnearest)
                    nonnearest_grad.add_(torch.autograd.grad(contribution, A)[0])
                    nonnearest_value += float(contribution.detach())
                    del contribution, terms
                del raw, scores, nonnearest
        require(math.isfinite(nonnearest_value) and nonnearest_value > 0 and
                torch.isfinite(nonnearest_grad).all().item() and nonnearest_grad.double().norm().item() > 0,
                'fixed firstB64 nonnearest positive gradient inactive')
        # Native ties/self and genuine singleton witnesses, without batch rescue.
        tied = torch.tensor([99., 0., 0., 0., 0.], dtype=torch.float32)
        require(torch.allclose(smooth_ap_terms(tied, [1, 2], [1, 2, 3, 4]),
                               torch.tensor([.4, .4]), rtol=1e-5, atol=1e-6), 'native tie/self rank algebra differs')
        singletons = [i for i, t in enumerate(bank['target']) if state['count_list'][t] == 1]
        require(len(singletons) == 12 and ranking_membership(bank, singletons)['valid'] == 0,
                'genuine singleton membership differs')
        raw = raw_features(context, candidate_state, state['views']['canonical'][singletons])
        _, singleton_rank, singleton_hinge, singleton_facts = loss_terms(context, candidate_state, raw, singletons, 0)
        singleton_grads = torch.autograd.grad(singleton_rank + singleton_hinge, (A, candidate_state['C']))
        require(singleton_rank.item() == singleton_hinge.item() == 0 and
                singleton_facts['active'] == singleton_facts['hinge_active'] == 0 and
                all(torch.count_nonzero(g).item() == 0 for g in singleton_grads),
                'singleton anchors must receive regression only')
        del raw, singleton_rank, singleton_hinge, singleton_grads, candidate_state
        native_live_top1_witness()
        require(A.grad is None and state['C'].grad is None and
                fingerprint(context, static_tree(state)) == frozen_before and
                torch.equal(A.detach(), context['initial']['A']) and
                torch.equal(state['C'].detach(), context['initial']['C']), 'falsifier changed A/frozen bytes')
        alignment = float(F.cosine_similarity(values['control'][1]['A']['regression'].flatten().double(),
                                              rank_grad.flatten().double(), dim=0))
        return {'seed': state['seed'], 'batch': batch, 'membership_sha256': json_sha256(membership),
                'mse': float(mse), 'rank': float(rank), 'active': active, 'K': K,
                'control_gradient_norm': float(control.double().norm()),
                'ranking_gradient_norm': float(rank_grad.double().norm()),
                'candidate_gradient_norm': float(candidate.double().norm()),
                'candidate_minus_control_gradient_norm': float((candidate - control).double().norm()),
                'gradient_alignment': alignment,
                'multi_positive_anchors': multi, 'nonnearest_positive_terms': nonnearest_count,
                'nonnearest_loss': nonnearest_value,
                'nonnearest_gradient_norm': float(nonnearest_grad.double().norm()),
                'native_mask_self_ties_singletons_exact': True,
                'arms': arms, 'initial_raw_unit_packed_exact': True,
                'initial_gallery_scores_matched': True, 'regression_gradients_identical': True,
                'original_active_objective_exact': True,
                'regression_difference_gradient_norm': 0.,
                'gallery_gradient_norm': roles_record['initial:candidate:A']['gallery_gradient_norm'],
                'gallery_gradient_sha256': roles_record['initial:candidate:A']['gallery_gradient_sha256'],
                'query_gradient_norm': roles_record['initial:candidate:A']['query_gradient_norm'],
                'query_gradient_sha256': roles_record['initial:candidate:A']['query_gradient_sha256'],
                'tied_gradient_equals_query_plus_gallery': True,
                'base_losses_and_A_C_gradients_matched': True, 'candidate_minus_control_equals_hinge': True,
                'hinge_only_gallery_detached_mutant_rejected': True, 'detached_gallery_mutant_rejected': True,
                'roles': roles_record, 'hinge_roles': hinge_roles,
                'live_top1_selection_exact': True,
                'hinge': arms['candidate']['hinge'], 'hinge_active': arms['candidate']['hinge_active'],
                'hinge_gradient_norm': arms['candidate']['gradients']['A']['hinge']['norm'],
                'control_C_gradient_norm': arms['control']['gradients']['C']['total']['norm'],
                'candidate_C_gradient_norm': arms['candidate']['gradients']['C']['total']['norm'],
                'C_candidate_minus_control_gradient_norm': float((values['candidate'][1]['C']['total'] -
                    values['control'][1]['C']['total']).double().norm()),
                'nonzero_C_oracle_exact': True, 'omitted_C_mutant_rejected': True, 'wrong_mu_mutant_rejected': True,
                'frozen_bytes_exact': True, 'micro16_global_reduction_exact': True}


def inference_members(context, state):
    return {'schema': INFERENCE_SCHEMA,
            'arm': state['arm'],
            'C': clone(context, state['C'].detach()), 'mu_train': clone(context, state['mu_train']),
            'mu_train_provenance': clone(context, state['mu_train_provenance']),
            'source': {'accepted_A_sha256': context['initial_A_sha256'],
                       'initial_C_sha256': context['initial_C_sha256'],
                       'mu_train_sha256': context['mu_train_sha256'],
                       'mu_train_provenance_sha256': context['mu_train_provenance_sha256'],
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
    batch_bound_files(guards, ((directory / name, digest) for name, digest in {**value['code'], **value['files']}.items()))
    for name, digest in {**value['code'], **value['files']}.items():
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
    batch_bound_files(guards, env['files'].items())
    return value, guards


def clone_inference_provenance(value):
    return strict_json(json.dumps(value, allow_nan=False))


def inference_readout_tree(endpoint):
    return {'arm': endpoint['arm'], 'A': endpoint['A'].detach(), 'C': endpoint['C'].detach(),
            'mu_train': endpoint['mu_train'], 'mu_train_provenance': endpoint['mu_train_provenance'],
            'means': endpoint['means'], 'head': dict(endpoint['head_object'].state_dict()),
            'A_trainable': endpoint['A'].requires_grad, 'C_trainable': endpoint['C'].requires_grad}


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
    parameter_roles(disk['arm'])
    check_mu_train_provenance(disk['mu_train_provenance'])
    primitive = modules['quadratic_readout.py']
    primitive._check_tensor(disk['C'], (128, 1152), disk['A'].device, frozen=True)
    primitive._check_tensor(disk['mu_train'], (1152,), disk['A'].device, frozen=True)
    require(torch.isfinite(disk['C']).all().item() and torch.isfinite(disk['mu_train']).all().item() and
            original.fingerprint(disk['mu_train']) == disk['source']['mu_train_sha256'] and
            original.fingerprint(disk['mu_train_provenance']) == disk['source']['mu_train_provenance_sha256'],
            'portable frozen mean/residual provenance differs')
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
    C = torch.nn.Parameter(disk['C'].to(device, copy=True), requires_grad=True)
    mu_train = disk['mu_train'].to(device, copy=True)
    endpoint = {'model': model, 'processor_object': processor, 'head_object': head, 'A': A, 'C': C,
                'arm': disk['arm'], 'mu_train': mu_train, 'mu_train_provenance': clone_inference_provenance(disk['mu_train_provenance']),
                'means': means,
                'device': device, 'modules': modules, 'guards': guards, 'flags': disk['numerical_flags'],
                'manifest': manifest}
    endpoint['readout_sha256'] = original.fingerprint(inference_readout_tree(endpoint))
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
    require(modules['train_siglip2_substrate_adaptation.py'].fingerprint(inference_readout_tree(endpoint)) ==
            endpoint['readout_sha256'], 'current portable A/C/mean/head bytes or roles differ')
    rng = torch.random.get_rng_state().clone()
    pixels = endpoint['processor_object'](images=images, return_tensors='pt')['pixel_values']
    require(pixels.shape == (len(images), 3, 256, 256) and pixels.dtype == torch.float32 and
            torch.isfinite(pixels).all().item(), 'portable pixels differ')
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device, enabled=False):
            features = F.normalize(pooled.float(), dim=1)
            raw = fullfeature_raw_features(features, endpoint['head_object'], endpoint['A'], endpoint['means'],
                endpoint['C'], endpoint['mu_train'], endpoint['arm'], modules['quadratic_readout.py'],
                modules['prototype_residual_readout.py'])
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


def authenticate_bundle_environment(context, environment):
    """Manifest allowances must equal the qualified source's serving inventory."""
    legacy = context['legacy']
    origins = legacy['origins']
    packages = origins['packages']
    require(packages == legacy['selected']['packages'] and
            set(packages) == NATIVE - {'sfora'}, 'preflight qualified packages differ')
    roots = [Path(v['root']) for v in packages.values()]
    native_paths = set(origins['native_files']) | {p for p in context['guards']
        if Path(p).name in context['nearest'].NATIVE_MEMBERS}
    files = {p: h for p, h in context['guards'].items()
             if any(Path(p).is_relative_to(r) for r in roots) or p in native_paths}
    constructor = legacy['prior']['sources']['native_environment']['vision_constructor']['path']
    require(constructor in files and environment == {'packages': packages, 'files': files,
            'native_files': {p: files[p] for p in native_paths}, 'vision_constructor': constructor},
            'preflight serving environment differs from qualified source')


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
                bundle = portable.read_json({'path': str(directory / 'bundle.json'), 'sha256': sha}, {})
                authenticate_bundle_environment(context, bundle['environment'])
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
                    base = raw
                    raw = base + F.linear(features - endpoint['mu_train'], endpoint['C'])
                    require(fingerprint(context, context['old'].packed_outputs(legacy, raw)) == fingerprint(context, output),
                            'same-role original helper raw/unit/packed/wire differs')
                    nonzero = torch.count_nonzero(endpoint['C']).item() > 0
                    require(nonzero is witness['residual_nonzero_witness'] and
                            fingerprint(context, endpoint['C']) == witness['current_C_sha256'] and
                            fingerprint(context, endpoint['mu_train']) == witness['mu_train_sha256'],
                            'independent portable C/mu endpoint differs')
                    if nonzero:
                        require(not torch.equal(base, output['raw'].to(device)), 'nonzero C omitted by portable API')
                        # Use a deterministic shift along the greatest nonzero C column;
                        # no fit, no calibration search, and no numerical cancellation of the mutant.
                        column = int(endpoint['C'].abs().sum(dim=0).argmax())
                        wrong_mu = endpoint['mu_train'].clone()
                        wrong_mu[column] += 1.
                        wrong = base + F.linear(features - wrong_mu, endpoint['C'])
                        require(not torch.equal(wrong, output['raw'].to(device)), 'wrong mu mutant accepted')
                        # The actual public API must reject current .data substitutions too.
                        for member in ('C', 'mu_train'):
                            value = endpoint[member]
                            saved, version = value.detach().clone(), value._version
                            try:
                                value.data.reshape(-1)[0] += .25
                                require(value._version == version, 'portable mutant must bypass version')
                                nearest.rejected(lambda: portable.inference_outputs(endpoint, images),
                                                 'portable current C/mu mutation accepted')
                            finally:
                                value.data.copy_(saved)
                        del wrong_mu, wrong, value, saved
                    del oracle_A, raw, base, features, pooled
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
            'bundle_original_dependencies_denied': True,
            'residual_nonzero_witness': witness['residual_nonzero_witness'],
            'omitted_C_mutant_rejected': True, 'wrong_mu_mutant_rejected': True, **drift}


def update(context, state, ident, step):
    import torch
    require(state['device'] == 'cuda' and state['counter'] == step - 1 and 1 <= step <= 128, 'fixed CUDA update required')
    own_A(context, state)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    full_membership = ranking_membership(state['ranking_bank'], batch)
    K = full_membership['valid']
    optimizer, scaler, A = state['optimizer_object'], state['scaler_object'], state['A']
    optimizer.zero_grad(set_to_none=True)
    before = fingerprint(context, A.detach())
    C_before = fingerprint(context, state['C'].detach())
    mse_sum = rank_sum = hinge_sum = 0.
    active, hinge_active, membership = 0, 0, []
    component_gradients = {term: {n: torch.zeros_like(state[n]) for n in ('A', 'C')}
                           for term in ('ranking', 'hinge')} if step == 1 else None
    for view in VIEWS:
        for offset in range(0, 64, 16):
            anchors = batch[offset:offset + 16]
            features = state['views'][view][anchors].to(state['device'])
            raw = raw_features(context, state, features)
            require(raw.requires_grad and raw.grad_fn is not None, 'sole-A graph detached')
            mse, rank, hinge, selected = loss_terms(context, state, raw, anchors, K)
            if step == 1:
                for label, term in (('ranking', rank), ('hinge', hinge)):
                    grads = torch.autograd.grad(term, (A, state['C']), retain_graph=True)
                    for n, grad in zip(('A', 'C'), grads, strict=True):
                        require(grad.dtype == torch.float32 and torch.isfinite(grad).all().item(),
                                'finite FP32 separate SmoothAP/hinge gradient required')
                        component_gradients[label][n].add_(grad.detach())
                    del grads, grad, term
            loss = mse + rank + hinge
            scaler.scale(loss).backward()
            mse_sum += float(mse.detach())
            rank_sum += float(rank.detach())
            hinge_sum += float(hinge.detach())
            active += selected['active']
            hinge_active += selected['hinge_active']
            membership.append({'view': view, 'batch': anchors, **selected})
            del raw, mse, rank, hinge, loss, features
    scaler.unscale_(optimizer)
    require(A.grad.dtype == torch.float32 and torch.isfinite(A.grad).all().item() and
            A.grad.double().norm().item() > 0, 'finite nonzero A gradient required')
    gradient = float(A.grad.double().norm())
    gradient_norms = {term + ('_C' if n == 'C' else '') + '_gradient_norm':
                      float(component_gradients[term][n].double().norm()) if step == 1 else None
                      for term in ('ranking', 'hinge') for n in ('A', 'C')}
    if step == 1:
        require(active > 0 and all(gradient_norms[k] > 0 for k in ('ranking_gradient_norm', 'ranking_C_gradient_norm')) and
                (hinge_active > 0 and all(gradient_norms[k] > 0 for k in ('hinge_gradient_norm', 'hinge_C_gradient_norm'))
                 if state['arm'] == 'candidate' else hinge_active == 0 and hinge_sum == 0 and
                 all(gradient_norms[k] == 0 for k in ('hinge_gradient_norm', 'hinge_C_gradient_norm'))),
                'fixed first-update separate SmoothAP/hinge activity required')
    del component_gradients
    members = optimizer.param_groups[0]['params']
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and
            torch.isfinite(p.grad).all().item() and p.grad.double().norm().item() > 0 for p in members),
            'finite actual A/C gradients required in both arms')
    C_gradient = float(state['C'].grad.double().norm())
    norm = torch.nn.utils.clip_grad_norm_(members, 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    require(scaler.get_scale() == scale == 128 and
            all(optimizer.state[p]['step'].item() == step for p in members),
            'skipped/nonfinite/rescaled update forbidden')
    state['counter'] = step
    after = fingerprint(context, A.detach())
    C_after = fingerprint(context, state['C'].detach())
    require(before != after and C_before != C_after,
            'actual role-aware A/C update required')
    own_A(context, state, advanced=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    core = time.perf_counter() - tick
    with timed(context, 'update_integrity'):
        integrity(context, state, ident)
        digest = fingerprint(context, payload(context, state, ident))
    row = {'step': step, 'batch': batch, 'membership': membership, 'full_membership_sha256': json_sha256(full_membership), 'full_valid': K, 'mse': mse_sum, 'rank': rank_sum,
           'hinge': hinge_sum, 'hinge_active_anchors': hinge_active,
           'loss': mse_sum + rank_sum + hinge_sum, 'active_anchors': active,
           'gradient_norm': gradient, **gradient_norms, 'preclip_norm': float(norm),
           'A_before_sha256': before, 'A_after_sha256': after,
           'C_before_sha256': C_before, 'C_after_sha256': C_after, 'C_gradient_norm': C_gradient,
           'arm': state['arm'], 'scale': scaler.get_scale(),
           'state_sha256': digest, 'core_seconds': core, 'seconds': time.perf_counter() - tick}
    print(json.dumps({'event': 'COMPACT_UPDATE', **row}, sort_keys=True, allow_nan=False), flush=True)
    return row


def diagnostic(row):
    return {k: v for k, v in row.items() if k not in ('seconds', 'core_seconds')}


def check_steps(rows, start, count, bank):
    check_ranking_bank(bank)
    require(isinstance(rows, list) and len(rows) == count and
            [r['step'] for r in rows] == list(range(start, start + count)), 'complete bounded steps required')
    for row in rows:
        full = ranking_membership(bank, row['batch'])
        require(len(row['batch']) == 64 and row['full_valid'] == full['valid'] and
                type(row['full_valid']) is int and row['full_membership_sha256'] == json_sha256(full) and
                row['scale'] == 128 and row['gradient_norm'] > 0 and row['arm'] in ARMS and
                type(row['C_gradient_norm']) in (int, float) and math.isfinite(row['C_gradient_norm']) and
                row['C_gradient_norm'] > 0 and row['C_before_sha256'] != row['C_after_sha256'] and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k])
                    for k in ('C_before_sha256', 'C_after_sha256')) and
                row['A_before_sha256'] != row['A_after_sha256'] and
                0 < row['core_seconds'] <= row['seconds'] and len(row['membership']) == 8 and
                [m['view'] for m in row['membership']] == ['canonical'] * 4 + ['augmented'] * 4,
                'complete paired-view ranking/update record differs')
        for i, member in enumerate(row['membership']):
            anchors = row['batch'][(i % 4) * 16:(i % 4 + 1) * 16]
            expected = ranking_membership(bank, anchors)
            require(member.keys() == {'view', 'batch', 'active', 'hinge_active', *expected} and member['batch'] == anchors and
                    json_sha256({k: member[k] for k in expected}) == json_sha256(expected) and
                    type(member['active']) is int and 0 <= member['active'] <= expected['valid'] and
                    type(member['hinge_active']) is int and 0 <= member['hinge_active'] <= expected['valid'] and
                    (row['arm'] == 'candidate' or member['hinge_active'] == 0),
                    'complete all-positive original-image membership differs')
        require(type(row['active_anchors']) is int and
                row['active_anchors'] == sum(m['active'] for m in row['membership']) and
                type(row['hinge_active_anchors']) is int and
                row['hinge_active_anchors'] == sum(m['hinge_active'] for m in row['membership']) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('mse', 'rank', 'hinge', 'loss', 'gradient_norm', 'preclip_norm', 'core_seconds', 'seconds')) and
                row['mse'] >= 0 and 0 <= row['rank'] <= 1 and 0 <= row['hinge'] <= 41 and
                (row['arm'] == 'candidate' or row['hinge'] == 0) and
                row['loss'] == row['mse'] + row['rank'] + row['hinge'], 'nonfinite update record')
        for term in ('ranking', 'hinge'):
            for suffix in ('', '_C'):
                value = row[term + suffix + '_gradient_norm']
                require((type(value) in (int, float) and math.isfinite(value) and
                         (value > 0 if term == 'ranking' or row['arm'] == 'candidate' else value == 0))
                        if row['step'] == 1 else value is None, 'separate first-step A/C gradient receipt differs')
    if start == 1:
        require(rows[0]['active_anchors'] > 0 and
                (rows[0]['hinge_active_anchors'] > 0 if rows[0]['arm'] == 'candidate' else
                 rows[0]['hinge_active_anchors'] == 0), 'first-step SmoothAP/hinge activity missing')


def tamper_witness(context, state, ident):
    import torch
    nearest = context['nearest']
    values = [state['A'], state['C'], state['mu_train'], next(state['head_object'].parameters()), state['means']['concat'],
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
    state['C'].requires_grad_(not (state['arm'] == 'candidate'))
    try:
        nearest.rejected(lambda: integrity(context, state, ident), 'C role mutation accepted')
    finally:
        state['C'].requires_grad_(state['arm'] == 'candidate')
    saved = payload(context, state, ident)
    for key, value in (('schema', 'wrong'), ('source', {}), ('teachers', {}), ('schedules', {}),
                       ('buffers', {}), ('C', saved['A']), ('mu_train', saved['A']), ('mu_train_provenance', {}),
                       ('optimizer', {'state': {}, 'param_groups': []}), ('counter', 1)):
        nearest.rejected(lambda k=key, v=value: check_payload(context, {**saved, k: v}, ident, 0),
                         'malformed complete state accepted')
    del saved


def inference_witness(context, state):
    import torch
    with torch.no_grad():
        raw = raw_features(context, state, state['views']['canonical'].to(state['device'])).detach().cpu()
    return {'seed': state['seed'], 'cache_raw': raw,
            'residual_nonzero_witness': state['counter'] > 0,
            'current_C_sha256': fingerprint(context, state['C'].detach()),
            'mu_train_sha256': fingerprint(context, state['mu_train'])}


def cpu_witnesses(context):
    import torch
    args = context['args']
    require(not torch.cuda.is_initialized(), 'CPU CUDA hidden required')
    gradients, first_witness, first_ident, first_digest = [], None, None, None
    members, native_witness = None, None
    context['active_original'] = authenticate_active_objective(context)
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
                tamper_witness(context, state, ident)
                candidate_checkpoint = args.output / f'candidate-initializer-{seed}.pt'
                candidate_sha, candidate_digest = save(context, state, ident, candidate_checkpoint)
                candidate_ident = ident
                require(ident['static_sha256'] == matched[0]['static_sha256'] and
                        ident['initial_cpu_rng_sha256'] == matched[0]['initial_cpu_rng_sha256'] and
                        witness == matched[1], 'independent matched CPU initialization differs')
                release(context, state)
                state = restore(context, candidate_checkpoint, candidate_sha, candidate_digest, candidate_ident, 0)
                require(fingerprint(context, cached_witness(context, state)) == matched[1], 'candidate CPU full reload differs')
                release(context, state)
    original = context.pop('active_original')
    require(sys.modules.pop(original.__name__, None) is original, 'archived objective registry changed')
    del original
    gc.collect()
    require_no_training(context)
    bundle = export_bundle(context, members, args.output / 'bundle')
    with timed(context, 'cpu_bundle_qualification'):
        native = qualify_bundle(context, args.output / 'bundle', bundle['sha256'], 'cpu', native_witness)
    require(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
    return {'ranking_bank': ranking_bank(context['initial']['target'].tolist(),
                                         context['initial']['original_rows'].tolist()),
            'initial_C_sha256': context['initial_C_sha256'], 'current_C_sha256': context['initial_C_sha256'],
            'mu_train_sha256': context['mu_train_sha256'],
            'mu_train_provenance_sha256': context['mu_train_provenance_sha256'],
            'C_exact_zero': True, 'C_trainable': True,
            'completed_step': 0, 'identity': first_ident, 'terminal_state_sha256': first_digest,
            'initial_A_sha256': context['initial_A_sha256'], 'initial_raw_unit_packed_sha256': first_witness,
            'checkpoint': {'path': str(args.output / f'initializer-{SEEDS[0]}.pt'),
                           'sha256': context['guards'][str(args.output / f'initializer-{SEEDS[0]}.pt')]},
            'bundle': bundle, 'gradients': gradients, 'active_objective_source': ACTIVE_OBJECTIVE_SOURCE, 'initial_arm_parity': True,
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
        final_residual = residual_facts(context, state)
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
        for name in ('C', 'mu_train'):
            original_value = context['initial'][name].to(state['device'])
            substitution = state[name].detach().clone()
            try:
                if name == 'C':
                    state[name].data.copy_(original_value)
                else:
                    state[name].data.reshape(-1)[0] += .25
                context['nearest'].rejected(lambda: integrity(context, state, ident), 'C/mu source substitution accepted')
            finally:
                state[name].data.copy_(substitution)
            del original_value, substitution
        integrity(context, state, ident)
        release(context, state)
        bundle_directory = temporary / 'bundle' if args.phase == 'mechanics' else args.output / 'bundle'
        bundle = export_bundle(context, members, bundle_directory)
        with timed(context, 'gpu_bundle_qualification'):
            native = qualify_bundle(context, bundle_directory, bundle['sha256'], 'cuda', native_witness)
        with timed(context, 'post_calibration_api_authentication'):
            api = context['nearest'].native_source_api(context)
        with timed(context, 'post_calibration_origin_audit'):
            api.audit_origins(context['legacy'], admission=context['legacy']['original'].FlatAdmission(), require_exact=True)
        for path in list(context['guards']):
            if Path(path).is_relative_to(temporary):
                context['guards'].pop(path)  # Discard only after full reload/parity qualification.
    bank = ranking_bank(context['initial']['target'].tolist(), context['initial']['original_rows'].tolist())
    check_steps(rows, 1, total, bank)
    if resumed:
        check_steps(resumed, 9, 9, bank)
    return {'ranking_bank': bank, 'completed_step': total, 'identity': ident, 'steps': rows, 'resumed_steps': resumed,
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
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), **final_residual, **native,
            'total_training_core_seconds': context['phase_seconds']['cache_target_preparation'] +
                sum(r['core_seconds'] for r in rows) + (sum(r['core_seconds'] for r in first8 + resumed) if resumed else 0.),
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:])}


def check_cpu_gradient(g, bank):
    members = ranking_membership(bank, g['batch'])
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    def digest(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
    require(len(g['batch']) == 64 and g['membership_sha256'] == json_sha256(members) and
            type(g['seed']) is int and g['seed'] in SEEDS and type(g['K']) is int and
            0 < g['K'] == members['valid'] <= 64 and type(g['active']) is int and
            0 < g['active'] <= 2*g['K'] and type(g['hinge_active']) is int and
            0 < g['hinge_active'] <= 2*g['K'] and
            all(positive(g[k]) for k in ('rank', 'hinge', 'control_gradient_norm', 'ranking_gradient_norm',
                'candidate_gradient_norm', 'candidate_minus_control_gradient_norm', 'hinge_gradient_norm',
                'control_C_gradient_norm', 'candidate_C_gradient_norm', 'C_candidate_minus_control_gradient_norm',
                'nonnearest_loss', 'nonnearest_gradient_norm', 'query_gradient_norm', 'gallery_gradient_norm')) and
            type(g['mse']) in (int,float) and math.isfinite(g['mse']) and g['mse'] >= 0 and
            g['rank'] <= 1 and g['hinge'] <= 41 and
            type(g['gradient_alignment']) in (int,float) and -1 <= g['gradient_alignment'] <= 1 and
            type(g['multi_positive_anchors']) is int and g['multi_positive_anchors'] > 0 and
            g['multi_positive_anchors'] == sum(len(p) > 1 for p in members['positive']) and
            type(g['nonnearest_positive_terms']) is int and
            g['nonnearest_positive_terms'] == 2*sum(max(len(p)-1,0) for p in members['positive']) and
            type(g['regression_difference_gradient_norm']) in (int,float) and g['regression_difference_gradient_norm'] == 0 and
            all(g[k] is True for k in ('native_mask_self_ties_singletons_exact', 'initial_raw_unit_packed_exact',
                'initial_gallery_scores_matched', 'regression_gradients_identical', 'original_active_objective_exact',
                'base_losses_and_A_C_gradients_matched', 'candidate_minus_control_equals_hinge',
                'hinge_only_gallery_detached_mutant_rejected', 'live_top1_selection_exact',
                'tied_gradient_equals_query_plus_gallery', 'detached_gallery_mutant_rejected',
                'nonzero_C_oracle_exact', 'omitted_C_mutant_rejected', 'wrong_mu_mutant_rejected',
                'frozen_bytes_exact', 'micro16_global_reduction_exact')),
            'fixed live-top1 firstB64 CPU falsifier differs')
    require(isinstance(g['arms'],dict) and g['arms'].keys() == set(ARMS), 'both objective arms required')
    terms = ('canonical_regression','augmented_regression','regression','ranking','hinge','total')
    for arm, row in g['arms'].items():
        numeric = ('canonical_mse','augmented_mse','mse','rank','hinge','loss')
        require(row.keys() == {*numeric,'active','hinge_active','micro16_global_reduction_exact','gradients'} and
                all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k] >= 0 for k in numeric) and
                0 < row['rank'] <= 1 and 0 <= row['hinge'] <= 41 and
                type(row['active']) is int and row['active'] == g['active'] and
                type(row['hinge_active']) is int and
                row['hinge_active'] == (g['hinge_active'] if arm == 'candidate' else 0) and
                row['hinge'] == (g['hinge'] if arm == 'candidate' else 0) and
                row['mse'] == row['canonical_mse'] + row['augmented_mse'] and
                row['loss'] == row['mse'] + row['rank'] + row['hinge'] and
                row['micro16_global_reduction_exact'] is True and
                row['gradients'].keys() == {'A','C'}, 'separate SmoothAP/hinge objective arithmetic differs')
        for name, gradients in row['gradients'].items():
            require(gradients.keys() == set(terms), 'complete A/C objective components required')
            for term, fact in gradients.items():
                require(fact.keys() == {'norm','sha256'} and digest(fact['sha256']) and
                        (positive(fact['norm']) if term != 'hinge' or arm == 'candidate' else
                         type(fact['norm']) in (int,float) and fact['norm'] == 0),
                        'finite separate A/C objective gradients required')
        require(row['gradients']['A']['total']['norm'] == g[arm+'_gradient_norm'] and
                row['gradients']['C']['total']['norm'] == g[arm+'_C_gradient_norm'],
                'complete total A/C gradient binding differs')
    control, candidate = (g['arms'][arm] for arm in ARMS)
    require(control['mse'] == g['mse'] and control['rank'] == g['rank'] and
            all(control[k] == candidate[k] for k in ('canonical_mse','augmented_mse','mse','rank','active')) and
            all(control['gradients'][n][k] == candidate['gradients'][n][k]
                for n in ('A','C') for k in terms[:4]) and
            control['gradients']['A']['ranking']['norm'] == g['ranking_gradient_norm'] and
            candidate['gradients']['A']['hinge']['norm'] == g['hinge_gradient_norm'] and
            all(math.isclose(g[key], candidate['gradients'][n]['hinge']['norm'], rel_tol=1e-5, abs_tol=1e-6)
                for n,key in (('A','candidate_minus_control_gradient_norm'),('C','C_candidate_minus_control_gradient_norm'))),
            'base objective correspondence or preclip auxiliary A/C difference differs')
    for collection, term, arms in (('roles','ranking',ARMS), ('hinge_roles','hinge',('candidate',))):
        roles = g[collection]
        require(isinstance(roles,dict) and roles.keys() == {
            stage+':'+arm+':'+n for stage in ('initial','nonzero') for arm in arms for n in ('A','C')},
            'complete separate query/gallery role inventory required')
        for key,row in roles.items():
            require(row.keys() == {*(r+'_gradient_'+k for r in ('query','gallery','tied') for k in ('norm','sha256')),
                                   'query_plus_gallery_exact','full64_micro16_exact','detached_gallery_mutant_rejected'} and
                    all(positive(row[r+'_gradient_norm']) and digest(row[r+'_gradient_sha256'])
                        for r in ('query','gallery','tied')) and
                    all(row[k] is True for k in ('query_plus_gallery_exact','full64_micro16_exact',
                                                'detached_gallery_mutant_rejected')),
                    'finite connected separate SmoothAP/hinge A/C roles required')
            stage,arm,n = key.split(':')
            if stage == 'initial':
                require(row['tied_gradient_norm'] == g['arms'][arm]['gradients'][n][term]['norm'] and
                        row['tied_gradient_sha256'] == g['arms'][arm]['gradients'][n][term]['sha256'],
                        'initial tied component gradient binding differs')
    require(all(g['roles'][stage+':control:'+n] == g['roles'][stage+':candidate:'+n]
                for stage in ('initial','nonzero') for n in ('A','C')) and
            all(g['roles']['initial:candidate:A'][r+'_gradient_'+k] == g[r+'_gradient_'+k]
                for r in ('query','gallery') for k in ('norm','sha256')),
            'SmoothAP role parity or top-level gallery witness binding differs')
    return True


def check_terminal_record(record, launch, phase, arm, seed):
    from types import SimpleNamespace
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == seed and method(record['launch']) == method(launch) and
            record['resource_policy'] == policy(phase) and record['optimizer_members'] == len(parameter_roles(arm)[0]) and
            record['trainable_scalars'] == parameter_roles(arm)[2] and record['frozen_vision_members'] == 448 and
            record['quality_read'] is False and all(record[k] is True for k in
                ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'sequential_model_ownership', 'forward_oracle_exact',
                 'native_training_inference_exact', 'inference_artifact_independent',
                 'bundle_original_dependencies_denied', 'both_locks_held_in_parent_authority')) and
            0 < record['total_training_core_seconds'] < record['wall_seconds'] < policy(phase)['seconds'] and
            0 < record['process_peak_rss_kib'] <= 8 * 1024**2, 'qualified compact whole-unit contract differs')
    check_launch(record['launch'], SimpleNamespace(execution_sha256=launch['execution_sha256'],
                                                  phase=phase, arm=arm, seed=seed))
    ident = record['identity']
    bank = record['ranking_bank']
    check_ranking_bank(bank)
    require(bank['sha256'] == ident['ranking_bank_sha256'], 'terminal complete ranking bank differs')
    require(record['code'].keys() == FILES and record['authority_sha256'] == record['authority']['sha256'] and
            record['invocation']['optimize'] == 0 and ident['method'] == method(launch) and
            ident['source'] == record['source'] and ident['arm'] == arm and ident['seed'] == seed and
            (ident['parameter_names'], ident['parameter_shapes']) == parameter_roles(ident['arm'])[:2] and
            record['numerical_flags'] == ident['numerical_flags'] and
            isinstance(record['inference_state_sha256'], str) and re.fullmatch('[0-9a-f]{64}', record['inference_state_sha256']),
            'whole-unit source/code/optimizer/inference identity differs')
    require(all(isinstance(record[k], str) and re.fullmatch('[0-9a-f]{64}', record[k])
                for k in ('initial_C_sha256', 'current_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256')) and
            all(record[k] == ident[k] for k in ('initial_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256')) and
            record['C_trainable'] is True and
            record['C_exact_zero'] is (phase == 'cpu') and
            ((record['current_C_sha256'] == record['initial_C_sha256']) if record['C_exact_zero'] else
             (record['current_C_sha256'] != record['initial_C_sha256'])) and
            record['residual_nonzero_witness'] is (phase != 'cpu') and
            record['omitted_C_mutant_rejected'] is True and record['wrong_mu_mutant_rejected'] is True,
            'qualified endpoint C roles/zero/nonzero/mean/oracle binding differs')
    if phase == 'cpu':
        require(record['completed_step'] == 0 and ident['device'] == 'cpu' and
                record['cuda_initialized'] is False and record['peak_cuda_allocated_bytes'] == 0 and
                record['invocation']['cuda_visible_devices'] == '' and
                isinstance(record['checkpoint'], dict) and isinstance(record['bundle'], dict) and
                all(record[k] is True for k in ('initial_arm_parity', 'cpu_serialization_exact',
                    'bypass_version_tamper_rejected', 'malformed_state_rejected', 'native_role_mutation_rejected',
                    'native_loss_reduction_exact')) and [g['seed'] for g in record['gradients']] == list(SEEDS) and
                record['active_objective_source'] == ACTIVE_OBJECTIVE_SOURCE and
                all(check_cpu_gradient(g, bank) for g in record['gradients']),
                'both-seed CPU qualification incomplete')
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
        check_steps(record['steps'], 1, count, bank)
        require(all(r['arm'] == arm for r in record['steps']) and
                record['steps'][0]['C_before_sha256'] == record['initial_C_sha256'] and
                record['steps'][-1]['C_after_sha256'] == record['current_C_sha256'] and
                all(a['C_after_sha256'] == b['C_before_sha256']
                    for a, b in zip(record['steps'], record['steps'][1:])), 'complete ordered C update bytes differ')
        require(all(r['mse'] >= 0 and r['rank'] >= 0 and
                    r['loss'] == r['mse'] + r['rank'] + r['hinge']
                    for r in record['steps']), 'matched objective arithmetic differs')
        if phase == 'mechanics':
            require(seed == SEEDS[0] and record['checkpoint'] is None and record['bundle'] is None and
                    record['training_state_discarded'] is True and record['replay_exact'] is True and
                    record['independent_first8_exact'] is True and all(diagnostic(a) == diagnostic(b)
                        for a, b in zip(record['steps'][8:], record['resumed_steps'], strict=True)), 'mechanics replay/discard differs')
            check_steps(record['resumed_steps'], 9, 9, bank)
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


def exit_admission_adapter(context, api, reader):
    """Private exact exit substitutions; one genuine reader in this boundary."""
    import ast
    import copy
    nearest, legacy = context['nearest'], context['legacy']
    require(nearest.native_source_api(context) is api, 'authenticated exit API required')
    original = legacy['original']
    require(type(reader) is original.FlatAdmission and reader is not legacy.get('admission') and
            vars(reader).keys() == {'entries', 'verified', 'json_bytes'} and
            reader.entries == {p: (h, Path(p).stat().st_size) for p, h in legacy['origins']['files'].items()} and
            reader.verified == set(legacy['origins']['files']) and reader.json_bytes == {},
            'fresh original exit reader required')
    reader_members = dict(vars(reader))
    dump = lambda node: ast.dump(node, include_attributes=False)

    class Substitute(ast.NodeTransformer):
        def __init__(self, before, after):
            self.before, self.after, self.count = before, after, 0

        def visit(self, node):
            if dump(node) == dump(self.before):
                self.count += 1
                return ast.copy_location(copy.deepcopy(self.after), node)
            return super().visit(node)

    private_functions, namespaces = [], []
    for module, changes in (
            (context['old'], [("context['original'].FlatAdmission()", '_compact_exit_reader')]),
            (context['fitter'], [("context['old'].exit_rehash(context['legacy'])", "_compact_quadratic_exit(context['legacy'])"),
                                 ('bound_file({}, path, digest)', '_compact_exit_reader.bound_file({}, path, digest)')])):
        path = Path(module.__file__)
        digest = context['guards'][str(path)]
        raw = bound_file({}, path, digest).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'exit source changed before compilation')
        tree = ast.parse(raw, filename=str(path))
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash']
        require(len(matches) == 1, 'exact original exit definition required')
        source_node = matches[0]
        node = copy.deepcopy(source_node)
        for before, after in changes:
            forward = Substitute(ast.parse(before, mode='eval').body, ast.parse(after, mode='eval').body)
            node = forward.visit(node)
            require(forward.count == 1, 'exact exit substitution required')
        inverse_node = copy.deepcopy(node)
        for before, after in reversed(changes):
            inverse = Substitute(ast.parse(after, mode='eval').body, ast.parse(before, mode='eval').body)
            inverse_node = inverse.visit(inverse_node)
            require(inverse.count == 1, 'exact exit inverse substitution required')
        require(dump(inverse_node) == dump(source_node), 'exit adapter changed retained predicates')
        namespace = dict(vars(module))
        require('_compact_exit_reader' not in namespace and '_compact_quadratic_exit' not in namespace,
                'fresh private exit namespace required')
        namespace['_compact_exit_reader'] = reader
        if private_functions:
            namespace['_compact_quadratic_exit'] = private_functions[0]
        else:
            namespace['audit_origins'] = api.audit_origins
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
        private_functions.append(namespace.pop('exit_rehash'))
        namespaces.append((namespace, dict(namespace)))
    fitter_exit = private_functions[1]
    fit_context = context['fit_context']
    owned_functions = []

    def authenticate():
        require(context['nearest'] is nearest and nearest.native_source_api(context) is api and
                context['legacy'] is legacy and context['fit_context'] is fit_context and
                type(reader) is original.FlatAdmission and vars(reader).keys() == reader_members.keys() and
                all(vars(reader)[n] is v for n, v in reader_members.items()), 'exit reader/context binding changed')
        for values, members in namespaces:
            require(values.keys() == members.keys() and all(values[n] is v for n, v in members.items()),
                    'exit private global binding changed')
        for fn, code, defaults, kwdefaults, values, cells in owned_functions:
            require(fn.__code__ is code and fn.__defaults__ == defaults and fn.__kwdefaults__ == kwdefaults and
                    fn.__globals__ is values and len(fn.__closure__ or ()) == len(cells) and
                    all(c.cell_contents is v for c, v in zip(fn.__closure__ or (), cells)),
                    'exit private function/closure changed')

    authentication_code = authenticate.__code__

    def admitted(value):
        try:
            require(authenticate.__code__ is authentication_code, 'exit authenticator code changed')
            authenticate()
            require(value is fit_context, 'owned exit fitter context required')
            return fitter_exit(value)
        finally:
            # This dispatcher is exit-local and used once. Break private
            # function/namespace/snapshot cycles on both success and rejection.
            for values, members in namespaces:
                values.clear()
                members.clear()
            owned_functions.clear()
            namespaces.clear()

    owned_functions.extend((fn, fn.__code__, fn.__defaults__, copy.deepcopy(fn.__kwdefaults__), fn.__globals__,
                            tuple(cell.cell_contents for cell in fn.__closure__ or ()))
                           for fn in (*private_functions, authenticate, admitted))
    return admitted


def exit_rehash(context):
    require_no_training(context)
    with timed(context, 'source_exit_rehash'):
        helper_guard(context)
        api = context['nearest'].native_source_api(context)
        exit_reader = context['legacy']['original'].FlatAdmission()
        api.audit_origins(context['legacy'], admission=exit_reader, require_exact=context['args'].phase != 'cpu')
        exit_admission_adapter(context, api, exit_reader)(context['fit_context'])
    with timed(context, 'own_exit_rehash'):
        for p, h in context['guards'].items():
            exit_reader.bound_file({}, p, h)
        require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'],
                'own exact2 exit closure differs')
        require(closure(NEAREST['root'], NEAREST['execution_sha256'], NEAREST['code'], {}) == NEAREST['code'],
                'external exact3 exit closure differs')
    del exit_reader
    with timed(context, 'post_exit_api_authentication'):
        api = context['nearest'].native_source_api(context)
    with timed(context, 'post_exit_origin_audit'):
        api.audit_origins(context['legacy'], admission=context['legacy']['original'].FlatAdmission(), require_exact=context['args'].phase != 'cpu')


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
        post_run_reader = legacy['original'].FlatAdmission()
        api.audit_origins(legacy, admission=post_run_reader, require_exact=args.phase != 'cpu')
    with timed(context, 'origin_guard_promotion'):
        for p, h in legacy['origins']['files'].items():
            post_run_reader.bound_file(context['guards'], p, h)
    del post_run_reader
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
        'optimizer_members': len(parameter_roles(args.arm)[0]),
        'trainable_scalars': parameter_roles(args.arm)[2], 'frozen_vision_members': 448,
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
