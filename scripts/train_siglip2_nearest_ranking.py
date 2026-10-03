#!/usr/bin/env python3
"""Frozen four-tensor nearest-ranking trial. Native qualification is UNRUN.

Exact3 execution.json owns this driver, its stdlib tests and the connected
nearest_ranking_readout.py. Original fitter3 and its transitive source closures
are separate authenticated dependencies. No quality scoring is implemented.
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
import statistics
import sys
import time
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
import weakref

UNIT_STARTED = time.perf_counter()
SCHEMA = 'siglip2-nearest-ranking-v1'
AUTHORITY_SCHEMA = 'siglip2-nearest-ranking-launch-v1'
INFERENCE_SCHEMA = 'siglip2-nearest-ranking-inference-v1'
FILES = {'train_siglip2_nearest_ranking.py', 'test_siglip2_nearest_ranking.py', 'nearest_ranking_readout.py'}
FITTER = {'code': {'fit_siglip2_prototype_residual.py': '95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b',
          'prototype_residual_readout.py': '2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68',
          'test_siglip2_prototype_residual.py': 'c89aa3855d60d9f11d0dd57b55311c5364304915bd4a5d22b26b4c1e0a29c7d4'},
 'execution_sha256': 'a47933beafd8a1c0e2d541624328c8fe1d7b9a90f68977eee83c3c82223a56fe',
 'root': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2'}
ACCEPTED = {'arm': 'concat',
 'checkpoint': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/resume.pt',
                'sha256': 'b702e03847be88540ee9711b420279e3bc131fafc475f361eca3d57cf3b8bbcf'},
 'launch': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/authority-fit-concat-v1.json',
            'sha256': '109b6fded3f4559fcee3822536abd5beeed5f4ffc23f4313b87904faa6c39630'},
 'terminal': {'both_locks_held': True,
              'invocation_id': '94a84de4194f42f0842cee5b5c8f932a',
              'log': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-source-v2/fit-concat-v1.log',
                      'sha256': '93c2f7024116fc63d8e7cacc1a4e0106d2d57e9ca57d794c866e9bcc77311197'},
              'native_peak_rss_kib': 2854356,
              'receipt': {'path': '/home/riomus/runs/sfora-so400-signed-concat-fit-concat-v1/receipt.json',
                          'sha256': 'b4af0fecfa1d5f750f0ac9cc995198690baf4397fd970c8192727644135c7d5c'},
              'service_seconds': 234.821,
              'unit': 'sfora-so400-signed-concat-fit-concat-v1'},
 'terminal_state_sha256': 'a118fd98cce0b8fafa51c897be70b2b6e2ec93ebb226b2382dd264721776b644'}
ARMS = ('control', 'candidate')
SEED = 179061
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
NAMES = ['encoder.layers.26.mlp.fc1.weight', 'encoder.layers.26.mlp.fc1.bias',
         'encoder.layers.26.mlp.fc2.weight', 'encoder.layers.26.mlp.fc2.bias']
SHAPES = [(4304, 1152), (4304,), (1152, 4304), (1152,)]
ADAM = {'lr': 1e-5, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05,
        'amsgrad': False, 'maximize': False, 'foreach': False, 'capturable': False,
        'differentiable': False, 'fused': False}
RECIPE = {'seed': SEED, 'rows': 6355, 'classes': 1008, 'singletons': 12,
          'updates': 32, 'batch': 64, 'microbatch': 16, 'margin': .05,
          'trainable_names': NAMES, 'trainable_shapes': [list(s) for s in SHAPES],
          'trainable_scalars': 9921872, 'adamw': {**ADAM, 'betas': list(ADAM['betas'])},
          'clip': 1., 'initial_scaler': 128., 'vision': 'eval FP32 parameters; FP16 CUDA autocast',
          'readout': 'FP32 normalize(pooled.float()); connected fixed concat',
          'teacher': 'accepted concat CPU full TRAIN normalization/readout; immutable T,V,P,e0',
          'regression': 'sum128 coordinates / (64*e0)',
          'ranking': 'hinge sum / (full B64 valid count * .05); detached indices/teacher',
          'mining': 'all6355; highest other same identity/highest wrong identity; ascending original-row ties',
          'images': 'original canonical native256; no augmentation',
          'core': 'target construction + mining + decode/preprocess + forward/backward + optimizer'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'fitter', 'accepted',
               'recipe', 'resource_policy', 'both_locks_held', 'selected_cpu', 'selected_mechanics'}
STATIC_KEYS = ('provenance', 'config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means',
               'partition', 'original_rows', 'target', 'schedule', 'teachers', 'panel')
PAYLOAD_KEYS = {'schema', 'identity', 'source', 'vision', *STATIC_KEYS, 'optimizer', 'scaler',
                'counter', 'cpu_rng', 'cuda_rng', 'numerical_flags'}
INFERENCE_KEYS = {'schema', 'identity', 'source', 'vision', 'config', 'buffers', 'processor',
                  'head', 'classifier', 'A', 'means', 'numerical_flags', 'vision_sha256', 'fixed_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        value = {}
        for k, v in items:
            require(k not in value, 'duplicate JSON key')
            value[k] = v
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def file_fact(fact):
    require(isinstance(fact, dict) and fact.keys() == {'path', 'sha256'} and
            isinstance(fact['path'], str) and Path(fact['path']).is_absolute() and
            isinstance(fact['sha256'], str) and re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'exact FILE required')


def bound_file(guards, path, expected):
    file_fact({'path': str(path), 'sha256': expected})
    path = Path(path)
    require(path.resolve() == path and path.is_file(), 'canonical current file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'current file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(fact, guards, *, admission=None):
    file_fact(fact)
    if admission is not None:
        return admission.descriptor_json(fact, guards)
    path = bound_file(guards, fact['path'], fact['sha256'])
    require(path.stat().st_size <= 64 * 1024**2, 'bounded JSON required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == fact['sha256'], 'JSON changed before parse')
    return strict_json(raw)


def closure(root, sha, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': sha}, guards)
    require(code.keys() == set(names), 'exact code closure required')
    for name, digest in code.items():
        require(Path(name).name == name, 'bare code member required')
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
            'original complete UNIT required')
    for k in ('receipt', 'log'):
        file_fact(unit[k])
    require(isinstance(unit['unit'], str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and
            isinstance(unit['invocation_id'], str) and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            all(type(unit[k]) in (int, float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')), 'actual UNIT identity/resources required')


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and args.arm in ARMS and launch['seed'] == args.seed == SEED and
            launch['fitter'] == FITTER and launch['accepted'] == ACCEPTED and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'frozen nearest-ranking launch differs')
    require((args.phase != 'cpu' or args.arm == 'control') and
            (launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if args.phase != 'cpu':
        check_unit(launch['selected_cpu'])
    if args.phase == 'train':
        require(launch['selected_mechanics'].keys() == set(ARMS), 'both mechanics required')
        for unit in launch['selected_mechanics'].values():
            check_unit(unit)


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'fitter', 'accepted', 'recipe', 'seed')}


def cli(root, authority, sha, execution, phase, arm, output):
    return [str(Path(root) / 'train_siglip2_nearest_ranking.py'), '--execution-sha256', execution,
            '--authority', str(authority), '--authority-sha256', sha, '--phase', phase,
            '--arm', arm, '--seed', str(SEED), '--output', str(output)]


@contextmanager
def timed(context, name):
    tick = time.perf_counter()
    print(json.dumps({'event': 'NEAREST_PHASE', 'phase': name, 'boundary': 'begin',
                      'seconds': tick - context['started']}), flush=True)
    try:
        yield
    finally:
        delta = time.perf_counter() - tick
        context['phase_seconds'][name] = context['phase_seconds'].get(name, 0.) + delta
        print(json.dumps({'event': 'NEAREST_PHASE', 'phase': name, 'boundary': 'end',
                          'delta_seconds': delta, 'seconds': time.perf_counter() - context['started']}), flush=True)


def authenticate_startup_reader(context, fitter):
    """Authenticate the genuine reader and its uncached SHA dependency at entry."""
    guards, legacy = context['guards'], context['legacy']
    admission, reader_source = legacy['admission'], legacy['original']
    require(type(admission) is reader_source.FlatAdmission and
            reader_source.FlatAdmission.__module__ == reader_source.__name__ and
            reader_source.FlatAdmission.__qualname__ == 'FlatAdmission' and
            reader_source.__spec__ is not None and
            Path(reader_source.__spec__.origin) == Path(reader_source.__file__) and
            guards.get(reader_source.__file__) == fitter.TERMINAL_SOURCE_SHA,
            'actual original startup reader/source required')
    raw = bound_file(guards, reader_source.__file__, fitter.TERMINAL_SOURCE_SHA).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == fitter.TERMINAL_SOURCE_SHA,
            'original startup reader source changed before compilation')
    source_code = compile(raw, reader_source.__file__, 'exec')
    reader_code = next(c for c in source_code.co_consts if getattr(c, 'co_name', None) == 'FlatAdmission')
    for name in ('__init__', 'canonical', 'digest_string', 'register', 'bound_file',
                 'read_json', 'descriptor_json', 'admit_terminal'):
        fn, bound = getattr(reader_source.FlatAdmission, name), getattr(admission, name)
        expected = next(c for c in reader_code.co_consts if getattr(c, 'co_name', None) == name)
        require(isinstance(fn, FunctionType) and fn.__globals__ is vars(reader_source) and
                fn.__code__ == expected and fn.__code__.co_filename == reader_source.__file__ and
                ((bound is fn) if name in ('canonical', 'digest_string') else
                 (getattr(bound, '__self__', None) is admission and getattr(bound, '__func__', None) is fn)),
                'original startup reader method changed: ' + name)
    fn = reader_source.bound_file
    expected = next(c for c in source_code.co_consts if getattr(c, 'co_name', None) == 'bound_file')
    require(isinstance(fn, FunctionType) and fn.__globals__ is vars(reader_source) and
            fn.__code__ == expected and fn.__code__.co_filename == reader_source.__file__,
            'original startup reader global bound_file changed')


def startup_admission_adapter(fitter, guards):
    """Compile only the two pinned inventory reader substitutions, startup-owned."""
    import ast
    import copy
    path = Path(fitter.__file__)
    require(fitter.__spec__ is not None and Path(fitter.__spec__.origin) == path,
            'startup fitter source origin differs')
    sha = FITTER['code']['fit_siglip2_prototype_residual.py']
    raw = bound_file(guards, path, sha).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'startup fitter changed before compilation')
    tree, source_code = ast.parse(raw, filename=str(path)), compile(raw, str(path), 'exec')
    pins = {'authority': ('ad95ed5ddb58e40230e4ef2a942569bddfdf1da727bd00ed6311372ab5ac98d6',
                          'ad95ed5ddb58e40230e4ef2a942569bddfdf1da727bd00ed6311372ab5ac98d6'),
            'admit_historical_linear': ('3637f43dd31f4802fcd68fdf96ee233f1011472a89694a3da7731a0c3295db24',
                                        '22293a825a9f8ad1c24e75abe41464b72c19aaa8e44c65b34ec35c4d247d05ec'),
            'admit_terminal': ('1b13b0de9887d7ce3bb76eda0717d5ca7111ac12a10dc47f33ede08c34e8dc76',
                               'c8722d6eed5469a76dce1a26a19036ff69a54da9f55d8b999f08d3e81b5302b0')}
    dump = lambda node: ast.dump(node, include_attributes=False)
    before = ast.parse('bound_file(guards, path, digest)', mode='eval').body
    after = ast.parse("context['legacy']['admission'].bound_file(guards, path, digest)", mode='eval').body

    class Substitute(ast.NodeTransformer):
        def __init__(self, source, target):
            self.source, self.target, self.count = source, target, 0

        def visit_Call(self, node):
            if dump(node) == dump(self.source):
                self.count += 1
                return ast.copy_location(copy.deepcopy(self.target), node)
            return self.generic_visit(node)

    nodes = {}
    for name, (original_sha, adapted_sha) in pins.items():
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(matches) == 1, 'exact startup fitter definition required: ' + name)
        original = matches[0]
        fn = getattr(fitter, name)
        expected = next(c for c in source_code.co_consts if getattr(c, 'co_name', None) == name)
        require(isinstance(fn, FunctionType) and fn.__globals__ is vars(fitter) and fn.__code__ == expected and
                fn.__code__.co_filename == str(path) and
                hashlib.sha256(dump(original).encode()).hexdigest() == original_sha,
                'actual startup fitter function/body differs: ' + name)
        node = copy.deepcopy(original)
        if name != 'authority':
            substitute = Substitute(before, after)
            node = substitute.visit(node)
            inverse = Substitute(after, before)
            restored = inverse.visit(copy.deepcopy(node))
            require(substitute.count == inverse.count == 1 and dump(restored) == dump(original),
                    'startup adapter changed inventory predicates: ' + name)
        require(hashlib.sha256(dump(node).encode()).hexdigest() == adapted_sha,
                'adapted startup fitter AST differs: ' + name)
        nodes[name] = node
    namespace = dict(vars(fitter))
    exec(compile(ast.fix_missing_locations(ast.Module(body=list(nodes.values()), type_ignores=[])),
                 str(path), 'exec'), namespace)

    def entry(function):
        def admitted(context, *args):
            authenticate_startup_reader(context, fitter)
            return function(context, *args)
        admitted.__startup_ast__ = nodes[function.__name__]
        return admitted

    for name in ('admit_historical_linear', 'admit_terminal'):
        namespace[name] = entry(namespace[name])
    namespace['authority'].__startup_ast__ = nodes['authority']
    return SimpleNamespace(**{name: namespace[name] for name in nodes})


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    oldroot, output = Path(FITTER['root']), args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink() and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (root, oldroot)) and not root.is_relative_to(oldroot) and not oldroot.is_relative_to(root),
            'exclusive separate output/source closures required')
    require(closure(oldroot, FITTER['execution_sha256'], FITTER['code'], guards) == FITTER['code'],
            'authenticated fitter3 required')
    fitter = load_authenticated('_nearest_fitter', oldroot / 'fit_siglip2_prototype_residual.py',
                                FITTER['code']['fit_siglip2_prototype_residual.py'], guards)
    startup = startup_admission_adapter(fitter, guards)
    original = startup.authority(SimpleNamespace(execution_sha256=FITTER['execution_sha256'],
        authority=Path(ACCEPTED['launch']['path']), authority_sha256=ACCEPTED['launch']['sha256'],
        phase='fit', arm='concat', output=output))
    accepted_record = startup.admit_terminal(original, ACCEPTED['terminal'], 'fit', 'concat')
    require(accepted_record['checkpoint'] == ACCEPTED['checkpoint'] and
            accepted_record['terminal_state_sha256'] == ACCEPTED['terminal_state_sha256'], 'accepted endpoint differs')
    for p, h in original['guards'].items():
        require(guards.setdefault(p, h) == h, 'source guard conflict')
    readout = load_authenticated('_nearest_connected', root / 'nearest_ranking_readout.py',
                                 code['nearest_ranking_readout.py'], guards)
    context = {'args': args, 'root': root, 'guards': guards, 'code': code, 'launch': launch,
               'fitter': fitter, 'fit_context': original, 'legacy': original['legacy'], 'old': original['old'],
               'source': original['source'], 'accepted_record': accepted_record, 'readout': readout,
               'phase_seconds': {}, 'terminals': {}, 'terminal_cgroups': {}, 'started': UNIT_STARTED}
    context['required_guards'] = {p: h for p, h in guards.items() if p != str(args.authority)}
    wanted = [] if args.phase == 'cpu' else [('cpu', 'control', launch['selected_cpu'])]
    if args.phase == 'train':
        wanted += [('mechanics', a, launch['selected_mechanics'][a]) for a in ARMS]
    for phase, arm, unit in wanted:
        admit_terminal(context, unit, phase, arm)
    if args.phase == 'train':
        c, a = (context['terminals']['mechanics:' + arm] for arm in ARMS)
        require(c['initial_model_sha256'] == a['initial_model_sha256'] and
                c['initial_raw_unit_packed_sha256'] == a['initial_raw_unit_packed_sha256'] and
                c['identity']['static_sha256'] == a['identity']['static_sha256'], 'matched mechanics initialization differs')
    return context


def select_nearest(scores, targets, original_rows, anchor):
    """Detached scalar miner, shared by native mining and stdlib falsifiers."""
    require(len(scores) == len(targets) == len(original_rows) and 0 <= anchor < len(scores) and
            len(set(original_rows)) == len(original_rows) and
            all(type(r) is int and r >= 0 for r in original_rows) and
            all(math.isfinite(float(s)) for s in scores), 'finite unique original-row miner required')
    positive = [i for i, c in enumerate(targets) if c == targets[anchor] and i != anchor]
    negative = [i for i, c in enumerate(targets) if c != targets[anchor]]
    require(bool(negative), 'wrong identity teacher required')
    best = lambda candidates: min(candidates, key=lambda i: (-float(scores[i]), original_rows[i]))
    return best(positive) if positive else -1, best(negative)


def mine(state, unit, anchors):
    import torch
    with torch.no_grad(), torch.autocast(unit.device.type, enabled=False):
        scores = (unit.detach() @ state['teachers']['V'].T).cpu().tolist()
        selected = [select_nearest(row, state['target_list'], state['row_list'], int(anchor))
                    for row, anchor in zip(scores, anchors, strict=True)]
    positive, negative = zip(*selected, strict=True)
    return torch.tensor(positive, device=unit.device), torch.tensor(negative, device=unit.device)


def loss_terms(state, raw, anchors, full_valid):
    import torch
    from torch.nn import functional as F
    with torch.autocast(raw.device.type, enabled=False):
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item(), 'finite FP32 raw required')
        index = torch.tensor(anchors, device=raw.device)
        mse = (raw - state['teachers']['P'][state['target'][index]]).square().sum() / (64 * state['teachers']['e0'])
        unit = F.normalize(raw, dim=1)
        positive, negative = mine(state, unit, anchors)
        valid = positive >= 0
        require(int(valid.sum()) == sum(state['count_list'][state['target_list'][i]] > 1 for i in anchors),
                'singleton/valid mining differs')
        # Filter before any indexing: -1 is never used as a teacher positive.
        if valid.any().item():
            require(full_valid > 0, 'global B64 valid denominator required')
            hinge = F.relu(.05 + (unit[valid] * state['teachers']['V'][negative[valid]]).sum(1) -
                          (unit[valid] * state['teachers']['V'][positive[valid]]).sum(1))
            rank = hinge.sum() / (full_valid * .05)
            active = int((hinge > 0).sum())
        else:
            rank, active = raw.sum() * 0., 0
        require(torch.isfinite(mse).item() and torch.isfinite(rank).item(), 'nonfinite objective')
    return mse, rank, {'positive': positive.tolist(), 'negative': negative.tolist(),
                       'valid': int(valid.sum()), 'active': active}


def current_cuda_occurrences(value):
    """Gather tensor references in the original traversal; never retain byte facts."""
    import torch
    occurrences = []
    supported = True
    tensor_type = getattr(torch, 'Tensor', ())
    def gather(item, canonical=False):
        nonlocal supported
        if isinstance(item, tensor_type):
            if item.is_cuda:
                occurrences.append((item, item.numel() * item.element_size()))
        elif isinstance(item, dict):
            if type(item) is not dict or any(type(k) not in (str, bytes, int, float, bool, complex, type(None))
                                            for k in item):
                supported = False
                return
            for key in sorted(item, key=repr) if canonical else item:
                gather(key, canonical); gather(item[key], canonical)
        elif isinstance(item, (tuple, list)):
            if type(item) not in (tuple, list):
                supported = False
                return
            for child in item:
                gather(child, canonical)
    try:
        # Do not sort or invoke repr on CPU-only or custom-container paths.
        gather(value)
        if not supported or not occurrences:
            occurrences.clear()
            return occurrences
        occurrences.clear()
        gather(value, canonical=True)
        return occurrences
    except BaseException:
        occurrences.clear()
        raise
    finally:
        gather = None


def current_cuda_bytes(occurrences):
    """Yield current raw bytes; at most one 64MiB group/device is resident."""
    import torch
    limit = 64 * 1024**2
    start = 0
    group, views = [], []
    gpu = host = buffer = snapshot = view = None
    try:
        while start < len(occurrences):
            device = occurrences[start][0].device
            size, end = 0, start
            while end < len(occurrences):
                tensor, count = occurrences[end]
                require(0 <= count <= limit, 'current CUDA tensor exceeds byte group limit')
                if tensor.device != device or size + count > limit:
                    break
                group.append((tensor, count))
                size += count
                end += 1
            try:
                if size:
                    for tensor, count in group:
                        view = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                        require(view.numel() == count, 'current CUDA byte size changed during traversal')
                        views.append(view)
                        view = None
                    gpu = torch.cat(views)
                    require(gpu.numel() == size, 'current CUDA byte group size differs')
                    host = gpu.cpu()  # One blocking D2H copy for this nonempty group.
                    gpu = None
                    views.clear()
                    buffer = memoryview(host.numpy())
                    require(len(buffer) == size, 'current CUDA host byte group size differs')
                else:
                    buffer = memoryview(b'')
                offset = 0
                for tensor, count in group:
                    snapshot = buffer[offset:offset + count]
                    try:
                        yield tensor, snapshot
                    finally:
                        snapshot.release()
                        snapshot = None
                    offset += count
            finally:
                if buffer is not None:
                    buffer.release()
                buffer = gpu = host = view = None
                views.clear()
                group.clear()
            start = end
    finally:
        # Clear also on failed cat/copy, generator close and retained tracebacks.
        if snapshot is not None:
            snapshot.release()
        if buffer is not None:
            buffer.release()
        snapshot = buffer = gpu = host = view = None
        views.clear()
        group.clear()


def current_byte_adapter(original, cuda_raw):
    """Authenticate live provenance and change only the original CUDA raw source."""
    import ast
    import builtins
    import copy
    import _hashlib
    from types import ModuleType
    require(type(original) is ModuleType and original.__spec__ is not None and
            original.__spec__.origin == original.__file__, 'actual original fingerprint source required')
    require(original.hashlib is hashlib and hashlib.sha256 is _hashlib.openssl_sha256,
            'original fingerprint SHA256 global differs')
    path = original.__file__
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() ==
            'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543',
            'original fingerprint source differs')
    tree, code = ast.parse(raw, filename=path), compile(raw, path, 'exec')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint']
    codes = [c for c in code.co_consts if getattr(c, 'co_name', None) == 'fingerprint']
    require(len(nodes) == len(codes) == 1, 'exact original fingerprint definition required')
    fn = original.fingerprint
    require(isinstance(fn, FunctionType) and fn.__globals__ is vars(original) and
            fn.__code__ == codes[0] and fn.__code__.co_filename == path and
            fn.__module__ == original.__name__ and fn.__qualname__ == 'fingerprint' and
            fn.__defaults__ == (None, None) and fn.__kwdefaults__ is None and fn.__closure__ is None and
            fn.__builtins__ is vars(builtins) and fn.__globals__.get('__builtins__') is vars(builtins),
            'actual original fingerprint function/globals differ')
    for name in ('__import__', 'isinstance', 'str', 'len', 'repr', 'sorted', 'type', 'tuple', 'memoryview'):
        require(fn.__globals__.get(name, getattr(builtins, name)) is getattr(builtins, name),
                'original fingerprint builtin global differs: ' + name)
    dump = lambda n: ast.dump(n, include_attributes=False)
    node = nodes[0]
    require(hashlib.sha256(dump(node).encode()).hexdigest() ==
            '3de225c57984a5ee3f292ecddce7a1516154938d5b4d415e692837abda5b2d0f',
            'original fingerprint AST differs')
    before = ast.parse('item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()', mode='eval').body
    after = ast.parse('_nearest_cuda_raw(item) if item.is_cuda else ' + ast.unparse(before), mode='eval').body
    class Substitute(ast.NodeTransformer):
        def __init__(self, source, target):
            self.source, self.target, self.count = source, target, 0

        def visit(self, node):
            if dump(node) == dump(self.source):
                self.count += 1
                return ast.copy_location(copy.deepcopy(self.target), node)
            return super().visit(node)
    substitute = Substitute(before, after)
    adapted = substitute.visit(copy.deepcopy(node))
    inverse = Substitute(after, before)
    restored = inverse.visit(copy.deepcopy(adapted))
    require(substitute.count == inverse.count == 1 and dump(restored) == dump(node),
            'current-byte adapter changed original serializer correspondence')
    require(hashlib.sha256(dump(adapted).encode()).hexdigest() ==
            'bcb97676f8800cd5d7e4048dde059d2d3a47d80673b8556f85616383701f3f12',
            'current-byte fingerprint AST differs')
    namespace = dict(vars(original), _nearest_cuda_raw=cuda_raw)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted], type_ignores=[])), path, 'exec'), namespace)
    namespace['fingerprint'].__current_byte_ast__ = adapted
    return namespace['fingerprint']


def fingerprint(context, value, **kwargs):
    # No stat, version, or tensor-hash cache: every call reads current bytes.
    original = context['legacy']['original']
    if kwargs:
        return original.fingerprint(value, **kwargs)
    occurrences = current_cuda_occurrences(value)
    if not occurrences:
        return original.fingerprint(value)
    chunks = current_cuda_bytes(occurrences)
    def cuda_raw(item):
        tensor, raw = next(chunks)
        require(tensor is item, 'current CUDA occurrence order differs')
        return raw
    try:
        adapted = current_byte_adapter(original, cuda_raw)
        if any(size > 64 * 1024**2 for _, size in occurrences):
            return original.fingerprint(value)
        result = adapted(value)
        require(next(chunks, None) is None, 'current CUDA occurrences not exhausted')
        return result
    finally:
        chunks.close()
        occurrences.clear()


def clone(context, value, device='cpu'):
    return context['old'].clone_tree(value, device)


def prepare_native(context):
    """Strictly admit COMPLETE accepted payload before constructing any vision."""
    import torch
    fitter, original, legacy = context['fitter'], context['fit_context'], context['legacy']
    fitter.prepare_original(original)
    original['flags'] = legacy['flags']
    path = bound_file(context['guards'], **{'path': ACCEPTED['checkpoint']['path'],
                                           'expected': ACCEPTED['checkpoint']['sha256']})
    with timed(context, 'accepted_payload_admission'):
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        with path.open('rb') as stream:
            pages = legacy['original'].CheckpointPages(stream)
            ident = context['accepted_record']['identity']
            fitter.check_payload(original, disk, ident)
            require(fingerprint(context, disk, consumed=pages.consume) == ACCEPTED['terminal_state_sha256'],
                    'complete accepted typed payload differs')
            initial = {k: clone(context, disk[k]) for k in ('encoder', 'config', 'buffers', 'head', 'classifier',
                        'A', 'means', 'partition', 'original_rows', 'target')}
            initial['processor'] = clone(context, disk['encoder']['export_runtime']['processor'])
            # Reconstruct exactly the accepted concat CPU output; never use old prototypes.
            with timed(context, 'target_construction'):
                head = legacy['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).train()
                A = initial['A'].detach()
                oracle_A = torch.nn.Parameter(A.clone(), requires_grad=True)
                with torch.no_grad(), torch.autocast('cpu', enabled=False):
                    T = context['readout'].raw_features(disk['features'], head, A, initial['means'], legacy['quadratic'])
                    oracle = fitter.prepare_readout(original).raw_features(
                        disk['features'], head, oracle_A, initial['means'], 'concat', legacy['quadratic'])
                    require(torch.equal(T, oracle) and fingerprint(context, context['old'].packed_outputs(legacy, T)) ==
                            ident['output_witness_sha256'], 'accepted full TRAIN concat forward/packing differs')
                    target = initial['target']
                    counts = torch.bincount(target, minlength=1008)
                    require(counts.shape == (1008,) and (counts > 0).all().item() and
                            counts.sum().item() == 6355 and (counts == 1).sum().item() == 12,
                            'TRAIN6355/1008/12singleton inventory differs')
                    P = torch.zeros((1008, 128), dtype=torch.float32)
                    P.index_add_(0, target, T)
                    P /= counts[:, None]
                    e0 = (T - P[target]).square().sum(1).mean()
                    require(torch.isfinite(e0).item() and e0.item() > 0, 'positive finite coordinate-summed e0 required')
                    from torch.nn import functional as F
                    initial['teachers'] = {'T': T.detach(), 'V': F.normalize(T, dim=1).detach(),
                                           'P': P.detach(), 'counts': counts, 'e0': e0.detach()}
                del head, oracle_A, oracle, A, T, P, e0
            schedule, _ = legacy['genuine'].schedule_and_masks(target.tolist(), SEED)
            schedule = torch.from_numpy(schedule[:32].copy())
            require(torch.equal(schedule, disk['warm_payload']['schedules'][str(SEED)][:32]),
                    'original seed179061 first32 exact schedule differs')
            initial['schedule'] = schedule
        del disk, pages
        gc.collect()
    initial['provenance'] = {'accepted': ACCEPTED, 'fitter': FITTER, 'encoder': initial.pop('encoder')}
    with timed(context, 'target_construction'):
        panel_state = {**initial, 'target_list': initial['target'].tolist(),
                       'row_list': initial['original_rows'].tolist()}
        triples = []
        order = sorted(range(6355), key=lambda i: panel_state['row_list'][i])
        for offset in range(0, len(order), 64):
            anchors = order[offset:offset + 64]
            positive, negative = mine(panel_state, initial['teachers']['V'][anchors], anchors)
            for i, p, n in zip(anchors, positive.tolist(), negative.tolist(), strict=True):
                if p >= 0:
                    margin = float((initial['teachers']['V'][i] * (initial['teachers']['V'][p] -
                                   initial['teachers']['V'][n])).sum())
                    if margin < .05 and len(triples) < 128:
                        triples.append([i, p, n])
            if len(triples) == 128:
                break
        initial['panel'] = torch.tensor(triples, dtype=torch.int64).reshape(-1, 3)
        del panel_state
    context['initial'] = initial
    context['flags'] = legacy['flags']
    # Every original preparation check is reused unchanged, with procedure-owned context.
    context['old'].audit_origins(legacy)
    context['initial_static_sha256'] = fingerprint(context, initial)
    return initial


def configure_roles(model):
    model.requires_grad_(False)
    pairs = dict(model.named_parameters())
    require(len(pairs) == 448 and model.state_dict().keys() == pairs.keys(), 'complete448 native state required')
    for name, shape in zip(NAMES, SHAPES, strict=True):
        require(name in pairs and tuple(pairs[name].shape) == shape and str(pairs[name].dtype) == 'torch.float32',
                'authorized lastMLP layout differs')
        pairs[name].requires_grad_(True)
    model.eval()
    require([n for n, p in model.named_parameters() if p.requires_grad] == NAMES and
            sum(p.numel() for p in pairs.values() if p.requires_grad) == 9921872, 'exact four roles required')
    return [(n, pairs[n]) for n in NAMES]


def optimizer_state(model):
    import torch
    pairs = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    require([n for n, _ in pairs] == NAMES, 'four-member optimizer order required')
    optimizer = torch.optim.AdamW([p for _, p in pairs], **ADAM)
    defaults = dict(ADAM)
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and len(optimizer.param_groups) == 1 and
            {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'} == defaults and
            all(a is b for a, (_, b) in zip(optimizer.param_groups[0]['params'], pairs, strict=True)),
            'fresh exact AdamW defaults/membership differs')
    return pairs, optimizer


def require_no_model(context):
    live = context.get('live_model')
    require(live is None or live() is None, 'one live vision model required; release before reload')


@contextmanager
def restore_memory_snapshot(context, phase):
    """Flushed raw diagnostics only; never reset or replace cgroup admission."""
    def snapshot(boundary):
        cgroup = {}
        try:
            unified = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                       if line.startswith('0::')]
            require(len(unified) == 1 and unified[0] != '/', 'enclosing cgroup v2 unit unavailable')
            root = Path('/sys/fs/cgroup') / unified[0].lstrip('/')
            cgroup['path'] = str(root)
            cgroup['values'] = {name: (root / name).read_text().strip() for name in
                               ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current',
                                'memory.swap.peak', 'memory.swap.max', 'memory.events', 'memory.stat')}
        except (OSError, ValueError) as error:
            cgroup['error'] = str(error)
        print(json.dumps({'event': 'NEAREST_RESTORE_MEMORY', 'phase': phase, 'boundary': boundary,
                          'seconds': time.perf_counter() - context['started'],
                          'invocation_id': os.environ.get('INVOCATION_ID'), 'cgroup': cgroup},
                         sort_keys=True, allow_nan=False), flush=True)
    snapshot('begin')
    try:
        yield
    finally:
        snapshot('end')


def fresh(context, arm, device, initial=None, *, defer_vision_transfer=False):
    import torch
    require_no_model(context)
    require(arm in ARMS and device in ('cpu', 'cuda'), 'fresh role required')
    initial = context['initial'] if initial is None else initial
    legacy, source = context['legacy'], context['legacy']['source_driver']
    with timed(context, 'source_construction'), restore_memory_snapshot(context, 'source_construction'):
        model, processor, roles = source.fresh_source(legacy['prior'])
        require(source.model_facts(model, processor, roles, legacy['selected']['packages']) ==
                initial['provenance']['encoder']['source_proof']['runtime'], 'actual full original source runtime differs')
        context['live_model'] = weakref.ref(model)
        configure_roles(model)
        if not defer_vision_transfer:
            with restore_memory_snapshot(context, 'vision_transfer'):
                model.to(device)
        head = legacy['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).to(device).train()
        pairs, optimizer = optimizer_state(model)
        state = {'model': model, 'processor_object': processor, 'head_object': head, 'params': pairs,
                 'optimizer_object': optimizer, 'scaler_object': torch.amp.GradScaler(device, init_scale=128),
                 'counter': 0, 'arm': arm, 'device': device,
                 **{k: clone(context, initial[k], device if k in ('classifier', 'A', 'means', 'target', 'teachers') else 'cpu')
                    for k in STATIC_KEYS}}
        state['target_list'], state['row_list'] = state['target'].tolist(), state['original_rows'].tolist()
        state['count_list'] = state['teachers']['counts'].tolist()
        state['head'] = dict(head.state_dict())
        require(source.numerical_flags() == context['flags'], 'source construction flags differ')
    return state


def current_buffers(state):
    return {n: v.detach() for n, v in state['model'].named_buffers()}


def static_tree(state):
    return {k: dict(state['head_object'].state_dict()) if k == 'head' else
            state['model'].config.to_dict() if k == 'config' else
            current_buffers(state) if k == 'buffers' else state[k] for k in STATIC_KEYS}


def frozen_vision(state):
    return {n: p.detach() for n, p in state['model'].named_parameters() if n not in NAMES}


def identity(context, state):
    import torch
    return {'method': method(context['launch']), 'source': context['source'], 'arm': state['arm'],
            'device': state['device'], 'seed': SEED, 'parameter_names': NAMES,
            'parameter_shapes': [list(s) for s in SHAPES], 'numerical_flags': context['flags'],
            'static_sha256': fingerprint(context, static_tree(state)),
            'frozen_vision_sha256': fingerprint(context, frozen_vision(state)),
            'initial_trainable_sha256': {n: fingerprint(context, p.detach()) for n, p in state['params']},
            'initial_model_sha256': fingerprint(context, state['model'].state_dict()),
            'optimizer_defaults': state['optimizer_object'].defaults,
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer_object'].param_groups],
            'initial_scaler': state['scaler_object'].state_dict(),
            'runtime': runtime(context, state),
            'initial_cpu_rng_sha256': fingerprint(context, torch.random.get_rng_state()),
            'initial_cuda_rng_sha256': fingerprint(context, torch.cuda.get_rng_state_all()) if state['device'] == 'cuda' else None}


def runtime(context, state):
    legacy = context['legacy']
    return legacy['original'].runtime({'source': legacy['source_driver'],
        'initialized': {'packages': legacy['selected']['packages']}},
        {'model': state['model'], 'processor': state['processor_object']})


def payload(context, state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'source': context['source'],
            'vision': dict(state['model'].state_dict()), **static_tree(state),
            'optimizer': state['optimizer_object'].state_dict(), 'scaler': state['scaler_object'].state_dict(),
            'counter': state['counter'], 'cpu_rng': torch.random.get_rng_state().clone(),
            'cuda_rng': [v.clone() for v in torch.cuda.get_rng_state_all()] if state['device'] == 'cuda' else [],
            'numerical_flags': context['legacy']['source_driver'].numerical_flags()}


def check_optimizer(saved, ident, step):
    import torch
    opt = saved['optimizer']
    require(opt.keys() == {'state', 'param_groups'} and len(opt['param_groups']) == 1 and
            opt['param_groups'][0]['params'] == list(range(4)) and
            {k: v for k, v in opt['param_groups'][0].items() if k != 'params'} == ident['optimizer_groups'][0] and
            opt['state'].keys() == (set(range(4)) if step else set()), 'four named moments/group ownership differs')
    for i, shape in enumerate(SHAPES):
        if not step:
            break
        member = opt['state'][i]
        require(member.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and member['step'].numel() == 1 and
                float(member['step']) == step and member['step'].dtype == torch.float32 and
                member['step'].device.type == 'cpu', 'AdamW step scalar differs')
        for k in ('exp_avg', 'exp_avg_sq'):
            value = member[k]
            require(tuple(value.shape) == shape and value.dtype == torch.float32 and
                    not value.requires_grad and value.grad_fn is None and torch.isfinite(value).all().item(),
                    'FP32 finite named moment shape/role differs')
    require(saved['scaler'] == (dict(ident['initial_scaler'], _growth_tracker=step) if
            ident['device'] == 'cuda' else ident['initial_scaler']) and
            saved['numerical_flags'] == ident['numerical_flags'], 'scaler128/counters/numerical flags differ')


def check_payload(context, saved, ident, step):
    import torch
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            saved['source'] == context['source'] == ident['source'] and ident['method'] == method(context['launch']) and
            ident['arm'] in ARMS and type(saved['counter']) is int and saved['counter'] == step and 0 <= step <= 32 and
            ident['parameter_names'] == NAMES and ident['parameter_shapes'] == [list(s) for s in SHAPES],
            'complete updated endpoint identity/counter differs')
    provenance = saved['provenance']
    require(provenance['accepted'] == ACCEPTED and provenance['fitter'] == FITTER, 'original provenance differs')
    context['old'].check_encoder(provenance['encoder'])
    inventory = provenance['encoder']['inventory']
    require(saved['vision'].keys() == {r['name'] for r in inventory} and len(saved['vision']) == 448,
            'complete updated vision448 required')
    for row in inventory:
        value = saved['vision'][row['name']]
        require(list(value.shape) == row['shape'] and value.dtype == torch.float32 and not value.requires_grad and
                value.grad_fn is None and torch.isfinite(value).all().item(), 'updated vision layout/finite/role differs')
    require(fingerprint(context, {k: saved[k] for k in STATIC_KEYS}) == ident['static_sha256'] and
            fingerprint(context, {n: p for n, p in saved['vision'].items() if n not in NAMES}) ==
            ident['frozen_vision_sha256'], 'frozen444/config/buffers/head/teachers/data current bytes changed')
    require(saved['schedule'].shape == (32, 64) and saved['schedule'].dtype == torch.int64 and
            torch.equal(saved['schedule'], context['initial']['schedule']) and
            torch.equal(saved['original_rows'].cpu(), context['initial']['original_rows']) and
            torch.equal(saved['target'].cpu(), context['initial']['target']), 'original rows/targets/first32 schedule differs')
    for n in NAMES:
        unchanged = fingerprint(context, saved['vision'][n]) == ident['initial_trainable_sha256'][n]
        require(unchanged if step == 0 else not unchanged, 'actual four updates/original vision substitution differs')
    require(saved['cpu_rng'].dtype == torch.uint8 and saved['cpu_rng'].ndim == 1 and
            fingerprint(context, saved['cpu_rng']) == ident['initial_cpu_rng_sha256'] and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0) and
            all(v.dtype == torch.uint8 and v.ndim == 1 for v in saved['cuda_rng']) and
            (ident['device'] != 'cuda' or fingerprint(context, saved['cuda_rng']) == ident['initial_cuda_rng_sha256']),
            'complete CPU/CUDA RNG differs')
    check_optimizer(saved, ident, step)
    context['old'].finite_tree(saved)


def integrity(context, state, ident):
    import torch
    model, head = state['model'], state['head_object']
    helper_guard(context)
    context['legacy']['quadratic']._check_base(head, state['A'].device)
    require([n for n, p in model.named_parameters() if p.requires_grad] == NAMES and
            all(p.dtype == torch.float32 and p.grad is None for p in model.parameters()) and
            all(not p.requires_grad and p.grad is None for p in head.parameters()) and
            not state['A'].requires_grad and state['A'].grad is None and
            all(not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in model.modules()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in head.modules()), 'native modes/hooks/frozen gradient/role differs')
    params = state['params']
    require([n for n, _ in params] == NAMES and
            all(a is b for a, (_, b) in zip(state['optimizer_object'].param_groups[0]['params'], params, strict=True)) and
            state['optimizer_object'].defaults == ident['optimizer_defaults'], 'active optimizer membership/defaults differs')
    source = context['legacy']['source_driver']
    require(source.numerical_flags() == context['flags'] == ident['numerical_flags'] and
            runtime(context, state) == ident['runtime'] and
            json.loads(state['processor_object'].to_json_string()) == state['processor']['config'] and
            state['processor_object'].backend == state['processor']['backend'], 'live processor/numerics changed')
    source.cgroup_memory()  # Whole-unit peak/swap/events, including construction; never reset.
    require(time.perf_counter() - context['started'] < policy(context['args'].phase)['seconds'], 'whole-unit deadline exceeded')
    # Current complete bytes at every boundary, including .data bypass mutations.
    check_payload(context, payload(context, state, ident), ident, state['counter'])
    if state['device'] == 'cuda':
        require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'whole-unit CUDA peak exceeded')


def release(context, state):
    state.clear()
    gc.collect()
    require_no_model(context)
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


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
        del saved
    return sha, digest


def restore(context, path, sha, digest, ident, step):
    """New source model; strict updated vision, saved teachers, fresh optimizer."""
    import torch
    require_no_model(context)
    with timed(context, 'independent_reload'):
        path = bound_file(context['guards'], path, sha)
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        with path.open('rb') as stream:
            pages = context['legacy']['original'].CheckpointPages(stream)
            with restore_memory_snapshot(context, 'payload_validation'):
                check_payload(context, disk, ident, step)
            with restore_memory_snapshot(context, 'payload_hash'):
                require(fingerprint(context, disk, consumed=pages.consume) == digest, 'complete serialized updated state differs')
            # The source factory is checked independently, then trained state is loaded strictly.
            state = fresh(context, ident['arm'], ident['device'], initial=disk,
                          defer_vision_transfer=ident['device'] == 'cuda')
            with restore_memory_snapshot(context, 'identity'):
                require(identity(context, state) == ident, 'independent original initialization differs')
            with restore_memory_snapshot(context, 'strict_load'):
                context['legacy']['original'].load_vision(state['model'], disk['vision'], pages)
                for name, value in state['model'].named_buffers():
                    with torch.no_grad():
                        value.copy_(disk['buffers'][name].to(value.device))
                    pages.consume(disk['buffers'][name])
            if state['device'] == 'cuda':
                # Transfer may replace Parameters. Drop CPU bindings before moving
                # vision, then bind fresh Adam to the transferred four Parameters.
                del state['optimizer_object'], state['params']
                with restore_memory_snapshot(context, 'vision_transfer'):
                    state['model'].to(state['device'])
                    state['params'], state['optimizer_object'] = optimizer_state(state['model'])
            # Only named moments move to CUDA; Adam step scalars stay independently owned CPU.
            with restore_memory_snapshot(context, 'moments_scaler_rng'):
                optimizer = {'state': {}, 'param_groups': clone(context, disk['optimizer']['param_groups'])}
                for i in disk['optimizer']['state']:
                    optimizer['state'][i] = {k: pages.copy(v, 'cpu' if k == 'step' else state['device'])
                                             for k, v in disk['optimizer']['state'][i].items()}
                state['optimizer_object'].load_state_dict(optimizer)
                del optimizer
                state['scaler_object'].load_state_dict(disk['scaler'])
                state['counter'] = step
                torch.random.set_rng_state(disk['cpu_rng'].clone())
                if state['device'] == 'cuda':
                    torch.cuda.set_rng_state_all([v.clone() for v in disk['cuda_rng']])
        with restore_memory_snapshot(context, 'archive_deletion'):
            del disk, pages
            gc.collect()
        integrity(context, state, ident)
        require(fingerprint(context, payload(context, state, ident)) == digest, 'strict independent all-state reload differs')
    return state


def canonical_row(context, state, ordinal):
    legacy = context['legacy']
    manifest, fit = legacy['selected']['genuine']['selected'], legacy['prior']['fit']
    require(type(ordinal) is int and 0 <= ordinal < 6355, 'TRAIN-only ordinal required')
    original = manifest['original_rows'][ordinal]
    row, path = manifest['rows'][ordinal], Path(manifest['resolved_paths'][ordinal])
    target = manifest['targets'][ordinal]
    require(original == int(state['original_rows'][ordinal]) and row == fit['rows'][original] and
            path == legacy['prior']['all_images'][original] and
            (Path(fit['dataset_root']) / row['relative_path']).resolve() == path and
            path.is_relative_to(Path(fit['dataset_root'])) and target == int(state['target'][ordinal]) and
            state['partition']['panels']['train']['original_class_ids'][target] == fit['targets'][original] and
            row['product'] == fit['class_names'][fit['targets'][original]], 'TRAIN original image/product mapping differs')
    return row, path, {'train_local': ordinal, 'fit_ordinal': original, 'train_row': row['train_row'],
        'path': str(path), 'relative_path': row['relative_path'], 'image_sha256': row['image_sha256'],
        'target': target, 'original_target': fit['targets'][original], 'product': row['product']}


def canonical_pixels(context, state, batch):
    import torch
    from PIL import Image
    require(0 < len(batch) <= 64 and all(type(i) is int and 0 <= i < 6355 for i in batch), 'TRAIN-only pixels required')
    images, mappings, rgb = [], [], hashlib.sha256()
    rng = torch.random.get_rng_state().clone()
    try:
        for ordinal in batch:
            row, path, fact = canonical_row(context, state, ordinal)
            bound_file(context['guards'], path, row['image_sha256'])
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            images.append(image)
            mappings.append(fact)
            rgb.update(str(image.size).encode())
            rgb.update(image.tobytes())
        pixels = state['processor_object'](images=images, return_tensors='pt')['pixel_values']
    finally:
        for image in images:
            image.close()
    require(pixels.dtype == torch.float32 and pixels.shape == (len(batch), 3, 256, 256) and
            not pixels.requires_grad and pixels.grad_fn is None and torch.isfinite(pixels).all().item() and
            torch.equal(rng, torch.random.get_rng_state()), 'canonical pixel/RNG contract differs')
    return pixels, rgb.hexdigest(), fingerprint(context, mappings)


def native_raw(context, state, pixels, oracle=False):
    import torch
    from torch.nn import functional as F
    device = state['device']
    with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
        pooled = state['model'](pixel_values=pixels.to(device)).pooler_output
    with torch.autocast(device, enabled=False):
        features = F.normalize(pooled.float(), dim=1)
        raw = context['readout'].raw_features(features, state['head_object'], state['A'],
                                              state['means'], context['legacy']['quadratic'])
        if oracle:
            # The active A remains frozen and outside AdamW throughout.
            isolated_A = torch.nn.Parameter(state['A'].detach().clone(), requires_grad=True)
            reference = context['fitter'].prepare_readout(context['fit_context']).raw_features(
                features.detach(), state['head_object'], isolated_A, state['means'], 'concat', context['legacy']['quadratic'])
            require(torch.equal(raw.detach(), reference.detach()), 'same-role exact forward oracle differs')
            del isolated_A, reference
    require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item() and
            (raw.norm(dim=1) > 0).all().item(), 'finite nonzero native raw128 required')
    return raw


def packed_outputs(context, raw):
    return context['old'].packed_outputs(context['legacy'], raw)


def calibration(context, state, oracle=False):
    import torch
    batch = state['schedule'][0].tolist()[:2 if state['device'] == 'cpu' else 16]
    with timed(context, 'calibration'):
        pixels, rgb, rows = canonical_pixels(context, state, batch)
        # Connected training forward and a separate native inference forward.
        raw = native_raw(context, state, pixels, oracle=oracle)
        require(raw.requires_grad and raw.grad_fn is not None, 'native downstream input graph detached')
        connected = raw.detach().clone()
        del raw
        with torch.no_grad():
            inference = native_raw(context, state, pixels, oracle=oracle)
        require(torch.equal(connected, inference), 'complete native training/inference path parity differs')
        outputs = packed_outputs(context, inference)
        drift = (inference - state['teachers']['T'][batch]).detach().float()
        from torch.nn import functional as F
        unit = F.normalize(inference, dim=1)
        positive, negative = mine(state, unit, batch)
        valid = positive >= 0
        # Same fixed selected roles expose teacher/native margin drift; no quality score.
        margin_drift, native_active, teacher_active = 0., 0, 0
        if valid.any().item():
            teacher = state['teachers']['V'][batch]
            difference = state['teachers']['V'][positive[valid]] - state['teachers']['V'][negative[valid]]
            native_margin = (unit[valid] * difference).sum(1)
            teacher_margin = (teacher[valid] * difference).sum(1)
            margin_drift = float((native_margin - teacher_margin).abs().max())
            native_active, teacher_active = int((native_margin < .05).sum()), int((teacher_margin < .05).sum())
        return {**outputs, 'batch': batch, 'rgb_sha256': rgb, 'row_mapping_sha256': rows,
                'pixels_sha256': fingerprint(context, pixels),
                'teacher_native_drift_max_abs': float(drift.abs().max()),
                'teacher_native_drift_l2': float(drift.double().norm()),
                'teacher_native_margin_drift_max_abs': margin_drift,
                'teacher_native_active_hinges': native_active, 'teacher_native_teacher_hinges': teacher_active,
                'arithmetic_role': 'CPU FP32 native' if state['device'] == 'cpu' else 'CUDA FP16 micro16 native'}


def inference_payload(context, state, ident):
    static = static_tree(state)
    result = {'schema': INFERENCE_SCHEMA, 'identity': ident, 'source': context['source'],
              'vision': dict(state['model'].state_dict()), 'numerical_flags': context['flags'],
              **{k: static[k] for k in ('config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means')}}
    result['vision_sha256'] = fingerprint(context, result['vision'])
    result['fixed_sha256'] = fingerprint(context, {k: result[k] for k in
        ('source', 'config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means', 'numerical_flags')})
    return result


def save_inference(context, state, ident, path):
    import torch
    with timed(context, 'inference_save'):
        integrity(context, state, ident)
        saved = inference_payload(context, state, ident)
        digest = fingerprint(context, saved)
        with context['legacy']['extract'].exclusive(path) as stream:
            writer = context['legacy']['original'].CheckpointWriter(stream)
            torch.save(saved, writer)
            writer.flush()
        sha = context['legacy']['extract'].sha(path)
        bound_file(context['guards'], path, sha)
        del saved
    return sha, digest


def check_inference(context, saved, digest):
    require(saved.keys() == INFERENCE_KEYS and saved['schema'] == INFERENCE_SCHEMA and
            saved['source'] == context['source'] and fingerprint(context, saved) == digest and
            fingerprint(context, saved['vision']) == saved['vision_sha256'] and
            fingerprint(context, {k: saved[k] for k in
                ('source', 'config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means', 'numerical_flags')}) ==
            saved['fixed_sha256'], 'updated inference artifact authentication differs')
    context['old'].finite_tree(saved)


def load_inference(context, path, sha, digest, device):
    """No TRAIN cache, teachers, labels, schedule or optimizer read by this API.

    Caller admits constructor/processor/head/readout/packing environment and
    endpoint FILE+typed digest. legacy holds those helpers, not required data.
    """
    import torch
    require_no_model(context)
    legacy, source = context['legacy'], context['legacy']['source_driver']
    with timed(context, 'inference_reload'):
        path = bound_file(context['guards'], path, sha)
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        check_inference(context, disk, digest)
        require(source.numerical_flags() == disk['numerical_flags'], 'inference flags differ')
        with path.open('rb') as stream:
            pages = legacy['original'].CheckpointPages(stream)
            # Constructor only: loading original source weights here is unnecessary.
            model = source.construct(disk['config'], legacy['prior'])
            context['live_model'] = weakref.ref(model)
            legacy['original'].load_vision(model, disk['vision'], pages)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(disk['buffers'][name])
                    pages.consume(disk['buffers'][name])
            model.requires_grad_(False).eval().to(device)
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(
                legacy['prior']['entry']['input']['preprocessor']['path'], local_files_only=True, backend='torchvision')
            head = legacy['selected']['cached'].head_from('control', tensors=clone(context, disk['head'])).requires_grad_(False).to(device).train()
            state = {'model': model, 'processor_object': processor, 'head_object': head, 'device': device,
                     **{k: clone(context, disk[k], device if k in ('A', 'means', 'classifier') else 'cpu') for k in
                        ('config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means')}}
            require(fingerprint(context, model.state_dict()) == disk['vision_sha256'] and
                    fingerprint(context, current_buffers(state)) == fingerprint(context, disk['buffers']) and
                    json.loads(processor.to_json_string()) == disk['processor']['config'],
                    'strict updated inference vision/buffers/processor differs')
        del disk, pages
        gc.collect()
    return state


def gradient_norms(pairs):
    import torch
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for _, p in pairs), 'four finite FP32 gradients required')
    return {n: float(p.grad.double().norm()) for n, p in pairs}


def update(context, state, ident, step):
    import torch
    device = state['device']
    require(device == 'cuda' and state['counter'] == step - 1 and 1 <= step <= 32, 'native CUDA fixed update required')
    torch.cuda.synchronize()
    tick = time.perf_counter()
    batch = state['schedule'][step - 1].tolist()
    full_valid = sum(state['count_list'][state['target_list'][i]] > 1 for i in batch)
    optimizer, scaler = state['optimizer_object'], state['scaler_object']
    optimizer.zero_grad(set_to_none=True)
    before = {n: fingerprint(context, p.detach()) for n, p in state['params']}
    ranking_grad = [torch.zeros_like(p) for _, p in state['params']] if step == 1 else None
    mined, images, mse_sum, rank_sum, active = [], [], 0., 0., 0
    for offset in range(0, 64, 16):
        anchors = batch[offset:offset + 16]
        pixels, rgb, rows = canonical_pixels(context, state, anchors)
        images.append({'batch': anchors, 'rgb_sha256': rgb, 'row_mapping_sha256': rows,
                       'pixels_sha256': fingerprint(context, pixels)})
        raw = native_raw(context, state, pixels, oracle=step == 1)
        require(raw.requires_grad and raw.grad_fn is not None, 'encoder/head graph detached')
        mse, rank, selected = loss_terms(state, raw, anchors, full_valid)
        if step == 1:
            gradients = torch.autograd.grad(rank, [p for _, p in state['params']], retain_graph=True)
            require(all(g.dtype == torch.float32 and torch.isfinite(g).all().item() for g in gradients),
                    'finite separate ranking gradients required')
            for total, grad in zip(ranking_grad, gradients, strict=True):
                total.add_(grad.detach())
            del gradients
        # Both arms use the identical miner/diagnostics; only this addition differs.
        loss = mse + rank if state['arm'] == 'candidate' else mse
        scaler.scale(loss).backward()
        mined.append(selected)
        mse_sum += float(mse.detach())
        rank_sum += float(rank.detach())
        active += selected['active']
        del raw, pixels, mse, rank, loss
    scaler.unscale_(optimizer)
    gradients = gradient_norms(state['params'])
    require(all(v > 0 for v in gradients.values()), 'all four nonzero native gradients required')
    rank_gradient_norms = ({n: float(g.double().norm()) for (n, _), g in zip(state['params'], ranking_grad, strict=True)}
                           if ranking_grad is not None else None)
    if step == 1:
        require(active > 0 and any(v > 0 for v in rank_gradient_norms.values()),
                'first-update active hinge/nonzero separate ranking-gradient witness required')
    del ranking_grad
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    require(scaler.get_scale() == scale == 128, 'nonfinite/skipped/rescaled update forbidden')
    state['counter'] = step
    after = {n: fingerprint(context, p.detach()) for n, p in state['params']}
    require(all(before[n] != after[n] for n in NAMES), 'each of four actual tensor updates required')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    core_seconds = time.perf_counter() - tick
    with timed(context, 'update_integrity'):
        integrity(context, state, ident)
        digest = fingerprint(context, payload(context, state, ident))
    row = {'step': step, 'batch': batch, 'images': images, 'mined': mined, 'full_valid': full_valid,
           'mse': mse_sum, 'rank': rank_sum, 'loss': mse_sum + (rank_sum if state['arm'] == 'candidate' else 0.),
           'active_hinges': active, 'gradient_norms': gradients, 'ranking_gradient_norms': rank_gradient_norms,
           'all_four_updated': True, 'four_before_sha256': before, 'four_after_sha256': after,
           'preclip_norm': float(norm), 'scale': scaler.get_scale(), 'state_sha256': digest,
           'core_seconds': core_seconds, 'seconds': time.perf_counter() - tick}
    print(json.dumps({'event': 'NEAREST_UPDATE', **row}, sort_keys=True, allow_nan=False), flush=True)
    return row


def diagnostic(row):
    return {k: v for k, v in row.items() if k not in ('seconds', 'core_seconds')}


def check_steps(rows, start, count):
    require(len(rows) == count and [r['step'] for r in rows] == list(range(start, start + count)), 'complete updates required')
    for row in rows:
        require(len(row['batch']) == 64 and all(type(i) is int and 0 <= i < 6355 for i in row['batch']) and
                len(row['images']) == len(row['mined']) == 4 and row['all_four_updated'] is True and
                row['gradient_norms'].keys() == set(NAMES) and all(v > 0 and math.isfinite(v) for v in row['gradient_norms'].values()) and
                row['scale'] == 128 and 0 < row['core_seconds'] <= row['seconds'] and
                math.isfinite(row['mse']) and math.isfinite(row['rank']) and row['mse'] >= 0 and row['rank'] >= 0,
                'finite complete native diagnostics required')
        if row['step'] == 1:
            require(row['active_hinges'] > 0 and row['ranking_gradient_norms'].keys() == set(NAMES) and
                    any(v > 0 for v in row['ranking_gradient_norms'].values()), 'separate ranking contribution absent')


def rejected(call, message):
    try:
        call()
    except (ValueError, TypeError, KeyError, RuntimeError):
        return True
    raise ValueError(message)


def tamper_witness(context, state, ident):
    import torch
    tensors = [next(p for n, p in state['model'].named_parameters() if n not in NAMES),
               state['A'], state['head_object'].primary.weight, state['teachers']['V'],
               dict(state['model'].named_buffers())['embeddings.position_ids']]
    for tensor in tensors:
        data = tensor.data.reshape(-1)
        original, version = data[0].item(), tensor._version
        data[0] = 1 if original == 0 else 0
        require(tensor._version == version and data[0].item() != original, 'actual version-bypass mutation required')
        try:
            rejected(lambda: integrity(context, state, ident), 'current-byte .data tamper accepted')
        finally:
            data[0] = original
        integrity(context, state, ident)
    return True


def source_substitution_witness(context, state, ident):
    import torch
    require(state['counter'] > 0, 'trained endpoint required for original-vision substitution witness')
    saved = payload(context, state, ident)
    fact = saved['provenance']['encoder']['checkpoint']
    path = bound_file(context['guards'], fact['path'], fact['sha256'])
    original = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = context['legacy']['original'].CheckpointPages(stream)
        # Same frozen444: replacing only the four owned updated tensors is full original substitution.
        substituted = {**saved, 'vision': {**saved['vision'], **{n: original['vision'][n] for n in NAMES}}}
        rejected(lambda: check_payload(context, substituted, ident, state['counter']),
                 'original vision substituted for trained state')
        for n in NAMES:
            pages.consume(original['vision'][n])
    del original, saved, substituted, pages
    gc.collect()
    return True


def cpu_witnesses(context):
    import torch
    args = context['args']
    require(not torch.cuda.is_initialized(), 'CPU CUDA hidden required')
    state = fresh(context, 'control', 'cpu')
    ident = identity(context, state)
    integrity(context, state, ident)
    witness = calibration(context, state, oracle=True)
    witness_sha = fingerprint(context, witness)
    native_loss_fixtures(context, state)
    tamper_witness(context, state, ident)
    parameter = state['params'][0][1]
    parameter.requires_grad_(False)
    try:
        rejected(lambda: integrity(context, state, ident), 'native role mutation accepted')
    finally:
        parameter.requires_grad_(True)
    integrity(context, state, ident)
    saved = payload(context, state, ident)
    for key, value in (('schema', 'wrong'), ('counter', 1), ('source', {}), ('vision', {})):
        rejected(lambda k=key, v=value: check_payload(context, {**saved, k: v}, ident, 0), 'malformed state accepted')
    del saved
    checkpoint = args.output / 'initializer.pt'
    sha, digest = save(context, state, ident, checkpoint)
    release(context, state)
    state = restore(context, checkpoint, sha, digest, ident, 0)
    require(fingerprint(context, calibration(context, state, oracle=True)) == witness_sha,
            'CPU strict fresh updated-vision/raw-unit-packed reload differs')
    release(context, state)
    state = fresh(context, 'candidate', 'cpu')
    candidate_ident = identity(context, state)
    require(candidate_ident['static_sha256'] == ident['static_sha256'] and
            candidate_ident['initial_model_sha256'] == ident['initial_model_sha256'] and
            fingerprint(context, calibration(context, state, oracle=True)) == witness_sha,
            'independent matched CPU arm initialization differs')
    release(context, state)
    require(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
    return {'completed_step': 0, 'checkpoint': {'path': str(checkpoint), 'sha256': sha},
            'terminal_state_sha256': digest, 'identity': ident,
            'initial_model_sha256': ident['initial_model_sha256'],
            'initial_raw_unit_packed_sha256': witness_sha, 'initial_arm_parity': True,
            'cuda_initialized': False, 'peak_cuda_allocated_bytes': 0,
            'forward_oracle_exact': True, 'native_training_inference_exact': True,
            'bypass_version_tamper_rejected': True, 'malformed_state_rejected': True,
            'teacher_native_drift': {k: v for k, v in witness.items() if k.startswith('teacher_native_')},
            'teachers_sha256': fingerprint(context, context['initial']['teachers']),
            'cpu_serialization_exact': True, 'training_state_discarded': True,
            'native_loss_reduction_exact': True, 'native_role_mutation_rejected': True,
            'total_training_core_seconds': context['phase_seconds']['target_construction']}


def native_loss_fixtures(context, state):
    """Actual CPU Torch reductions/indices on admitted teachers, no vision update."""
    import torch
    from torch.nn import functional as F
    with timed(context, 'cpu_loss_fixtures'):
        valid = next(i for i, c in enumerate(state['target_list']) if state['count_list'][c] > 1)
        singleton = next(i for i, c in enumerate(state['target_list']) if state['count_list'][c] == 1)
        groups = [[valid] * 16, [valid] * 8 + [singleton] * 8, [singleton] * 16, [valid] + [singleton] * 15]
        anchors = sum(groups, [])
        raw = state['teachers']['T'][anchors].clone().requires_grad_(True)
        terms = [loss_terms(state, raw[o:o + 16], batch, 25) for o, batch in zip(range(0, 64, 16), groups, strict=True)]
        require([v['valid'] for _, _, v in terms] == [16, 8, 0, 1], 'unequal native micro valid fixture differs')
        expected_mse = (raw - state['teachers']['P'][state['target'][anchors]]).square().sum() / (64 * state['teachers']['e0'])
        selected = [(offset + i, p, n) for offset, (_, _, v) in zip(range(0, 64, 16), terms, strict=True)
                    for i, (p, n) in enumerate(zip(v['positive'], v['negative'], strict=True)) if p >= 0]
        indices, positives, negatives = map(list, zip(*selected, strict=True))
        unit = F.normalize(raw, dim=1)
        expected_rank = F.relu(.05 + (unit[indices] * state['teachers']['V'][negatives]).sum(1) -
                               (unit[indices] * state['teachers']['V'][positives]).sum(1)).sum() / (25 * .05)
        actual = sum(mse + rank for mse, rank, _ in terms)
        expected = expected_mse + expected_rank
        actual_grad = torch.autograd.grad(actual, raw, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, raw)[0]
        require(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7) and
                torch.allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-7), 'native B64 objective/reduction gradient differs')
        invalid = loss_terms(state, raw[:16], [singleton] * 16, 0)
        require(invalid[1].item() == 0 and invalid[2]['positive'] == [-1] * 16, 'native invalid-positive/zero rank differs')


def gpu_run(context):
    import torch
    args = context['args']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible fresh CUDA device required')
    torch.cuda.manual_seed_all(SEED)
    state = fresh(context, args.arm, 'cuda')
    ident = identity(context, state)
    integrity(context, state, ident)
    initial = calibration(context, state, oracle=True)
    initial_witness = fingerprint(context, initial)
    cpu = context['terminals']['cpu:control']
    require(ident['static_sha256'] == cpu['identity']['static_sha256'] and
            ident['initial_model_sha256'] == cpu['initial_model_sha256'], 'new CPU initial source/teacher state differs')
    initial_state_sha = fingerprint(context, payload(context, state, ident))
    rows, resumed, total = [], [], 17 if args.phase == 'mechanics' else 32
    with TemporaryDirectory(prefix='discard-mechanics-', dir=args.output) as directory:
        temporary = Path(directory)
        sha8 = digest8 = None
        for step in range(1, total + 1):
            row = update(context, state, ident, step)
            if args.phase == 'train' and step <= 17:
                require(diagnostic(row) == diagnostic(context['terminals']['mechanics:' + args.arm]['steps'][step - 1]),
                        'fresh first17 mechanics replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
        witness = calibration(context, state, oracle=True)
        witness_sha = fingerprint(context, witness)
        source_substitution_witness(context, state, ident)
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else args.output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        inference_path = temporary / 'inference.pt' if args.phase == 'mechanics' else args.output / 'inference.pt'
        inference_sha, inference_digest = save_inference(context, state, ident, inference_path)
        release(context, state)
        if args.phase == 'mechanics':
            state = restore(context, temporary / 'step8.pt', sha8, digest8, ident, 8)
            resumed = [update(context, state, ident, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    fingerprint(context, payload(context, state, ident)) == digest,
                    '17 vs independent serialized8+9 complete replay differs')
            release(context, state)
        state = restore(context, checkpoint, sha, digest, ident, total)
        require(fingerprint(context, calibration(context, state, oracle=True)) == witness_sha,
                'strict independent final updated vision/raw-unit-packed reload differs')
        release(context, state)
        inference = load_inference(context, inference_path, inference_sha, inference_digest, 'cuda')
        # No teacher/cache dependency in the inference artifact or forward API.
        batch = witness['batch']
        mapping_state = {**inference, **{k: context['initial'][k] for k in ('original_rows', 'target', 'partition')}}
        pixels, rgb, mapping = canonical_pixels(context, mapping_state, batch)
        with torch.no_grad():
            raw = native_raw(context, inference, pixels, oracle=True)
            packed = packed_outputs(context, raw)
        require(fingerprint(context, packed) == fingerprint(context, {k: witness[k] for k in packed}) and
                rgb == witness['rgb_sha256'] and mapping == witness['row_mapping_sha256'],
                'strict self-contained updated inference raw/unit/packed/native parity differs')
        del mapping_state, pixels, raw, packed
        release(context, inference)
        for path in list(context['guards']):
            if Path(path).is_relative_to(temporary):
                context['guards'].pop(path)  # Discarded states are authenticated above before removal.
    check_steps(rows, 1, total)
    if resumed:
        check_steps(resumed, 9, 9)
    return {'completed_step': total, 'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'inference_checkpoint': None if args.phase == 'mechanics' else {'path': str(inference_path), 'sha256': inference_sha},
            'inference_state_sha256': inference_digest, 'terminal_state_sha256': digest,
            'initial_state_sha256': initial_state_sha, 'initial_model_sha256': ident['initial_model_sha256'],
            'initial_raw_unit_packed_sha256': initial_witness, 'identity': ident,
            'steps': rows, 'resumed_steps': resumed, 'replay_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics', 'fresh_first17_exact': args.phase == 'train',
            'source_substitution_rejected': True, 'forward_oracle_exact': True,
            'native_training_inference_exact': True, 'inference_artifact_independent': True,
            'teacher_native_drift': {k: v for k, v in initial.items() if k.startswith('teacher_native_')},
            'total_training_core_seconds': context['phase_seconds']['target_construction'] +
                sum(r['core_seconds'] for r in rows + resumed),
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
            'cuda_initialized': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def check_terminal_record(record, launch, phase, arm):
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == SEED and method(record['launch']) == method(launch) and
            record['resource_policy'] == policy(phase) and record['optimizer_members'] == 4 and
            record['trainable_scalars'] == 9921872 and record['quality_read'] is False and
            all(record[k] is True for k in ('pass', 'strict_reload_exact', 'exit_rehash_pass',
                'sequential_model_ownership', 'forward_oracle_exact', 'native_training_inference_exact',
                'both_locks_held_in_parent_authority')) and
            0 < record['total_training_core_seconds'] < record['wall_seconds'] < policy(phase)['seconds'] and
            0 < record['process_peak_rss_kib'] <= 8 * 1024**2, 'qualified whole-unit contract differs')
    check_launch(record['launch'], SimpleNamespace(execution_sha256=launch['execution_sha256'], phase=phase, arm=arm, seed=SEED))
    require(record['code'].keys() == FILES and record['authority_sha256'] == record['authority']['sha256'] and
            record['invocation']['optimize'] == 0 and record['numerical_flags'] == record['identity']['numerical_flags'] and
            record['identity']['method'] == method(launch) and record['identity']['parameter_names'] == NAMES and
            record['identity']['source'] == record['source'], 'new closure/state/source identity differs')
    if phase == 'cpu':
        require(record['completed_step'] == 0 and record['cuda_initialized'] is False and
                record['peak_cuda_allocated_bytes'] == 0 and record['initial_arm_parity'] is True and
                record['cpu_serialization_exact'] is True and record['bypass_version_tamper_rejected'] is True and
                record['malformed_state_rejected'] is True and record['native_loss_reduction_exact'] is True and
                record['native_role_mutation_rejected'] is True and record['invocation']['cuda_visible_devices'] == '',
                'new CPU qualification incomplete')
    else:
        require(record['launch']['selected_cpu'] == launch['selected_cpu'] and
                record['cuda_initialized'] is True and 0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000 and
                record['source_substitution_rejected'] is True and record['inference_artifact_independent'] is True and
                record['invocation']['cuda_visible_devices'] not in (None, '') and
                record['invocation']['cublas_workspace_config'] == ':4096:8', 'native numerical/update/inference witnesses incomplete')
        count = 17 if phase == 'mechanics' else 32
        require(record['completed_step'] == count, 'bounded native update count differs')
        check_steps(record['steps'], 1, count)
        if phase == 'mechanics':
            require(record['checkpoint'] is None and record['inference_checkpoint'] is None and
                    record['training_state_discarded'] is True and record['replay_exact'] is True and
                    all(diagnostic(a) == diagnostic(b) for a, b in
                        zip(record['steps'][8:], record['resumed_steps'], strict=True)), 'mechanics replay/discard differs')
            check_steps(record['resumed_steps'], 9, 9)
        else:
            require(record['fresh_first17_exact'] is True and record['training_state_discarded'] is False and
                    record['launch']['selected_mechanics'] == launch['selected_mechanics'] and
                    record['resumed_steps'] == [], 'fresh paired TRAIN prerequisites differ')
    if record['checkpoint'] is not None:
        file_fact(record['checkpoint'])
        require(record['input_guards'].get(record['checkpoint']['path']) == record['checkpoint']['sha256'],
                'serialized endpoint FILE binding differs')


def admit_terminal(context, unit, phase, arm):
    check_unit(unit)
    guards, fitter, original, legacy = (context[k] for k in ('guards', 'fitter', 'fit_context', 'legacy'))
    admission, reader_source = legacy['admission'], legacy['original']
    require(type(admission) is reader_source.FlatAdmission and
            reader_source.FlatAdmission.__module__ == reader_source.__name__ and
            reader_source.FlatAdmission.__qualname__ == 'FlatAdmission' and
            reader_source.__spec__ is not None and
            Path(reader_source.__spec__.origin) == Path(reader_source.__file__) and
            original['guards'].get(reader_source.__file__) == fitter.TERMINAL_SOURCE_SHA,
            'actual original startup reader/source required')
    raw = bound_file(guards, reader_source.__file__, fitter.TERMINAL_SOURCE_SHA).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == fitter.TERMINAL_SOURCE_SHA,
            'original startup reader source changed before compilation')
    source_code = compile(raw, reader_source.__file__, 'exec')
    reader_code = next(c for c in source_code.co_consts if getattr(c, 'co_name', None) == 'FlatAdmission')
    for name in ('__init__', 'canonical', 'digest_string', 'register', 'bound_file',
                 'read_json', 'descriptor_json', 'admit_terminal'):
        fn, bound = getattr(reader_source.FlatAdmission, name), getattr(admission, name)
        expected = next(c for c in reader_code.co_consts if getattr(c, 'co_name', None) == name)
        require(isinstance(fn, FunctionType) and fn.__globals__ is vars(reader_source) and
                fn.__code__ == expected and fn.__code__.co_filename == reader_source.__file__ and
                ((bound is fn) if name in ('canonical', 'digest_string') else
                 (getattr(bound, '__self__', None) is admission and getattr(bound, '__func__', None) is fn)),
                'original startup reader method changed: ' + name)
    record = read_json(unit['receipt'], guards, admission=admission)
    check_terminal_record(record, context['launch'], phase, arm)
    prior = legacy['selected']['source_cpu']['invocation']
    require(record['code'] == context['code'] and record['source'] == context['source'] and
            record['execution_sha256'] == context['args'].execution_sha256 and
            record['numerical_flags'] == legacy['selected']['source_cpu']['numerical_flags'] and
            read_json(record['authority'], guards, admission=admission) == record['launch'] and
            all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items()) and
            all(record['invocation'][k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
            record['invocation']['argv'] == cli(context['root'], record['authority']['path'], record['authority']['sha256'],
                context['args'].execution_sha256, phase, arm, Path(unit['receipt']['path']).parent),
            'new original UNIT source/CLI/complete guards differ')
    final = fitter.original_terminal_reader(original)(legacy['admission'], record, unit, policy(phase)['seconds'], guards)
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['old'].zero_events(value)
    require(unit['invocation_id'] not in legacy['invocations'], 'reused original unit invocation')
    legacy['invocations'].add(unit['invocation_id'])
    context['terminals'][phase + ':' + arm] = record
    context['terminal_cgroups'][phase + ':' + arm] = final
    for p, h in record['input_guards'].items():
        admission.bound_file(guards, p, h)
    return record


def helper_guard(context):
    fitter, original = context['fitter'], context['fit_context']
    fitter.prepare_readout(original)
    cached = original.get('original_preparation')
    if cached is not None:
        for owner, name, fn, code in cached['functions']:
            require(getattr(owner, name, None) is fn and fn.__code__ is code, 'original live helper changed: ' + name)
        for values, members in cached['globals']:
            require(values.keys() == members.keys() and all(values[k] is v for k, v in members.items()),
                    'original helper global changed')
    module = context['readout']
    cached = context.get('readout_objects')
    if cached is None:
        cached = {'members': dict(vars(module)), 'functions': [(v, v.__code__) for v in vars(module).values()
                  if isinstance(v, FunctionType)], 'spec': module.__spec__, 'loader': module.__spec__.loader}
        context['readout_objects'] = cached
    require(sys.modules.get('_nearest_connected') is module and module.__spec__ is cached['spec'] and
            module.__spec__.loader is cached['loader'] and
            Path(module.__file__) == Path(module.__spec__.origin) == context['root'] / 'nearest_ranking_readout.py' and
            vars(module).keys() == cached['members'].keys() and
            all(vars(module)[k] is v for k, v in cached['members'].items()) and
            all(fn.__code__ is code for fn, code in cached['functions']), 'connected helper object/code/global changed')
    bound_file(context['guards'], module.__file__, context['code']['nearest_ranking_readout.py'])


def exit_rehash(context):
    require_no_model(context)
    with timed(context, 'source_exit_rehash'):
        helper_guard(context)
        context['fitter'].exit_rehash(context['fit_context'])
    with timed(context, 'own_exit_rehash'):
        for p, h in context['guards'].items():
            bound_file({}, p, h)
        require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'],
                'new exact3 exit closure differs')


def run(args):
    started = UNIT_STARTED
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
            os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'explicit CUDA-hidden CPU or original deterministic CUDA launch required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original systemd invocation required')
    require(sys.argv == cli(Path(__file__).absolute().parent, args.authority, args.authority_sha256,
                           args.execution_sha256, args.phase, args.arm, args.output), 'fixed canonical CLI order required')
    context = authority(args)
    context['phase_seconds']['authority'] = time.perf_counter() - started
    legacy, source = context['legacy'], context['legacy']['source_driver']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before, unit)
    prior = legacy['selected']['source_cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and legacy['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'original qualified interpreter required')
    import torch
    require(not torch.cuda.is_initialized(), 'admission must precede CUDA')
    flags = legacy['selected']['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original source flags differ')
    torch.random.default_generator.manual_seed(SEED)
    rng = torch.random.get_rng_state().clone()
    args.output.mkdir()
    helper_guard(context)
    with timed(context, 'preparation'):
        prepare_native(context)
    # Original fitter markers are maintained in its own context, never helper globals.
    context['fit_context']['unit_started'] = started
    result = cpu_witnesses(context) if args.phase == 'cpu' else gpu_run(context)
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'constructor/forward/update/reload RNG/flags changed')
    context['old'].audit_origins(legacy)
    origins = legacy['origins']
    for p, h in origins['files'].items():
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
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': SEED, 'pass': True,
        'quality_read': False, 'strict_reload_exact': True, 'exit_rehash_pass': True,
        'sequential_model_ownership': True, 'optimizer_members': 4, 'trainable_scalars': 9921872,
        'frozen_vision_members': 444, 'source': context['source'], 'launch': context['launch'],
        'execution_sha256': args.execution_sha256, 'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
        'authority_sha256': args.authority_sha256, 'code': context['code'], 'output': str(args.output),
        'resource_policy': policy(args.phase), 'numerical_flags': flags, 'wall_seconds': wall,
        'process_peak_rss_kib': rss, 'cgroup_before': before, 'cgroup_after': after,
        'origins': origins, 'input_guards': context['guards'], 'phase_seconds': context['phase_seconds'],
        'terminal_cgroups': context['terminal_cgroups'], 'both_locks_held_in_parent_authority': True,
        'terminal_exit_and_both_locks_require_parent_receipt': True,
        'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                      'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                      'invocation_id': os.environ['INVOCATION_ID'],
                      'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}, **result}
    # Parent admits whole-service duration, final normal exit and lifetime locks.
    legacy['selected']['genuine']['reference'].write_json(legacy['extract'], args.output / 'receipt.json', receipt)
    require(time.perf_counter() - started < policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('cpu', 'mechanics', 'train'), required=True)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--seed', type=int, choices=(SEED,), required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Nearest-ranking rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
