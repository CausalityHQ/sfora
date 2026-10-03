#!/usr/bin/env python3
"""Discarded native current-byte comparison; native execution is parent-owned.

Freeze exactly FILES in DRIVER_ROOT/execution.json. Run unoptimized Python -B:
DRIVER_ROOT/compare_nearest_ranking_current_bytes.py --execution-sha256 SHA
 --authority FILE --authority-sha256 SHA --output NEW_ABSOLUTE_DIRECTORY
The parent supplies an exact AUTHORITY_KEYS launch, actual candidate TRAIN3
manifest/code, its CPU authority FILE and accepted fresh CPU UNIT. No future
hashes are built in. Genuine CPU-phase authority and terminal admission precede
native imports. Three alternating pairs compare the complete frozen tree and
complete payload on one fresh control CUDA state. Exact bytes and resources are
required; timings are observations, with no speed gate. No update runs here.
Full original exit and whole-service terminal/lock footers remain mandatory.
This receipt never qualifies mechanics, training, quality or state reuse.
"""
if not __debug__:
    raise SystemExit('Comparison requires assertions; optimized mode is forbidden')

import argparse
import builtins
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
import traceback
from types import FunctionType, SimpleNamespace

UNIT_STARTED = time.perf_counter()
SCHEMA = 'nearest-ranking-current-bytes-native-receipt-v1'
AUTHORITY_SCHEMA = 'nearest-ranking-current-bytes-native-launch-v1'
FILES = {'compare_nearest_ranking_current_bytes.py', 'test_compare_nearest_ranking_current_bytes.py'}
TRAIN_FILES = {'train_siglip2_nearest_ranking.py', 'test_siglip2_nearest_ranking.py', 'nearest_ranking_readout.py'}
ORIGINAL_SHA = 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'
POLICY = {'seconds': 300, 'host_bytes': 8589934592, 'swap_bytes': 0,
          'cuda_allocated_bytes_exclusive': 10000000000}
BODY_SECONDS = 90
SEED = 179061
LOCKS = ['/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock']
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
AUTHORITY_KEYS = {'schema', 'execution_sha256', 'python', 'python_version', 'candidate',
                  'cpu_authority', 'selected_cpu', 'output', 'unit', 'both_locks_held',
                  'resource_policy', 'comparison_seconds', 'qualification_eligible',
                  'state_reuse_eligible', 'quality_read'}
UNIT_KEYS = {'receipt', 'log', 'unit', 'invocation_id', 'service_seconds',
             'native_peak_rss_kib', 'both_locks_held'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path.parent.resolve() == path.parent and not path.is_symlink(),
            'canonical absolute path required: ' + str(path))
    return path


def file_fact(value):
    require(type(value) is dict and value.keys() == {'path', 'sha256'} and
            type(value['path']) is str and type(value['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', value['sha256']), 'exact FILE required')
    canonical(value['path'])


def source_bytes(binding, guards):
    file_fact(binding)
    path = canonical(binding['path'])
    require(path.is_file(), 'regular source FILE required')
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'source size/SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), binding['sha256']) == binding['sha256'], 'conflicting FILE authority')
    return raw


def strict_json(raw):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def closure(root, digest, names, guards):
    root = canonical(root)
    require(root.is_dir() and root.resolve() == root, 'canonical closure root required')
    code = strict_json(source_bytes({'path': str(root / 'execution.json'), 'sha256': digest}, guards))
    require(type(code) is dict and code.keys() == set(names), 'exact code closure required')
    for name, sha in code.items():
        source_bytes({'path': str(root / name), 'sha256': sha}, guards)
    return code


def check_authority(value, args):
    require(type(value) is dict and value.keys() == AUTHORITY_KEYS and
            value['schema'] == AUTHORITY_SCHEMA and value['execution_sha256'] == args.execution_sha256 and
            type(value['resource_policy']) is dict and value['resource_policy'] == POLICY and
            all(type(v) is int for v in value['resource_policy'].values()) and value['comparison_seconds'] == BODY_SECONDS and
            type(value['comparison_seconds']) is int and value['both_locks_held'] is True and
            all(value[k] is False for k in ('qualification_eligible', 'state_reuse_eligible', 'quality_read')) and
            value['output'] == str(args.output) and type(value['python_version']) is str and
            type(value['unit']) is str and re.fullmatch('[A-Za-z0-9_.@-]+', value['unit']) and
            not value['unit'].endswith('.service'), 'fixed comparison authority differs')
    for key in ('python', 'cpu_authority'):
        try:
            file_fact(value[key])
        except (ValueError, TypeError):
            raise ValueError('comparison FILE authority differs: ' + key) from None
    candidate, unit = value['candidate'], value['selected_cpu']
    require(type(candidate) is dict and candidate.keys() == {'root', 'execution_sha256', 'code'} and
            type(candidate['root']) is str and Path(candidate['root']).is_absolute() and
            type(candidate['execution_sha256']) is str and re.fullmatch('[0-9a-f]{64}', candidate['execution_sha256']) and
            type(candidate['code']) is dict and candidate['code'].keys() == TRAIN_FILES and
            all(type(s) is str and re.fullmatch('[0-9a-f]{64}', s) for s in candidate['code'].values()),
            'actual candidate TRAIN3 authority differs')
    require(type(unit) is dict and unit.keys() == UNIT_KEYS and unit['both_locks_held'] is True and
            type(unit['unit']) is str and re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and
            not unit['unit'].endswith('.service') and unit['unit'] != value['unit'] and
            type(unit['invocation_id']) is str and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            type(unit['service_seconds']) in (int, float) and math.isfinite(unit['service_seconds']) and
            0 < unit['service_seconds'] <= 500 and type(unit['native_peak_rss_kib']) in (int, float) and
            math.isfinite(unit['native_peak_rss_kib']) and 0 < unit['native_peak_rss_kib'] <= 8 * 1024**2,
            'fresh CPU UNIT authority differs')
    for key in ('receipt', 'log'):
        try:
            file_fact(unit[key])
        except (ValueError, TypeError):
            raise ValueError('fresh CPU FILE authority differs: ' + key) from None


def exclusive_output(value, roots):
    output = canonical(value)
    require(not output.exists() and all(not output.is_relative_to(root) and
            not Path(root).is_relative_to(output) for root in roots), 'exclusive separate output required')
    return output


def cli(root, args):
    return [str(Path(root) / 'compare_nearest_ranking_current_bytes.py'), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--output', str(args.output)]


def load_candidate(path, digest, guards):
    name = '_nearest_current_bytes_candidate'
    require(name not in sys.modules, 'fresh candidate namespace required')
    raw = source_bytes({'path': str(path), 'sha256': digest}, guards)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'candidate origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(module.__file__ == str(path) and module.__spec__ is spec and
            not any(n.split('.')[0] in NATIVE for n in sys.modules), 'candidate source-only origin/startup differs')
    return module


def original_source(candidate, context):
    original = context['legacy']['original']
    path = Path(original.__file__)
    require(context['fitter'].TERMINAL_SOURCE_SHA == ORIGINAL_SHA and
            context['guards'].get(str(path)) == ORIGINAL_SHA and original.__spec__ is not None and
            Path(original.__spec__.origin) == path and sys.modules.get(original.__name__) is original,
            'actual full original source/origin differs')
    raw = candidate.bound_file(context['guards'], path, ORIGINAL_SHA).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'original source SHA256 changed before compilation')
    code = next(c for c in compile(raw, str(path), 'exec').co_consts if getattr(c, 'co_name', None) == 'fingerprint')
    fn = original.fingerprint
    require(isinstance(fn, FunctionType) and fn.__globals__ is vars(original) and fn.__code__ == code and
            fn.__defaults__ == (None, None) and fn.__kwdefaults__ is None, 'genuine original fingerprint changed')
    guard = context.get('comparison_original_guard')
    if guard is None:
        names = ('__import__', 'isinstance', 'str', 'tuple', 'dict', 'sorted', 'repr', 'list', 'type', 'len', 'memoryview')
        guard = context['comparison_original_guard'] = {'hashlib': hashlib, 'sha256': hashlib.sha256,
            'global_builtins': vars(original).get('__builtins__'),
            'builtins': {name: getattr(builtins, name) for name in names}}
    require(vars(original).get('hashlib') is guard['hashlib'] and hashlib.sha256 is guard['sha256'] and
            vars(original).get('__builtins__') is guard['global_builtins'] and
            all(fn.__builtins__.get(k) is v for k, v in guard['builtins'].items()), 'original live global changed')
    return fn


def prepare(args, owned=None):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None and
            sys.gettrace() is None and not any(n.split('.')[0] in NATIVE for n in sys.modules),
            'unoptimized -B source-only startup required')
    root = Path(__file__).absolute().parent
    require(sys.argv == cli(root, args), 'fixed canonical CLI order required')
    guards = {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = strict_json(source_bytes({'path': str(args.authority), 'sha256': args.authority_sha256}, guards))
    check_authority(launch, args)
    binding = launch['candidate']
    candidate_root = canonical(binding['root'])
    require(not root.is_relative_to(candidate_root) and not candidate_root.is_relative_to(root),
            'separate source closures required')
    exclusive_output(args.output, [root, candidate_root])
    require(closure(candidate_root, binding['execution_sha256'], TRAIN_FILES, guards) == binding['code'],
            'actual candidate manifest/code differs')
    source_bytes(launch['cpu_authority'], guards)
    python = canonical(launch['python']['path'])
    require(str(Path(sys.executable).resolve()) == str(python) and sys.version == launch['python_version'],
            'original interpreter path/version differs')
    with python.open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == launch['python']['sha256'],
                'original interpreter SHA256 differs')
    guards[str(python)] = launch['python']['sha256']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')) and
            os.environ['INVOCATION_ID'] != launch['selected_cpu']['invocation_id'],
            'fixed CUDA/fresh systemd launch required')
    path = candidate_root / 'train_siglip2_nearest_ranking.py'
    candidate = load_candidate(path, binding['code'][path.name], guards)
    require(candidate.FILES == TRAIN_FILES and candidate.SEED == SEED, 'actual candidate API closure differs')
    cpu_args = SimpleNamespace(execution_sha256=binding['execution_sha256'],
        authority=Path(launch['cpu_authority']['path']), authority_sha256=launch['cpu_authority']['sha256'],
        phase='cpu', arm='control', seed=SEED, output=args.output)
    context = candidate.authority(cpu_args)
    if owned is not None:
        owned.update(candidate=candidate, context=context, launch=launch, code=code)
    # Admit before adding diagnostic guards: CPU must cover its genuine original
    # required inventory, rather than an invented driver-dependent inventory.
    cpu = candidate.admit_terminal(context, launch['selected_cpu'], 'cpu', 'control')
    require(cpu['authority'] == launch['cpu_authority'] and cpu['launch'] == context['launch'] and
            not any(n.split('.')[0] in NATIVE for n in sys.modules), 'fresh candidate CPU launch/startup differs')
    prior = context['legacy']['selected']['source_cpu']['invocation']
    require(all(prior[k] == v for k, v in (('python', str(python)),
            ('python_sha256', launch['python']['sha256']), ('python_version', launch['python_version']))) and
            os.environ['INVOCATION_ID'] not in context['legacy']['invocations'],
            'original interpreter/fresh invocation differs')
    original_source(candidate, context)
    for path, digest in guards.items():
        require(context['guards'].setdefault(path, digest) == digest, 'source/driver guard conflict')
    return candidate, context, launch, code


def cleanup_steps(primary, steps, record):
    for label, action in steps:
        try:
            action()
            record[label + '_pass'] = True
        except BaseException as error:
            record[label + '_pass'] = False
            record.setdefault('cleanup_errors', []).append({'step': label, 'type': type(error).__name__, 'error': str(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note(label + ' cleanup failed: ' + str(error))
    return primary


def serializer_parity(original, proposed, value):
    expected = original(value)
    require(proposed(value) == expected, 'native complete typed serializer parity differs')
    consumed = []
    require(original(value, consumed=lambda t: consumed.append(id(t))) == expected, 'original consumed parity differs')
    order = []
    require(proposed(value, consumed=lambda t: order.append(id(t))) == expected and order == consumed,
            'native consumed fallback parity/order differs')
    require(proposed(value, frozen={}) == original(value, frozen={}) == expected, 'native frozen fallback parity differs')
    return expected


def data_probe(original, proposed, value, tensor):
    version, saved = tensor._version, tensor[0, 0].item()
    expected = original(value)
    require(proposed(value) == expected, 'native mutation initial parity differs')
    primary, changed, restored = None, None, None
    try:
        tensor.data[0, 0] += 7
        require(tensor._version == version and tensor[0, 0].item() != saved,
                'isolated .data mutation must change actual scalar without version')
        changed = original(value)
        require(changed != expected and proposed(value) == proposed(value) == changed,
                'fresh unchanged-version .data mutation missed')
    except BaseException as error:
        primary = error
    finally:
        def restore():
            tensor.data[0, 0] = saved
        def exact_restore():
            nonlocal restored
            restored = original(value)
            require(tensor._version == version and tensor[0, 0].item() == saved and
                    restored == proposed(value) == expected, 'unchanged-version .data restoration/digest recovery failed')
        primary = cleanup_steps(primary, [('data_restore', restore), ('data_restoration', exact_restore)], {})
    if primary is not None:
        raise primary
    return {'unchanged_version_data_mutation_detected': True, 'unchanged_version_data_restoration_exact': True,
            'isolated_probe_only': True, 'original_digest': expected, 'mutated_digest': changed,
            'restored_digest': restored}


def cpu_probe(original, proposed):
    import torch
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    view = base.T[1::2, 1:]
    require(view.storage_offset() > 0 and not view.is_contiguous() and
            view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr(),
            'CPU nonzero-offset shared-storage stride probe differs')
    value = {1: [view, view], '1': (torch.tensor(2, dtype=torch.int16), torch.empty((0, 3))),
             'bf16': torch.arange(4).to(torch.bfloat16)}
    digest = serializer_parity(original, proposed, value)
    require(not torch.cuda.is_initialized(), 'CPU small serializer probe initialized CUDA')
    return {'exact': True, 'digest': digest}


def native_probes(original, proposed):
    import torch
    cpu_base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cuda_base = torch.arange(24, dtype=torch.float32, device='cuda').reshape(4, 6)
    cpu, cuda = cpu_base.T[1::2, 1:], cuda_base.T[1::2, 1:]
    require(all(view.storage_offset() > 0 and not view.is_contiguous() and
                view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
                for view, base in ((cpu, cpu_base), (cuda, cuda_base))),
            'native nonzero-offset shared-storage stride probes required')
    dtypes = [torch.float32, torch.bfloat16, torch.int16, torch.int64, torch.uint8, torch.bool]
    value = {1: (cpu, cuda, cuda, cpu), '1': [cuda, cpu],
             'dtype': [(torch.arange(4).to(dtype), torch.arange(4, device='cuda').to(dtype)) for dtype in dtypes],
             'scalar': [torch.tensor(3., device='cuda'), torch.tensor(2, dtype=torch.int16)],
             'empty': [torch.empty((0, 3)), torch.empty((0, 3), device='cuda')]}
    digest = serializer_parity(original, proposed, value)
    result = data_probe(original, proposed, value, cuda)
    return {**result, 'mixed_cpu_cuda_dtypes_exact': True, 'alias_empty_scalar_exact': True,
            'nonzero_offset_shared_storage_stride_exact': True, 'consumed_frozen_fallback_exact': True,
            'digest': digest}


def paired_samples(baseline, proposed, synchronize, rows, clock=time.perf_counter, expected=None, body_started=None):
    started, reference = clock() if body_started is None else body_started, expected
    for pair in range(3):
        names = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
        row = {'pair': pair, 'order': list(names)}
        rows.append(row)
        for name in names:
            require(clock() - started < BODY_SECONDS, '90-second comparison body exceeded')
            synchronize()
            tick = clock()
            try:
                digests = tuple((baseline if name == 'baseline' else proposed)())
            finally:
                primary = sys.exception()
                final = cleanup_steps(primary, [('synchronize', synchronize)], {})
                if primary is None and final is not None:
                    raise final
            row[name] = {'seconds': clock() - tick, 'digests': digests}
            require(len(digests) == 2 and all(type(d) is str and re.fullmatch('[0-9a-f]{64}', d) for d in digests) and
                    (reference is None or digests == reference), 'complete frozen/payload paired digests differ')
            reference = digests
            require(clock() - started <= BODY_SECONDS, '90-second comparison body exceeded')
    old = statistics.median(row['baseline']['seconds'] for row in rows)
    new = statistics.median(row['candidate']['seconds'] for row in rows)
    require(old > 0 and new > 0, 'positive synchronized timings required')
    return {'paired_body_seconds': clock() - started, 'baseline_median_seconds': old,
            'candidate_median_seconds': new, 'candidate_to_baseline_ratio': new / old,
            'paired_median_delta_seconds': statistics.median(row['candidate']['seconds'] - row['baseline']['seconds'] for row in rows),
            'exact_digests_every_pair': True}


def frozen_tree(candidate, state):
    # The complete actual identity complement, with current buffers/static state.
    return {'vision': candidate.frozen_vision(state), **candidate.static_tree(state)}


def live_snapshot(candidate, context, state, ident):
    import torch
    def versions(value):
        if isinstance(value, torch.Tensor):
            return [(value.data_ptr(), value._version, str(value.device), str(value.dtype), tuple(value.shape),
                     tuple(value.stride()), value.storage_offset(), value.requires_grad,
                     value.grad is None, value.grad_fn is None)]
        if isinstance(value, dict):
            return [v for k in sorted(value, key=repr) for child in (k, value[k]) for v in versions(child)]
        if isinstance(value, (tuple, list)):
            return [v for child in value for v in versions(child)]
        return []
    original = context['legacy']['original'].fingerprint
    actual = [dict(state['model'].named_parameters()), dict(state['model'].named_buffers()),
              dict(state['head_object'].named_parameters()), dict(state['head_object'].named_buffers()),
              candidate.static_tree(state), state['params'], state['optimizer_object'].state_dict()]
    return {'complete_payload_sha256': original(candidate.payload(context, state, ident)),
            'complete_frozen_tree_sha256': original(frozen_tree(candidate, state)), 'live_versions': versions(actual),
            'counter': state['counter'], 'optimizer_sha256': original([state['optimizer_object'].state_dict(),
                state['optimizer_object'].defaults]), 'scaler_sha256': original(state['scaler_object'].state_dict()),
            'cpu_rng_sha256': original(torch.random.get_rng_state()),
            'cuda_rng_sha256': original(torch.cuda.get_rng_state_all()),
            'numerical_flags': context['legacy']['source_driver'].numerical_flags()}


def compare_live(candidate, context, state, ident, record):
    import torch
    tick = time.perf_counter()
    require(tick - UNIT_STARTED + BODY_SECONDS < POLICY['seconds'],
            'prospective 90-second comparison cannot fit whole 300-second cap')
    original = original_source(candidate, context)
    proposed = lambda value, **kwargs: candidate.fingerprint(context, value, **kwargs)
    before = live_snapshot(candidate, context, state, ident)
    record['state_before'] = before
    primary = None
    try:
        record['native_differential'] = native_probes(original, proposed)
        require(time.perf_counter() - tick < BODY_SECONDS, '90-second comparison body exceeded')
        def work(fingerprint):
            def sample():
                require(time.perf_counter() - tick < BODY_SECONDS, '90-second comparison body exceeded')
                frozen = fingerprint(frozen_tree(candidate, state))
                require(time.perf_counter() - tick < BODY_SECONDS, '90-second comparison body exceeded')
                return frozen, fingerprint(candidate.payload(context, state, ident))
            return sample
        record['pairs'] = []
        record.update(paired_samples(work(original), work(proposed), torch.cuda.synchronize, record['pairs'],
            expected=(before['complete_frozen_tree_sha256'], before['complete_payload_sha256']), body_started=tick))
    except BaseException as error:
        primary = error
    def unchanged():
        record['state_after'] = live_snapshot(candidate, context, state, ident)
        require(before == record['state_after'], 'complete RNG/live bytes/versions/optimizer/scaler/counter changed')
    primary = cleanup_steps(primary, [('state_preserved', unchanged),
        ('final_integrity', lambda: candidate.integrity(context, state, ident))], record)
    record['body_seconds'] = time.perf_counter() - tick
    primary = cleanup_steps(primary, [('body_cap', lambda: require(record['body_seconds'] <= BODY_SECONDS,
        '90-second comparison body exceeded'))], record)
    if primary is not None:
        raise primary


def check_resources(before, after, wall, rss, cuda):
    for value in (before, after):
        values = value['values']
        events = dict(line.split() for line in values['memory.events'].splitlines())
        require(values['memory.max'] == str(POLICY['host_bytes']) and
                all(values[k] == '0' for k in ('memory.swap.current', 'memory.swap.peak', 'memory.swap.max')) and
                0 < int(values['memory.current']) <= int(values['memory.peak']) <= POLICY['host_bytes'] and
                all(events.get(k) == '0' for k in ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')),
                'whole-service cgroup resource cap/events differ')
    require(before['path'] == after['path'] and int(after['values']['memory.peak']) >= int(before['values']['memory.peak']) and
            0 < wall < POLICY['seconds'] and 0 < rss <= 8 * 1024**2 and
            0 <= cuda < POLICY['cuda_allocated_bytes_exclusive'], 'whole-service time/RSS/lifetime CUDA resource cap differs')


def write_receipt(path, value):
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args):
    owned, state, primary, before, rng, cuda_rng = {}, None, None, None, None, None
    output_created = False
    record = {'schema': SCHEMA, 'pass': False, 'qualification_eligible': False,
        'state_reuse_eligible': False, 'quality_read': False, 'mechanics_qualified': False,
        'training_qualified': False, 'training_state_discarded': True, 'trained_state_reused': False,
        'completed_step': 0, 'checkpoint': None, 'resource_policy': POLICY,
        'output': str(args.output), 'execution_sha256': args.execution_sha256,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
        'locks': LOCKS, 'both_locks_held_in_parent_authority': True,
        'terminal_exit_and_both_locks_require_parent_receipt': True}
    try:
        candidate, context, launch, code = prepare(args, owned)
        context['phase_seconds']['authority'] = time.perf_counter() - UNIT_STARTED
        source, legacy = context['legacy']['source_driver'], context['legacy']
        before = source.cgroup_memory()
        unit = Path(before['path']).name.removesuffix('.service')
        require(unit == launch['unit'], 'actual comparison unit differs')
        legacy['selected']['genuine']['reference'].admit_cgroup(before, unit)
        context['old'].zero_events(before)
        args.output.mkdir()
        output_created = True
        record.update(launch=launch, code=code, candidate_code=context['code'], candidate_launch=context['launch'],
            cpu_authority=launch['cpu_authority'], selected_cpu=launch['selected_cpu'], cgroup_before=before,
            original_cpu_argv=candidate.cli(context['root'], context['args'].authority, context['args'].authority_sha256,
                context['args'].execution_sha256, 'cpu', 'control', args.output),
            invocation={'argv': list(sys.argv), 'python': launch['python']['path'],
                'python_sha256': launch['python']['sha256'], 'python_version': sys.version,
                'optimize': sys.flags.optimize, 'dont_write_bytecode': sys.dont_write_bytecode,
                'pid': os.getpid(), 'unit': unit, 'invocation_id': os.environ['INVOCATION_ID'],
                'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']})
        import torch
        require(not torch.cuda.is_initialized(), 'source/CPU admission must precede CUDA')
        original = original_source(candidate, context)
        record['cpu_small_serializer'] = cpu_probe(original, lambda v, **kw: candidate.fingerprint(context, v, **kw))
        flags = legacy['selected']['source_cpu']['numerical_flags']
        torch.set_num_threads(flags['threads'])
        if torch.get_num_interop_threads() != flags['interop_threads']:
            torch.set_num_interop_threads(flags['interop_threads'])
        require(source.numerical_flags() == flags, 'original source flags differ')
        torch.random.default_generator.manual_seed(SEED)
        rng = torch.random.get_rng_state().clone()
        torch.cuda.manual_seed_all(SEED)  # Original lazy CUDA seeding; no CUDA state before preparation.
        candidate.helper_guard(context)
        candidate.prepare_native(context)
        context['fit_context']['unit_started'] = UNIT_STARTED
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
                'one visible fresh CUDA device required')
        state = candidate.fresh(context, 'control', 'cuda')
        cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
        ident = candidate.identity(context, state)
        cpu = context['terminals']['cpu:control']
        require(ident['static_sha256'] == cpu['identity']['static_sha256'] and
                ident['initial_model_sha256'] == cpu['initial_model_sha256'], 'fresh CPU-qualified source/static identity differs')
        require(state['counter'] == 0 and len(list(state['model'].named_parameters())) == 448 and
                len(candidate.frozen_vision(state)) == 444, 'complete fresh source parameter inventory differs')
        record['identity'] = ident
        candidate.integrity(context, state, ident)
        witness = candidate.calibration(context, state, oracle=True)
        calibration_digest = original(witness)
        require(candidate.fingerprint(context, witness) == calibration_digest, 'complete calibration serializer parity differs')
        record['calibration'] = {'complete_typed_sha256': calibration_digest,
            'payload_keys': sorted(witness),
            **{k: v for k, v in witness.items() if k not in {'raw', 'unit', 'codes', 'inverse_norms', 'wire'}}}
        del witness
        compare_live(candidate, context, state, ident, record)
    except BaseException as error:
        primary = error
    candidate, context = owned.get('candidate'), owned.get('context')
    if context is not None:
        # Inactive native frames can retain a model after construction/integrity
        # fails. Drop their locals before genuine weakref-based release/exit.
        pending, seen = [primary] if primary is not None else [], set()
        while pending:
            error = pending.pop()
            if id(error) not in seen:
                seen.add(id(error))
                traceback.clear_frames(error.__traceback__)
                pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
        source, legacy = context['legacy']['source_driver'], context['legacy']
        def original_rng_flags():
            if rng is not None:
                import torch
                require(torch.equal(rng, torch.random.get_rng_state()) and
                        source.numerical_flags() == legacy['selected']['source_cpu']['numerical_flags'] and
                        (cuda_rng is None or all(torch.equal(a, b) for a, b in
                            zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))),
                        'original constructor/calibration/comparison RNG or flags changed')
        def origins():
            context['old'].audit_origins(legacy)
            for path, digest in legacy['origins']['files'].items():
                candidate.bound_file(context['guards'], path, digest)
        primary = cleanup_steps(primary, [('original_rng_flags', original_rng_flags),
            ('release', lambda: candidate.release(context, state) if state is not None else candidate.require_no_model(context)),
            ('origins', origins), ('exit_rehash', lambda: candidate.exit_rehash(context))], record)
        def union_rehash():
            # Genuine exit_rehash has just freshly read this augmented union,
            # including driver/launch/interpreter guards. Check exact2 topology
            # separately without repeating every original image/package read.
            original_source(candidate, context)
            require(closure(Path(__file__).absolute().parent, args.execution_sha256, FILES, {}) == owned['code'],
                    'own exact2 exit closure differs')
            require(record['exit_rehash_pass'], 'genuine source/driver union exit failed')
        def resources():
            after = source.cgroup_memory()
            record['cgroup_after'] = after
            legacy['selected']['genuine']['reference'].admit_cgroup(after, owned['launch']['unit'])
            context['old'].zero_events(after)
            record['wall_seconds'] = time.perf_counter() - UNIT_STARTED
            record['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            torch = sys.modules.get('torch')
            record['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch is not None else 0
            require(before is not None, 'original whole-service cgroup baseline absent')
            check_resources(before, after, record['wall_seconds'], record['process_peak_rss_kib'],
                            record['peak_cuda_allocated_bytes'])
        primary = cleanup_steps(primary, [('union_rehash', union_rehash), ('resources', resources)], record)
        record.update(input_guards=context['guards'], origins=legacy.get('origins'),
            numerical_flags=context.get('flags'), phase_seconds=context['phase_seconds'],
            source=context['source'], terminal_cgroups=context['terminal_cgroups'])
    record['pass'] = primary is None
    record['wall_seconds'] = time.perf_counter() - UNIT_STARTED
    if primary is not None:
        record['error'] = {'type': type(primary).__name__, 'message': str(primary), 'notes': getattr(primary, '__notes__', [])}
    if output_created:
        primary = cleanup_steps(primary, [('receipt', lambda: write_receipt(args.output / 'receipt.json', record))], record)
    if primary is not None:
        raise primary
    require(time.perf_counter() - UNIT_STARTED < POLICY['seconds'], 'receipt included whole-service cap exceeded')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Current-byte comparison rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'pass': receipt['pass'], 'output': str(args.output),
        'qualification_eligible': False, 'state_reuse_eligible': False, 'quality_read': False}), flush=True)


if __name__ == '__main__':
    main()
