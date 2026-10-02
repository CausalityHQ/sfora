#!/usr/bin/env python3
"""One discarded native fresh-copy diagnostic; parent owns frozen launch and terminal.

python -B DRIVER_ROOT/compare_quadratic_fresh_copy.py --execution-sha256 SHA
 --authority FILE --authority-sha256 SHA --output NEW_ABSOLUTE_DIRECTORY
Freeze exactly FILES in DRIVER_ROOT/execution.json. Authority has AUTHORITY_KEYS;
FILE={path:canonical_absolute_file,sha256:lowercase64}. The parent supplies the
proposed_helper FILE, driver closure/authority hashes, fresh output and unit.
Original mechanics authority bytes are copied unchanged; no training runs here.
Eight alternating pairs hash two complete frozen complements and one complete
payload on identical live tensors and separately owned equivalent capsules.
Every digest must match; body <=10s and proposed paired median must improve.
Full original admission, integrity, release and fresh uncached exit remain.
This receipt can never qualify training, quality, or state reuse.
"""
if not __debug__:
    raise SystemExit('Diagnostic requires assertions; optimized mode is forbidden')

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time
from types import SimpleNamespace

SCHEMA = 'quadratic-fresh-copy-native-receipt-v1'
AUTHORITY_SCHEMA = 'quadratic-fresh-copy-native-launch-v1'
FILES = {'compare_quadratic_fresh_copy.py', 'test_compare_quadratic_fresh_copy.py'}
BASELINE = {'root': '/home/riomus/runs/sfora-so400-quadratic-readout-source-v5',
            'execution_sha256': 'a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3'}
BASELINE_CODE = {
    'quadratic_encoder_frames.py': '1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db',
    'quadratic_readout.py': '12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6',
    'test_siglip2_quadratic_readout.py': 'd1991583c2e6aff00f875d30d458f29d2c3e0ee113d033ed45172c8caf19b242',
    'train_siglip2_quadratic_readout.py': 'f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b'}
ORIGINAL_AUTHORITY_SHA = 'eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e'
PYTHON = {'path': '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13',
          'sha256': '9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'}
PYTHON_VERSION = '3.13.9 (main, Oct 14 2025, 21:26:54) [Clang 20.1.4 ]'
POLICY = {'seconds': 300, 'host_bytes': 8589934592, 'swap_bytes': 0,
          'cuda_allocated_bytes_exclusive': 10000000000}
LOCKS = ['/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock']
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
AUTHORITY_KEYS = {'schema', 'execution_sha256', 'python', 'baseline', 'original_authority',
                  'proposed_helper', 'output', 'unit', 'both_locks_held', 'resource_policy',
                  'qualification_eligible', 'state_reuse_eligible', 'quality_read'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path.parent.resolve() == path.parent and not path.is_symlink(),
            'canonical absolute path required: ' + str(path))
    return path


def source_bytes(binding, guards):
    require(type(binding) is dict and binding.keys() == {'path', 'sha256'} and
            type(binding['sha256']) is str and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
            'exact FILE required')
    path = canonical(binding['path'])
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'source size/SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), binding['sha256']) == binding['sha256'], 'conflicting FILE authority')
    return raw


def strict_json(raw):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'duplicate JSON key')
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def closure(root, sha, names, guards):
    root = canonical(root)
    require(root.is_dir() and root.resolve() == root, 'canonical closure root required')
    code = strict_json(source_bytes({'path': str(root / 'execution.json'), 'sha256': sha}, guards))
    require(type(code) is dict and code.keys() == set(names), 'exact code closure required')
    for name, digest in code.items():
        source_bytes({'path': str(root / name), 'sha256': digest}, guards)
    return code


def check_authority(value, args):
    require(type(value) is dict and value.keys() == AUTHORITY_KEYS and
            value['schema'] == AUTHORITY_SCHEMA and value['execution_sha256'] == args.execution_sha256 and
            value['python'] == PYTHON and value['baseline'] == BASELINE and
            value['resource_policy'] == POLICY and value['both_locks_held'] is True and
            all(value[k] is False for k in ('qualification_eligible', 'state_reuse_eligible', 'quality_read')) and
            value['output'] == str(args.output) and type(value['unit']) is str and
            re.fullmatch('[A-Za-z0-9_.@-]+', value['unit']) and not value['unit'].endswith('.service'),
            'fixed diagnostic authority differs')
    for key in ('original_authority', 'proposed_helper'):
        binding = value[key]
        require(type(binding) is dict and binding.keys() == {'path', 'sha256'} and
                type(binding['sha256']) is str and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
                'diagnostic FILE authority differs')
        canonical(binding['path'])
    require(value['original_authority']['sha256'] == ORIGINAL_AUTHORITY_SHA and
            Path(value['proposed_helper']['path']).name == 'quadratic_encoder_frames.py',
            'original/proposed source authority differs')


def exclusive_output(value, roots):
    output = canonical(value)
    require(not output.exists() and all(not output.is_relative_to(root) and
            not Path(root).is_relative_to(output) for root in roots), 'exclusive separate output required')
    return output


def cli(root, args):
    return [str(Path(root) / 'compare_quadratic_fresh_copy.py'), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--output', str(args.output)]


def prepare(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None and
            not any(n.split('.')[0] in NATIVE for n in sys.modules), 'unoptimized -B source-only startup required')
    root = Path(__file__).absolute().parent
    require(sys.argv == cli(root, args), 'fixed canonical CLI order required')
    guards = {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = strict_json(source_bytes({'path': str(args.authority), 'sha256': args.authority_sha256}, guards))
    check_authority(launch, args)
    original_root = Path(BASELINE['root'])
    helper_root = canonical(launch['proposed_helper']['path']).parent
    require(not root.is_relative_to(original_root) and not original_root.is_relative_to(root) and
            not helper_root.is_relative_to(original_root) and not original_root.is_relative_to(helper_root),
            'separate authenticated source roots required')
    exclusive_output(args.output, [root, original_root, helper_root])
    require(closure(original_root, BASELINE['execution_sha256'], BASELINE_CODE, guards) == BASELINE_CODE,
            'fixed original four-file baseline differs')
    original_authority = canonical(launch['original_authority']['path'])
    require(not original_authority.is_relative_to(original_root), 'copied original authority required')
    source_bytes(launch['original_authority'], guards)
    source_bytes(launch['proposed_helper'], guards)
    require(str(Path(sys.executable).resolve()) == PYTHON['path'] and sys.version == PYTHON_VERSION,
            'original interpreter path/version differs')
    with canonical(PYTHON['path']).open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == PYTHON['sha256'], 'original interpreter SHA256 differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'fixed CUDA/fresh systemd launch required')
    guards[PYTHON['path']] = PYTHON['sha256']
    path = original_root / 'train_siglip2_quadratic_readout.py'
    raw = source_bytes({'path': str(path), 'sha256': BASELINE_CODE[path.name]}, guards)
    name = '_quadratic_fresh_copy_baseline'
    require(name not in sys.modules, 'bare baseline already loaded')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'bare baseline origin differs')
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    exec(compile(raw, str(path), 'exec'), vars(old))
    original_args = SimpleNamespace(execution_sha256=BASELINE['execution_sha256'],
        authority=original_authority, authority_sha256=ORIGINAL_AUTHORITY_SHA,
        phase='mechanics', arm='control', seed=179061, output=args.output)
    context = old.authority(original_args)
    for path, digest in guards.items():
        require(context['guards'].setdefault(path, digest) == digest, 'original/driver guard conflict')
    binding = launch['proposed_helper']
    helper = context['genuine'].load_helper('_quadratic_fresh_copy_proposed', Path(binding['path']),
                                          binding['sha256'], context['guards'])
    require(helper.ORIGINAL_SHA256 == context['frames'].ORIGINAL_SHA256 and callable(helper.seal) and
            not any(n.split('.')[0] in NATIVE for n in sys.modules), 'proposed helper provenance/startup differs')
    return old, context, helper, launch, code


def replace_capsules(value, baseline, replacements):
    require(type(value) is dict and all(value.get(key) is capsule for key, capsule in
            zip(('encoder', 'partition'), baseline, strict=True)), 'owned comparison capsule differs')
    return {**value, 'encoder': replacements[0], 'partition': replacements[1]}


def paired_samples(baseline, proposed, synchronize, rows, clock=time.perf_counter, expected=None):
    started, reference = clock(), expected
    for pair in range(8):
        names = ('baseline', 'proposed') if pair % 2 == 0 else ('proposed', 'baseline')
        row = {'pair': pair, 'order': list(names)}
        rows.append(row)
        for name in names:
            synchronize()
            tick = clock()
            try:
                digests = tuple((baseline if name == 'baseline' else proposed)())
            finally:
                primary = sys.exception()
                final_error = cleanup_steps(primary, [('synchronize', synchronize)], {})
                if primary is None and final_error is not None:
                    raise final_error
            row[name] = {'seconds': clock() - tick, 'digests': digests}
            require(len(digests) == 3 and all(type(d) is str and re.fullmatch('[0-9a-f]{64}', d) for d in digests) and
                    digests[0] == digests[1], 'two complement/complete-payload digests differ')
            require(reference is None or reference == digests, 'paired or consecutive digests differ')
            reference = digests
            require(clock() - started <= 10, '10-second comparison body exceeded')
    delta = statistics.median(row['proposed']['seconds'] - row['baseline']['seconds'] for row in rows)
    old_median = statistics.median(row['baseline']['seconds'] for row in rows)
    new_median = statistics.median(row['proposed']['seconds'] for row in rows)
    duration = clock() - started
    require(duration <= 10, '10-second comparison body exceeded')
    return {'body_seconds': duration, 'paired_median_delta_seconds': delta,
            'baseline_median_seconds': old_median, 'proposed_median_seconds': new_median,
            'cost_improved': delta < 0 and new_median < old_median, 'exact_digests_every_pair': True}


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
                try:
                    primary.add_note(label + ' cleanup failed: ' + str(error))
                except BaseException:
                    pass
    return primary


def native_probes(original, baseline, proposed, capsules, ordinary_capsules):
    import torch
    cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6).T[::2]
    cuda = torch.arange(24, dtype=torch.float32, device='cuda').reshape(4, 6).T[::2]
    scalar = torch.tensor(3., dtype=torch.float64, device='cuda')
    cpu_scalar = torch.tensor(2, dtype=torch.int16)
    dtypes = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
              torch.float16, torch.bfloat16, torch.float32, torch.float64]
    ordinary = {'encoder': ordinary_capsules[0], 'partition': ordinary_capsules[1],
                'mixed': {1: (cpu, cuda, scalar, cpu_scalar, cuda), '1': [cuda, cpu]},
                'dtype': [(torch.arange(4).to(dtype), torch.arange(4, device='cuda').to(dtype)) for dtype in dtypes],
                'empty': [torch.empty((0, 3)), torch.empty((0, 3), device='cuda')]}
    old_value = {**ordinary, 'encoder': capsules[0][0], 'partition': capsules[0][1]}
    new_value = replace_capsules(old_value, capsules[0], capsules[1])
    expected = original(ordinary)
    require(baseline(old_value) == proposed(new_value) == expected, 'native dtype/scalar/empty/strided/alias differential differs')
    consumed = []
    original(ordinary, consumed=lambda t: consumed.append(id(t)))
    for fingerprint, value in ((baseline, old_value), (proposed, new_value)):
        order = []
        require(fingerprint(value, consumed=lambda t: order.append(id(t))) == expected and order == consumed,
                'native consumed fallback/order differs')
    version = cuda._version
    cuda.data[0, 0] += 7
    require(cuda._version == version, 'isolated .data probe unexpectedly changed version')
    changed = original(ordinary)
    require(changed != expected and baseline(old_value) == proposed(new_value) == changed and
            baseline(old_value) == proposed(new_value), 'fresh unchanged-version .data mutation missed')
    return {'ordinary_baseline_proposed_exact': True, 'mixed_cpu_cuda_dtypes': True,
            'scalar_empty_strided_repeated_alias_exact': True, 'consumed_fallback_order_exact': True,
            'unchanged_version_data_mutation_detected': True, 'isolated_probe_only': True,
            'original_digest': expected, 'mutated_digest': changed}


def live_snapshot(old, context, state, ident, capsules, metadata):
    import torch
    def versions(value):
        if isinstance(value, torch.Tensor):
            return [(value.data_ptr(), value._version, str(value.dtype), tuple(value.shape),
                     tuple(value.stride()), value.requires_grad, value.grad is None, value.grad_fn is None)]
        if type(value) is dict:
            return [v for key in sorted(value, key=repr) for child in (key, value[key]) for v in versions(child)]
        if type(value) in (tuple, list):
            return [v for child in value for v in versions(child)]
        return []
    original = context['original'].fingerprint
    full = replace_capsules(old.payload(context, state, ident), capsules, metadata)
    frozen = replace_capsules(old.frozen_tree(context, state), capsules, metadata)
    actual = [old.frozen_tensors(state), state['params'], state['bank'], state['target'], state['positive'],
              state['schedules'], state['masks'], state['original_rows'], state['pca'], state['optimizer'].state_dict()]
    return {'complete_payload_sha256': original(full), 'frozen_live_bytes_sha256': original(frozen),
            'live_versions': versions(actual), 'counter': state['counter'],
            'optimizer_sha256': original([state['optimizer'].state_dict(), state['optimizer'].defaults]),
            'cpu_rng_sha256': original(torch.random.get_rng_state()),
            'cuda_rng_sha256': original(torch.cuda.get_rng_state_all()),
            'numerical_flags': context['source_driver'].numerical_flags()}


def compare_live(old, context, helper, state, ident, record):
    import torch
    baseline_capsules = context['encoder'], context['partition']
    metadata = tuple(c.materialize() for c in baseline_capsules)
    proposed_capsules, proposed, check_owned = helper.seal(context['original'].fingerprint, *metadata)
    require(len(proposed_capsules) == 2 and all(check_owned(i, c) is c for i, c in enumerate(proposed_capsules)) and
            all(c.materialize() == m for c, m in zip(proposed_capsules, metadata, strict=True)),
            'equivalent separately owned proposed metadata differs')
    baseline = context['encoder_fingerprint']
    for check, foreign in ((context['encoder_check'], proposed_capsules[0]),
                           (context['partition_check'], proposed_capsules[1]),
                           (lambda c: check_owned(0, c), baseline_capsules[0]),
                           (lambda c: check_owned(1, c), baseline_capsules[1])):
        try:
            check(foreign)
        except ValueError:
            continue
        raise ValueError('foreign comparison capsule admitted')
    before = live_snapshot(old, context, state, ident, baseline_capsules, metadata)
    record['state_before'] = before
    primary = None
    try:
        record['native_differential'] = native_probes(context['original'].fingerprint, baseline, proposed,
                                                     (baseline_capsules, proposed_capsules), metadata)
        def work(fingerprint, replacements):
            def sample():
                first = fingerprint(replace_capsules(old.frozen_tree(context, state), baseline_capsules, replacements))
                second = fingerprint(replace_capsules(old.frozen_tree(context, state), baseline_capsules, replacements))
                complete = fingerprint(replace_capsules(old.payload(context, state, ident), baseline_capsules, replacements))
                return first, second, complete
            return sample
        record['pairs'] = []
        tick = time.perf_counter()
        try:
            record.update(paired_samples(work(baseline, baseline_capsules), work(proposed, proposed_capsules),
                torch.cuda.synchronize, record['pairs'], expected=(before['frozen_live_bytes_sha256'],
                    before['frozen_live_bytes_sha256'], before['complete_payload_sha256'])))
        finally:
            record['body_seconds'] = time.perf_counter() - tick
        require(record['body_seconds'] <= 10, '10-second comparison body exceeded')
        require(record['cost_improved'], 'no lower proposed paired median; intervention rejected')
    except BaseException as error:
        primary = error
    def unchanged():
        record['state_after'] = live_snapshot(old, context, state, ident, baseline_capsules, metadata)
        require(before == record['state_after'], 'complete RNG/live bytes/versions/optimizer/counter changed')
    primary = cleanup_steps(primary, [('state_preserved', unchanged),
        ('final_integrity', lambda: old.integrity(context, state, ident, fresh_bytes=True))], record)
    if primary is not None:
        raise primary


def write_receipt(path, value):
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args):
    started = time.perf_counter()
    old, context, helper, launch, code = prepare(args)
    source, prior = context['source_driver'], context['selected']['source_cpu']['invocation']
    before = source.cgroup_memory()
    old.zero_events(before)
    unit = Path(before['path']).name.removesuffix('.service')
    require(unit == launch['unit'], 'actual diagnostic unit differs')
    context['admission'].init.admit_cgroup(before, unit)
    require(os.environ['INVOCATION_ID'] not in context['invocations'] and
            all(prior[k] == v for k, v in (('python', PYTHON['path']), ('python_sha256', PYTHON['sha256']),
                                          ('python_version', PYTHON_VERSION))), 'original interpreter/fresh invocation differs')
    args.output.mkdir()
    record = {'schema': SCHEMA, 'pass': False, 'qualification_eligible': False, 'state_reuse_eligible': False,
        'quality_read': False, 'training_qualified': False, 'training_state_discarded': True,
        'completed_step': 0, 'checkpoint': None, 'trained_state_reused': False,
        'output': str(args.output), 'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
        'launch': launch, 'execution_sha256': args.execution_sha256, 'code': code, 'baseline': BASELINE,
        'baseline_code': context['code'], 'original_launch': context['launch'],
        'original_authority': launch['original_authority'], 'proposed_helper': launch['proposed_helper'],
        'resource_policy': POLICY, 'locks': LOCKS, 'both_locks_held_in_parent_authority': True,
        'terminal_exit_and_both_locks_require_parent_receipt': True, 'cgroup_before': before,
        'source': context['source'], 'terminal_cgroups': context['terminal_cgroups'],
        'invocation': {'argv': list(sys.argv), 'original_mechanics_argv': old.cli(context['root'],
            context['args'].authority, ORIGINAL_AUTHORITY_SHA, BASELINE['execution_sha256'],
            'mechanics', 'control', 179061, args.output), 'python': PYTHON['path'],
            'python_sha256': PYTHON['sha256'], 'python_version': sys.version, 'optimize': sys.flags.optimize,
            'dont_write_bytecode': sys.dont_write_bytecode, 'pid': os.getpid(),
            'unit': unit, 'invocation_id': os.environ['INVOCATION_ID'], 'cuda_visible_devices': '0',
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}}
    state, primary, cpu_rng = None, None, None
    try:
        tick = time.perf_counter()
        old.prepare_native(context)
        old.add_seconds(context, 'native_admission', tick)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
                'one uninitialized CUDA device required')
        # Identical original startup seeds; the entire subsequent diagnostic preserves RNG.
        torch.random.default_generator.manual_seed(179061)
        torch.cuda.manual_seed_all(179061)
        cpu_rng = torch.random.get_rng_state().clone()
        state = old.fresh(context, 'control', 179061, 'cuda')
        require(torch.equal(cpu_rng, torch.random.get_rng_state()), 'original constructor CPU RNG changed')
        ident = old.identity(context, state)
        cpu_ident = context['terminals']['cpu:control']['arms']['control']['identity']
        require(ident == {**cpu_ident, 'device': 'cuda', 'seed': 179061,
            'schedule_sha256': context['original'].fingerprint(state['schedules']['179061']),
            'full_schedule_sha256': context['original'].fingerprint(state['schedules']['179061'])},
            'original CPU-qualified native identity differs')
        record['identity'] = ident
        old.integrity(context, state, ident, fresh_bytes=True)
        compare_live(old, context, helper, state, ident, record)
    except BaseException as error:
        primary = error
    def original_rng_flags():
        if cpu_rng is not None:
            import torch
            require(torch.equal(cpu_rng, torch.random.get_rng_state()) and source.numerical_flags() == context['flags'],
                    'original constructor/diagnostic RNG or flags changed')
    primary = cleanup_steps(primary, [
        ('original_rng_flags', original_rng_flags),
        ('release', lambda: old.release(context, state) if state is not None else old.require_no_model(context)),
        ('exit_rehash', lambda: old.exit_rehash(context))], record)
    def resources():
        import torch
        after = source.cgroup_memory()
        record['cgroup_after'] = after
        old.zero_events(after)
        context['admission'].init.admit_cgroup(after, unit)
        record['wall_seconds'] = time.perf_counter() - started
        record['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        require(before['path'] == after['path'] and int(after['values']['memory.peak']) >= int(before['values']['memory.peak']) and
                0 < record['wall_seconds'] < POLICY['seconds'] and
                0 < record['process_peak_rss_kib'] <= 8 * 1024**2 and
                record['peak_cuda_allocated_bytes'] < POLICY['cuda_allocated_bytes_exclusive'],
                'whole-service time/RSS/lifetime CUDA cap differs')
    primary = cleanup_steps(primary, [('resources', resources)], record)
    record.update(input_guards=context['guards'], origins=context.get('origins'),
                  numerical_flags=context.get('flags'), phase_seconds=context['phase_seconds'],
                  wall_seconds=time.perf_counter() - started)
    record['pass'] = primary is None
    if primary is not None:
        record['error'] = {'type': type(primary).__name__, 'message': str(primary),
                           'notes': getattr(primary, '__notes__', [])}
    primary = cleanup_steps(primary, [('receipt', lambda: write_receipt(args.output / 'receipt.json', record))], record)
    if primary is not None:
        raise primary
    require(time.perf_counter() - started < POLICY['seconds'], 'receipt included whole-service cap exceeded')
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
        raise SystemExit('Fresh-copy diagnostic rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'pass': receipt['pass'], 'output': str(args.output),
                      'qualification_eligible': False, 'state_reuse_eligible': False, 'quality_read': False}), flush=True)


if __name__ == '__main__':
    main()
