#!/usr/bin/env python3
"""Original image-PCA128 initialization of ONE fresh native256 FIT export.

Parent launch schema native256-fit-initializer-launch-v1 has exactly:
  schema, execution_sha256, pca_helper_sha256, export_root,
  export_execution_sha256, export_authority: {path, sha256},
  selected_export, startup, resource_policy, both_locks_held (true).
selected_export/startup each have exactly:
  receipt: {path, sha256}, log: {path, sha256}, unit, invocation_id,
  service_seconds, native_peak_rss_kib, both_locks_held (true).
All SHA256 values come from actual parent-frozen bytes, never future guesses.
resource_policy equals POLICY below. Export/source authority and original
CPU proof/log/checkpoint metadata are admitted by the separately pinned
exporter's two-file closure and immutable original source's three-file closure.
No source model/checkpoint is constructed or torch-loaded here.

Freeze execution.json beside this script with exactly FILES below. Copy the
unchanged src/sfora/representation_ceiling.py as representation_ceiling.py;
pca_helper_sha256 independently pins those original bytes. The helper loads
as a bare file after admission, without sfora.__init__ or training libraries.
External exporter/source closures stay separate and unchanged.

CLI: --execution-sha256 SHA --authority PATH --authority-sha256 SHA
     --arm {large,so400} --output ABSOLUTE_NEW_DIRECTORY
Output: exclusive initializers.npz and receipt.json. The parent encloses ONE
CPU stage in 120s/8GiB/noSwap/CUDA-hidden with BOTH admission locks, retains
the original log and attaches normal-exit/final-cgroup evidence afterwards.
There is no startup mode, chunking, rescue or quality read. Costs describe
preparation only. Full initialized-model CPU/reload and native arithmetic
parity are later parent-owned gates, not qualified by this receipt.
Local help/tests need only python3 -B -S, with no native imports.
"""
if not __debug__:
    raise SystemExit('Initialization requires assertions; optimized mode is forbidden')

import argparse
import ast
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
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'siglip2-substrate-fit-initializer-v1'
AUTHORITY_SCHEMA = 'native256-fit-initializer-launch-v1'
FILES = {'initialize_siglip2_substrate_fit.py', 'test_siglip2_substrate_initializer.py',
         'representation_ceiling.py'}
EXPORT_FILES = {'export_siglip2_substrate_fit.py', 'test_siglip2_substrate_fit.py'}
SOURCE_FILES = {'extract_siglip2_vision_source.py', 'qualify_siglip2_substrate_cpu.py',
                'test_siglip2_substrate_cpu.py'}
POLICY = {'seconds': 120, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0,
          'cuda_visible_devices': ''}
WIDTHS = {'large': 1024, 'so400': 1152}
NATIVE_PACKAGES = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision'}
EXPORT_ARITHMETIC = ('unaugmented stock native256; B32+tail; FP32 vision; CUDA FP16 autocast; '
                     'pooled.float(); CUDA F.normalize(dim=1); CPU NumPy FP32')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            'canonical regular file required: ' + str(path))
    return path


def digest_string(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256 required')


def sha(path):
    with canonical(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_file(guards, path, expected):
    digest_string(expected)
    path = canonical(path)
    require(sha(path) == expected, 'bound file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(path, expected, guards):
    path = bound_file(guards, path, expected)
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected,
            'authority JSON size/SHA256 changed during read')
    return strict_json(raw)


def descriptor_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'file descriptor differs')
    return read_json(value['path'], value['sha256'], guards)


def closure(root, expected, names):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    guards = {}
    code = read_json(root / 'execution.json', expected, guards)
    require(isinstance(code, dict) and code.keys() == names, 'execution requires exactly declared files')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def load_bare(name, path, expected):
    """Authenticate bytes before execution and reject preloaded/shadow helpers."""
    path = bound_file({}, path, expected)
    require(name not in sys.modules, 'helper already loaded; origin is not admissible: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path,
            'bare helper import origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need their real module registered.
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'helper SHA256 changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))  # Execute pinned bytes, never an unpinned .pyc.
    require(Path(module.__file__) == path and Path(module.__spec__.origin) == path and sha(path) == expected,
            'bare helper loaded origin/SHA256 differs')
    return module


def admit_cgroup(value, unit):
    path, values = Path(value['path']), value['values']
    require(path.is_absolute() and '..' not in path.parts and path.name == unit + '.service',
            'enclosing unit cgroup differs')
    require(values['memory.max'] == str(POLICY['host_bytes']) and
            0 < int(values['memory.current']) <= int(values['memory.peak']) <= POLICY['host_bytes'] and
            all(values[k] == '0' for k in ('memory.swap.current', 'memory.swap.peak', 'memory.swap.max')),
            'whole-cgroup memory/swap caps differ')
    events = dict(line.split() for line in values['memory.events'].splitlines())
    require(all(events.get(k) == '0' for k in ('max', 'oom', 'oom_kill')), 'whole-cgroup memory failure events')


def admit_terminal(record, descriptor, seconds, guards):
    require(descriptor.keys() == {'receipt', 'log', 'unit', 'invocation_id', 'service_seconds',
                                  'native_peak_rss_kib', 'both_locks_held'} and
            descriptor['both_locks_held'] is True, 'terminal descriptor/both locks differ')
    identity, unit = descriptor['invocation_id'], descriptor['unit']
    require(isinstance(identity, str) and re.fullmatch('[0-9a-f]{32}', identity) is not None and
            isinstance(unit, str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit) is not None and
            record['invocation']['invocation_id'] == identity and record['invocation']['optimize'] == 0,
            'original terminal invocation differs')
    require(0 < record['wall_seconds'] <= descriptor['service_seconds'] <= seconds and
            0 < record['process_peak_rss_kib'] <= descriptor['native_peak_rss_kib'] <= 8 * 1024**2,
            'original whole-service duration/RSS caps differ')
    log_descriptor = descriptor['log']
    require(log_descriptor.keys() == {'path', 'sha256'}, 'log descriptor differs')
    log = bound_file(guards, log_descriptor['path'], log_descriptor['sha256']).read_text()
    lines = log.splitlines()
    required = [f'Running as unit: {unit}.service; invocation ID: {identity}', '\tExit status: 0',
                'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
                '\tSwaps: 0', 'Memory swap peak: 0B',
                f"\tMaximum resident set size (kbytes): {descriptor['native_peak_rss_kib']}"]
    require(all(lines.count(line) == 1 for line in required), 'original normal-exit log differs')
    runtimes = [line.removeprefix('Service runtime: ') for line in lines if line.startswith('Service runtime: ')]
    require(len(runtimes) == 1, 'original service runtime line differs')
    match = re.fullmatch(r'(?:(\d+)min )?(\d+(?:\.\d+)?)s', runtimes[0])
    require(match is not None, 'original service runtime format differs')
    minutes, native_seconds = match.groups()
    duration = Decimal(minutes or '0') * 60 + Decimal(native_seconds)
    require((minutes is None or Decimal(native_seconds) < 60) and
            duration == Decimal(str(descriptor['service_seconds'])), 'original service runtime numeric binding differs')
    footers = [strict_json(line[len('FINAL_CGROUP '):]) for line in lines if line.startswith('FINAL_CGROUP ')]
    require(len(footers) == 1 and footers[0]['invocation_id'] == identity, 'original final cgroup footer differs')
    final = footers[0]
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        admit_cgroup(value, unit)
        require(value['path'] == final['path'], 'enclosing cgroup changed')
    require(int(final['values']['memory.peak']) >= int(record['cgroup_after']['values']['memory.peak']) >=
            int(record['cgroup_before']['values']['memory.peak']), 'complete whole-unit peak differs')
    return final


def cache_facts(path, expected, width, guards):
    """Check complete .npy FP32 data, shape and unit rows using only stdlib."""
    path = bound_file(guards, path, expected)
    with path.open('rb') as stream:
        magic = stream.read(8)
        require(magic in (b'\x93NUMPY\x01\x00', b'\x93NUMPY\x02\x00'), 'cache NPY version differs')
        size = 2 if magic[-2] == 1 else 4
        length = int.from_bytes(stream.read(size), 'little')
        require(0 < length <= 65536, 'cache NPY header size differs')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        require(isinstance(header, dict) and header.keys() == {'descr', 'fortran_order', 'shape'} and header['descr'] == '<f4' and
                header['fortran_order'] is False and header['shape'] == (13283, width),
                'cache shape/dtype/layout differs')
        require(path.stat().st_size == 8 + size + length + 13283 * width * 4, 'cache payload size differs')
        unpack = struct.Struct('<' + str(width) + 'f')
        max_error = 0.0
        for _ in range(13283):
            values = unpack.unpack(stream.read(unpack.size))
            require(all(math.isfinite(value) for value in values), 'cache nonfinite FP32 row')
            error = abs(math.sqrt(math.fsum(value * value for value in values)) - 1.0)
            require(error <= 1e-5, 'cache row unit norm differs')
            max_error = max(max_error, error)
        require(stream.read(1) == b'', 'cache trailing payload differs')
    require(sha(path) == expected, 'cache SHA256 changed during validation')
    return {'shape': [13283, width], 'dtype': 'float32', 'finite': True,
            'unit_norm_atol': 1e-5, 'maximum_unit_norm_error': max_error}


def admit_export(exporter, context, launch, record, startup, guards):
    require(record['schema'] == exporter.SCHEMA and record['phase'] == 'export' and
            record['binding'] == exporter.binding(context) and record['resource_policy'] == exporter.EXPORT_POLICY and
            record['arithmetic'] == EXPORT_ARITHMETIC and
            all(record[k] is True for k in ('pass', 'exported', 'fresh_source', 'exit_rehash_pass',
                                            'source_cpu_runtime_and_first2_exact', 'constructor_rng_preserved',
                                            'terminal_exit_and_both_locks_require_parent_receipt')) and
            all(record[k] is False for k in ('gradients_created', 'optimizer_created', 'source_features_reused',
                                           'teacher_state_reused', 'quality_read', 'initializer_qualified',
                                           'training_qualified', 'quality_qualified', 'cuda_peak_reset')) and
            record['updates'] == 0, 'actual fresh export admission/profile differs')
    require(record['startup'] == launch['startup']['receipt'] and
            record['source_checkpoint_metadata_only'] == context['proof']['checkpoint'],
            'export startup/source CPU binding differs')
    cache, selected = record['cache'], launch['selected_export']['receipt']
    receipt = canonical(selected['path'])
    require(receipt.name == 'receipt.json' and cache.keys() ==
            {'path', 'sha256', 'shape', 'dtype', 'normalized', 'raw_pooled_cache'} and
            Path(cache['path']) == receipt.parent / 'fit.npy' and
            cache['shape'] == [13283, WIDTHS[context['export_args'].arm]] and cache['dtype'] == 'float32' and
            cache['normalized'] is True and cache['raw_pooled_cache'] is False,
            'selected export cache role/shape/dtype differs')
    counters = {'images': 13283, 'classes': 2004, 'batches': 416,
                'batch_sizes': [32] * (13283 // 32) + [13283 % 32],
                'calibration_images': 4, 'cpu_witness_images': 2, 'optimizer_updates': 0}
    require(record['counters'] == counters and 0 < record['export_seconds'] <= record['wall_seconds'] <= 300 and
            0 < record['complete_unit_peak_cuda_allocated_bytes'] < 10_000_000_000,
            'export counters/cost/complete-unit CUDA allocation differs')
    cosines = record['fp32_autocast_first4_cosines']
    require(len(cosines) == 4 and all(type(c) in (int, float) and math.isfinite(c) and .999 <= c <= 1.00001
                                    for c in cosines), 'source numerical calibration differs')
    identity, prior = record['invocation'], startup['invocation']
    require(identity['cuda_visible_devices'] not in (None, '') and
            all(identity[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
            record['cpu_numerical_flags'] == context['proof']['numerical_flags'], 'export CPU/interpreter differs')
    args = context['export_args']
    expected_argv = [str(context['own_root'] / 'export_siglip2_substrate_fit.py'),
                     '--execution-sha256', args.execution_sha256,
                     '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
                     '--arm', args.arm, '--startup', startup_path(launch),
                     '--startup-sha256', launch['startup']['receipt']['sha256'], '--output', str(receipt.parent)]
    require(identity['argv'] == expected_argv, 'selected export original argv differs')
    expected_guards = {**context['guards'], startup_path(launch): launch['startup']['receipt']['sha256']}
    require(all(record['input_guards'].get(path) == digest for path, digest in expected_guards.items()),
            'export source/startup input guards differ')
    origins = record['origins']
    require(origins['packages'] == startup['packages'] == context['proof']['origins']['packages'] and
            all(record['input_guards'].get(path) == digest for path, digest in origins['files'].items()),
            'export original package/file origins differ')
    for name, path in origins['modules'].items():
        package = name.split('.')[0]
        require(package in origins['packages'] and Path(path).is_relative_to(Path(origins['packages'][package]['root'])) and
                path in origins['files'], 'export loaded module origin differs')
    require(set(origins['native_files']) <= origins['files'].keys(), 'export native origin guards differ')
    for path, digest in record['input_guards'].items():
        bound_file(guards, path, digest)
    rgb, fit = record['rgb_manifest'], context['fit']
    require(len(rgb) == 13283 and record['ordered_rgb_sha256'] == exporter.object_sha(rgb),
            'complete ordered RGB binding differs')
    for index, (actual, row, target, path) in enumerate(zip(rgb, fit['rows'], fit['targets'], context['all_images'])):
        require(actual.keys() == {'ordinal', 'train_row', 'target', 'relative_path', 'path',
                                  'image_sha256', 'mode', 'size', 'rgb_sha256'} and
                actual['ordinal'] == index and actual['train_row'] == row['train_row'] and
                actual['target'] == target and actual['relative_path'] == row['relative_path'] and
                actual['path'] == str(path) and actual['image_sha256'] == row['image_sha256'] and
                actual['mode'] == 'RGB' and len(actual['size']) == 2 and
                all(type(n) is int and n > 0 for n in actual['size']), 'ordered FIT/RGB row differs')
        digest_string(actual['rgb_sha256'])
    require([{k: row[k] for k in context['proof']['sample']['images'][0]} for row in rgb[:2]] ==
            context['proof']['sample']['images'], 'source CPU/export first-two RGB differ')
    ordered = {'rows': fit['rows'], 'targets': fit['targets'], 'class_names': fit['class_names'],
               'resolved_paths': [str(path) for path in context['all_images']]}
    require(record['ordered_input_sha256'] == exporter.object_sha(ordered), 'original ordered FIT authority differs')


def startup_path(launch):
    return launch['startup']['receipt']['path']


def authority(args):
    require(not any(name.split('.')[0] in NATIVE_PACKAGES or name == 'sfora' for name in sys.modules),
            'native packages must not precede authority admission')
    root = Path(__file__).absolute().parent
    code = closure(root, args.execution_sha256, FILES)
    guards = {str(root / 'execution.json'): args.execution_sha256,
              **{str(root / name): digest for name, digest in code.items()}}
    launch = read_json(args.authority, args.authority_sha256, guards)
    require(launch.keys() == {'schema', 'execution_sha256', 'pca_helper_sha256', 'export_root',
                             'export_execution_sha256', 'export_authority', 'selected_export', 'startup',
                             'resource_policy', 'both_locks_held'} and
            launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
            launch['pca_helper_sha256'] == code['representation_ceiling.py'] and
            launch['resource_policy'] == POLICY and launch['both_locks_held'] is True,
            'parent initializer launch authority/profile/locks differ')
    export_root = Path(launch['export_root'])
    export_code = closure(export_root, launch['export_execution_sha256'], EXPORT_FILES)
    export_launch = descriptor_json(launch['export_authority'], guards)
    source_root = Path(export_launch['source_root'])
    source_code = closure(source_root, export_launch['source_execution_sha256'], SOURCE_FILES)
    require(len({root, export_root, source_root}) == 3 and
            not any(args.output.is_relative_to(path) for path in (root, export_root, source_root)),
            'initializer/export/original source closures and output must be separate')
    for origin, expected, manifest in ((export_root, launch['export_execution_sha256'], export_code),
                                       (source_root, export_launch['source_execution_sha256'], source_code)):
        guards[str(origin / 'execution.json')] = expected
        guards.update({str(origin / name): digest for name, digest in manifest.items()})
    exporter = load_bare('_siglip2_pinned_fit_export', export_root / 'export_siglip2_substrate_fit.py',
                         export_code['export_siglip2_substrate_fit.py'])
    require(exporter.FILES == EXPORT_FILES and exporter.SOURCE_FILES == SOURCE_FILES and
            exporter.STARTUP_POLICY == POLICY, 'pinned exporter closure/policy differs')
    # The original factory's bootstrap will reuse these exact registered origins.
    # Keep its immutable implementation while excluding unpinned bytecode caches.
    for name in ('extract_siglip2_vision_source', 'qualify_siglip2_substrate_cpu'):
        load_bare(name, source_root / (name + '.py'), source_code[name + '.py'])
    export_args = SimpleNamespace(execution_sha256=launch['export_execution_sha256'],
        authority=Path(launch['export_authority']['path']), authority_sha256=launch['export_authority']['sha256'],
        arm=args.arm, output=args.output)
    context = exporter.authority(export_args)  # stdlib provenance only; NEVER fresh_source().
    startup = descriptor_json(launch['startup']['receipt'], guards)
    exporter.admit_startup(context, startup)
    startup_final = admit_terminal(startup, launch['startup'], 120, guards)
    expected_startup_argv = [str(export_root / 'export_siglip2_substrate_fit.py'),
        '--execution-sha256', export_args.execution_sha256, '--authority', str(export_args.authority),
        '--authority-sha256', export_args.authority_sha256, '--arm', args.arm,
        '--check-startup-only', '--output', startup_path(launch)]
    require(startup['invocation']['argv'] == expected_startup_argv, 'startup original argv differs')
    record = descriptor_json(launch['selected_export']['receipt'], guards)
    admit_export(exporter, context, launch, record, startup, guards)
    export_final = admit_terminal(record, launch['selected_export'], 300, guards)
    facts = cache_facts(record['cache']['path'], record['cache']['sha256'], WIDTHS[args.arm], guards)
    source = context['source_driver']
    packages = source.package_origins(context)  # installed-origin admission still precedes Torch.
    require(packages == startup['packages'], 'initializer installed package origins differ')
    for path, digest in context['guards'].items():
        require(guards.setdefault(path, digest) == digest, 'conflicting source file authority')
    context['guards'] = guards  # One shared exit guard inventory, not duplicate full hash passes.
    require(not any(name.split('.')[0] in NATIVE_PACKAGES for name in sys.modules),
            'authority admission imported native packages')
    return {'root': root, 'code': code, 'guards': guards, 'launch': launch, 'exporter': exporter,
            'source_context': context, 'source': source, 'export': record, 'startup': startup,
            'export_final_cgroup': export_final, 'startup_final_cgroup': startup_final,
            'packages': packages, 'cache_facts_before_native': facts}


def rehash(context):
    context['exporter'].rehash(context['source_context'])


def initialize(args):
    started = time.perf_counter()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.flags.optimize == 0,
            'initializer requires explicit CUDA hidden/unoptimized Python')
    identity = os.environ.get('INVOCATION_ID')
    require(re.fullmatch('[0-9a-f]{32}', identity or '') is not None, 'original systemd invocation ID required')
    context = authority(args)
    source, extract = context['source'], context['source_context']['extract']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    admit_cgroup(before, unit)
    python = canonical(Path(sys.executable).resolve())
    prior = context['startup']['invocation']
    require(str(python) == prior['python'] and sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'initializer source interpreter differs')
    bound_file(context['guards'], python, prior['python_sha256'])
    require(time.perf_counter() - started < 120, 'initializer admission exceeds 120s')
    args.output.mkdir()  # Failure is never reusable as an admitted initializer.
    import torch
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    require(not torch.cuda.is_initialized(), 'initializer must remain CPU-only')
    flags = context['source_context']['proof']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags and torch.get_default_dtype() == torch.float32 and
            str(torch.get_default_device()) == 'cpu' and not torch.is_autocast_enabled('cpu'),
            'original CPU numerical flags differ')
    helper = load_bare('_siglip2_pinned_representation_ceiling', context['root'] / 'representation_ceiling.py',
                       context['launch']['pca_helper_sha256'])
    width = WIDTHS[args.arm]
    cache = np.load(context['export']['cache']['path'], allow_pickle=False)
    require(cache.shape == (13283, width) and cache.dtype == np.float32 and cache.flags.c_contiguous and
            np.isfinite(cache).all() and np.allclose(np.linalg.norm(cache, axis=1), 1, rtol=0, atol=1e-5),
            'native cache shape/dtype/finite/unit norms differ')
    features = torch.from_numpy(cache).float()
    labels = tuple(context['source_context']['fit']['targets'])
    names = tuple(sorted(set(labels)))
    require(names == tuple(range(2004)), 'original FIT class order differs')
    indexes = {label: index for index, label in enumerate(names)}
    rng = torch.random.get_rng_state().clone()
    pca_started = time.perf_counter()
    # Keep the original operations and their order, including repeat normalization.
    normalized = F.normalize(features, dim=1)
    pca = helper.fit_centered_pca(normalized, dimensions=128)
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(width, 128)
    with torch.no_grad():
        head.weight.copy_(pca.components)
        head.bias.copy_(-(pca.components @ pca.mean))
    projected = pca.apply(normalized)  # Original float64 projection/norm -> FP32.
    sums = torch.zeros(len(names), 128)
    counts = torch.zeros(len(names), dtype=torch.int64)
    for row, label in enumerate(labels):
        sums[indexes[label]] += projected[row]
        counts[indexes[label]] += 1
    require(int(counts.min()) >= 1, 'original FIT class proxy inventory differs')
    classifier = nn.Parameter(F.normalize(sums, dim=1))
    # DISTINCT bank path: actual FP32 Linear, outside autocast, then normalization.
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        bank = F.normalize(head(F.normalize(features.float(), dim=1)), dim=1).detach()
    pca_seconds = time.perf_counter() - pca_started
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and
            not torch.cuda.is_initialized() and all(p.grad is None for p in head.parameters()) and
            classifier.grad is None and not bank.requires_grad, 'initializer RNG/flags/gradients/CUDA changed')
    tensors = {'mean': pca.mean, 'components': pca.components, 'head.weight': head.weight.detach(),
               'head.bias': head.bias.detach(), 'classifier': classifier.detach(),
               'bank': bank, 'target': torch.tensor(labels, dtype=torch.int64)}
    expected_shapes = {'mean': [width], 'components': [128, width], 'head.weight': [128, width],
                       'head.bias': [128], 'classifier': [2004, 128], 'bank': [13283, 128], 'target': [13283]}
    require(all(list(value.shape) == expected_shapes[name] and value.device.type == 'cpu' and
                value.dtype == (torch.int64 if name == 'target' else torch.float32) and
                torch.isfinite(value).all().item() for name, value in tensors.items()), 'initializer finite inventory differs')
    require(all(torch.allclose(value.norm(dim=1), torch.ones(len(value)), rtol=0, atol=1e-5)
                for value in (classifier, bank)), 'initializer proxy/bank unit rows differ')
    facts = {name: source.tensor_fact(value) for name, value in tensors.items()}
    pca_sha = hashlib.sha256(pca.mean.numpy().tobytes() + pca.components.numpy().tobytes()).hexdigest()
    origins = source.imported_origins(extract, context['packages'])
    for path, digest in origins['files'].items():
        bound_file(context['guards'], path, digest)
    require(time.perf_counter() - started < 120, 'complete PCA initializer exceeds 120s')
    arrays = {name: value.numpy() for name, value in tensors.items()}
    artifact = args.output / 'initializers.npz'
    with extract.exclusive(artifact) as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(artifact, allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays) and all(np.array_equal(saved[name], value)
                and saved[name].dtype == value.dtype for name, value in arrays.items()), 'serialized initializer arrays differ')
    artifact_sha = sha(artifact)
    rehash(context)
    after = source.cgroup_memory()
    admit_cgroup(after, unit)
    require(before['path'] == after['path'] and
            int(before['values']['memory.peak']) <= int(after['values']['memory.peak']), 'initializer enclosing cgroup changed')
    wall = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(0 < wall < 120 and 0 < rss <= 8 * 1024**2 and not torch.cuda.is_initialized() and
            torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'complete initializer duration/RSS/RNG/flags/CUDA differs')
    result = {'schema': SCHEMA, 'phase': 'pca', 'pass': True, 'prepared': True,
              'arm': args.arm, 'width': width, 'output_dim': 128,
              'authority_sha256': args.authority_sha256, 'execution_sha256': args.execution_sha256,
              'code': context['code'], 'pca_helper_sha256': context['launch']['pca_helper_sha256'],
              'export_root': context['launch']['export_root'],
              'export_execution_sha256': context['launch']['export_execution_sha256'],
              'export_authority': context['launch']['export_authority'],
              'selected_export': context['launch']['selected_export'], 'startup': context['launch']['startup'],
              'source_binding': context['export']['binding'],
              'source_checkpoint_metadata_only': context['export']['source_checkpoint_metadata_only'],
              'source_roles': context['source_context']['proof']['runtime']['roles'],
              'source_roles_sha256': context['exporter'].object_sha(context['source_context']['proof']['runtime']['roles']),
              'cache': context['export']['cache'], 'cache_facts_before_native': context['cache_facts_before_native'],
              'ordered_input_sha256': context['export']['ordered_input_sha256'],
              'ordered_rgb_sha256': context['export']['ordered_rgb_sha256'],
              'class_names': context['source_context']['fit']['class_names'],
              'class_counts': counts.tolist(), 'arrays': facts, 'pca_sha256': pca_sha,
              'artifact': {'path': str(artifact), 'sha256': artifact_sha}, 'serialized_arrays_exact': True,
              'arithmetic': {'pca': 'CPU F.normalize(FP32 cache, dim=1); original centered float64 full SVD/rank/degeneracy/maxabs-positive signs; FP32 mean/components',
                             'head': 'fork_rng CPU nn.Linear(width,128); W=components; bias=-(components@mean)',
                             'proxies': 'original pca.apply(normalized): float64 projection/norm -> FP32; class-ordered sequential FP32 sums/int64 counts; F.normalize(sums,dim=1)',
                             'bank': 'no_grad/outside autocast; head(F.normalize(cache.float(),dim=1)); F.normalize(dim=1); detach'},
              'initializer_roles': [{'name': name, 'shape': expected_shapes[name], 'dtype': facts[name]['dtype'],
                                     'role': 'trainable'} for name in ('head.weight', 'head.bias', 'classifier')],
              'cpu_rng_before': source.tensor_fact(rng), 'cpu_rng_after': source.tensor_fact(torch.random.get_rng_state()),
              'constructor_rng_preserved': True, 'numerical_flags': flags, 'origins': origins,
              'input_guards': context['guards'], 'exit_rehash_pass': True, 'all_arrays_finite': True,
              'gradients_created': False, 'optimizer_created': False, 'updates': 0, 'head_updates': 0,
              'bank_detached': True, 'source_model_loaded': False, 'teacher_state_reused': False,
              'source_features_reused': False, 'proxies_reused': False, 'head_reused': False, 'quality_read': False,
              'initializer_qualified': False, 'training_qualified': False, 'quality_qualified': False,
              'preparation_cost_only': True, 'pca_seconds': pca_seconds, 'wall_seconds': wall,
              'resource_policy': POLICY, 'cgroup_before': before, 'cgroup_after': after,
              'process_peak_rss_kib': rss, 'cuda_initialized': False,
              'both_locks_held_in_parent_authority': context['launch']['both_locks_held'],
              'export_final_cgroup': context['export_final_cgroup'], 'startup_final_cgroup': context['startup_final_cgroup'],
              'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                             'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                             'invocation_id': identity, 'cuda_visible_devices': ''},
              'terminal_exit_and_both_locks_require_parent_receipt': True}
    context['exporter'].write_json(extract, args.output / 'receipt.json', result)
    return result


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
        result = initialize(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError, struct.error) as error:
        raise SystemExit('PCA initializer rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
