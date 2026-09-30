#!/usr/bin/env python3
"""Fresh paired native256 FIT export (one independently capped arm per unit).

Parent launch authority schema native256-fit-export-launch-v1, exact keys:
  schema, execution_sha256, source_root, source_execution_sha256,
  source_cpu_authority: {path, sha256}, sources: {path, sha256},
  fit_manifest: {path, sha256}, startup_policy, export_policy, source_cpu.
source_cpu has exactly large/so400 descriptors, each with exact keys:
  proof: {path, sha256}, log: {path, sha256}, unit, invocation_id,
  service_seconds, native_peak_rss_kib, both_locks_held (true).
The parent supplies all hashes from actual frozen files. Policy dictionaries
must equal STARTUP_POLICY/EXPORT_POLICY below; no configurable rescue caps.
The source_root is the ORIGINAL immutable v4 three-file closure, never this
exporter's separate two-file closure. sources/fit_manifest remain there too.
The source checkpoint is authenticated metadata, never a model loader input.

Freeze exactly this script and test_siglip2_substrate_fit.py in execution.json.
Both modes require --execution-sha256 --authority PATH --authority-sha256 SHA
--arm {large,so400} --output ABSOLUTE_NEW_PATH. --check-startup-only writes one
exclusive startup JSON (120s/8GiB/noSwap, CUDA hidden, no native imports/model).
Actual export additionally requires --startup PATH --startup-sha256 SHA and
writes a new directory containing fit.npy and receipt.json (300s/8GiB/noSwap,
CUDA allocated <10GB, B32 including tail). Parent supplies both admission locks
and enclosing unit caps, retains original logs, and attaches final cgroup and
normal-exit evidence. Neither startup nor an internal receipt replaces that.
No teacher/features/F5/PCA/head/optimizer/updates/quality read, chunks or rescue.
Local tests/help use only python3 -B -S; native imports occur only in export.
"""
if not __debug__:
    raise SystemExit('Export requires assertions; optimized mode is forbidden')

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'siglip2-substrate-fit-export-v1'
AUTHORITY_SCHEMA = 'native256-fit-export-launch-v1'
FILES = {'export_siglip2_substrate_fit.py', 'test_siglip2_substrate_fit.py'}
SOURCE_FILES = {'extract_siglip2_vision_source.py', 'qualify_siglip2_substrate_cpu.py',
                'test_siglip2_substrate_cpu.py'}
SOURCE_ROOT = Path('/home/riomus/runs/sfora-native256-source-cpu-v4')
STARTUP_POLICY = {'seconds': 120, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0,
                  'cuda_visible_devices': ''}
EXPORT_POLICY = {'seconds': 300, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0,
                 'cuda_allocated_bytes_exclusive': 10_000_000_000, 'batch_size': 32,
                 'fit_images': 13283, 'fit_identities': 2004, 'native_size': 256,
                 'fp32_autocast_cosine_min': .999, 'norm_atol': 1e-5}
WIDTHS = {'large': 1024, 'so400': 1152}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            'canonical regular file required: ' + str(path))
    return path


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


def read_json(path, expected):
    require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None, 'SHA256 required')
    path = canonical(path)
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2, 'authority JSON too large')
    require(hashlib.sha256(raw).hexdigest() == expected, 'authority SHA256 differs: ' + str(path))
    return strict_json(raw)


def closure(root, expected, names):
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json(root / 'execution.json', expected)
    require(isinstance(code, dict) and code.keys() == names,
            'execution requires exactly ' + ('two' if names == FILES else 'three') + ' files')
    for name, digest in code.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                'execution file SHA256 required')
        require(sha(root / name) == digest, 'execution file SHA256 differs: ' + name)
    return code


def bootstrap(root, expected):
    try:
        return closure(root, expected, FILES)
    except ValueError as error:
        raise ValueError('export execution SHA256/closure rejected: ' + str(error)) from error


def load_source(root, expected):
    code = closure(root, expected, SOURCE_FILES)
    sys.path.insert(0, str(root))
    name = 'qualify_siglip2_substrate_cpu'
    path = root / (name + '.py')
    spec = importlib.util.find_spec(name)
    require(spec is not None and Path(spec.origin).absolute() == path, 'source factory import origin differs')
    loaded = sys.modules.get(name)
    if loaded is None:
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    require(Path(loaded.__file__).absolute() == path and loaded.FILES == SOURCE_FILES,
            'source factory loaded origin/closure differs')
    return loaded, code


def all_fit_images(context):
    source = context['source_driver']
    source.fit_rows(context['extract'], context['fit'])
    root = Path(context['fit']['dataset_root'])
    images = [(root / row['relative_path']).resolve() for row in context['fit']['rows']]
    require(all(path.is_relative_to(root) for path in images), 'FIT image escaped dataset root')
    require(len(set(images)) == 13283, 'FIT resolved image aliases collide')
    for path, row in zip(images, context['fit']['rows']):
        source.bound_file(context['extract'], context['guards'], path, row['image_sha256'])
    return images


def admit_cgroup(value, unit):
    path, values = value['path'], value['values']
    require(Path(path).is_absolute() and '..' not in Path(path).parts and
            Path(path).name == unit + '.service', 'source CPU enclosing unit differs')
    require(values['memory.max'] == str(STARTUP_POLICY['host_bytes']) and
            0 < int(values['memory.current']) <= int(values['memory.peak']) <= STARTUP_POLICY['host_bytes'] and
            all(values[k] == '0' for k in ('memory.swap.current', 'memory.swap.peak', 'memory.swap.max')),
            'source CPU whole-cgroup caps differ')
    events = dict(line.split() for line in values['memory.events'].splitlines())
    require(all(events.get(k) == '0' for k in ('max', 'oom', 'oom_kill')), 'source CPU memory failure events')


def admit_cpu(context, descriptor, original):
    source, extract, guards = context['source_driver'], context['extract'], context['guards']
    require(descriptor.keys() == {'proof', 'log', 'unit', 'invocation_id', 'service_seconds',
                                 'native_peak_rss_kib', 'both_locks_held'} and
            descriptor['both_locks_held'] is True, 'source CPU descriptor/locks differ')
    proof = source.read_json(extract, guards, descriptor['proof']['path'], descriptor['proof']['sha256'])
    require(proof['schema'] == source.SCHEMA and proof['arm'] == context['args'].arm and
            all(proof[k] is True for k in ('pass', 'source_qualified', 'reload_exact', 'exit_rehash_pass',
                                         'constructor_rng_preserved')) and
            all(proof[k] is False for k in ('model_qualified', 'initializer_qualified', 'training_qualified',
                                          'quality_qualified', 'quality_read', 'gradients_created', 'optimizer_created')) and
            proof['updates'] == 0, 'actual source-only CPU proof required')
    require(proof['execution_sha256'] == original['execution_sha256'] and proof['code'] == original['code'] and
            proof['sources_sha256'] == original['sources_sha256'] and
            proof['fit_manifest_sha256'] == original['fit_manifest_sha256'] and
            proof['selected_source'] == context['entry'] and
            proof['source_model'] == context['entry']['source_model'] and
            proof['revision'] == context['entry']['revision'], 'source CPU authority/source binding differs')
    require(proof['resource_policy'] == source.POLICY == STARTUP_POLICY and
            0 < proof['wall_seconds'] <= descriptor['service_seconds'] <= 120 and
            0 < proof['process_peak_rss_kib'] <= descriptor['native_peak_rss_kib'] <= 8 * 1024**2,
            'source CPU service/RSS/caps differ')
    invocation = proof['invocation']
    require(re.fullmatch('[0-9a-f]{32}', descriptor['invocation_id']) is not None and
            re.fullmatch('[A-Za-z0-9_.@-]+', descriptor['unit']) is not None and
            invocation['invocation_id'] == descriptor['invocation_id'] and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0,
            'source CPU original invocation differs')
    checkpoint = canonical(proof['checkpoint']['path'])
    require(checkpoint.name == 'fresh_vision.pt' and Path(descriptor['proof']['path']) == checkpoint.parent / 'proof.json',
            'source CPU checkpoint/proof path role differs')
    expected_argv = [str(context['root'] / 'qualify_siglip2_substrate_cpu.py'),
                     '--execution-sha256', original['execution_sha256'],
                     '--sources', str(context['root'] / 'sources.json'), '--sources-sha256', original['sources_sha256'],
                     '--fit-manifest', str(context['root'] / 'fit.json'), '--fit-manifest-sha256', original['fit_manifest_sha256'],
                     '--arm', context['args'].arm, '--output', str(checkpoint.parent)]
    require(invocation['argv'] == expected_argv, 'source CPU argv differs')
    source.bound_file(extract, guards, checkpoint, proof['checkpoint']['sha256'])  # Metadata only: never torch.load.
    source.bound_file(extract, guards, invocation['python'], invocation['python_sha256'])
    for path, digest in proof['input_guards'].items():
        source.bound_file(extract, guards, path, digest)
    for path, digest in proof['origins']['files'].items():
        require(proof['input_guards'].get(path) == digest, 'source CPU origin guard differs')
    require(proof['input_guards'].get(str(checkpoint)) == proof['checkpoint']['sha256'] and
            all(proof['input_guards'].get(str(path)) == row['image_sha256'] for path, row in
                zip(context['images'], context['fit']['rows'][:2])), 'source CPU input guards differ')
    runtime = proof['runtime']
    require(runtime['vision'].keys() == context['expected'].keys(), 'source CPU vision inventory differs')
    for name, shape in context['expected'].items():
        require(runtime['vision'][name] == {'shape': shape, 'dtype': 'torch.float32',
                                         'sha256': context['mapping'][name]['sha256']},
                'source CPU tensor provenance differs: ' + name)
    roles = runtime['roles']
    boundary = context['config']['vision_config']['num_hidden_layers'] - 12
    frozen = ('embeddings.',) + tuple(f'encoder.layers.{i}.' for i in range(boundary))
    require(len(roles) == len(context['expected']) and {r['name'] for r in roles} == context['expected'].keys() and
            sum(r['role'] == 'trainable' for r in roles) == 205, 'source CPU 205 roles differ')
    for row in roles:
        require(row == {'name': row['name'], 'shape': context['expected'][row['name']], 'dtype': 'torch.float32',
                        'role': 'frozen' if row['name'].startswith(frozen) else 'trainable'}, 'source CPU role differs')
    require(runtime['buffers'].keys() == {'embeddings.position_ids'} and
            runtime['buffers']['embeddings.position_ids']['persistent'] is False and
            runtime['buffers']['embeddings.position_ids']['dtype'] == 'torch.int64' and
            runtime['buffers']['embeddings.position_ids']['shape'] == [1, 256] and
            proof['sample']['pixels']['shape'] == [2, 3, 256, 256] and
            proof['sample']['raw']['shape'] == [2, WIDTHS[context['args'].arm]], 'source CPU buffer/sample differs')
    log = source.bound_file(extract, guards, descriptor['log']['path'], descriptor['log']['sha256']).read_text()
    lines = log.splitlines()
    required = [f"Running as unit: {descriptor['unit']}.service; invocation ID: {descriptor['invocation_id']}",
                '\tExit status: 0', 'Finished with result: success',
                'Main processes terminated with: code=exited/status=0', '\tSwaps: 0', 'Memory swap peak: 0B',
                f"Service runtime: {descriptor['service_seconds']}s",
                f"\tMaximum resident set size (kbytes): {descriptor['native_peak_rss_kib']}"]
    require(all(lines.count(line) == 1 for line in required), 'source CPU original normal-exit log differs')
    footers = [strict_json(line[len('FINAL_CGROUP '):]) for line in lines if line.startswith('FINAL_CGROUP ')]
    require(len(footers) == 1 and footers[0]['invocation_id'] == descriptor['invocation_id'],
            'source CPU final cgroup footer differs')
    final = footers[0]
    for value in (proof['cgroup_before'], proof['cgroup_after'], final):
        admit_cgroup(value, descriptor['unit'])
        require(value['path'] == final['path'], 'source CPU cgroup path changed')
    require(int(final['values']['memory.peak']) >= int(proof['cgroup_after']['values']['memory.peak']) >=
            int(proof['cgroup_before']['values']['memory.peak']), 'source CPU complete peak differs')
    return proof


def authority(args):
    own_root = Path(__file__).absolute().parent
    code = bootstrap(own_root, args.execution_sha256)
    launch = read_json(args.authority, args.authority_sha256)
    require(launch.keys() == {'schema', 'execution_sha256', 'source_root', 'source_execution_sha256',
                             'source_cpu_authority', 'sources', 'fit_manifest', 'source_cpu',
                             'startup_policy', 'export_policy'} and
            launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
            launch['startup_policy'] == STARTUP_POLICY and launch['export_policy'] == EXPORT_POLICY and
            launch['source_cpu'].keys() == WIDTHS.keys(), 'parent launch authority/profile differs')
    require(Path(launch['source_root']) == SOURCE_ROOT and not own_root.is_relative_to(SOURCE_ROOT) and
            not args.output.is_relative_to(SOURCE_ROOT),
            'original immutable source root required')
    require(launch['sources']['path'] == str(SOURCE_ROOT / 'sources.json') and
            launch['fit_manifest']['path'] == str(SOURCE_ROOT / 'fit.json'), 'original source input path role differs')
    source, source_code = load_source(SOURCE_ROOT, launch['source_execution_sha256'])
    original = read_json(launch['source_cpu_authority']['path'], launch['source_cpu_authority']['sha256'])
    require(original['schema'] == 'native256-source-cpu-launch-v1' and original['code'] == source_code and
            original['execution_sha256'] == launch['source_execution_sha256'] and
            original['sources_sha256'] == launch['sources']['sha256'] and
            original['fit_manifest_sha256'] == launch['fit_manifest']['sha256'] and
            original['resource_policy'] == source.POLICY == STARTUP_POLICY,
            'original source CPU launch authority differs')
    source_args = SimpleNamespace(execution_sha256=original['execution_sha256'],
        sources=Path(launch['sources']['path']), sources_sha256=original['sources_sha256'],
        fit_manifest=Path(launch['fit_manifest']['path']), fit_manifest_sha256=original['fit_manifest_sha256'],
        arm=args.arm, output=args.output)
    context = source.authority(source_args)
    context.update(source_driver=source, own_root=own_root, own_code=code, launch=launch,
                   export_args=args, authority_sha256=args.authority_sha256)
    for path, digest in [(args.authority, args.authority_sha256),
                         (Path(launch['source_cpu_authority']['path']), launch['source_cpu_authority']['sha256']),
                         (own_root / 'execution.json', args.execution_sha256),
                         *[(own_root / name, digest) for name, digest in code.items()]]:
        source.bound_file(context['extract'], context['guards'], path, digest)
    context['proof'] = admit_cpu(context, launch['source_cpu'][args.arm], original)
    context['all_images'] = all_fit_images(context)
    return context


def rehash(context):
    require(all_fit_images(context) == context['all_images'], 'FIT image resolution changed')
    context['source_driver'].rehash(context)
    args = context['export_args']
    require(bootstrap(context['own_root'], args.execution_sha256) == context['own_code'], 'exit exporter closure differs')


def binding(context):
    return {'arm': context['export_args'].arm, 'authority_sha256': context['authority_sha256'],
            'execution_sha256': context['export_args'].execution_sha256, 'code': context['own_code'],
            'source_root': str(context['root']), 'source_execution_sha256': context['args'].execution_sha256,
            'sources_sha256': context['args'].sources_sha256,
            'fit_manifest_sha256': context['args'].fit_manifest_sha256,
            'source_cpu': context['launch']['source_cpu'][context['export_args'].arm]}


def invocation(context):
    identity = os.environ.get('INVOCATION_ID')
    require(re.fullmatch('[0-9a-f]{32}', identity or '') is not None, 'original systemd invocation ID required')
    python = canonical(Path(sys.executable).resolve())
    return {'argv': sys.argv, 'python': str(python), 'python_sha256': context['extract'].sha(python),
            'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
            'invocation_id': identity, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def write_json(extract, path, value):
    with extract.exclusive(path) as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def startup(args):
    started = time.perf_counter()
    context = authority(args)
    source = context['source_driver']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'startup requires CUDA explicitly hidden')
    before = source.cgroup_memory()
    packages = source.package_origins(context)
    identity = invocation(context)
    require(identity['python'] == context['proof']['invocation']['python'] and
            identity['python_sha256'] == context['proof']['invocation']['python_sha256'] and
            identity['python_version'] == context['proof']['invocation']['python_version'], 'source CPU interpreter differs')
    rehash(context)
    after = source.cgroup_memory()
    require(before['path'] == after['path'], 'startup cgroup changed')
    wall = time.perf_counter() - started
    require(wall < 120, 'startup exceeds 120s')
    result = {'schema': SCHEMA, 'phase': 'startup', 'pass': True, 'binding': binding(context),
              'native_imported': False, 'model_constructed': False, 'exported': False,
              'packages': packages, 'input_guards': context['guards'], 'exit_rehash_pass': True,
              'resource_policy': STARTUP_POLICY, 'cgroup_before': before, 'cgroup_after': after,
              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'wall_seconds': wall, 'invocation': identity,
              'terminal_exit_and_both_locks_require_parent_receipt': True}
    write_json(context['extract'], args.output, result)
    return result


def admit_startup(context, record):
    require(record['schema'] == SCHEMA and record['phase'] == 'startup' and record['pass'] is True and
            record['binding'] == binding(context) and record['exit_rehash_pass'] is True and
            all(record[k] is False for k in ('native_imported', 'model_constructed', 'exported')) and
            record['resource_policy'] == STARTUP_POLICY and 0 < record['wall_seconds'] < 120 and
            record['input_guards'] == context['guards'], 'pinned startup admission/binding differs')
    identity, prior = record['invocation'], context['proof']['invocation']
    require(identity['cuda_visible_devices'] == '' and identity['optimize'] == 0 and
            re.fullmatch('[0-9a-f]{32}', identity['invocation_id'] or '') is not None and
            all(identity[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')),
            'pinned startup interpreter/invocation differs')
    before, after = record['cgroup_before'], record['cgroup_after']
    unit = Path(after['path']).name.removesuffix('.service')
    for memory in (before, after):
        admit_cgroup(memory, unit)
    require(before['path'] == after['path'] and
            int(before['values']['memory.peak']) <= int(after['values']['memory.peak']), 'startup cgroup changed')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def export(args):
    started = time.perf_counter()
    context = authority(args)
    source, extract, guards = context['source_driver'], context['extract'], context['guards']
    require(args.startup is not None and args.startup_sha256 is not None, 'export requires pinned startup')
    prior = read_json(args.startup, args.startup_sha256)
    admit_startup(context, prior)
    source.bound_file(extract, guards, args.startup, args.startup_sha256)
    before = source.cgroup_memory()
    packages = source.package_origins(context)
    require(packages == prior['packages'] == context['proof']['origins']['packages'],
            'startup/source CPU package origins differ')
    context['packages'] = packages
    identity = invocation(context)
    require(all(identity[k] == prior['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
            identity['cuda_visible_devices'] not in (None, ''), 'export interpreter/explicit CUDA visibility differs')
    args.output.mkdir()  # Failure leaves a non-reusable directory, never a partial accepted cache.
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA was initialized before fresh CPU verification')
    cpu_flags = context['proof']['numerical_flags']
    torch.set_num_threads(cpu_flags['threads'])
    if torch.get_num_interop_threads() != cpu_flags['interop_threads']:
        torch.set_num_interop_threads(cpu_flags['interop_threads'])
    require(source.numerical_flags() == cpu_flags, 'source CPU numerical defaults differ')
    rng = torch.random.get_rng_state().clone()  # Current process RNG, never another process's saved seed.
    model, processor, roles = source.fresh_source(context)
    pixels, raw, sample = source.pixels_and_raw(context, model, processor)
    runtime = source.model_facts(model, processor, roles, packages)
    require(runtime == context['proof']['runtime'] and sample == context['proof']['sample'],
            'fresh FP32 CPU runtime/first2 actual sample differs from selected source proof')
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == cpu_flags and
            not torch.cuda.is_initialized(), 'fresh CPU verification changed RNG/flags or initialized CUDA')
    del pixels, raw
    # Authenticate actual loaded CPU code before admitting any CUDA construction.
    cpu_origins = source.imported_origins(extract, packages)
    for name, path in cpu_origins['modules'].items():
        require(context['proof']['origins']['modules'].get(name) == path, 'loaded CPU module differs from proof: ' + name)
    for path, digest in cpu_origins['files'].items():
        require(context['proof']['origins']['files'].get(path) == digest, 'loaded CPU file differs from proof: ' + path)
    import numpy as np
    from PIL import Image
    from torch.nn import functional as F
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    export_flags = source.numerical_flags()
    model = model.cuda().eval()  # No peak reset: construction and both calibration forwards remain included.
    require(all(p.dtype == torch.float32 and p.device.type == 'cuda' and p.grad is None
                for p in model.parameters()), 'FP32 CUDA source required')
    width = WIDTHS[args.arm]
    require(model.config.hidden_size == width, 'selected source width differs')
    input_manifest = {'rows': context['fit']['rows'], 'targets': context['fit']['targets'],
                      'class_names': context['fit']['class_names'], 'resolved_paths': [str(p) for p in context['all_images']]}
    input_digest = object_sha(input_manifest)

    def limits():
        require(time.perf_counter() - started < 300, 'complete FIT export exceeds 300s')
        require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'complete-unit CUDA allocation exceeds <10GB')

    def batch_pixels(start, count):
        images, facts = [], []
        try:
            for index in range(start, start + count):
                path, row = context['all_images'][index], context['fit']['rows'][index]
                resolved = (Path(context['fit']['dataset_root']) / row['relative_path']).resolve()
                require(resolved == path and path.is_relative_to(Path(context['fit']['dataset_root'])),
                        'FIT image resolution changed during export')
                require(extract.sha(path) == row['image_sha256'], 'FIT image SHA256 changed during export')
                with Image.open(path) as opened:
                    image = opened.convert('RGB')
                images.append(image)
                facts.append({'ordinal': index, 'train_row': row['train_row'], 'target': context['fit']['targets'][index],
                              'relative_path': row['relative_path'], 'path': str(path),
                              'image_sha256': row['image_sha256'], 'mode': image.mode, 'size': list(image.size),
                              'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            pixels = processor(images=images, return_tensors='pt')['pixel_values']
            require(pixels.dtype == torch.float32 and list(pixels.shape) == [count, 3, 256, 256] and
                    torch.isfinite(pixels).all().item(), 'unaugmented stock native256 pixels differ')
            return pixels.cuda(), facts
        finally:
            for image in images:
                image.close()

    with torch.no_grad():
        calibration_pixels, calibration_images = batch_pixels(0, 4)
        fp32 = model(pixel_values=calibration_pixels).pooler_output.float()
        with torch.autocast('cuda', dtype=torch.float16):
            fp16 = model(pixel_values=calibration_pixels).pooler_output.float()
        require(fp32.shape == fp16.shape == (4, width) and torch.isfinite(fp32).all().item() and
                torch.isfinite(fp16).all().item() and (fp32.norm(dim=1) > 0).all().item() and
                (fp16.norm(dim=1) > 0).all().item(), 'first4 raw outputs must be finite/nonzero')
        cosines = F.cosine_similarity(fp32, fp16, dim=1).cpu().tolist()
        require(min(cosines) >= .999, 'first4 FP32/autocast cosine below .999')
        del calibration_pixels, fp32, fp16
        limits()
        features = np.empty((13283, width), dtype=np.float32)
        rgb_manifest, batch_sizes = [], []
        export_started = time.perf_counter()
        for start in range(0, 13283, 32):
            count = min(32, 13283 - start)
            pixels, facts = batch_pixels(start, count)
            with torch.autocast('cuda', dtype=torch.float16):
                raw = model(pixel_values=pixels).pooler_output.float()
            require(raw.dtype == torch.float32 and raw.shape == (count, width) and
                    torch.isfinite(raw).all().item() and (raw.norm(dim=1) > 0).all().item(),
                    'FIT raw features shape/finite/nonzero differ')
            normalized = F.normalize(raw, dim=1)  # Original arithmetic: CUDA FP32 normalization, then CPU NumPy.
            features[start:start + count] = normalized.cpu().numpy()
            rgb_manifest.extend(facts)
            batch_sizes.append(count)
            del pixels, raw, normalized
            limits()
        torch.cuda.synchronize()
        export_seconds = time.perf_counter() - export_started
    require(features.dtype == np.float32 and features.shape == (13283, width) and np.isfinite(features).all() and
            np.allclose(np.linalg.norm(features, axis=1), 1, rtol=0, atol=1e-5), 'complete FIT cache validation differs')
    require(batch_sizes == [32] * (13283 // 32) + [13283 % 32] and len(rgb_manifest) == 13283 and
            [r['ordinal'] for r in rgb_manifest] == list(range(13283)) and
            rgb_manifest[:4] == calibration_images and
            [r['train_row'] for r in rgb_manifest] == [r['train_row'] for r in context['fit']['rows']] and
            [r['target'] for r in rgb_manifest] == context['fit']['targets'], 'ordered FIT/RGB/counters differ')
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == export_flags,
            'export changed CPU RNG/numerical flags')
    peak_cuda = torch.cuda.max_memory_allocated()
    # Reuse the CPU facts guard to verify unchanged weights, strict buffers and
    # 205 roles after all forwards; peak includes the entire unit, never reset.
    model.cpu()
    require(source.model_facts(model, processor, roles, packages) == runtime,
            'source state/runtime/roles/buffers changed during export')
    origins = source.imported_origins(extract, packages)
    for path, digest in origins['files'].items():
        require(guards.setdefault(path, digest) == digest, 'loaded export origin hash changed: ' + path)
    rehash(context)
    require(object_sha(input_manifest) == input_digest, 'ordered input changed during export')
    after = source.cgroup_memory()
    require(before['path'] == after['path'], 'export enclosing cgroup changed')
    limits()
    cache = args.output / 'fit.npy'
    with extract.exclusive(cache) as stream:
        np.save(stream, features, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    cache_sha = extract.sha(cache)
    limits()
    after = source.cgroup_memory()
    require(before['path'] == after['path'], 'export enclosing cgroup changed after save')
    wall = time.perf_counter() - started
    result = {'schema': SCHEMA, 'phase': 'export', 'pass': True, 'exported': True,
              'binding': binding(context), 'startup': {'path': str(args.startup), 'sha256': args.startup_sha256},
              'cache': {'path': str(cache), 'sha256': cache_sha, 'shape': list(features.shape), 'dtype': 'float32',
                        'normalized': True, 'raw_pooled_cache': False},
              'arithmetic': 'unaugmented stock native256; B32+tail; FP32 vision; CUDA FP16 autocast; pooled.float(); CUDA F.normalize(dim=1); CPU NumPy FP32',
              'source_checkpoint_metadata_only': context['proof']['checkpoint'], 'fresh_source': True,
              'source_cpu_runtime_and_first2_exact': True, 'constructor_rng_preserved': True,
              'cpu_rng': source.tensor_fact(rng), 'cpu_numerical_flags': cpu_flags, 'export_numerical_flags': export_flags,
              'fp32_autocast_first4_cosines': cosines, 'ordered_input_sha256': input_digest,
              'ordered_rgb_sha256': object_sha(rgb_manifest), 'rgb_manifest': rgb_manifest,
              'counters': {'images': len(rgb_manifest), 'classes': 2004, 'batches': len(batch_sizes),
                           'batch_sizes': batch_sizes, 'calibration_images': 4, 'cpu_witness_images': 2,
                           'optimizer_updates': 0},
              'input_guards': guards, 'origins': origins, 'exit_rehash_pass': True,
              'gradients_created': False, 'optimizer_created': False, 'updates': 0,
              'source_features_reused': False, 'teacher_state_reused': False, 'quality_read': False,
              'initializer_qualified': False, 'training_qualified': False, 'quality_qualified': False,
              'resource_policy': EXPORT_POLICY, 'cgroup_before': before, 'cgroup_after': after,
              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'complete_unit_peak_cuda_allocated_bytes': peak_cuda, 'cuda_peak_reset': False,
              'wall_seconds': wall, 'export_seconds': export_seconds, 'invocation': identity,
              'terminal_exit_and_both_locks_require_parent_receipt': True}
    write_json(extract, args.output / 'receipt.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--arm', choices=sorted(WIDTHS), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--check-startup-only', action='store_true')
    result.add_argument('--startup', type=Path)
    result.add_argument('--startup-sha256')
    return result


def main():
    args = parser().parse_args()
    try:
        require((args.startup is None and args.startup_sha256 is None) if args.check_startup_only else
                (args.startup is not None and args.startup_sha256 is not None),
                'startup mode has no prior startup; export requires pinned startup path/SHA256')
        result = startup(args) if args.check_startup_only else export(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as error:
        raise SystemExit('FIT export rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
