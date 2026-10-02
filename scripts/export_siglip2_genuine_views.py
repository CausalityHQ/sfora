#!/usr/bin/env python3
"""Export one fresh pair of genuine mild TRAIN views, never fit held pixels.

Freeze exactly this file + test_siglip2_genuine_views.py in execution.json.
Launch schema native256-genuine-view-launch-v1 has exactly: schema,
execution_sha256, reference, partition, image_rows, startup_policy, export_policy.
FILE = {path: canonical absolute file, sha256: lowercase SHA256}.
reference = {root: ORIGINAL_REF_ROOT, execution_sha256: REF_EXECUTION_SHA,
authority: FILE with ORIGINAL_AUTHORITY_SHA}. It is the unchanged TWO-file old
FIT exporter; its ORIGINAL THREE-file source closure remains separate. All
original source proof/log/config/processor/environment/origin/image SHA guards
are admitted unchanged. Historical full FIT metadata is authenticated only;
no historical feature cache or original source witness pixels are loaded.
partition is FILE pinned by PARTITION_SHA. image_rows is the original
train_sop_siglip2_compact.py FILE; its exact ImageRows class AST is pinned.
Only that class is compiled, avoiding unrelated trainer imports. No helper
is patched. startup_policy/export_policy must equal the constants below.

python -B ROOT/export_siglip2_genuine_views.py --execution-sha256 NEW_TWO_SHA
 --authority FILE --authority-sha256 SHA --phase startup|export --output NEWDIR
Startup hides CUDA explicitly, imports no native modules, and writes proof.json.
Export additionally requires --startup-terminal FILE --startup-terminal-sha256
SHA. TERMINAL has exactly proof:FILE, log:FILE, unit, invocation_id,
service_seconds, native_peak_rss_kib, both_locks_held:true, from an ACTUAL
normal-exit original startup120 unit, including one FINAL_CGROUP log footer.
Parent owns both lifetime locks, enclosing startup120/export300, 8GiB/noSwap,
zero memory failures and actual terminal receipts. No guessed future evidence.

Export writes canonical.npy + augmented.npy: normalized F32[6355,1152], same
original FIT-row order, B32+tail19, native256, frozen FP32 So400, F16 autocast,
pooled.float(), CUDA F.normalize then CPU NumPy F32. Exactly one ImageRows mild
view per image, isolated CPU RNG179081+original_train_row, restored globally.
Ordered original/canonical/augmented RGB size+bytes and processed pixel hashes
bind both views. First4 both views calibrate FP32/F16 cosine>=.999. Sequential
independent complete strict checkpoint/config/nonpersistent-buffer/processor
reload repeats first B32 + tail19 pixels/raw/unit/cache and first4 calibration.
Both full source state witnesses and all uncached original/new input/output
rehashes are inside the one300 lifecycle; no peak reset or partial reuse.
receipt.json binds caches, ordered mapping, startup terminal and complete
resource evidence. It is preparation, never training/quality qualification.
"""
if not __debug__:
    raise SystemExit('Genuine export requires assertions; optimized mode is forbidden')

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import re
import resource
import time
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace

SCHEMA = 'siglip2-genuine-views-v1'
AUTHORITY_SCHEMA = 'native256-genuine-view-launch-v1'
FILES = {'export_siglip2_genuine_views.py', 'test_siglip2_genuine_views.py'}
REF_FILES = {'export_siglip2_substrate_fit.py', 'test_siglip2_substrate_fit.py'}
ORIGINAL_REF_ROOT = Path('/home/riomus/runs/sfora-native256-fit-export-source-v1')
REF_EXECUTION_SHA = 'ad32df859712e3d608341ed9eda24fd96db375ec85c61779265a9c16cb8dd3b5'
ORIGINAL_AUTHORITY_SHA = '065b32e841e8a05bc71aa36e5c9b32db07b139ba7159f428dc21eba23a57361c'
PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'
MANIFEST_SHA = 'd32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251'
OLD_CACHE_SHA = 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'
SOURCE_CHECKPOINT_SHA = '5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189'
IMAGE_ROWS_AST_SHA = '09e080b36b7fe3059e15e47ff5795a9390fb9cf5c28c075b955762f0795e4103'
ROWS, CLASSES, WIDTH, BATCH = 6355, 1008, 1152, 32
VIEWS = ('canonical', 'augmented')
STARTUP_POLICY = {'seconds': 120, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0,
                  'cuda_visible_devices': ''}
EXPORT_POLICY = {'seconds': 300, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0,
                 'cuda_allocated_bytes_exclusive': 10_000_000_000, 'batch_size': BATCH,
                 'train_images': ROWS, 'train_identities': CLASSES, 'native_size': 256,
                 'width': WIDTH, 'views': list(VIEWS), 'augmentation_seed': 179081,
                 'fp32_autocast_cosine_min': .999, 'norm_atol': 1e-5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical regular file required')
    return path


def sha(path):
    with canonical(path).open('rb') as stream:
        digest = hashlib.sha256()
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


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


def read_json(path, expected, guards):
    require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None, 'SHA256 required')
    path = canonical(path)
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected, 'JSON SHA256/size differs')
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return strict_json(raw)


def file_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'exact FILE descriptor required')
    return read_json(value['path'], value['sha256'], guards)


def closure(root, expected, names, guards):
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json(root / 'execution.json', expected, guards)
    require(isinstance(code, dict) and code.keys() == names, 'exact execution closure required')
    for name, digest in code.items():
        require(re.fullmatch('[0-9a-f]{64}', digest or '') is not None and sha(root / name) == digest,
                'execution file SHA256 differs: ' + name)
        guards[str(root / name)] = digest
    return code


def check_launch(launch, execution_sha256):
    require(launch.keys() == {'schema', 'execution_sha256', 'reference', 'partition', 'image_rows',
                             'startup_policy', 'export_policy'} and
            launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == execution_sha256 and
            launch['startup_policy'] == STARTUP_POLICY and launch['export_policy'] == EXPORT_POLICY,
            'genuine launch profile differs')
    ref = launch['reference']
    require(ref.keys() == {'root', 'execution_sha256', 'authority'} and
            ref['root'] == str(ORIGINAL_REF_ROOT) and ref['execution_sha256'] == REF_EXECUTION_SHA and
            ref['authority'] == {'path': str(ORIGINAL_REF_ROOT / 'authority.json'), 'sha256': ORIGINAL_AUTHORITY_SHA},
            'original immutable reference differs')
    require(launch['partition'].keys() == {'path', 'sha256'} and
            launch['partition']['sha256'] == PARTITION_SHA and
            launch['image_rows'].keys() == {'path', 'sha256'}, 'partition/ImageRows descriptor differs')


def selected_manifest(partition, fit):
    """Validate original metadata fully; return ONLY the frozen TRAIN rows."""
    require(partition.keys() == {'schema', 'global_class_names', 'original_cache', 'original_fit',
                                 'panels', 'partition_seeds'} and
            partition['schema'] == 'siglip2-identity-mix-partition-v1' and
            partition['original_cache']['sha256'] == OLD_CACHE_SHA and
            partition['original_fit']['sha256'] == MANIFEST_SHA and
            partition['global_class_names'] == fit['class_names'] and
            len(fit['rows']) == len(fit['targets']) == 13283 and
            partition['partition_seeds'] == [179071, 179072] and
            partition['panels'].keys() == {'train', 'selection', 'validation'}, 'partition source differs')
    rows_seen, classes_seen = set(), set()
    for name, (count, classes) in {'train': (ROWS, CLASSES), 'selection': (3449, 498),
                                  'validation': (3479, 498)}.items():
        panel = partition['panels'][name]
        rows, ids = panel['original_rows'], panel['original_class_ids']
        require(panel.keys() == ({'original_rows', 'original_class_ids'} if name == 'train' else
                                 {'original_rows', 'original_class_ids', 'query', 'gallery'}) and
                len(rows) == count and len(ids) == classes and
                all(type(n) is int and 0 <= n < 13283 for n in rows) and
                all(type(n) is int and 0 <= n < 2004 for n in ids) and
                rows == sorted(set(rows)) and ids == sorted(set(ids)) and
                not rows_seen.intersection(rows) and not classes_seen.intersection(ids) and
                set(fit['targets'][r] for r in rows) == set(ids), 'partition leakage/order differs')
        if name != 'train':
            query, gallery = panel['query'], panel['gallery']
            require(all(type(n) is int and 0 <= n < count for n in query + gallery) and
                    query == sorted(set(query)) and gallery == sorted(set(gallery)) and
                    not set(query).intersection(gallery) and set(query + gallery) == set(range(count)) and
                    len(query) == (1734 if name == 'selection' else 1749) and
                    set(fit['targets'][rows[r]] for r in query) == set(ids) ==
                    set(fit['targets'][rows[r]] for r in gallery), 'held metadata query/gallery differs')
        rows_seen.update(rows)
        classes_seen.update(ids)
    require(rows_seen == set(range(13283)) and classes_seen == set(range(2004)), 'partition coverage differs')
    ids = partition['panels']['train']['original_class_ids']
    dense = {c: i for i, c in enumerate(ids)}
    rows = partition['panels']['train']['original_rows']
    return {'original_rows': rows, 'rows': [fit['rows'][r] for r in rows],
            'targets': [dense[fit['targets'][r]] for r in rows],
            'class_names': [fit['class_names'][c] for c in ids]}


def image_rows_node(path):
    nodes = [n for n in ast.parse(canonical(path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ImageRows']
    require(len(nodes) == 1 and hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() ==
            IMAGE_ROWS_AST_SHA, 'original ImageRows class differs')
    return nodes[0]


def authority(args):
    guards = {}
    root = Path(__file__).absolute().parent
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json(args.authority, args.authority_sha256, guards)
    check_launch(launch, args.execution_sha256)
    ref_root = Path(launch['reference']['root'])
    require(root != ref_root and not root.is_relative_to(ref_root) and not ref_root.is_relative_to(root),
            'separate immutable reference closure required')
    ref_code = closure(ref_root, REF_EXECUTION_SHA, REF_FILES, guards)
    spec = importlib.util.spec_from_file_location('_genuine_fit_reference', ref_root / 'export_siglip2_substrate_fit.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    require(Path(reference.__file__).absolute() == ref_root / 'export_siglip2_substrate_fit.py' and
            reference.FILES == REF_FILES, 'reference import origin differs')
    prior = reference.authority(SimpleNamespace(execution_sha256=REF_EXECUTION_SHA,
        authority=Path(launch['reference']['authority']['path']), authority_sha256=ORIGINAL_AUTHORITY_SHA,
        arm='so400', output=args.output))
    require(prior['args'].fit_manifest_sha256 == MANIFEST_SHA and
            prior['proof']['checkpoint']['sha256'] == SOURCE_CHECKPOINT_SHA, 'original source/FIT pin differs')
    partition = file_json(launch['partition'], guards)
    require(partition['original_fit'] == launch_descriptor(prior['args'].fit_manifest, MANIFEST_SHA),
            'original FIT path differs')
    selected = selected_manifest(partition, prior['fit'])
    selected['resolved_paths'] = [str(prior['all_images'][r]) for r in selected['original_rows']]
    node_path = canonical(launch['image_rows']['path'])
    require(sha(node_path) == launch['image_rows']['sha256'], 'ImageRows file SHA256 differs')
    guards[str(node_path)] = launch['image_rows']['sha256']
    image_rows_node(node_path)
    prior['extract'].new_output(args.output)
    require(not args.output.is_relative_to(root) and not root.is_relative_to(args.output), 'output must be separate from code')
    return {'args': args, 'root': root, 'code': code, 'launch': launch, 'guards': guards,
            'reference': reference, 'reference_code': ref_code, 'prior': prior, 'selected': selected}


def launch_descriptor(path, digest):
    return {'path': str(path), 'sha256': digest}


def binding(context):
    args = context['args']
    return {'authority_sha256': args.authority_sha256, 'execution_sha256': args.execution_sha256,
            'code': context['code'], 'reference': context['launch']['reference'],
            'source': context['reference'].binding(context['prior']),
            'partition': context['launch']['partition'], 'image_rows': context['launch']['image_rows'],
            'ordered_input_sha256': object_sha(context['selected'])}


def rehash(context):
    context['reference'].rehash(context['prior'])  # Original validators/guards remain unchanged.
    for path, digest in context['guards'].items():
        require(sha(path) == digest, 'uncached exit SHA256 differs: ' + path)
    require(closure(context['root'], context['args'].execution_sha256, FILES, {}) == context['code'],
            'exit genuine closure differs')
    selected = selected_manifest(file_json(context['launch']['partition'], {}), context['prior']['fit'])
    selected['resolved_paths'] = [str(context['prior']['all_images'][r]) for r in selected['original_rows']]
    require(selected == context['selected'], 'exit TRAIN mapping changed')
    image_rows_node(context['launch']['image_rows']['path'])


def invocation(context):
    return context['reference'].invocation(context['prior'])


def write_json(context, path, value):
    context['reference'].write_json(context['prior']['extract'], path, value)


def admit_terminal(context, descriptor, proof, phase):
    """Require a real complete original startup/export service, never a claim."""
    require(descriptor.keys() == {'proof', 'log', 'unit', 'invocation_id', 'service_seconds',
                                 'native_peak_rss_kib', 'both_locks_held'} and
            descriptor['both_locks_held'] is True, 'terminal descriptor/locks differ')
    require(phase in ('startup', 'export') and file_json(descriptor['proof'], {}) == proof and
            descriptor['log'].keys() == {'path', 'sha256'}, 'terminal proof bytes/FILE descriptors differ')
    policy = STARTUP_POLICY if phase == 'startup' else EXPORT_POLICY
    require(proof['schema'] == SCHEMA and proof['phase'] == phase and proof['pass'] is True and
            proof['binding'] == binding(context) and proof['exit_rehash_pass'] is True and
            proof['resource_policy'] == policy and
            0 < proof['wall_seconds'] < descriptor['service_seconds'] <= policy['seconds'] and
            0 < proof['process_peak_rss_kib'] <= descriptor['native_peak_rss_kib'] <= 8 * 1024**2,
            'terminal proof/binding/resources differ')
    identity = proof['invocation']
    require(re.fullmatch('[A-Za-z0-9_.@-]+', descriptor['unit'] or '') is not None and
            re.fullmatch('[0-9a-f]{32}', descriptor['invocation_id'] or '') is not None and
            identity['invocation_id'] == descriptor['invocation_id'] and identity['optimize'] == 0,
            'terminal original invocation differs')
    prior_identity = context['prior']['proof']['invocation']
    require(all(identity[k] == prior_identity[k] for k in ('python', 'python_sha256', 'python_version')),
            'terminal original interpreter differs')
    proof_path = canonical(descriptor['proof']['path'])
    require(proof_path.name == ('proof.json' if phase == 'startup' else 'receipt.json'), 'terminal proof path role differs')
    expected_argv = [str(context['root'] / 'export_siglip2_genuine_views.py'),
                     '--execution-sha256', context['args'].execution_sha256,
                     '--authority', str(context['args'].authority), '--authority-sha256', context['args'].authority_sha256,
                     '--phase', phase, '--output', str(proof_path.parent)]
    if phase == 'export':
        expected_argv += ['--startup-terminal', proof['startup_terminal']['path'],
                          '--startup-terminal-sha256', proof['startup_terminal']['sha256']]
    require(identity['argv'] == expected_argv, 'terminal argv differs')
    if phase == 'startup':
        require(identity['cuda_visible_devices'] == '' and
                all(proof[k] is False for k in ('native_imported', 'model_constructed', 'exported')) and
                proof['input_guards'] == context['guards'] and
                proof['original_input_guards'] == context['prior']['guards'], 'startup guards/native state differ')
    else:
        require(proof['complete_unit_peak_cuda_allocated_bytes'] < 10_000_000_000 and
                proof['cuda_peak_reset'] is False and proof['strict_independent_reload_exact'] is True,
                'export terminal reload/CUDA differs')
    log_path = canonical(descriptor['log']['path'])
    require(sha(log_path) == descriptor['log']['sha256'], 'terminal log SHA256 differs')
    lines = log_path.read_text().splitlines()
    required = [f"Running as unit: {descriptor['unit']}.service; invocation ID: {descriptor['invocation_id']}",
                '\tExit status: 0', 'Finished with result: success',
                'Main processes terminated with: code=exited/status=0', '\tSwaps: 0', 'Memory swap peak: 0B',
                f"\tMaximum resident set size (kbytes): {descriptor['native_peak_rss_kib']}"]
    require(all(lines.count(line) == 1 for line in required), 'original normal-exit log differs')
    runtimes = [line.removeprefix('Service runtime: ') for line in lines if line.startswith('Service runtime: ')]
    require(len(runtimes) == 1, 'original service runtime line differs')
    match = re.fullmatch(r'(?:(\d+)min )?(\d+(?:\.\d+)?)s', runtimes[0])
    require(match is not None, 'original service runtime format differs')
    minutes, seconds = match.groups()
    require((minutes is None or Decimal(seconds) < 60) and
            Decimal(minutes or '0') * 60 + Decimal(seconds) == Decimal(str(descriptor['service_seconds'])),
            'original service runtime numeric binding differs')
    footers = [strict_json(line.removeprefix('FINAL_CGROUP ')) for line in lines if line.startswith('FINAL_CGROUP ')]
    require(len(footers) == 1 and footers[0]['invocation_id'] == descriptor['invocation_id'], 'terminal final cgroup differs')
    final = footers[0]
    for record in (proof['cgroup_before'], proof['cgroup_after'], final):
        context['reference'].admit_cgroup(record, descriptor['unit'])
        require(record['path'] == final['path'], 'terminal cgroup path changed')
    require(int(final['values']['memory.peak']) >= int(proof['cgroup_after']['values']['memory.peak']) >=
            int(proof['cgroup_before']['values']['memory.peak']), 'terminal complete cgroup peak differs')


def startup(context, started):
    source = context['prior']['source_driver']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'startup requires CUDA explicitly hidden')
    before = source.cgroup_memory()
    packages = source.package_origins(context['prior'])
    require(packages == context['prior']['proof']['origins']['packages'], 'source packages differ')
    identity = invocation(context)
    require(all(identity[k] == context['prior']['proof']['invocation'][k]
                for k in ('python', 'python_sha256', 'python_version')), 'original interpreter differs')
    context['args'].output.mkdir()
    rehash(context)
    after = source.cgroup_memory()
    require(before['path'] == after['path'] and time.perf_counter() - started < 120, 'startup lifetime/cgroup differs')
    result = {'schema': SCHEMA, 'phase': 'startup', 'pass': True, 'binding': binding(context),
              'native_imported': False, 'model_constructed': False, 'exported': False,
              'packages': packages, 'input_guards': context['guards'], 'original_input_guards': context['prior']['guards'],
              'exit_rehash_pass': True, 'resource_policy': STARTUP_POLICY,
              'cgroup_before': before, 'cgroup_after': after, 'wall_seconds': time.perf_counter() - started,
              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'invocation': identity, 'terminal_exit_and_both_locks_require_parent_receipt': True}
    write_json(context, context['args'].output / 'proof.json', result)
    require(time.perf_counter() - started < 120, 'complete startup exceeds120')
    return result


def isolated_view(torch, image, transform, original_train_row):
    require(type(original_train_row) is int and original_train_row >= 0, 'original train row required')
    rng = torch.random.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(179081 + original_train_row)
            return transform(image)
    finally:
        require(torch.equal(rng, torch.random.get_rng_state()), 'view changed global CPU RNG')


def validate_views(manifest, mapping):
    require(mapping.keys() == set(VIEWS), 'paired view inventory differs')
    for view, rows in mapping.items():
        require(len(rows) == ROWS, 'complete paired view row count differs')
        for i, (fact, row, path, original, target) in enumerate(zip(rows, manifest['rows'],
                manifest['resolved_paths'], manifest['original_rows'], manifest['targets'])):
            require(fact['view'] == view and fact['ordinal'] == i and fact['original_row'] == original and
                    fact['train_row'] == row['train_row'] and fact['target'] == target and fact['path'] == path and
                    fact['relative_path'] == row['relative_path'] and fact['image_sha256'] == row['image_sha256'] and
                    fact['rng_seed'] == (None if view == 'canonical' else 179081 + row['train_row']) and
                    fact['original_rgb'] == mapping['canonical'][i]['original_rgb'] and
                    (fact['rgb'] == fact['original_rgb'] if view == 'canonical' else fact['rgb']['size'] == [256, 256]),
                    'swapped/leaked image/view mapping')
            for rgb in (fact['original_rgb'], fact['rgb']):
                require(rgb.keys() == {'mode', 'size', 'sha256'} and rgb['mode'] == 'RGB' and
                        len(rgb['size']) == 2 and all(type(n) is int and n > 0 for n in rgb['size']) and
                        re.fullmatch('[0-9a-f]{64}', rgb['sha256'] or '') is not None, 'RGB witness differs')
            pixel = fact['pixels']
            require(pixel.keys() == {'dtype', 'shape', 'sha256'} and pixel['dtype'] == 'torch.float32' and
                    pixel['shape'] == [3, 256, 256] and re.fullmatch('[0-9a-f]{64}', pixel['sha256'] or '') is not None,
                    'processed pixel witness differs')


def limits(started, torch):
    require(time.perf_counter() - started < 300, 'complete genuine export exceeds300')
    require(torch.cuda.max_memory_allocated() < 10_000_000_000, 'complete CUDA allocation exceeds<10GB')


def export(context, started):
    args, prior = context['args'], context['prior']
    source, extract = prior['source_driver'], prior['extract']
    descriptor = read_json(args.startup_terminal, args.startup_terminal_sha256, {})
    startup_proof = file_json(descriptor['proof'], {})
    # package_origins mutates original guards identically to startup BEFORE comparison.
    before = source.cgroup_memory()
    packages = source.package_origins(prior)
    require(packages == startup_proof['packages'] == prior['proof']['origins']['packages'], 'startup/source packages differ')
    admit_terminal(context, descriptor, startup_proof, 'startup')
    file_json(descriptor['proof'], context['guards'])
    context['guards'][str(canonical(descriptor['log']['path']))] = descriptor['log']['sha256']
    context['guards'][str(canonical(args.startup_terminal))] = args.startup_terminal_sha256
    identity = invocation(context)
    require(identity['cuda_visible_devices'] not in (None, '') and
            all(identity[k] == startup_proof['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
            identity['invocation_id'] not in (descriptor['invocation_id'], prior['proof']['invocation']['invocation_id']),
            'fresh export invocation/interpreter/CUDA differs')
    args.output.mkdir()  # Any failure leaves a non-reusable output, never an accepted partial cache.
    prior['packages'] = packages
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA initialized before CPU source verification')
    cpu_flags = prior['proof']['numerical_flags']
    torch.set_num_threads(cpu_flags['threads'])
    if torch.get_num_interop_threads() != cpu_flags['interop_threads']:
        torch.set_num_interop_threads(cpu_flags['interop_threads'])
    require(source.numerical_flags() == cpu_flags, 'original numerical flags differ')
    rng = torch.random.get_rng_state().clone()
    model, processor, roles = source.fresh_source(prior)
    require(source.model_facts(model, processor, roles, packages) == prior['proof']['runtime'], 'fresh original source runtime differs')
    cpu_origins = source.imported_origins(extract, packages)
    for name, path in cpu_origins['modules'].items():
        require(prior['proof']['origins']['modules'].get(name) == path, 'loaded CPU module origin differs: ' + name)
    for path, digest in cpu_origins['files'].items():
        require(prior['proof']['origins']['files'].get(path) == digest, 'loaded CPU file origin differs: ' + path)
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from torch.nn import functional as F
    namespace = {'Dataset': torch.utils.data.Dataset, 'Path': Path, 'Image': Image, 'transforms': transforms}
    node = image_rows_node(context['launch']['image_rows']['path'])
    exec(compile(ast.Module(body=[node], type_ignores=[]), context['launch']['image_rows']['path'], 'exec'), namespace)
    manifest = context['selected']
    dataset = namespace['ImageRows'](tuple(Path(p) for p in manifest['resolved_paths']), tuple(manifest['targets']), augment=True)
    model.requires_grad_(False)
    roles = [{**r, 'role': 'frozen'} for r in roles]
    runtime = source.model_facts(model, processor, roles, packages)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA device required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = source.numerical_flags()
    model = model.cuda().eval()  # Whole peak includes construction, calibration and reload; never reset.
    require(model.config.hidden_size == WIDTH and all(p.dtype == torch.float32 and p.device.type == 'cuda' and
            not p.requires_grad and p.grad is None for p in model.parameters()), 'frozen FP32 CUDA So400 required')
    features = {v: np.empty((ROWS, WIDTH), dtype=np.float32) for v in VIEWS}
    mapping, witnesses, calibration = {v: [] for v in VIEWS}, {}, {}

    def pixels_for(start, count, view):
        images, facts = [], []
        try:
            for i in range(start, start + count):
                row, path = manifest['rows'][i], Path(manifest['resolved_paths'][i])
                require((Path(prior['fit']['dataset_root']) / row['relative_path']).resolve() == path and
                        path.is_relative_to(Path(prior['fit']['dataset_root'])) and extract.sha(path) == row['image_sha256'],
                        'TRAIN path/image bytes changed before decode')
                with Image.open(path) as opened:
                    original = opened.convert('RGB')
                def rgb_fact(image):
                    return {'mode': image.mode, 'size': list(image.size), 'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
                original_rgb = rgb_fact(original)
                try:
                    image = original.copy() if view == 'canonical' else isolated_view(torch, original, dataset.augment, row['train_row'])
                finally:
                    original.close()
                images.append(image)
                facts.append({'view': view, 'ordinal': i, 'original_row': manifest['original_rows'][i],
                              'train_row': row['train_row'], 'target': manifest['targets'][i], 'path': str(path),
                              'relative_path': row['relative_path'], 'image_sha256': row['image_sha256'],
                              'rng_seed': None if view == 'canonical' else 179081 + row['train_row'],
                              'original_rgb': original_rgb, 'rgb': rgb_fact(image)})
            pixels = processor(images=images, return_tensors='pt')['pixel_values']
            require(pixels.dtype == torch.float32 and list(pixels.shape) == [count, 3, 256, 256] and
                    torch.isfinite(pixels).all().item(), 'actual native256 processed pixels differ')
            for i, fact in enumerate(facts):
                fact['pixels'] = source.tensor_fact(pixels[i])
            return pixels.cuda(), facts
        finally:
            for image in images:
                image.close()

    def infer(pixels):
        with torch.autocast('cuda', dtype=torch.float16):
            raw = model(pixel_values=pixels).pooler_output.float()
        require(raw.dtype == torch.float32 and raw.shape == (pixels.shape[0], WIDTH) and
                torch.isfinite(raw).all().item() and (raw.norm(dim=1) > 0).all().item(), 'native raw source outputs differ')
        return raw, F.normalize(raw, dim=1)

    def calibrate(view):
        pixels, facts = pixels_for(0, 4, view)
        fp32 = model(pixel_values=pixels).pooler_output.float()
        raw, unit = infer(pixels)
        require(torch.isfinite(fp32).all().item() and (fp32.norm(dim=1) > 0).all().item(), 'FP32 calibration differs')
        cosines = F.cosine_similarity(fp32, raw, dim=1).cpu().tolist()
        require(min(cosines) >= .999, 'first4 FP32/autocast cosine below.999')
        return {'images': facts, 'fp32': source.tensor_fact(fp32), 'fp16': source.tensor_fact(raw), 'cosines': cosines}

    def witness(pixels, facts, raw, unit):
        return {'images': facts, 'pixels': source.tensor_fact(pixels), 'raw': source.tensor_fact(raw), 'unit': source.tensor_fact(unit)}

    with torch.no_grad():
        for view in VIEWS:
            calibration[view] = calibrate(view)
            limits(started, torch)
        extraction_started = time.perf_counter()
        batch_sizes = []
        for start in range(0, ROWS, BATCH):
            count = min(BATCH, ROWS - start)
            for view in VIEWS:
                pixels, facts = pixels_for(start, count, view)
                raw, unit = infer(pixels)
                features[view][start:start + count] = unit.cpu().numpy()
                mapping[view].extend(facts)
                if start in (0, ROWS - ROWS % BATCH):
                    witnesses[f'{view}:{start}'] = witness(pixels, facts, raw, unit)
                del pixels, raw, unit
                limits(started, torch)
            batch_sizes.append(count)
        torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - extraction_started
    validate_views(manifest, mapping)
    require(batch_sizes == [BATCH] * (ROWS // BATCH) + [ROWS % BATCH], 'identical B32/tail19 required')
    for view in VIEWS:
        require(calibration[view]['images'] == mapping[view][:4] and features[view].shape == (ROWS, WIDTH) and
                features[view].dtype == np.float32 and np.isfinite(features[view]).all() and
                np.allclose(np.linalg.norm(features[view], axis=1), 1, rtol=0, atol=1e-5), 'complete normalized cache differs')
    model.cpu()
    require(source.model_facts(model, processor, roles, packages) == runtime, 'complete frozen source state changed')
    del model, processor
    gc.collect()
    limits(started, torch)
    # Independent constructor/strict complete original checkpoint reload, never two live models.
    checkpoint = canonical(prior['proof']['checkpoint']['path'])
    require(extract.sha(checkpoint) == SOURCE_CHECKPOINT_SHA, 'source checkpoint changed before reload')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    require(saved.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
            saved['runtime'] == prior['proof']['runtime'] and saved['config'] == prior['proof']['runtime']['config'] and
            source.tensor_fact(saved['cpu_rng']) == prior['proof']['cpu_rng'], 'complete original checkpoint differs')
    model = source.construct(saved['config'], prior)
    require(saved['vision'].keys() == prior['expected'].keys(), 'strict independent vision inventory differs')
    for name, value in saved['vision'].items():
        require(source.tensor_fact(value) == prior['proof']['runtime']['vision'][name], 'serialized vision provenance differs')
    model.load_state_dict(saved['vision'], strict=True)
    buffers = dict(model.named_buffers())
    require(buffers.keys() == saved['buffers'].keys(), 'strict complete buffer inventory differs')
    with torch.no_grad():
        for name, value in buffers.items():
            require(source.tensor_fact(value) == source.tensor_fact(saved['buffers'][name]), 'independent nonpersistent buffer differs')
            value.copy_(saved['buffers'][name])
    model.requires_grad_(False).eval()
    del saved, buffers, value
    gc.collect()
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(prior['entry']['input']['preprocessor']['path'], local_files_only=True, backend='torchvision')
    require(source.model_facts(model, processor, roles, packages) == runtime, 'complete strict independent source reload differs')
    model = model.cuda().eval()
    with torch.no_grad():
        for view in VIEWS:
            require(calibrate(view) == calibration[view], 'independent first4 numerical calibration differs')
            for start, count in ((0, BATCH), (ROWS - ROWS % BATCH, ROWS % BATCH)):
                pixels, facts = pixels_for(start, count, view)
                raw, unit = infer(pixels)
                require(witness(pixels, facts, raw, unit) == witnesses[f'{view}:{start}'] and
                        np.array_equal(unit.cpu().numpy(), features[view][start:start + count]),
                        'independent pixels/raw/unit/cache witness differs')
                del pixels, raw, unit
                limits(started, torch)
    model.cpu()
    require(source.model_facts(model, processor, roles, packages) == runtime and
            torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'independent forward changed source state/CPU RNG/flags')
    del model, processor
    gc.collect()
    cache_facts = {}
    for view in VIEWS:
        cache = args.output / (view + '.npy')
        with extract.exclusive(cache) as stream:
            np.save(stream, features[view], allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        digest = extract.sha(cache)
        context['guards'][str(cache)] = digest
        loaded = np.load(cache, allow_pickle=False)
        require(loaded.dtype == np.float32 and loaded.shape == (ROWS, WIDTH) and
                np.array_equal(loaded, features[view]), 'serialized paired cache differs')
        del loaded
        cache_facts[view] = {'path': str(cache), 'sha256': digest, 'shape': [ROWS, WIDTH], 'dtype': 'float32',
                             'normalized': True, 'raw_pooled_cache': False}
        limits(started, torch)
    origins = source.imported_origins(extract, packages)
    for path, digest in origins['files'].items():
        require(prior['guards'].setdefault(path, digest) == digest, 'loaded origin changed: ' + path)
    rehash(context)
    after = source.cgroup_memory()
    require(before['path'] == after['path'], 'whole export cgroup changed')
    limits(started, torch)
    result = {'schema': SCHEMA, 'phase': 'export', 'pass': True, 'exported': True,
              'binding': binding(context), 'startup_terminal': launch_descriptor(args.startup_terminal, args.startup_terminal_sha256),
              'caches': cache_facts, 'ordered_input': manifest, 'view_mapping': mapping,
              'ordered_view_sha256': {v: object_sha(mapping[v]) for v in VIEWS},
              'arithmetic': 'native256 B32+tail19 FP32 frozen vision; F16 autocast; pooled.float(); CUDA F.normalize; CPU NumPy F32',
              'original_source_witness_pixels_decoded': False, 'held_pixels_decoded': 0,
              'strict_independent_reload_exact': True, 'source_runtime': runtime,
              'source_checkpoint': prior['proof']['checkpoint'], 'witnesses': witnesses,
              'fp32_autocast_first4': calibration, 'constructor_and_view_rng_preserved': True,
              'cpu_rng': source.tensor_fact(rng), 'cpu_numerical_flags': cpu_flags, 'export_numerical_flags': flags,
              'counters': {'images_per_view': ROWS, 'classes': CLASSES, 'batch_sizes_per_view': batch_sizes,
                           'views': list(VIEWS), 'optimizer_updates': 0},
              'input_guards': context['guards'], 'original_input_guards': prior['guards'], 'origins': origins,
              'exit_rehash_pass': True, 'quality_read': False, 'training_qualified': False,
              'quality_qualified': False, 'initializer_qualified': False,
              'gradients_created': False, 'optimizer_created': False, 'updates': 0, 'historical_cache_reused': False,
              'resource_policy': EXPORT_POLICY, 'cgroup_before': before, 'cgroup_after': after,
              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'complete_unit_peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reset': False,
              'wall_seconds': time.perf_counter() - started, 'extraction_seconds': extraction_seconds,
              'invocation': identity, 'terminal_exit_and_both_locks_require_parent_receipt': True}
    write_json(context, args.output / 'receipt.json', result)
    limits(started, torch)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('startup', 'export'), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--startup-terminal', type=Path)
    result.add_argument('--startup-terminal-sha256')
    return result


def main():
    args = parser().parse_args()
    started = time.perf_counter()
    try:
        require((args.startup_terminal is None and args.startup_terminal_sha256 is None) if args.phase == 'startup' else
                (args.startup_terminal is not None and args.startup_terminal_sha256 is not None), 'startup terminal mode differs')
        context = authority(args)
        result = startup(context, started) if args.phase == 'startup' else export(context, started)
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as error:
        raise SystemExit('Genuine view export rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
