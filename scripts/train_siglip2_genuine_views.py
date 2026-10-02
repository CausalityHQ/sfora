#!/usr/bin/env python3
"""Matched affine So400 heads on fresh genuine mild TRAIN view caches.

Freeze exactly FILES in execution.json; schema/keys are LAUNCH_KEYS below.
FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}.
export_reference={root:EXPORT_ROOT,execution_sha256:EXPORT_EXECUTION_SHA,
code:EXPORT_CODE,authority:FILE pinned EXPORT_AUTHORITY_SHA}; its exact TWO-file
code hashes and authority come from the parent's actual corrected freeze.
The known failed v1 closure is explicitly inadmissible. selected_export is the complete EXPORT
TERMINAL with proof:FILE (receipt.json), log:FILE, unit, invocation_id,
service_seconds,native_peak_rss_kib,both_locks_held:true. Its original startup
terminal is independently admitted unchanged. No partial/failed cache reuse.
Both fresh normalized F32[6355,1152] caches and complete ordered image/view
mapping are authenticated BEFORE native imports; original source, processor,
package, runtime, image and uncached exit predicates remain genuine.

cached_reference={root,execution_sha256:CACHED_EXECUTION_SHA} is the original
TWO-file cached trainer: only its fresh affine head factory is used, never its
authority/old cache/initializers. training_reference={root,execution_sha256:
REFERENCE_EXECUTION_SHA} is the original FIVE-file adaptation/math closure.
helpers={pca:FILE,packing:FILE} independently pinned by HELPER_SHAS. Helpers
load ONCE after admission; initializer reconstruction uses the same PCA helper
and a fresh native head. No imported helper's globals or predicates change.
partition:FILE is pinned PARTITION_SHA and must equal the exporter partition.
selected_cpu:null for cpu, otherwise complete NEW TERMINAL with receipt:FILE;
selected_mechanics:null except train, then {control:TERMINAL,candidate:TERMINAL}.
Recipe/resource_policy must equal RECIPE/policy(phase), both_locks_held:true.
Parent supplies future cache/terminal hashes ONLY after actual completion.

python -B ROOT/train_siglip2_genuine_views.py --execution-sha256 SHA
 --authority FILE --authority-sha256 SHA --phase cpu|mechanics|train
 --arm control|candidate --seed 179061|179069 --output NEW_ABSOLUTE_DIRECTORY
CPU=control061/CUDA explicitly hidden/120s; NEW initializer, objective,
gradients, exact resume and raw/unit/CPU-pack qualify BOTH arms. Mechanics=
eacharm061/300s/full17 vs independent8+9 with complete strict reload; discard.
TRAIN=1000 fresh updates/300s only after BOTH accepted mechanics. CUDA requires
one visible device and CUBLAS_WORKSPACE_CONFIG=:4096:8. All units8GiB/noSwap,
zero disallowed memory events/CUDAallocated<10GB/both lifetime locks; sources,
admission, save, strict reload, outputs and uncached exit are inside the cap.

ONLY canonical TRAIN builds NEW PCA128/center/scaling/proxies/bank, init179074,
zero up, phi=.5*z and exact five FP32 optimizer members. Original class-balanced
1000x64/micro16 anchors use PCG64(seed); independent mask PCG64(seed+3000001)
<.5. Control consumes canonical; candidate consumes actual augmented same-row
features on the mask. Both banks refresh CANONICAL PRE-update detached values
with original last-duplicate semantics. CE margin.3/scale64 plus8 valid rank.
No donors/interpolation/old trained state/held pixels/serving augmentation.

initializer.pt contains the fresh complete initial tree and discarded CPU
fixtures. resume.pt has exactly PAYLOAD_KEYS: both cache paths/hashes/roles,
ordered TRAIN/image/view mapping digests, masks, PCA, initializer, head/buffers,
classifier, bank, targets, positives, schedules, AdamW/scaler, CPU+CUDA RNG,
source identity, numeric flags and counter. Future terminal/quality/native
qualification remains UNRUN until actual original receipts exist.
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
import re
import resource
import statistics
import struct
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCHEMA = 'siglip2-genuine-view-trainer-v1'
AUTHORITY_SCHEMA = 'siglip2-genuine-view-trainer-launch-v1'
FILES = {'train_siglip2_genuine_views.py', 'test_siglip2_genuine_view_training.py'}
CACHED_FILES = {'train_siglip2_cached_readout.py', 'test_siglip2_cached_readout.py'}
CACHED_EXECUTION_SHA = '907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598'
CACHED_TRAINER_SHA = 'a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c'
REFERENCE_EXECUTION_SHA = '236327289f110ed2bf2012ea6e0d1ea48cf822b4e4094d9cf544081a2dc476f7'
FAILED_EXPORT_EXECUTION_SHA = '2e4bcececc3b32c01544cdd3b075bf3c6079a8e3680bcb4e668b1e080f030031'
FAILED_EXPORT_AUTHORITY_SHA = '862cdec3df3b12af88d00c3fa9412fa01804222fc6defbaff39c84d532a51577'
EXPORT_ROOT = Path('/home/riomus/runs/sfora-so400-genuine-view-export-source-v2')
EXPORT_EXECUTION_SHA = '8ec7f2687f7d1e7962de4f9753cefae989d0b6310d325b5e19da33a9961af3dc'
EXPORT_AUTHORITY_SHA = 'a0072203febf1ef3d16ea329deb0a52dccef34a63223d7dc4b5a430d43d8b6a4'
EXPORT_CODE = {'export_siglip2_genuine_views.py': 'e5e98f9bc85680cab013d53752e7fa140da9d5e7f7ecd4537413b29b1047f65e',
               'test_siglip2_genuine_views.py': '980fe18f9fd2baaf7e1bb5c19738bc7971c3010f6b4e66c83d284dd7c79afb2b'}
EXPORT_FILES = {'export_siglip2_genuine_views.py', 'test_siglip2_genuine_views.py'}
TRAINING_FILES = {'train_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_adaptation.py',
                  'deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py'}
TRAINING_SHA = 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'
HELPER_SHAS = {'pca': '1608181a9c7ba18d1ae1016804898551b2bba9ab40fec9bd4cc2eaf27981771c',
               'packing': '4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67'}
VIEWS = ('canonical', 'augmented')
PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'
FIT_SHA = 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'
MANIFEST_SHA = 'd32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251'
ROWS, CLASSES, WIDTH, DIM = 6355, 1008, 1152, 128
SEEDS, ARMS = (179061, 179069), ('control', 'candidate')
PARAMETERS = ['compact_head.primary.weight', 'compact_head.primary.bias',
              'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
SHAPES = [(128, 1152), (128,), (32, 1152), (128, 32), (1008, 128)]
RECIPE = {'width': WIDTH, 'rows': ROWS, 'classes': CLASSES, 'output_dim': DIM,
          'rank': 32, 'initialization_seed': 179074, 'steps': 1000, 'batch': 64,
          'microbatch': 16, 'seeds': list(SEEDS), 'margin': .3, 'scale': 64,
          'rank_weight': 8, 'learning_rate': 1e-4, 'weight_decay': .05, 'clip': 1,
          'initial_scaler': 128, 'control': '0.5*z', 'candidate': '0.5*z',
          'view_probability': .5, 'mask_rng': 'PCG64', 'mask_seed_offset': 3000001,
          'bank': 'canonical pre-update detached last duplicate',
          'encoder_updates': False, 'augmentation': 'one genuine cached mild TRAIN view'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'export_reference',
               'selected_export', 'cached_reference', 'training_reference', 'helpers', 'partition',
               'selected_cpu', 'selected_mechanics', 'resource_policy', 'both_locks_held', 'recipe'}
STATIC_KEYS = ('pca', 'target', 'positive', 'schedules', 'masks', 'original_rows', 'initializer', 'partition', 'views')
INITIAL_KEYS = {'head', 'classifier', 'bank', 'pca', 'target', 'positive', 'schedules', 'masks', 'original_rows', 'views'}
PAYLOAD_KEYS = {'schema', 'identity', 'head', 'classifier', 'bank', *STATIC_KEYS,
                'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng', 'cuda_rng',
                'counter', 'seed', 'numerical_flags', 'source'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(phase):
    require(phase in ('cpu', 'mechanics', 'train'), 'fixed phase required')
    return {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'SHA256 required')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def descriptor_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'exact FILE descriptor required')
    raw = bound_file(guards, value['path'], value['sha256']).read_bytes()
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)


def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = descriptor_json({'path': str(root / 'execution.json'), 'sha256': expected}, guards)
    require(code.keys() == names, 'exact execution closure required')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def check_launch(launch, args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and launch['seed'] == args.seed and args.arm in ARMS and
            type(args.seed) is int and args.seed in SEEDS and launch['recipe'] == RECIPE and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True,
            'launch profile differs')
    require((args.phase != 'cpu' or (args.arm, args.seed) == ('control', SEEDS[0])) and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]), 'CPU/mechanics seed/arm differs')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train'), 'phase prerequisites differ')
    if args.phase == 'train':
        require(isinstance(launch['selected_mechanics'], dict) and
                launch['selected_mechanics'].keys() == set(ARMS), 'BOTH mechanics required')
    for key, digest in (('cached_reference', CACHED_EXECUTION_SHA),
                        ('training_reference', REFERENCE_EXECUTION_SHA)):
        require(launch[key].keys() == {'root', 'execution_sha256'} and
                launch[key]['execution_sha256'] == digest, 'unchanged helper closure pin differs')
    require(launch['export_reference'].keys() == {'root', 'execution_sha256', 'code', 'authority'} and
            launch['export_reference']['root'] == str(EXPORT_ROOT) and
            launch['export_reference']['execution_sha256'] == EXPORT_EXECUTION_SHA and
            launch['export_reference']['code'] == EXPORT_CODE and
            launch['export_reference']['authority'].keys() == {'path', 'sha256'} and
            launch['export_reference']['authority']['sha256'] == EXPORT_AUTHORITY_SHA and
            launch['export_reference']['authority']['path'] == str(Path(launch['export_reference']['root']) / 'authority.json') and
            launch['partition'].keys() == {'path', 'sha256'} and
            launch['partition']['sha256'] == PARTITION_SHA, 'genuine export/partition pins differ')
    require(launch['helpers'].keys() == HELPER_SHAS.keys() and
            all(value.keys() == {'path', 'sha256'} and value['sha256'] == HELPER_SHAS[name]
                for name, value in launch['helpers'].items()), 'PCA/packing helper pins differ')
    terminal = launch['selected_export']
    require(terminal.keys() == {'proof', 'log', 'unit', 'invocation_id', 'service_seconds',
                                'native_peak_rss_kib', 'both_locks_held'} and terminal['both_locks_held'] is True,
            'complete original export terminal required')


def method(launch):
    return {k: launch[k] for k in ('execution_sha256', 'export_reference', 'selected_export',
            'cached_reference', 'training_reference', 'helpers', 'partition', 'recipe')}


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def check_steps(rows, start, count):
    keys = {'step', 'batch', 'schedule_sha256', 'feature_rows_sha256', 'mask_sha256', 'clean_bank_sha256',
            'ce', 'rank', 'loss', 'scale', 'preclip_norm', 'gradient_norms', 'state_sha256', 'seconds'}
    require(len(rows) == count, 'complete update records required')
    for step, row in enumerate(rows, start):
        require(row.keys() == keys and row['step'] == step and len(row['batch']) == 64 and
                all(type(n) is int and 0 <= n < ROWS for n in row['batch']) and
                all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k]) for k in
                    ('schedule_sha256', 'feature_rows_sha256', 'mask_sha256', 'clean_bank_sha256', 'state_sha256')) and
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                    ('ce', 'rank', 'loss', 'scale', 'preclip_norm', 'seconds')) and
                row['loss'] == row['ce'] + 8 * row['rank'] and row['scale'] == 128 and row['seconds'] > 0 and
                row['gradient_norms'].keys() == set(PARAMETERS) and
                all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in row['gradient_norms'].values()),
                'complete finite update record differs')


def check_terminal_record(record, launch, phase, arm):
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == SEEDS[0] and record['pass'] is True and record['quality_read'] is False and
            record['exit_rehash_pass'] is True and record['strict_reload_exact'] is True and
            record['optimizer_members'] == 5 and method(record['launch']) == method(launch) and
            record['resource_policy'] == policy(phase) and record['trained_state_reused'] is False,
            'new method terminal binding differs')
    check_launch(record['launch'], SimpleNamespace(phase=phase, arm=arm, seed=SEEDS[0],
                                                execution_sha256=launch['execution_sha256']))
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['initial_arm_parity'] is True and
                record['gradient_witnesses'] is True and record['cpu_resume_exact'] is True and
                record['training_only_fit'] is True and record['completed_step'] == 0,
                'CPU witnesses incomplete')
    else:
        require(record['launch']['selected_cpu'] == launch['selected_cpu'] and
                record['completed_step'] == 17 and record['checkpoint'] is None and
                record['training_state_discarded'] is True and record['replay_exact'] is True and
                len(record['steps']) == 17 and len(record['resumed_steps']) == 9 and
                all(diagnostic(a) == diagnostic(b) for a, b in
                    zip(record['steps'][8:], record['resumed_steps'], strict=True)) and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'discarded17 replay incomplete')
        check_steps(record['steps'], 1, 17); check_steps(record['resumed_steps'], 9, 9)


def load_helper(name, path, digest, guards):
    path = bound_file(guards, path, digest)
    require(name not in sys.modules, 'helper already loaded; origin is not admissible')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'bare helper origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == Path(module.__spec__.origin) == path, 'loaded helper origin differs')
    bound_file(guards, path, digest)
    return module


def check_export_record(context, record):
    exporter, genuine = context['exporter'], context['genuine']
    require(record['schema'] == exporter.SCHEMA and record['phase'] == 'export' and
            record['pass'] is True and record['exported'] is True and
            record['binding'] == exporter.binding(genuine) and record['ordered_input'] == genuine['selected'] and
            record['historical_cache_reused'] is False and record['held_pixels_decoded'] == 0 and
            record['original_source_witness_pixels_decoded'] is False and record['quality_read'] is False and
            all(record[k] is False for k in ('training_qualified', 'quality_qualified', 'initializer_qualified',
                                            'gradients_created', 'optimizer_created', 'cuda_peak_reset')) and
            record['updates'] == 0 and record['strict_independent_reload_exact'] is True and
            record['constructor_and_view_rng_preserved'] is True and record['exit_rehash_pass'] is True and
            record['source_checkpoint'] == genuine['prior']['proof']['checkpoint'] and
            record['cpu_numerical_flags'] == genuine['prior']['proof']['numerical_flags'] and
            record['counters'] == {'images_per_view': ROWS, 'classes': CLASSES,
                'batch_sizes_per_view': [32] * (ROWS // 32) + [ROWS % 32],
                'views': list(VIEWS), 'optimizer_updates': 0}, 'fresh complete paired export differs')
    exporter.validate_views(genuine['selected'], record['view_mapping'])
    require(record['ordered_view_sha256'] == {v: exporter.object_sha(record['view_mapping'][v]) for v in VIEWS},
            'ordered view mapping digest differs')
    caches = record['caches']
    parent = Path(context['launch']['selected_export']['proof']['path']).parent
    require(caches.keys() == set(VIEWS), 'BOTH genuine caches required')
    for view in VIEWS:
        fact = caches[view]
        require(fact.keys() == {'path', 'sha256', 'shape', 'dtype', 'normalized', 'raw_pooled_cache'} and
                fact['path'] == str(parent / (view + '.npy')) and fact['shape'] == [ROWS, WIDTH] and
                fact['dtype'] == 'float32' and fact['normalized'] is True and fact['raw_pooled_cache'] is False and
                fact['sha256'] != FIT_SHA, 'fresh cache role/layout differs')
    require(caches['canonical']['path'] != caches['augmented']['path'], 'cache roles alias')


def cache_facts(fact, guards):
    """Admit the actual complete fresh NPY payload before native imports."""
    import ast
    path = bound_file(guards, fact['path'], fact['sha256'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        magic = stream.read(8); digest.update(magic)
        require(magic in (b'\x93NUMPY\x01\x00', b'\x93NUMPY\x02\x00'), 'cache NPY version differs')
        size = 2 if magic[-2] == 1 else 4
        raw_length = stream.read(size); digest.update(raw_length)
        length = int.from_bytes(raw_length, 'little')
        require(0 < length <= 65536, 'cache header size differs')
        raw_header = stream.read(length); digest.update(raw_header)
        header = ast.literal_eval(raw_header.decode('latin1'))
        require(isinstance(header, dict) and header.keys() == {'descr', 'fortran_order', 'shape'} and
                header['descr'] == '<f4' and header['fortran_order'] is False and header['shape'] == (ROWS, WIDTH) and
                path.stat().st_size == 8 + size + length + ROWS * WIDTH * 4, 'fresh cache shape/dtype/payload differs')
        unpack = struct.Struct('<' + str(WIDTH) + 'f')
        advised = 0
        for _ in range(ROWS):
            raw = stream.read(unpack.size); digest.update(raw)
            values = unpack.unpack(raw)
            require(all(math.isfinite(v) for v in values) and
                    abs(math.sqrt(math.fsum(v * v for v in values)) - 1) <= 1e-5, 'fresh cache finite/unit row differs')
            if stream.tell() - advised >= 1024**2:
                os.posix_fadvise(stream.fileno(), advised, stream.tell() - advised, os.POSIX_FADV_DONTNEED)
                advised = stream.tell()
        require(stream.read(1) == b'', 'cache trailing bytes differ')
        os.posix_fadvise(stream.fileno(), advised, stream.tell() - advised, os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == fact['sha256'], 'cache changed during payload admission')


def authority(args):
    require(not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'safetensors',
                                       'torchvision', 'sfora'} for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = descriptor_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_launch(launch, args)
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and
            not output.exists() and not output.is_symlink(), 'exclusive canonical output required')
    roots = [Path(launch[k]['root']) for k in ('export_reference', 'cached_reference', 'training_reference')]
    require(all(not root.is_relative_to(p) and not p.is_relative_to(root) and
                not output.is_relative_to(p) for p in roots) and not output.is_relative_to(root),
            'separate immutable closures required')
    export_root, cached_root, training_root = roots
    export_ref = launch['export_reference']
    export_code = closure(export_root, export_ref['execution_sha256'], EXPORT_FILES, guards)
    require(export_code == export_ref['code'], 'parent-frozen exact export source hashes differ')
    exporter = load_helper('_genuine_training_exporter', export_root / 'export_siglip2_genuine_views.py',
                           export_code['export_siglip2_genuine_views.py'], guards)
    require(exporter.FILES == EXPORT_FILES and exporter.SCHEMA == 'siglip2-genuine-views-v1' and
            exporter.AUTHORITY_SCHEMA == 'native256-genuine-view-launch-v1' and exporter.PARTITION_SHA == PARTITION_SHA and
            exporter.REF_EXECUTION_SHA == 'ad32df859712e3d608341ed9eda24fd96db375ec85c61779265a9c16cb8dd3b5' and
            exporter.ORIGINAL_AUTHORITY_SHA == '065b32e841e8a05bc71aa36e5c9b32db07b139ba7159f428dc21eba23a57361c' and
            exporter.SOURCE_CHECKPOINT_SHA == '5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189' and
            exporter.MANIFEST_SHA == MANIFEST_SHA and exporter.OLD_CACHE_SHA == FIT_SHA and
            exporter.IMAGE_ROWS_AST_SHA == '09e080b36b7fe3059e15e47ff5795a9390fb9cf5c28c075b955762f0795e4103',
            'unchanged genuine original source/method pins differ')
    genuine = exporter.authority(SimpleNamespace(execution_sha256=export_ref['execution_sha256'],
        authority=Path(export_ref['authority']['path']), authority_sha256=export_ref['authority']['sha256'],
        phase='export', output=output))
    require(genuine['launch']['partition'] == launch['partition'], 'export/trainer partition differs')
    prior = genuine['prior']
    packages = prior['source_driver'].package_origins(prior)
    require(packages == prior['proof']['origins']['packages'], 'original source package origins differ')
    record = descriptor_json(launch['selected_export']['proof'], guards)
    check_export_record({'exporter': exporter, 'genuine': genuine, 'launch': launch}, record)
    startup = exporter.file_json(record['startup_terminal'], {})
    startup_proof = exporter.file_json(startup['proof'], {})
    exporter.admit_terminal(genuine, startup, startup_proof, 'startup')
    require(startup_proof['packages'] == packages, 'original startup packages differ')
    exporter.file_json(startup['proof'], genuine['guards'])
    for value in (startup['log'], record['startup_terminal']):
        bound_file(genuine['guards'], value['path'], value['sha256'])
    for fact in record['caches'].values():
        cache_facts(fact, genuine['guards'])
    require(record['input_guards'] == genuine['guards'] and
            all(record['original_input_guards'].get(p) == h for p, h in prior['guards'].items()),
            'original complete preparation guards differ')
    exporter.admit_terminal(genuine, launch['selected_export'], record, 'export')
    ids = {prior['proof']['invocation']['invocation_id'], startup['invocation_id'],
           launch['selected_export']['invocation_id']}
    require(len(ids) == 3 and record['invocation']['cuda_visible_devices'] not in (None, ''),
            'fresh original export invocation/CUDA differs')
    for path, digest in {**genuine['guards'], **record['original_input_guards']}.items():
        bound_file(guards, path, digest)
    bound_file(guards, launch['selected_export']['log']['path'], launch['selected_export']['log']['sha256'])
    # Reuse ONLY immutable source code and math: their archived authority is never called.
    cached_code = closure(cached_root, CACHED_EXECUTION_SHA, CACHED_FILES, guards)
    require(cached_code['train_siglip2_cached_readout.py'] == CACHED_TRAINER_SHA, 'cached head factory pin differs')
    cached = load_helper('_genuine_training_cached', cached_root / 'train_siglip2_cached_readout.py',
                         CACHED_TRAINER_SHA, guards)
    training_code = closure(training_root, REFERENCE_EXECUTION_SHA, TRAINING_FILES, guards)
    require(training_code['train_siglip2_substrate_adaptation.py'] == TRAINING_SHA, 'original math trainer pin differs')
    original = load_helper('_genuine_training_original', training_root / 'train_siglip2_substrate_adaptation.py',
                           TRAINING_SHA, guards)
    require(training_code['deployed_code_rank.py'] == original.RANK_SHA256, 'original rank helper pin differs')
    for name, pin in original.REFERENCES.items():
        require(training_code[name] == pin['source'], 'original math source pin differs')
        original.selected_ast(training_root / name, pin)
    for name, value in launch['helpers'].items():
        bound_file(guards, value['path'], HELPER_SHAS[name])
        require(not Path(value['path']).is_relative_to(root) and not output.is_relative_to(Path(value['path']).parent),
                'separate immutable native helpers required')
    partition = descriptor_json(launch['partition'], guards)
    source = {'export_binding': record['binding'], 'preparation_terminal': launch['selected_export'],
              'caches': record['caches'], 'ordered_input_sha256': record['binding']['ordered_input_sha256'],
              'ordered_view_sha256': record['ordered_view_sha256'], 'source_checkpoint': record['source_checkpoint'],
              'source_runtime_sha256': exporter.object_sha(record['source_runtime']), 'arithmetic': record['arithmetic']}
    views = {k: source[k] for k in ('caches', 'ordered_input_sha256', 'ordered_view_sha256')}
    context = {'args': args, 'root': root, 'code': code, 'launch': launch, 'cached': cached,
               'exporter': exporter, 'genuine': genuine, 'export_record': record, 'original': original,
               'math_context': {'root': training_root}, 'source': source, 'views': views,
               'partition': partition, 'target': genuine['selected']['targets'], 'guards': guards,
               'source_driver': prior['source_driver'], 'extract': prior['extract'],
               'source_cpu': prior['proof'], 'packages': packages,
               'terminals': {}, 'terminal_cgroups': {}}
    admission = original.FlatAdmission()
    admission.init = genuine['reference']  # unchanged admit_cgroup, used only by admit_terminal
    prior_guards = {p: h for p, h in guards.items() if p != str(args.authority)}
    wanted = [] if args.phase == 'cpu' else [('cpu', 'control', launch['selected_cpu'])]
    if args.phase == 'train':
        wanted += [('mechanics', arm, launch['selected_mechanics'][arm]) for arm in ARMS]
    for phase, arm, terminal in wanted:
        require(terminal['receipt'].keys() == {'path', 'sha256'} and
                Path(terminal['receipt']['path']).name == 'receipt.json', 'new trainer receipt path role differs')
        receipt = descriptor_json(terminal['receipt'], guards)
        check_terminal_record(receipt, launch, phase, arm)
        invocation = receipt['invocation']
        identity = prior['proof']['invocation']
        require(descriptor_json(receipt['authority'], guards) == receipt['launch'] and
                receipt['code'] == code and receipt['authority']['sha256'] == receipt['authority_sha256'] and
                receipt['execution_sha256'] == args.execution_sha256 and receipt['source'] == source and
                receipt['partition_sha256'] == PARTITION_SHA and receipt['numerical_flags'] == prior['proof']['numerical_flags'] and
                invocation['argv'] == [str(root / 'train_siglip2_genuine_views.py'), '--execution-sha256',
                    args.execution_sha256, '--authority', receipt['authority']['path'], '--authority-sha256',
                    receipt['authority']['sha256'], '--phase', phase, '--arm', arm, '--seed', str(SEEDS[0]),
                    '--output', str(Path(terminal['receipt']['path']).parent)] and
                all(invocation[k] == identity[k] for k in ('python', 'python_sha256', 'python_version')) and
                (invocation['cuda_visible_devices'] == '' if phase == 'cpu' else
                 invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8') and
                all(receipt['input_guards'].get(p) == h for p, h in prior_guards.items()),
                'original new trainer launch/source/input closure differs')
        if phase == 'cpu':
            checkpoint = receipt['checkpoint']
            require(checkpoint.keys() == {'path', 'sha256'} and
                    checkpoint['path'] == str(Path(terminal['receipt']['path']).parent / 'initializer.pt') and
                    receipt['input_guards'].get(checkpoint['path']) == checkpoint['sha256'], 'NEW CPU initializer differs')
        require(invocation['invocation_id'] not in ids, 'duplicate original unit invocation')
        ids.add(invocation['invocation_id'])
        context['terminal_cgroups'][phase + ':' + arm] = admission.admit_terminal(
            receipt, terminal, policy(phase)['seconds'], guards)
        for path, digest in receipt['input_guards'].items():
            admission.bound_file(guards, path, digest)
        context['terminals'][phase + ':' + arm] = receipt
    return context


def check_views(views, source):
    require(views.keys() == {'caches', 'ordered_input_sha256', 'ordered_view_sha256'} and
            views == {k: source[k] for k in views} and views['caches'].keys() == set(VIEWS) and
            views['ordered_view_sha256'].keys() == set(VIEWS), 'complete paired cache/mapping binding differs')
    for view in VIEWS:
        fact = views['caches'][view]
        require(fact.keys() == {'path', 'sha256', 'shape', 'dtype', 'normalized', 'raw_pooled_cache'} and
                Path(fact['path']).is_absolute() and Path(fact['path']).name == view + '.npy' and
                fact['shape'] == [ROWS, WIDTH] and fact['dtype'] == 'float32' and fact['normalized'] is True and
                fact['raw_pooled_cache'] is False and re.fullmatch('[0-9a-f]{64}', fact['sha256']) and
                re.fullmatch('[0-9a-f]{64}', views['ordered_view_sha256'][view]), 'paired cache role/layout differs')
    require(re.fullmatch('[0-9a-f]{64}', views['ordered_input_sha256']), 'ordered TRAIN mapping SHA required')


def check_masks(anchors, masks):
    require(len(anchors) == len(masks) == RECIPE['steps'] and
            all(len(a) == len(m) == RECIPE['batch'] and
                all(type(v) is int and 0 <= v < ROWS for v in a) and
                all(type(v) is bool for v in m) for a, m in zip(anchors, masks, strict=True)),
            'complete independent TRAIN masks/schedule differ')


def schedule_and_masks(target, seed):
    import numpy as np
    require(type(seed) is int and seed in SEEDS and len(target) == ROWS and
            all(type(v) is int for v in target) and sorted(set(target)) == list(range(CLASSES)),
            'dense TRAIN target/seed differs')
    members = {}
    for row, label in enumerate(target):
        members.setdefault(label, []).append(row)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(list(range(CLASSES)))
    anchors = np.asarray([[int(rng.choice(members[int(c)])) for c in
        order[(step * 64 + np.arange(64)) % CLASSES]] for step in range(1000)], dtype=np.int64)
    mask_rng = np.random.Generator(np.random.PCG64(seed + RECIPE['mask_seed_offset']))
    masks = mask_rng.random((1000, 64)) < .5
    check_masks(anchors.tolist(), masks.tolist())
    require(all(len(set(target[a] for a in batch)) == 64 for batch in anchors.tolist()),
            'class-balanced TRAIN schedule differs')
    return anchors, masks


def view_inputs(canonical, augmented, anchors, masks, arm):
    """Same-row genuine cache selection; no interpolation or renormalization."""
    require(arm in ARMS, 'fixed genuine view arm required')
    result = canonical[anchors].clone()
    if arm == 'candidate' and bool(masks.any()):
        result[masks] = augmented[anchors[masks]]
    return result


def normalize_nonzero(value):
    import torch
    from torch.nn import functional as F
    require(value.dtype == torch.float32 and torch.isfinite(value).all().item() and
            (value.norm(dim=1) > 0).all().item(), 'cache must be finite nonzero FP32')
    result = F.normalize(value, dim=1)
    require(torch.isfinite(result).all().item(), 'normalized cache nonfinite')
    return result


def clean_bank_descriptors(state, index):
    import torch
    with torch.no_grad(), torch.autocast(device_type=state['features'].device.type, enabled=False):
        return state['head'](state['features'][index]).detach()


def optimizer_state(head, classifier):
    import torch
    params = [('compact_head.' + n, p) for n, p in head.named_parameters()] + [('classifier', classifier)]
    require([n for n, _ in params] == PARAMETERS and len({id(p) for _, p in params}) == 5 and
            [tuple(p.shape) for _, p in params] == SHAPES and
            all(p.requires_grad and p.dtype == torch.float32 for _, p in params), 'EXACT five optimizer members required')
    optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': 1e-4},
                                   {'params': [classifier], 'lr': 1e-4}], weight_decay=.05)
    defaults = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in optimizer.defaults:
        defaults['decoupled_weight_decay'] = True
    require(optimizer.defaults == defaults and not optimizer.state and
            all({k: v for k, v in g.items() if k != 'params'} == {**defaults, 'lr': 1e-4}
                for g in optimizer.param_groups), 'original AdamW defaults/groups differ')
    return params, optimizer


def training_features(context):
    import numpy as np
    import torch
    result = {}
    for view in VIEWS:
        fact = context['source']['caches'][view]
        bound_file({}, fact['path'], fact['sha256'])
        cache = np.load(fact['path'], allow_pickle=False, mmap_mode='r')
        require(cache.shape == (ROWS, WIDTH) and cache.dtype == np.float32, 'fresh TRAIN cache layout differs')
        features = torch.from_numpy(cache.copy())
        del cache
        # Preserve original shared CPU normalization for BOTH caches and arms.
        result[view] = normalize_nonzero(features)
    return result


def initializer(context, ref, features, pca=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    require(features.shape == (ROWS, WIDTH) and features.device.type == 'cpu', 'TRAIN-only CPU PCA required')
    pca_helper = context['pca_helper']
    normalized = F.normalize(features, dim=1)
    if pca is None:
        pca = pca_helper.fit_centered_pca(normalized, dimensions=DIM)
    else:
        pca = pca_helper.CenteredPcaTransform(**pca)
    head = context['cached'].head_from('control', tensors={
        'primary.weight': pca.components, 'primary.bias': -(pca.components @ pca.mean),
        'down.weight': torch.zeros(32, WIDTH), 'up.weight': torch.zeros(DIM, 32),
        'center': normalized.mean(0), 'preactivation_std': torch.ones(())})
    with torch.no_grad():
        nn.init.kaiming_uniform_(head.down.weight, a=5**.5,
                                generator=torch.Generator().manual_seed(RECIPE['initialization_seed']))
        std = head.down(normalized - head.center).std(unbiased=False)
        require(torch.isfinite(std).item() and float(std) > 0, 'TRAIN preactivation std undefined')
        head.preactivation_std.copy_(std); head.down.weight.div_(std)
        projected = pca.apply(normalized)
        target = torch.tensor(context['target'], dtype=torch.int64)
        sums = torch.zeros(CLASSES, DIM)
        for row, label in enumerate(context['target']):
            sums[label] += projected[row]
        classifier = F.normalize(sums, dim=1)
        bank = F.normalize(head(features), dim=1).detach()
    schedules, masks = {}, {}
    for seed in SEEDS:
        anchors, mask = schedule_and_masks(context['target'], seed)
        schedules[str(seed)] = torch.from_numpy(anchors)
        masks[str(seed)] = torch.from_numpy(mask)
    result = {'head': {n: v.detach().clone() for n, v in head.state_dict().items()},
              'pca': {'mean': pca.mean.clone(), 'components': pca.components.clone()},
              'classifier': classifier, 'bank': bank, 'target': target,
              'positive': ref.member_bank_positive_ordinals(target.numpy(), allow_singletons=True),
              'schedules': schedules, 'masks': masks, 'views': context['views'],
              'original_rows': torch.tensor(context['partition']['panels']['train']['original_rows'])}
    require(all(torch.isfinite(v).all().item() for v in (classifier, bank)) and
            (classifier.norm(dim=1) > 0).all().item() and (bank.norm(dim=1) > 0).all().item(), 'initial proxy/bank nonfinite/zero')
    return result


def fresh(context, ref, arm, seed, device, initial, features):
    import torch
    head = context['cached'].head_from('control', tensors=initial['head']).to(device).train()
    classifier = torch.nn.Parameter(initial['classifier'].to(device).clone())
    params, optimizer = optimizer_state(head, classifier)
    return {'head': head, 'classifier': classifier, 'bank': initial['bank'].to(device).detach().clone(),
            'target': initial['target'].to(device).clone(), 'positive': initial['positive'].to(device).clone(),
            'pca': {n: v.clone() for n, v in initial['pca'].items()}, 'schedules': initial['schedules'],
            'masks': initial['masks'], 'views': initial['views'], 'original_rows': initial['original_rows'],
            'partition': context['partition'], 'initializer': initial, 'features': features['canonical'].to(device),
            'augmented': features['augmented'].to(device),
            'params': params, 'optimizer': optimizer, 'seed': seed, 'arm': arm, 'counter': 0,
            'scaler': torch.amp.GradScaler(device, init_scale=128)}


def identity(context, state, flags):
    original = context['original']
    return {'method': method(context['launch']), 'arm': state['arm'], 'seed': state['seed'],
            'selected_cpu': context['launch']['selected_cpu'], 'source': context['source'],
            'parameter_names': PARAMETERS, 'device': state['features'].device.type,
            'feature_versions': {v: state[k]._version for v, k in zip(VIEWS, ('features', 'augmented'), strict=True)},
            'feature_state_sha256': original.fingerprint({v: state[k] for v, k in zip(VIEWS, ('features', 'augmented'), strict=True)}),
            'optimizer_defaults': state['optimizer'].defaults.copy(),
            'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
            'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
            'static_sha256': original.fingerprint({n: state[n] for n in STATIC_KEYS}),
            'buffers_sha256': original.fingerprint(dict(state['head'].named_buffers())),
            'positive_shape': tuple(state['positive'].shape),
            'schedule_sha256': original.fingerprint(state['schedules'][str(state['seed'])]),
            'mask_sha256': original.fingerprint(state['masks'][str(state['seed'])]), 'numerical_flags': flags}


def payload(state, ident):
    import torch
    return {'schema': SCHEMA, 'identity': ident, 'head': dict(state['head'].state_dict()),
            'classifier': state['classifier'].detach(), 'bank': state['bank'],
            **{n: state[n] for n in STATIC_KEYS},
            'optimizer': state['optimizer'].state_dict(), 'optimizer_defaults': state['optimizer'].defaults.copy(),
            'scaler': state['scaler'].state_dict(), 'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if ident['device'] == 'cuda' else [],
            'counter': state['counter'], 'seed': state['seed'],
            'numerical_flags': ident['numerical_flags'], 'source': ident['source']}


def check_payload(saved, ident, step):
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == ident and
            type(saved['counter']) is int and saved['counter'] == step and saved['seed'] == ident['seed'] and
            saved['source'] == ident['source'] and saved['numerical_flags'] == ident['numerical_flags'] and
            saved['optimizer_defaults'] == ident['optimizer_defaults'] and ident['parameter_names'] == PARAMETERS,
            'complete resume identity/state differs')
    optimizer = saved['optimizer']
    require(optimizer.keys() == {'state', 'param_groups'} and optimizer['param_groups'] == ident['optimizer_serial_groups'] and
            [i for g in optimizer['param_groups'] for i in g['params']] == list(range(5)) and
            all(type(i) is int for g in optimizer['param_groups'] for i in g['params']) and
            all(type(i) is int for i in optimizer['state']) and
            set(optimizer['state']) == (set(range(5)) if step else set()), 'EXACT five optimizer states/order required')
    require(all(v.keys() == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == step for v in optimizer['state'].values()) and
            saved['scaler'].keys() == {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'} and
            saved['scaler']['scale'] == 128 and saved['scaler']['growth_factor'] == 2 and
            saved['scaler']['backoff_factor'] == .5 and saved['scaler']['growth_interval'] == 2000 and
            saved['scaler']['_growth_tracker'] == step and ident['device'] in ('cpu', 'cuda') and
            len(saved['cuda_rng']) == (1 if ident['device'] == 'cuda' else 0), 'optimizer counters/scaler/RNG differ')
    def tensor(value, shape, dtype):
        require(tuple(value.shape) == shape and str(value.dtype) == dtype, 'complete tensor layout differs')
    head_shapes = dict(zip(('primary.weight', 'primary.bias', 'down.weight', 'up.weight', 'center', 'preactivation_std'),
                           (*SHAPES[:4], (WIDTH,), ()), strict=True))
    def static(tree):
        require(tree['pca'].keys() == {'mean', 'components'} and
                tree['schedules'].keys() == tree['masks'].keys() == {str(s) for s in SEEDS},
                'complete PCA/schedules/masks inventory differs')
        for value, shape in ((tree['pca']['mean'], (WIDTH,)), (tree['pca']['components'], (DIM, WIDTH))):
            tensor(value, shape, 'torch.float32')
        tensor(tree['target'], (ROWS,), 'torch.int64'); tensor(tree['positive'], ident['positive_shape'], 'torch.int64')
        tensor(tree['original_rows'], (ROWS,), 'torch.int64')
        for seed in SEEDS:
            tensor(tree['schedules'][str(seed)], (1000, 64), 'torch.int64')
            tensor(tree['masks'][str(seed)], (1000, 64), 'torch.bool')
        check_views(tree['views'], ident['source'])
    for tree in (saved, saved['initializer']):
        require(tree['head'].keys() == head_shapes.keys(), 'complete head/buffers required')
        for name, shape in head_shapes.items():
            tensor(tree['head'][name], shape, 'torch.float32')
        tensor(tree['classifier'], SHAPES[4], 'torch.float32'); tensor(tree['bank'], (ROWS, DIM), 'torch.float32')
        static(tree)
    require(saved['initializer'].keys() == INITIAL_KEYS, 'complete NEW initializer required')
    for value in [saved['cpu_rng'], *saved['cuda_rng']]:
        require(len(value.shape) == 1 and value.shape[0] > 0 and str(value.dtype) == 'torch.uint8', 'complete RNG layout differs')
    for i, moments in optimizer['state'].items():
        tensor(moments['step'], (), 'torch.float32')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor(moments[name], SHAPES[i], 'torch.float32')


def integrity(context, state, ident):
    import torch
    actual = [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]
    require([n for n, _ in actual] == PARAMETERS and [tuple(p.shape) for _, p in actual] == SHAPES and
            len({id(p) for _, p in actual}) == 5 and [id(p) for _, p in actual] ==
            [id(p) for _, p in state['params']] == [id(p) for g in state['optimizer'].param_groups for p in g['params']] and
            all(p.dtype == torch.float32 and p.device.type == ident['device'] and p.requires_grad and
                torch.isfinite(p).all().item() for _, p in actual), 'live five-member inventory differs')
    require(state['head'].training and all(m.training and not m._forward_hooks and not m._forward_pre_hooks and
            not m._backward_hooks for m in state['head'].modules()), 'head modes/hooks differ')
    for view, key in zip(VIEWS, ('features', 'augmented'), strict=True):
        value = state[key]
        require(value.shape == (ROWS, WIDTH) and value.dtype == torch.float32 and
                value.device.type == ident['device'] and not value.requires_grad and value.grad is None and
                value._version == ident['feature_versions'][view], 'immutable paired TRAIN cache changed')
    require(context['original'].fingerprint({n: state[n] for n in STATIC_KEYS}) == ident['static_sha256'] and
            context['original'].fingerprint(dict(state['head'].named_buffers())) == ident['buffers_sha256'] and
            state['bank'].shape == (ROWS, DIM) and state['bank'].dtype == torch.float32 and
            not state['bank'].requires_grad and state['bank'].grad is None and torch.isfinite(state['bank']).all().item(),
            'complete static state/buffers/bank differ')
    require(state['optimizer'].defaults == ident['optimizer_defaults'] and
            [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups] == ident['optimizer_groups'],
            'live optimizer options differ')
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    for moments in saved['optimizer']['state'].values():
        require(all(torch.isfinite(v).all().item() for v in moments.values()), 'optimizer moments nonfinite')


def save(context, state, ident, path):
    import torch
    saved = payload(state, ident)
    check_payload(saved, ident, state['counter'])
    with context['extract'].exclusive(path) as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    return context['exporter'].sha(path), context['original'].fingerprint(saved)


def restore(context, ref, path, sha, digest, ident, step, features):
    import torch
    bound_file({}, path, sha)
    disk = torch.load(path, map_location='cpu', weights_only=True)
    check_payload(disk, ident, step)
    require(context['original'].fingerprint(disk) == digest, 'serialized complete state fingerprint differs')
    state = fresh(context, ref, ident['arm'], ident['seed'], ident['device'], disk['initializer'], features)
    state['head'].load_state_dict(disk['head'], strict=True)
    with torch.no_grad():
        state['classifier'].copy_(disk['classifier'].to(ident['device']))
    state['bank'] = disk['bank'].to(ident['device']).detach().clone()
    for name in STATIC_KEYS:
        state[name] = disk[name]
    state['target'] = state['target'].to(ident['device']); state['positive'] = state['positive'].to(ident['device'])
    state['optimizer'].load_state_dict(disk['optimizer']); state['scaler'].load_state_dict(disk['scaler'])
    state['counter'] = step
    require(torch.equal(state['positive'], ref.member_bank_positive_ordinals(state['target'].cpu().numpy(),
                allow_singletons=True).to(ident['device'])), 'strict positive ordinal reload differs')
    torch.random.set_rng_state(disk['cpu_rng'].clone())
    if ident['device'] == 'cuda':
        torch.cuda.set_rng_state_all([v.clone() for v in disk['cuda_rng']])
    del disk
    gc.collect()
    integrity(context, state, ident)
    require(context['original'].fingerprint(payload(state, ident)) == digest, 'independent complete resume reload differs')
    return state


def terms(context, ref, state, index, mask):
    import torch
    inputs = view_inputs(state['features'], state['augmented'], index, mask, state['arm'])
    raw = state['head'](inputs)
    ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                     torch.arange(DIM, device=raw.device).unsqueeze(0), margin=.3, scale=64)
    rank = context['original'].valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
    return raw, ce, rank


def update(context, ref, state, ident, step):
    import torch
    device = ident['device']
    if device == 'cuda':
        torch.cuda.synchronize()
    tick = time.perf_counter()
    require(state['counter'] == step - 1 and 1 <= step <= 1000, 'update counter differs')
    batch = state['schedules'][str(state['seed'])][step - 1].tolist()
    masks = state['masks'][str(state['seed'])]
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version, rows, ce_sum, rank_sum = state['bank']._version, [], 0., 0.
    for offset in range(0, 64, 16):
        index = torch.tensor(batch[offset:offset + 16], device=device)
        mask = masks[step - 1, offset:offset + 16].to(device)
        # Clean forward is detached BEFORE any optimizer mutation, for BOTH arms.
        rows.append(clean_bank_descriptors(state, index))
        with torch.autocast(device_type=device, enabled=False):
            raw, ce, rank = terms(context, ref, state, index, mask)
            loss = (ce + 8 * rank) * .25
        require(raw.dtype == torch.float32 and torch.isfinite(raw).all().item() and torch.isfinite(loss).item(),
                'nonfinite cached update')
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach()) * .25; rank_sum += float(rank.detach()) * .25
    require(state['bank']._version == version, 'bank changed during backward')
    scaler.unscale_(optimizer)
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for _, p in state['params']), 'all five finite FP32 gradients required')
    gradients = {n: float(p.grad.double().norm()) for n, p in state['params']}
    if step <= 2:
        require(gradients['compact_head.up.weight'] > 0 and
                (gradients['compact_head.down.weight'] == 0 if step == 1 else gradients['compact_head.down.weight'] > 0),
                'initial zero/subsequent nonzero residual gradient differs')
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    require(scaler.get_scale() == scale == 128, 'skipped/scaled update forbidden')
    state['counter'] += 1
    refresh, positions = ref.member_bank_refresh_rows(batch)
    clean = torch.cat(rows)
    state['bank'][torch.tensor(refresh, device=device)] = ref.member_bank_refresh_values(
        clean, clean, torch.tensor(positions, device=device), live_head=False)
    require(state['bank']._version == version + 1, 'last duplicate detached refresh differs')
    optimizer.zero_grad(set_to_none=True)
    integrity(context, state, ident)
    require(context['source_driver'].numerical_flags() == ident['numerical_flags'] and
            (device != 'cuda' or torch.cuda.max_memory_allocated() < 10_000_000_000), 'flags/whole-unit CUDA peak differs')
    row = {'step': step, 'batch': batch, 'schedule_sha256': ident['schedule_sha256'],
           'feature_rows_sha256': context['original'].fingerprint({v: state[k][batch]
                for v, k in zip(VIEWS, ('features', 'augmented'), strict=True)}),
           'mask_sha256': context['original'].fingerprint(masks[step - 1]),
           'clean_bank_sha256': context['original'].fingerprint(clean),
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients,
           'state_sha256': context['original'].fingerprint(payload(state, ident))}
    if device == 'cuda':
        torch.cuda.synchronize()
    row['seconds'] = time.perf_counter() - tick
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def calibration(context, state):
    import torch
    from torch.nn import functional as F
    with torch.no_grad(), torch.autocast(device_type=state['features'].device.type, enabled=False):
        raw = state['head'](state['features'][:64])
        unit = F.normalize(raw, dim=1)
        packed = context['packing_helper'].pack_int8_unit_embeddings(unit.cpu())
    return {'raw': raw.cpu(), 'unit': unit.cpu(), 'codes': packed.codes.cpu(),
            'inverse_norms': packed.inverse_norms.cpu(), 'wire': packed.to_bytes()}


def cpu_witnesses(context, ref, output, flags):
    import torch
    rng = torch.random.get_rng_state().clone()
    features = training_features(context)
    initial = initializer(context, ref, features['canonical'])
    digest = context['original'].fingerprint(initial)
    reconstructed = initializer(context, ref, features['canonical'], pca=initial['pca'])
    require(context['original'].fingerprint(reconstructed) == digest, 'independent NEW initializer geometry differs')
    del reconstructed
    witnesses, gradients, resumes = {}, {}, {}
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu', initial, features)
        ident = identity(context, state, flags)
        integrity(context, state, ident)
        witnesses[arm] = calibration(context, state)
        gradients[arm] = [update(context, ref, state, ident, 1)]
        with TemporaryDirectory(prefix='discard-cpu-', dir=output) as directory:
            checkpoint = Path(directory) / 'step1.pt'
            sha, state_digest = save(context, state, ident, checkpoint)
            gradients[arm].append(update(context, ref, state, ident, 2))
            final_digest = context['original'].fingerprint(payload(state, ident))
            final_witness = context['original'].fingerprint(calibration(context, state))
            del state
            gc.collect()
            state = restore(context, ref, checkpoint, sha, state_digest, ident, 1, features)
            row = update(context, ref, state, ident, 2)
            require(diagnostic(row) == diagnostic(gradients[arm][1]) and
                    context['original'].fingerprint(payload(state, ident)) == final_digest and
                    context['original'].fingerprint(calibration(context, state)) == final_witness,
                    'CPU full objective/gradient/resume differs')
            resumes[arm] = {'state_sha256': final_digest, 'raw_unit_packed_sha256': final_witness}
            del state
            gc.collect()
    require(context['original'].fingerprint(witnesses['control']) == context['original'].fingerprint(witnesses['candidate']),
            'initial matched arm raw/unit/packed parity differs')
    checkpoint = output / 'initializer.pt'
    saved = {'schema': SCHEMA, 'source': context['source'], 'partition': context['partition'], 'initial': initial,
             'cpu_rng': rng, 'numerical_flags': flags, 'counter': 0,
             'fixtures': {'calibration': witnesses, 'gradients': gradients, 'resumes': resumes}}
    saved_digest = context['original'].fingerprint(saved)
    with context['extract'].exclusive(checkpoint) as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())
    del saved, initial
    gc.collect()
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(context['original'].fingerprint(disk) == saved_digest, 'CPU complete initializer reload differs')
    initial = disk['initial']
    for arm in ARMS:
        state = fresh(context, ref, arm, SEEDS[0], 'cpu', initial, features)
        require(context['original'].fingerprint(calibration(context, state)) ==
                context['original'].fingerprint(witnesses[arm]), 'CPU strict raw/unit/packed initializer reload differs')
        del state
    require(torch.equal(rng, torch.random.get_rng_state()) and
            context['source_driver'].numerical_flags() == flags and not torch.cuda.is_initialized(),
            'CPU constructor RNG/flags/CUDA differs')
    bound_file(context['guards'], checkpoint, context['exporter'].sha(checkpoint))
    return {'completed_step': 0, 'checkpoint': {'path': str(checkpoint), 'sha256': context['guards'][str(checkpoint)]},
            'cuda_initialized': False, 'initial_arm_parity': True, 'gradient_witnesses': True,
            'cpu_resume_exact': True, 'training_only_fit': True, 'initial_state_sha256': digest,
            'cpu_state_sha256': saved_digest, 'cpu_witnesses_sha256': context['original'].fingerprint(disk['fixtures'])}


def gpu_run(context, ref, output, flags):
    import torch
    args = context['args']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized(),
            'one visible uninitialized CUDA device required')
    proof = context['terminals']['cpu:control']
    bound_file({}, proof['checkpoint']['path'], proof['checkpoint']['sha256'])
    disk = torch.load(proof['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(context['original'].fingerprint(disk) == proof['cpu_state_sha256'] and disk['schema'] == SCHEMA and
            disk['source'] == context['source'] and disk['partition'] == context['partition'] and
            disk['counter'] == 0 and disk['numerical_flags'] == flags and
            context['original'].fingerprint(disk['initial']) == proof['initial_state_sha256'], 'accepted NEW initializer differs')
    initial = disk['initial']
    features = training_features(context)
    torch.cuda.manual_seed_all(SEEDS[0])
    state = fresh(context, ref, args.arm, args.seed, 'cuda', initial, features)
    del initial, disk
    ident = identity(context, state, flags)
    integrity(context, state, ident)
    start_digest = context['original'].fingerprint(payload(state, ident))
    features = {'canonical': state['features'], 'augmented': state['augmented']}
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    rows, resumed = [], []
    total = 17 if args.phase == 'mechanics' else 1000
    with TemporaryDirectory(prefix='discard-mechanics-', dir=output) as directory:
        temporary = Path(directory)
        tick = time.perf_counter()
        for step in range(1, total + 1):
            row = update(context, ref, state, ident, step)
            if args.phase == 'train' and args.seed == SEEDS[0] and step <= 17:
                require(diagnostic(row) == diagnostic(context['terminals']['mechanics:' + args.arm]['steps'][step - 1]),
                        'fresh first17 mechanics replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                sha8, digest8 = save(context, state, ident, temporary / 'step8.pt')
        training_seconds = time.perf_counter() - tick
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else output / 'resume.pt'
        sha, digest = save(context, state, ident, checkpoint)
        witness = context['original'].fingerprint(calibration(context, state))
        del state
        gc.collect(); torch.cuda.empty_cache()
        if args.phase == 'mechanics':
            state = restore(context, ref, temporary / 'step8.pt', sha8, digest8, ident, 8, features)
            resumed = [update(context, ref, state, ident, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    context['original'].fingerprint(payload(state, ident)) == digest, '17 versus independent8+9 differs')
            del state
            gc.collect(); torch.cuda.empty_cache()
        state = restore(context, ref, checkpoint, sha, digest, ident, total, features)
        require(context['original'].fingerprint(calibration(context, state)) == witness and
                context['original'].fingerprint(features) == ident['feature_state_sha256'], 'strict final raw/unit/packed/TRAIN reload differs')
        require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)), 'CUDA RNG changed')
        del state, features
        gc.collect(); torch.cuda.empty_cache()
    if args.phase == 'train':
        bound_file(context['guards'], checkpoint, sha)
    return {'completed_step': total, 'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': sha},
            'initial_state_sha256': start_digest, 'terminal_state_sha256': digest, 'identity': ident,
            'steps': rows, 'resumed_steps': resumed, 'replay_exact': args.phase == 'mechanics',
            'training_state_discarded': args.phase == 'mechanics', 'training_wall_seconds': training_seconds,
            'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def run(args):
    started = time.perf_counter()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
            os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'explicit CUDA-hidden CPU or deterministic CUDA launch required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original systemd invocation required')
    require(sys.argv == [str(Path(__file__).absolute()), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--phase', args.phase, '--arm', args.arm, '--seed', str(args.seed), '--output', str(args.output)],
            'fixed canonical CLI order required')
    context = authority(args)
    source = context['source_driver']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    context['genuine']['reference'].admit_cgroup(before, unit)
    prior = context['source_cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and context['exporter'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'original qualified interpreter differs')
    for name in ('pca', 'packing'):
        helper = context['launch']['helpers'][name]
        context[name + '_helper'] = load_helper('_genuine_training_' + name, helper['path'],
                                               HELPER_SHAS[name], context['guards'])
    import torch
    require(not torch.cuda.is_initialized(), 'admission must precede CUDA')
    flags = context['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original numerical flags differ')
    torch.random.default_generator.manual_seed(SEEDS[0])
    cpu_rng = torch.random.get_rng_state().clone()
    cpu_origins = source.imported_origins(context['extract'], context['packages'])
    require(all(context['source_cpu']['origins']['files'].get(p) == h for p, h in cpu_origins['files'].items()),
            'actual CPU imports differ from original qualified origins')
    ref = context['original'].reference_math(context['math_context'])
    args.output.mkdir()
    result = cpu_witnesses(context, ref, args.output, flags) if args.phase == 'cpu' else gpu_run(context, ref, args.output, flags)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'constructor/update/reload RNG/flags differ')
    origins = source.imported_origins(context['extract'], context['packages'])
    for path, digest in origins['files'].items():
        bound_file(context['guards'], path, digest)
        require(context['genuine']['prior']['guards'].setdefault(path, digest) == digest, 'loaded origin changed')
    context['exporter'].rehash(context['genuine'])
    for path, digest in context['guards'].items():
        bound_file({}, path, digest)  # fresh uncached source/cache/output/closure exit pass
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'], 'exit trainer closure differs')
    after = source.cgroup_memory()
    context['genuine']['reference'].admit_cgroup(after, unit)
    wall, rss = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and 0 < rss <= 8 * 1024**2 and
            (args.phase != 'cpu' or not torch.cuda.is_initialized()), 'whole-unit time/RSS/CUDA caps differ')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed,
               'pass': True, 'quality_read': False, 'strict_reload_exact': True, 'optimizer_members': 5,
               'training_qualified': args.phase == 'train', 'trained_state_reused': False,
               'encoder_updates': 0, 'head_scalars': 188544, 'trainable_scalars': 317568,
               'source': context['source'], 'partition_sha256': PARTITION_SHA,
               'launch': context['launch'], 'execution_sha256': args.execution_sha256,
               'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
               'authority_sha256': args.authority_sha256, 'code': context['code'],
               'resource_policy': policy(args.phase), 'numerical_flags': flags, 'wall_seconds': wall,
               'process_peak_rss_kib': rss, 'cgroup_before': before, 'cgroup_after': after,
               'origins': origins, 'input_guards': context['guards'], 'exit_rehash_pass': True,
               'terminal_cgroups': context['terminal_cgroups'], 'both_locks_held_in_parent_authority': True,
               'terminal_exit_and_both_locks_require_parent_receipt': True, **result,
               'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                              'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                              'invocation_id': os.environ['INVOCATION_ID'],
                              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}}
    context['genuine']['reference'].write_json(context['extract'], args.output / 'receipt.json', receipt)
    require(time.perf_counter() - started < policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
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
        raise SystemExit('Genuine view training rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
