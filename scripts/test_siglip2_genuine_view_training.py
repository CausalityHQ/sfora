#!/usr/bin/env python3
"""One stdlib genuine-view trainer falsifier; native/resource/quality UNRUN.

No Torch/NumPy/native/image/cache consumption or quality execution.
Synthetic FP32 NPY data, scalar initializer and bank doubles qualify only checks.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import struct
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError, TypeError):
        return
    raise AssertionError('accepted ' + message)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainerArray:
    """Selection/bank protocol double; no native arithmetic qualification."""
    device = SimpleNamespace(type='cpu')

    def __init__(self, rows):
        self.rows = rows

    def clone(self):
        return TrainerArray(copy.deepcopy(self.rows))

    def detach(self):
        return self

    def float(self):
        return self

    def __getitem__(self, index):
        values = index.rows if isinstance(index, TrainerArray) else index
        if values and type(values[0]) is bool:
            values = [i for i, yes in enumerate(values) if yes]
        return TrainerArray([self.rows[i] for i in values])

    def __setitem__(self, mask, values):
        for i, row in zip([i for i, yes in enumerate(mask.rows) if yes], values.rows, strict=True):
            self.rows[i] = row

    def any(self):
        return any(self.rows)


class MetadataTensor:
    def __init__(self, shape, dtype='torch.float32', value=None):
        self.shape, self.dtype, self.value = shape, dtype, value

    def __float__(self):
        return float(self.value)


def trainer_launch(d, phase='train', arm='candidate', seed=179061):
    file = {'path': '/synthetic/receipt.json', 'sha256': 'e' * 64}
    terminal = {'receipt': file, 'log': file, 'unit': 'fixture', 'invocation_id': '1' * 32,
                'service_seconds': 10, 'native_peak_rss_kib': 10, 'both_locks_held': True}
    return {'schema': d.AUTHORITY_SCHEMA, 'execution_sha256': 'a' * 64, 'phase': phase, 'arm': arm, 'seed': seed,
            'export_reference': {'root': str(d.EXPORT_ROOT), 'execution_sha256': d.EXPORT_EXECUTION_SHA,
                'code': copy.deepcopy(d.EXPORT_CODE),
                'authority': {'path': str(d.EXPORT_ROOT / 'authority.json'), 'sha256': d.EXPORT_AUTHORITY_SHA}},
            'selected_export': {**{k: v for k, v in terminal.items() if k != 'receipt'}, 'proof': file},
            'cached_reference': {'root': '/synthetic/cached', 'execution_sha256': d.CACHED_EXECUTION_SHA},
            'training_reference': {'root': '/synthetic/math', 'execution_sha256': d.REFERENCE_EXECUTION_SHA},
            'helpers': {n: {'path': '/synthetic/' + n + '.py', 'sha256': h} for n, h in d.HELPER_SHAS.items()},
            'partition': {'path': '/synthetic/partition.json', 'sha256': d.PARTITION_SHA},
            'selected_cpu': None if phase == 'cpu' else copy.deepcopy(terminal),
            'selected_mechanics': {a: copy.deepcopy(terminal) for a in d.ARMS} if phase == 'train' else None,
            'recipe': copy.deepcopy(d.RECIPE), 'resource_policy': d.policy(phase), 'both_locks_held': True}


def trainer_views(d):
    return {'caches': {v: {'path': '/synthetic/' + v + '.npy', 'sha256': str(i + 1) * 64,
                         'shape': [d.ROWS, d.WIDTH], 'dtype': 'float32', 'normalized': True,
                         'raw_pooled_cache': False} for i, v in enumerate(d.VIEWS)},
            'ordered_input_sha256': 'b' * 64, 'ordered_view_sha256': dict.fromkeys(d.VIEWS, 'c' * 64)}


def trainer_resume(d):
    groups = [{'lr': .0001, 'params': [0, 1, 2, 3]}, {'lr': .0001, 'params': [4]}]
    views = trainer_views(d)
    ident = {'seed': d.SEEDS[0], 'source': copy.deepcopy(views), 'parameter_names': d.PARAMETERS.copy(),
             'numerical_flags': {}, 'optimizer_defaults': {}, 'optimizer_serial_groups': groups,
             'positive_shape': (d.ROWS, 8), 'device': 'cuda'}
    initial = {'head': {n: MetadataTensor(shape) for n, shape in zip(
        ('primary.weight', 'primary.bias', 'down.weight', 'up.weight', 'center', 'preactivation_std'),
        (*d.SHAPES[:4], (d.WIDTH,), ()), strict=True)},
        'classifier': MetadataTensor(d.SHAPES[4]), 'bank': MetadataTensor((d.ROWS, d.DIM)),
        'target': MetadataTensor((d.ROWS,), 'torch.int64'), 'positive': MetadataTensor(ident['positive_shape'], 'torch.int64'),
        'original_rows': MetadataTensor((d.ROWS,), 'torch.int64'), 'views': views,
        'pca': {'mean': MetadataTensor((d.WIDTH,)), 'components': MetadataTensor((d.DIM, d.WIDTH))},
        'schedules': {str(s): MetadataTensor((1000, 64), 'torch.int64') for s in d.SEEDS},
        'masks': {str(s): MetadataTensor((1000, 64), 'torch.bool') for s in d.SEEDS}}
    saved = {'schema': d.SCHEMA, 'identity': ident, **copy.deepcopy(initial), 'initializer': initial,
             'partition': {}, 'counter': 8, 'seed': ident['seed'], 'source': ident['source'], 'numerical_flags': {},
             'optimizer_defaults': {}, 'optimizer': {'param_groups': copy.deepcopy(groups), 'state': {
                 i: {'step': MetadataTensor((), value=8), 'exp_avg': MetadataTensor(shape), 'exp_avg_sq': MetadataTensor(shape)}
                 for i, shape in enumerate(d.SHAPES)}},
             'scaler': {'scale': 128, 'growth_factor': 2., 'backoff_factor': .5, 'growth_interval': 2000, '_growth_tracker': 8},
             'cpu_rng': MetadataTensor((5056,), 'torch.uint8'), 'cuda_rng': [MetadataTensor((16,), 'torch.uint8')]}
    return saved, ident


def trainer_metadata_checks(d):
    for phase, arm, seed in (('cpu', 'control', 179061), ('mechanics', 'candidate', 179061),
                            ('train', 'control', 179061), ('train', 'candidate', 179069)):
        launch = trainer_launch(d, phase, arm, seed)
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256='a' * 64)
        d.check_launch(launch, args)
        for mutation in (
            lambda m: m.update(extra=True), lambda m: m.update(both_locks_held=False),
            lambda m: m['recipe'].update(rank_weight=.8), lambda m: m['recipe'].update(candidate='gelu-exact'),
            lambda m: m['recipe'].update(initialization_seed=179034),
            lambda m: m['recipe'].update(mask_seed_offset=1), lambda m: m['resource_policy'].update(seconds=301),
            lambda m: m['export_reference'].update(execution_sha256=d.FAILED_EXPORT_EXECUTION_SHA),
            lambda m: m['export_reference']['authority'].update(sha256=d.FAILED_EXPORT_AUTHORITY_SHA),
            lambda m: m['export_reference'].update(root='/different/export'),
            lambda m: m['export_reference']['code'].update(extra='b' * 64),
            lambda m: m['helpers']['pca'].update(sha256='0' * 64),
            lambda m: m['training_reference'].update(execution_sha256='0' * 64),
            lambda m: m['partition'].update(sha256='0' * 64),
            lambda m: m['selected_export'].update(both_locks_held=False),
            lambda m: m.update(selected_cpu={} if phase == 'cpu' else None),
        ):
            changed = copy.deepcopy(launch); mutation(changed)
            rejects(lambda: d.check_launch(changed, args), 'trainer launch mutation')
    launch = trainer_launch(d); launch['selected_mechanics'].pop('control')
    rejects(lambda: d.check_launch(launch, args), 'missing control mechanics')
    saved, ident = trainer_resume(d)
    d.check_payload(saved, ident, 8)
    for key in d.PAYLOAD_KEYS:
        changed = copy.deepcopy(saved); del changed[key]
        rejects(lambda: d.check_payload(changed, ident, 8), 'missing ' + key)
    for mutation in (
        lambda m: m['masks'].pop(str(d.SEEDS[1])),
        lambda m: m['masks'].update({str(d.SEEDS[0]): MetadataTensor((1000, 64))}),
        lambda m: m['initializer'].pop('bank'), lambda m: m['initializer']['head'].pop('center'),
        lambda m: m['initializer']['views']['caches']['augmented'].update(path='/synthetic/canonical.npy'),
        lambda m: m['views']['ordered_view_sha256'].update(augmented='0' * 64),
        lambda m: m.update(classifier=MetadataTensor((2004, 128))),
        lambda m: m.update(original_rows=MetadataTensor((13283,), 'torch.int64')),
        lambda m: m['optimizer']['state'].pop(4),
        lambda m: m['optimizer']['state'][0].update(step=MetadataTensor((), value=7)),
        lambda m: m['scaler'].update(_growth_tracker=7), lambda m: m.update(cuda_rng=[]),
    ):
        changed = copy.deepcopy(saved); mutation(changed)
        rejects(lambda: d.check_payload(changed, ident, 8), 'complete trainer state')
    cpu = copy.deepcopy(saved); cpu_ident = copy.deepcopy(ident); cpu_ident['device'] = 'cpu'
    cpu.update(identity=cpu_ident, cuda_rng=[]); d.check_payload(cpu, cpu_ident, 8)
    cpu['counter'] = 0; cpu['optimizer']['state'] = {}; cpu['scaler']['_growth_tracker'] = 0
    d.check_payload(cpu, cpu_ident, 0)


def trainer_selection_checks(d, scripts):
    canonical = TrainerArray([[1., 0.], [0., 1.], [-1., 0.]])
    augmented = TrainerArray([[.6, .8], [.8, .6], [0., -1.]])
    anchors, mask = TrainerArray([0, 1, 0, 2]), TrainerArray([True, False, False, True])
    control = d.view_inputs(canonical, augmented, anchors, mask, 'control')
    candidate = d.view_inputs(canonical, augmented, anchors, mask, 'candidate')
    check(control.rows == [[1., 0.], [0., 1.], [1., 0.], [-1., 0.]] and
          candidate.rows == [[.6, .8], [0., 1.], [1., 0.], [0., -1.]], 'same-row masked genuine input')
    check(d.view_inputs(canonical, augmented, anchors, TrainerArray([False] * 4), 'candidate').rows == control.rows,
          'all false mask identity')
    check(d.view_inputs(canonical, augmented, anchors, TrainerArray([True] * 4), 'candidate').rows ==
          [augmented.rows[i] for i in anchors.rows], 'all true mask, including singleton')
    check(canonical.rows == [[1., 0.], [0., 1.], [-1., 0.]], 'view selection mutated cache')
    rejects(lambda: d.view_inputs(canonical, augmented, anchors, mask, 'mixed'), 'invalid arm')
    anchors_all = [list(range(64)) for _ in range(1000)]
    masks_all = [[True, False] * 32 for _ in range(1000)]
    d.check_masks(anchors_all, masks_all)
    for changed in (masks_all[:-1], [[0] * 64] * 1000, [[False] * 63] * 1000):
        rejects(lambda: d.check_masks(anchors_all, changed), 'mask inventory/type')
    fake_torch = SimpleNamespace(no_grad=nullcontext, autocast=lambda **kw: nullcontext())
    events = []
    def head(value):
        events.append(copy.deepcopy(value.rows))
        return value.clone()
    state = {'head': head, 'features': canonical, 'augmented': augmented, 'arm': 'candidate'}
    with patch.dict(sys.modules, torch=fake_torch):
        clean = d.clean_bank_descriptors(state, anchors)
    check(clean.rows == control.rows and events == [control.rows], 'canonical PRE-update detached bank path')
    # Exercise the actual unchanged original duplicate and detached refresh helpers.
    path = scripts / 'train_sop_siglip2_compact.py'
    tree = ast.parse(path.read_text())
    names = ('member_bank_refresh_rows', 'member_bank_refresh_values')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    import __future__
    namespace = {'F': SimpleNamespace(normalize=lambda value, dim: value)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    rows, positions = namespace['member_bank_refresh_rows'](tuple(anchors.rows))
    selected = namespace['member_bank_refresh_values'](augmented, clean, TrainerArray(list(positions)), live_head=False)
    check(rows == (0, 1, 2) and positions == (2, 1, 3) and selected.rows == [control.rows[i] for i in positions],
          'original last-duplicate canonical bank refresh')


def trainer_file_checks(d, root):
    cache = root / 'canonical.npy'
    def npy(shape, first=1.):
        header = repr({'descr': '<f4', 'fortran_order': False, 'shape': shape}).encode() + b'\n'
        with cache.open('wb') as stream:
            stream.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header)
            row = struct.pack('<f', first) + b'\0' * ((d.WIDTH - 1) * 4)
            for _ in range(d.ROWS):
                stream.write(row)
        return {'path': str(cache), 'sha256': d.sha(cache) if hasattr(d, 'sha') else hashlib.sha256(cache.read_bytes()).hexdigest()}
    fact = npy((d.ROWS, d.WIDTH))
    d.cache_facts(fact, {})
    for shape, value in (((13283, d.WIDTH), 1.), ((d.ROWS, d.WIDTH), float('nan')), ((d.ROWS, d.WIDTH), 0.)):
        wrong = npy(shape, value)
        rejects(lambda: d.cache_facts(wrong, {}), 'wrong/old/nonunit cache')
    fact = npy((d.ROWS, d.WIDTH))
    cache.write_bytes(cache.read_bytes() + b'extra')
    rejects(lambda: d.cache_facts(fact, {}), 'changed bytes')
    guards = {}
    for name in d.FILES:
        (root / name).write_bytes(Path(d.__file__).with_name(name).read_bytes())
    code = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in d.FILES}
    digest = write(root / 'execution.json', code)
    check(d.closure(root, digest, d.FILES, guards) == code, 'trainer exact closure')
    digest = write(root / 'execution.json', {**code, 'extra.py': '0' * 64})
    rejects(lambda: d.closure(root, digest, d.FILES, {}), 'extra source')
    helper = root / 'helper.py'; helper.write_text('calls = 0\n')
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    name = '_genuine_lifetime_fixture'
    module = d.load_helper(name, helper, digest, {})
    check(module.__file__ == module.__spec__.origin == str(helper), 'actual helper origin')
    rejects(lambda: d.load_helper(name, helper, digest, {}), 'second helper load')
    helper.write_text('calls = 1\n')
    rejects(lambda: d.load_helper('_genuine_changed_fixture', helper, digest, {}), 'changed helper SHA')
    del sys.modules[name]


def trainer_export_checks(d, scripts):
    path = scripts / 'export_siglip2_genuine_views.py'
    spec = importlib.util.spec_from_file_location('_genuine_metadata_exporter', path)
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
    evidence = scripts.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    fit = json.loads((evidence / 'late-dense-v1/native256-fit-manifest-v1.json').read_text())
    partition = json.loads((evidence / 'identity-mix-v1/partition.json').read_text())
    manifest = exporter.selected_manifest(partition, fit)
    manifest['resolved_paths'] = [str(Path(fit['dataset_root']) / r['relative_path']) for r in manifest['rows']]
    rgb = {'mode': 'RGB', 'size': [256, 256], 'sha256': 'b' * 64}
    mapping = {v: [{'view': v, 'ordinal': i, 'original_row': original, 'train_row': row['train_row'],
                    'target': manifest['targets'][i], 'path': manifest['resolved_paths'][i],
                    'relative_path': row['relative_path'], 'image_sha256': row['image_sha256'],
                    'original_rgb': rgb, 'rgb': rgb, 'rng_seed': None if v == 'canonical' else 179081 + row['train_row'],
                    'pixels': {'shape': [3, 256, 256], 'dtype': 'torch.float32', 'sha256': 'c' * 64}}
                   for i, (original, row) in enumerate(zip(manifest['original_rows'], manifest['rows']))] for v in d.VIEWS}
    launch = trainer_launch(d)
    original = {'checkpoint': {'path': '/synthetic/source.pt', 'sha256': 'f' * 64}, 'numerical_flags': {}}
    genuine = {'selected': manifest, 'prior': {'proof': original}}
    wrapper = SimpleNamespace(SCHEMA=exporter.SCHEMA, binding=lambda ctx: {'synthetic': True},
                              validate_views=exporter.validate_views, object_sha=exporter.object_sha)
    context = {'launch': launch, 'genuine': genuine, 'exporter': wrapper}
    record = {'schema': exporter.SCHEMA, 'phase': 'export', 'pass': True, 'exported': True,
              'binding': wrapper.binding(genuine), 'ordered_input': manifest, 'historical_cache_reused': False,
              'held_pixels_decoded': 0, 'original_source_witness_pixels_decoded': False, 'quality_read': False,
              **dict.fromkeys(('training_qualified', 'quality_qualified', 'initializer_qualified', 'gradients_created',
                              'optimizer_created', 'cuda_peak_reset'), False),
              'updates': 0, 'strict_independent_reload_exact': True, 'constructor_and_view_rng_preserved': True,
              'exit_rehash_pass': True, 'source_checkpoint': original['checkpoint'], 'cpu_numerical_flags': {},
              'counters': {'images_per_view': d.ROWS, 'classes': d.CLASSES,
                  'batch_sizes_per_view': [32] * (d.ROWS // 32) + [d.ROWS % 32], 'views': list(d.VIEWS), 'optimizer_updates': 0},
              'view_mapping': mapping, 'ordered_view_sha256': {v: exporter.object_sha(mapping[v]) for v in d.VIEWS},
              'caches': trainer_views(d)['caches']}
    d.check_export_record(context, record)
    for mutation in (
        lambda m: m.update(pass_=False, held_pixels_decoded=1),
        lambda m: m.update(strict_independent_reload_exact=False), lambda m: m.update(cuda_peak_reset=True),
        lambda m: m['caches']['augmented'].update(path='/synthetic/canonical.npy'),
        lambda m: m['caches']['canonical'].update(sha256=d.FIT_SHA),
        lambda m: m['caches']['canonical'].update(shape=[13283, d.WIDTH]),
        lambda m: m['ordered_view_sha256'].update(augmented='0' * 64),
        lambda m: m['view_mapping']['augmented'][0].update(original_row=manifest['original_rows'][1]),
    ):
        changed = copy.deepcopy(record); mutation(changed)
        rejects(lambda: d.check_export_record(context, changed), 'preparation cache/mapping/role')


def trainer_source_checks(d, path):
    tree = ast.parse(path.read_bytes())
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    clean, update = functions['clean_bank_descriptors'], functions['update']
    check('torch.no_grad()' in clean and "state['features'][index]" in clean and '.detach()' in clean and
          update.index('clean_bank_descriptors(') < update.index('scaler.step(') and
          'member_bank_refresh_values(clean, clean,' in update and 'member_bank_refresh_rows(batch)' in update,
          'canonical pre-update last-duplicate bank order')
    schedule = functions['schedule_and_masks']
    check("PCG64(seed + RECIPE['mask_seed_offset'])" in schedule and 'mask_rng.random((1000, 64)) < 0.5' in schedule and
          'rng.permutation(list(range(CLASSES)))' in schedule and 'rng.choice(members[int(c)])' in schedule,
          'original anchor arithmetic and independent 50% PCG64 masks')
    check("initializer(context, ref, features['canonical'])" in functions['cpu_witnesses'] and
          "fit_centered_pca(normalized, dimensions=DIM)" in functions['initializer'] and
          "context['pca_helper']" in functions['initializer'] and 'load_helper(' not in functions['initializer'],
          'fresh canonical-only PCA and once-loaded initializer helper')
    check("np.load(fact['path'], allow_pickle=False, mmap_mode='r')" in functions['training_features'] and
          'cache.copy()' in functions['training_features'] and 'normalize_nonzero(features)' in functions['training_features'],
          'shared fresh cache normalization arithmetic')
    check(all(text in functions['restore'] for text in ('weights_only=True', 'load_state_dict', 'set_rng_state',
                                                       'set_rng_state_all', 'check_payload')), 'complete strict resume')
    check('17 versus independent8+9 differs' in functions['gpu_run'] and
          'fresh first17 mechanics replay differs' in functions['gpu_run'], 'mechanics and fresh TRAIN gates')
    check("context['exporter'].rehash(context['genuine'])" in functions['run'] and
          'bound_file({}, path, digest)' in functions['run'], 'uncached complete exit')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            check(not ast.unparse(node.func).endswith(('reset_peak_memory_stats', 'fresh_source', 'load_initializers')),
                  'unexpected native encoder/old initializer/peak reset')
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            check(not any('__globals__' in ast.unparse(t) for t in targets), 'helper globals patch')
    for name in ('torch', 'numpy', 'PIL', 'transformers', 'torchvision', 'sfora'):
        check(not any(name in ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))),
              'native import before admission')


def trainer_terminal_checks(d, scripts, root):
    launch = trainer_launch(d)
    record = {'schema': d.SCHEMA, 'phase': 'mechanics', 'arm': 'candidate', 'seed': 179061,
              'pass': True, 'quality_read': False, 'exit_rehash_pass': True, 'strict_reload_exact': True,
              'optimizer_members': 5, 'launch': trainer_launch(d, 'mechanics'),
              'resource_policy': d.policy('mechanics'), 'trained_state_reused': False,
              'completed_step': 17, 'checkpoint': None, 'training_state_discarded': True,
              'replay_exact': True, 'peak_cuda_allocated_bytes': 1000000}
    record['steps'] = [{'step': i, 'batch': list(range(64)), 'ce': 1., 'rank': .1, 'loss': 1.8,
                        'scale': 128., 'preclip_norm': 1., 'seconds': .01,
                        **{k: 'b' * 64 for k in ('schedule_sha256', 'feature_rows_sha256', 'mask_sha256',
                                                'clean_bank_sha256', 'state_sha256')},
                        'gradient_norms': dict.fromkeys(d.PARAMETERS, 0.)} for i in range(1, 18)]
    record['resumed_steps'] = copy.deepcopy(record['steps'][8:])
    d.check_terminal_record(record, launch, 'mechanics', 'candidate')
    for mutation in (lambda m: m.update(optimizer_members=208), lambda m: m.update(completed_step=16),
                     lambda m: m['steps'][0]['batch'].__setitem__(0, 6355),
                     lambda m: m['resumed_steps'][0].update(clean_bank_sha256='c' * 64),
                     lambda m: m['steps'][0].update(loss=1.08),
                     lambda m: m.update(checkpoint={'path': 'partial'}),
                     lambda m: m.update(peak_cuda_allocated_bytes=10_000_000_000)):
        changed = copy.deepcopy(record); mutation(changed)
        rejects(lambda: d.check_terminal_record(changed, launch, 'mechanics', 'candidate'), 'mechanics complete replay')
    modules = []
    for name in ('train_siglip2_substrate_adaptation', 'export_siglip2_substrate_fit'):
        spec = importlib.util.spec_from_file_location('_genuine_terminal_' + name, scripts / (name + '.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); modules.append(module)
    original, exporter = modules
    check(hashlib.sha256((scripts / 'train_siglip2_substrate_adaptation.py').read_bytes()).hexdigest() == d.TRAINING_SHA,
          'unchanged original terminal/math helper bytes')
    admission = original.FlatAdmission(); admission.init = exporter
    unit, invocation = 'genuine-trainer-fixture', '4' * 32
    memory = {'path': '/sys/fs/cgroup/' + unit + '.service',
              'values': {'memory.max': str(8 * 1024**3), 'memory.current': '100', 'memory.peak': '200',
                         'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                         'memory.events': 'max 0\noom 0\noom_kill 0'}}
    log = root / 'trainer.log'
    text = f'Running as unit: {unit}.service; invocation ID: {invocation}\n' + '\n'.join([
        '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
        '\tSwaps: 0', 'Memory swap peak: 0B', 'Service runtime: 11s', '\tMaximum resident set size (kbytes): 20',
        'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation})]) + '\n'
    def terminal(raw):
        log.write_text(raw)
        return {'receipt': {'path': str(root / 'receipt.json'), 'sha256': 'e' * 64},
                'log': {'path': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()},
                'unit': unit, 'invocation_id': invocation, 'service_seconds': 11,
                'native_peak_rss_kib': 20, 'both_locks_held': True}
    proof = {'invocation': {'invocation_id': invocation, 'optimize': 0}, 'wall_seconds': 10,
             'process_peak_rss_kib': 10, 'cgroup_before': memory, 'cgroup_after': memory}
    check(admission.admit_terminal(proof, terminal(text), 120, {})['path'] == memory['path'], 'original whole unit admission')
    for mutation in ('lock', 'cap', 'rss', 'swap', 'events', 'invocation', 'exit', 'footer', 'runtime'):
        changed, descriptor = copy.deepcopy(proof), terminal(text)
        if mutation == 'lock': descriptor['both_locks_held'] = False
        elif mutation == 'cap': descriptor['service_seconds'] = 121
        elif mutation == 'rss': descriptor['native_peak_rss_kib'] = 8 * 1024**2 + 1
        elif mutation == 'swap': changed['cgroup_after']['values']['memory.swap.peak'] = '1'
        elif mutation == 'events': changed['cgroup_after']['values']['memory.events'] = 'max 1\noom 0\noom_kill 0'
        elif mutation == 'invocation': changed['invocation']['invocation_id'] = '5' * 32
        elif mutation == 'exit': descriptor = terminal(text.replace('\tExit status: 0', '\tExit status: 1'))
        elif mutation == 'footer': descriptor = terminal(text + 'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation}) + '\n')
        else: descriptor = terminal(text.replace('Service runtime: 11s', 'Service runtime: 11.001s'))
        # Each independent unit has an independent original admission cache.
        fresh = original.FlatAdmission(); fresh.init = exporter
        rejects(lambda: fresh.admit_terminal(changed, descriptor, 120, {}), 'complete terminal ' + mutation)


def trainer_initializer_lifetime(d, root):
    """Run the real admission/helper/initializer boundary with scalar doubles."""
    pca_path = root / 'pca.py'
    pca_path.write_text('''fit_calls = 0
transforms = []
class CenteredPcaTransform:
    def __init__(self, mean, components):
        self.mean, self.components = mean, components
        transforms.append(self)
    def apply(self, value): return value.clone()
def fit_centered_pca(value, dimensions):
    global fit_calls
    fit_calls += 1
    return CenteredPcaTransform(value.mean(0), value.clone())
''')
    packing_path = root / 'packing.py'; packing_path.write_text('fixture = True\n')
    class Scalar:
        shape, device = (d.ROWS, d.WIDTH), SimpleNamespace(type='cpu')
        def __init__(self, value=1): self.value = value
        def clone(self): return Scalar(self.value)
        def __matmul__(self, other): return Scalar(self.value * other.value)
        def __neg__(self): return Scalar(-self.value)
        def __sub__(self, other): return Scalar(self.value - other.value)
        def __add__(self, other): return Scalar(self.value + other.value)
        def __getitem__(self, index): return self
        def __setitem__(self, index, value): self.value = value.value
        def __gt__(self, other): return Scalar(self.value > other)
        def __float__(self): return float(self.value)
        def all(self): return self
        def item(self): return self.value
        def mean(self, dimension): return self.clone()
        def std(self, unbiased): return Scalar()
        def norm(self, dim): return Scalar()
        def detach(self): return self
        def numpy(self): return [0]
        def copy_(self, other): self.value = other.value
        def div_(self, other): self.value /= other.value
    heads = []
    def factory(arm, tensors):
        check(arm == 'control' and float(tensors['primary.bias']) == -1, 'fresh native affine factory')
        head = SimpleNamespace(state_dict=lambda: tensors)
        head.down = lambda value: value.clone(); head.down.weight = tensors['down.weight']
        head.center, head.preactivation_std = tensors['center'], tensors['preactivation_std']
        heads.append(head)
        # SimpleNamespace cannot define __call__; the tiny wrapper preserves the factory interface.
        class Head:
            def __call__(self, value): return value.clone()
            def __getattr__(self, name): return getattr(head, name)
        return Head()
    nn = SimpleNamespace(init=SimpleNamespace(kaiming_uniform_=lambda *a, **kw: None),
                         functional=SimpleNamespace(normalize=lambda value, dim: value.clone()))
    torch = SimpleNamespace(nn=nn, zeros=lambda *a: Scalar(0), ones=lambda *a: Scalar(),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed), no_grad=nullcontext,
        tensor=lambda *a, **kw: Scalar(), from_numpy=lambda value: Scalar(), int64='int64',
        isfinite=lambda value: Scalar(True), cuda=SimpleNamespace(is_initialized=lambda: False),
        set_num_threads=lambda n: None, get_num_interop_threads=lambda: 1,
        random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None), get_rng_state=lambda: Scalar()))
    flags = {'threads': 1, 'interop_threads': 1}
    context = {'source_driver': SimpleNamespace(cgroup_memory=lambda: {'path': '/fixture.service'},
        numerical_flags=lambda: flags, imported_origins=lambda *a: {'files': {}}),
        'genuine': {'reference': SimpleNamespace(admit_cgroup=lambda *a: None)},
        'source_cpu': {'invocation': {'python': str(Path(sys.executable).resolve()), 'python_sha256': 'e' * 64,
            'python_version': sys.version}, 'numerical_flags': flags, 'origins': {'files': {}}},
        'exporter': SimpleNamespace(sha=lambda path: 'e' * 64),
        'launch': {'helpers': {n: {'path': str(p), 'sha256': d.HELPER_SHAS[n]}
            for n, p in (('pca', pca_path), ('packing', packing_path))}}, 'guards': {}, 'extract': None, 'packages': {},
        'original': SimpleNamespace(reference_math=lambda ctx: SimpleNamespace(member_bank_positive_ordinals=lambda *a, **kw: Scalar())),
        'math_context': {}, 'cached': SimpleNamespace(head_from=factory), 'target': [0], 'views': {},
        'partition': {'panels': {'train': {'original_rows': [0]}}}}
    args = SimpleNamespace(phase='cpu', arm='control', seed=d.SEEDS[0], output=root / 'output',
                          execution_sha256='a' * 64, authority=root / 'authority', authority_sha256='b' * 64)
    argv = [str(Path(d.__file__).absolute()), '--execution-sha256', args.execution_sha256, '--authority', str(args.authority),
            '--authority-sha256', args.authority_sha256, '--phase', args.phase, '--arm', args.arm,
            '--seed', str(args.seed), '--output', str(args.output)]
    real_import, real_load = __import__, d.load_helper
    loaded, results = [], []
    def synthetic_load(name, path, digest, guards):
        check(digest == d.HELPER_SHAS[name.removeprefix('_genuine_training_')], 'run independently pinned helpers')
        loaded.append(name)
        return real_load(name, path, hashlib.sha256(Path(path).read_bytes()).hexdigest(), guards)
    def synthetic_import(name, *a, **kw):
        if name in ('torch', 'torch.nn'):
            check('_genuine_training_pca' in sys.modules and '_genuine_training_packing' in sys.modules, 'helpers before initializer')
            return torch if name == 'torch' else nn
        return real_import(name, *a, **kw)
    class BoundaryComplete(Exception): pass
    def witnesses(ctx, ref, output, numeric):
        results.append(d.initializer(ctx, ref, Scalar()))
        results.append(d.initializer(ctx, ref, Scalar(), pca=results[0]['pca']))
        raise BoundaryComplete
    with patch.object(d, 'authority', return_value=context), patch.object(d, 'load_helper', synthetic_load), \
         patch.object(d, 'cpu_witnesses', witnesses), patch.object(d, 'schedule_and_masks', return_value=(None, None)), \
         patch.object(sys, 'argv', argv), patch.dict(d.os.environ, CUDA_VISIBLE_DEVICES='', INVOCATION_ID='c' * 32), \
         patch('builtins.__import__', synthetic_import):
        try:
            d.run(args)
        except BoundaryComplete:
            pass
        else:
            raise AssertionError('actual initializer boundary did not execute')
    helper = context['pca_helper']
    check(loaded == ['_genuine_training_pca', '_genuine_training_packing'] and helper.fit_calls == 1 and
          len(helper.transforms) == len(heads) == len(results) == 2 and helper.transforms[0] is not helper.transforms[1] and
          results[0]['head'] is not results[1]['head'] and results[0]['bank'] is not results[1]['bank'] and
          results[0]['classifier'] is not results[1]['classifier'], 'once-loaded helper, fresh independent initializer reconstruction')
    for name in loaded: del sys.modules[name]


def trainer_checks(scripts):
    path = scripts / 'train_siglip2_genuine_views.py'
    spec = importlib.util.spec_from_file_location('_genuine_trainer_metadata', path)
    d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    trainer_metadata_checks(d)
    trainer_selection_checks(d, scripts)
    trainer_export_checks(d, scripts)
    with TemporaryDirectory() as directory:
        trainer_file_checks(d, Path(directory))
        trainer_terminal_checks(d, scripts, Path(directory))
    with TemporaryDirectory() as directory:
        trainer_initializer_lifetime(d, Path(directory))
    trainer_source_checks(d, path)
    for file in (path, Path(__file__).absolute()):
        compile(file.read_bytes(), str(file), 'exec')
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, '-B', '-S', flag, str(file), '--help'], capture_output=True, text=True)
            check(result.returncode != 0 and 'optimized mode is forbidden' in result.stderr, 'optimized trainer/test rejection')
    result = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    check(result.returncode == 0 and all('--' + n in result.stdout for n in
          ('execution-sha256', 'authority-sha256', 'phase', 'arm', 'seed', 'output')), 'trainer stdlib help')
    check(not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'sfora'} for n in sys.modules),
          'native imports during stdlib checks')


if __name__ == '__main__':
    trainer_checks(Path(__file__).absolute().parent)
    print('genuine-view trainer stdlib falsifier PASS; native/resource/quality qualification unrun')
