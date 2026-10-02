#!/usr/bin/env python3
"""Stdlib contract falsifier; synthetic fixtures are never native evidence.

python3 -B -S scripts/test_siglip2_genuine_view_evaluation.py
Optional --trainer-root DIRECTORY tests the separate authentic trainer2 source.
Requires the integrated original helpers and trainer2 check; no native imports.
No Torch/NumPy/native imports, real image/cache reads, GPU, SSH or quality runs.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import copy
import ast
import builtins
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejects(call, message=None):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        if message:
            assert message in str(error), (message, str(error))
        return
    raise AssertionError('invalid contract accepted: ' + str(message))


def file_ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def terminal_ref(prefix):
    return {'receipt': {'path': prefix + '/receipt.json', 'sha256': 'a' * 64},
            'log': {'path': prefix + '/unit.log', 'sha256': 'b' * 64},
            'unit': prefix.rsplit('/', 1)[-1], 'invocation_id': 'c' * 32,
            'service_seconds': 12., 'native_peak_rss_kib': 1000, 'both_locks_held': True}


def spec_fixture(driver, stage='first', panel='selection'):
    endpoints = [{'seed': seed, 'arm': arm,
                  'launch': {'path': f'/train/{arm}-{seed}/launch.json', 'sha256': 'a' * 64},
                  'checkpoint': {'path': f'/train/{arm}-{seed}/resume.pt', 'sha256': 'b' * 64},
                  'terminal_state_sha256': 'c' * 64, 'terminal': terminal_ref(f'/train/{arm}-{seed}')}
                 for seed, arm in driver.endpoint_order(stage)]
    return {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
            'training': {'root': driver.TRAIN_ROOT, 'execution_sha256': driver.TRAIN_EXECUTION_SHA,
                         'code': copy.deepcopy(driver.TRAIN_PINS)},
            'evaluation_reference': {'root': driver.REFERENCE_ROOT,
                                     'execution_sha256': driver.REFERENCE_EXECUTION_SHA},
            'partition': {'path': '/immutable/partition.json', 'sha256': driver.PARTITION_SHA},
            'endpoints': endpoints, 'stage': stage, 'panel': panel,
            'first_selection': None if stage == 'first' else terminal_ref('/first-score'),
            'selection_go': terminal_ref('/full-score') if panel == 'validation' else None,
            'resource_policies': {p: driver.policy(p) for p in ('cpu', 'score')},
            'cost_policy': copy.deepcopy(driver.COST_POLICY), 'both_locks_held': True,
            'selection_previously_exposed': True}


def authority_checks(driver):
    args = SimpleNamespace(execution_sha256='d' * 64)
    for stage, panel in (('first', 'selection'), ('full', 'selection'), ('full', 'validation')):
        spec = spec_fixture(driver, stage, panel)
        driver.check_spec(spec, args)
        for mutation in (
            lambda m: m.update(extra=True), lambda m: m.update(both_locks_held=False),
            lambda m: m['resource_policies']['score'].update(seconds=301),
            lambda m: m['cost_policy'].update(whole_service_ratio_max=1.51),
            lambda m: m['partition'].update(sha256='0' * 64),
            lambda m: m['training'].update(execution_sha256='future'),
            lambda m: m['training'].update(execution_sha256='0' * 64),
            lambda m: m['training']['code'].update({'train_siglip2_genuine_views.py': '0' * 64}),
            lambda m: m['training'].update(root='/wrong/frozen-trainer'),
            lambda m: m['training']['code'].pop('test_siglip2_genuine_view_training.py'),
            lambda m: m.update(selection_previously_exposed=False),
            lambda m: m['evaluation_reference'].update(execution_sha256='0' * 64),
            lambda m: m['endpoints'].reverse(), lambda m: m['endpoints'].pop(),
            lambda m: m['endpoints'][0].update(terminal_state_sha256='future'),
            lambda m: m['endpoints'][0]['terminal'].update(service_seconds=float('nan')),
            lambda m: m['endpoints'][0]['terminal'].update(both_locks_held=False),
        ):
            changed = copy.deepcopy(spec); mutation(changed)
            rejects(lambda: driver.check_spec(changed, args))
    for spec in (spec_fixture(driver, 'first', 'validation'),
                 dict(spec_fixture(driver, 'full', 'validation'), selection_go=None),
                 dict(spec_fixture(driver, 'full'), first_selection=None)):
        rejects(lambda: driver.check_spec(spec, args))


def metric_fixture(driver, stage='full', panel='selection', gains=(.004, .004)):
    count = driver.PANELS[panel][1]
    return {str(seed): {arm: {'per_query_r1': [.1 + (gain if arm == 'candidate' else 0)] * count,
                            'per_query_ap': [.2 + (gain if arm == 'candidate' else 0)] * count}
                       for arm in driver.ARMS}
            for seed, gain in zip(driver.seeds(stage), gains, strict=False)}


def metric_checks(driver):
    quality = metric_fixture(driver)
    deltas, average = driver.averaged_deltas(quality, 'full', 'selection')
    intervals = {m: {'mean_delta': sum(average[m]) / len(average[m]),
                      'product_lower95': .001, 'product_upper95': .01,
                      'query_lower95': -.001, 'query_upper95': .01} for m in driver.METRICS}
    assert driver.quality_gate(deltas, intervals) == (True, True)
    for key, value in (('mean_delta', .00199), ('product_lower95', 0), ('product_upper95', float('nan'))):
        changed = copy.deepcopy(intervals); changed[driver.METRICS[0]][key] = value
        if key == 'mean_delta':
            rejects(lambda: driver.quality_gate(deltas, changed))
        else:
            assert driver.quality_gate(deltas, changed) == (True, False)
    changed = copy.deepcopy(deltas); changed[str(driver.SEEDS[1])]['per_query_r1'] = [0.] * 1734
    changed_intervals = copy.deepcopy(intervals)
    for m in driver.METRICS:
        changed_intervals[m]['mean_delta'] = sum(sum(changed[s][m]) / len(changed[s][m])
                                                for s in changed) / 2
    assert driver.quality_gate(changed, changed_intervals) == (False, False)
    for gain, expected in ((0., False), (-.001, False), (.00001, True)):
        first, _ = driver.averaged_deltas(metric_fixture(driver, 'first', gains=(gain,)), 'first', 'selection')
        assert driver.first_gate(first) is expected
    first = metric_fixture(driver, 'first', gains=(.001,))
    first[str(driver.SEEDS[0])]['candidate']['per_query_ap'] = [.199] * 1734
    assert driver.first_gate(driver.averaged_deltas(first, 'first', 'selection')[0]) is False
    for mutation in (lambda m: m.pop(str(driver.SEEDS[1])),
                     lambda m: m[str(driver.SEEDS[0])]['candidate']['per_query_ap'].pop(),
                     lambda m: m[str(driver.SEEDS[0])]['candidate']['per_query_ap'].__setitem__(0, float('nan'))):
        changed = copy.deepcopy(quality); mutation(changed)
        rejects(lambda: driver.averaged_deltas(changed, 'full', 'selection'))
    assert driver.averaged_deltas(metric_fixture(driver, panel='validation'), 'full', 'validation')
    rejects(lambda: driver.averaged_deltas(quality, 'full', 'validation'))


def view_source():
    caches = {v: {'path': '/preparation/' + v + '.npy', 'sha256': ('1' if v == 'canonical' else '2') * 64,
                  'shape': [6355, 1152], 'dtype': 'float32', 'normalized': True, 'raw_pooled_cache': False}
              for v in ('canonical', 'augmented')}
    return {'caches': caches, 'ordered_input_sha256': '3' * 64,
            'ordered_view_sha256': {'canonical': '4' * 64, 'augmented': '5' * 64},
            'preparation_terminal': {'service_seconds': 20., 'native_peak_rss_kib': 1000}}


def launch_fixture(trainer, phase, arm, seed):
    return {'schema': trainer.AUTHORITY_SCHEMA, 'execution_sha256': 'e' * 64,
            'phase': phase, 'arm': arm, 'seed': seed, 'recipe': copy.deepcopy(trainer.RECIPE),
            'export_reference': {'root': str(trainer.EXPORT_ROOT), 'execution_sha256': trainer.EXPORT_EXECUTION_SHA,
                'code': trainer.EXPORT_CODE, 'authority': {'path': str(trainer.EXPORT_ROOT / 'authority.json'),
                                                        'sha256': trainer.EXPORT_AUTHORITY_SHA}},
            'selected_export': {'proof': {'path': '/preparation/receipt.json', 'sha256': 'a' * 64},
                'log': {'path': '/preparation/unit.log', 'sha256': 'b' * 64}, 'unit': 'export',
                'invocation_id': 'a' * 32, 'service_seconds': 20., 'native_peak_rss_kib': 1000,
                'both_locks_held': True},
            'cached_reference': {'root': '/cached', 'execution_sha256': trainer.CACHED_EXECUTION_SHA},
            'training_reference': {'root': '/original', 'execution_sha256': trainer.REFERENCE_EXECUTION_SHA},
            'helpers': {n: {'path': '/helpers/' + n + '.py', 'sha256': h} for n, h in trainer.HELPER_SHAS.items()},
            'partition': {'path': '/partition.json', 'sha256': trainer.PARTITION_SHA},
            'selected_cpu': None if phase == 'cpu' else {'receipt': {'path': '/cpu/receipt.json', 'sha256': 'a' * 64}},
            'selected_mechanics': None if phase != 'train' else dict.fromkeys(trainer.ARMS, {}),
            'resource_policy': trainer.policy(phase), 'both_locks_held': True}


def record_fixture(driver, trainer, seed=179061, arm='control'):
    launch = launch_fixture(trainer, 'train', arm, seed)
    defaults = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    groups = [dict(defaults, lr=1e-4)] * 2
    ident = {'method': trainer.method(launch), 'arm': arm, 'seed': seed, 'source': view_source(),
             'selected_cpu': launch['selected_cpu'], 'parameter_names': trainer.PARAMETERS,
             'numerical_flags': {}, 'optimizer_defaults': defaults, 'optimizer_groups': groups, 'device': 'cuda',
             'optimizer_serial_groups': [dict(groups[0], params=[0, 1, 2, 3]), dict(groups[1], params=[4])],
             **dict.fromkeys(('static_sha256', 'buffers_sha256', 'feature_state_sha256', 'schedule_sha256', 'mask_sha256'), 'a' * 64)}
    rows = [{'step': i, 'batch': list(range(64)), 'ce': 1., 'rank': .1, 'loss': 1.8, 'scale': 128.,
             'preclip_norm': 1., 'seconds': .001, 'gradient_norms': dict.fromkeys(trainer.PARAMETERS, 0.),
             'schedule_sha256': 'a' * 64,
             **dict.fromkeys(('feature_rows_sha256', 'mask_sha256', 'clean_bank_sha256', 'state_sha256'), 'b' * 64)}
            for i in range(1, 1001)]
    endpoint = {'seed': seed, 'arm': arm, 'launch': {'path': '/fixture/launch.json', 'sha256': 'c' * 64},
                'checkpoint': {'path': '/fixture/resume.pt', 'sha256': 'd' * 64},
                'terminal': {}, 'terminal_state_sha256': 'b' * 64}
    record = {'schema': trainer.SCHEMA, 'phase': 'train', 'arm': arm, 'seed': seed, 'launch': launch,
              'completed_step': 1000, 'optimizer_members': 5, 'source': view_source(), 'terminal_cgroups': {},
              'checkpoint': endpoint['checkpoint'], 'terminal_state_sha256': endpoint['terminal_state_sha256'],
              'partition_sha256': driver.PARTITION_SHA, 'code': {}, 'resource_policy': trainer.policy('train'),
              'numerical_flags': {}, 'head_scalars': 188544, 'trainable_scalars': 317568, 'encoder_updates': 0,
              'resumed_steps': [], 'peak_cuda_allocated_bytes': 1000000, 'identity': ident, 'steps': rows,
              'median_update_seconds': .001, 'training_wall_seconds': 2., 'wall_seconds': 10.,
              'service_seconds': 12., 'initial_state_sha256': 'e' * 64,
              **dict.fromkeys(('pass', 'training_qualified', 'strict_reload_exact', 'exit_rehash_pass'), True),
              **dict.fromkeys(('quality_read', 'trained_state_reused', 'training_state_discarded', 'replay_exact'), False)}
    selected = {'launch': launch, 'source': view_source(), 'terminal_cgroups': {}, 'code': {},
                'source_cpu': {'numerical_flags': {}},
                'export_record': {'extraction_seconds': 15., 'wall_seconds': 18.},
                'terminals': {'mechanics:' + a: {'initial_state_sha256': record['initial_state_sha256'],
                             'steps': copy.deepcopy(rows[:17])} for a in driver.ARMS}}
    return endpoint, record, selected


def endpoint_checks(driver, trainer):
    records = {}
    for seed, arm in driver.ORDER:
        endpoint, record, selected = record_fixture(driver, trainer, seed, arm)
        driver.check_endpoint(trainer, record, endpoint, selected)
        for mutation in (
            lambda m: m.update(completed_step=999), lambda m: m.update(trainable_scalars=445056),
            lambda m: m.update(quality_read=True), lambda m: m.update(strict_reload_exact=False),
            lambda m: m.update(peak_cuda_allocated_bytes=10000000000), lambda m: m['steps'].pop(),
            lambda m: m['steps'][-1].update(state_sha256='f' * 64),
            lambda m: m['steps'][19]['batch'].__setitem__(0, trainer.ROWS),
            lambda m: m['identity']['optimizer_serial_groups'][0].update(params=[0, 1, 2, 4]),
            lambda m: m['launch']['recipe'].update(candidate='gelu-exact'),
            lambda m: m['launch']['selected_mechanics'].pop('control'),
        ):
            changed = copy.deepcopy(record); mutation(changed)
            rejects(lambda: driver.check_endpoint(trainer, changed, endpoint, selected))
        if seed == driver.SEEDS[0]:
            changed = copy.deepcopy(record); changed['steps'][0]['clean_bank_sha256'] = 'f' * 64
            rejects(lambda: driver.check_endpoint(trainer, changed, endpoint, selected), 'first17 mechanics replay')
        records[seed, arm] = record
    costs = driver.paired_cost(records, 'full')
    assert all(v['pass'] for v in costs.values())
    for key in ('service_seconds', 'median_update_seconds'):
        changed = copy.deepcopy(records); changed[driver.ORDER[1]][key] *= 1.50
        assert driver.paired_cost(changed, 'full')[str(driver.SEEDS[0])]['pass']
        changed[driver.ORDER[1]][key] *= 1.0001
        assert driver.paired_cost(changed, 'full')[str(driver.SEEDS[0])]['pass'] is False
        changed[driver.ORDER[1]][key] = float('nan')
        rejects(lambda: driver.paired_cost(changed, 'full'), 'finite original endpoint cost')
    changed = copy.deepcopy(records); changed[driver.ORDER[1]]['steps'][0]['mask_sha256'] = 'f' * 64
    rejects(lambda: driver.paired_cost(changed, 'full'), 'paired TRAIN inputs')
    return records, selected


def origin_checks(driver):
    packages, files = {'torch': {'root': '/qualified/torch'}}, {'/qualified/base.so': 'a' * 64}
    records = [{'origins': {'packages': packages, 'files': files.copy()}, 'input_guards': files.copy()},
               {'origins': {'packages': packages, 'files': {'/qualified/head.so': 'b' * 64}},
                'input_guards': {'/qualified/head.so': 'b' * 64}}]
    context = {'selected': {'packages': packages}, 'origin_records': records}
    actual = {'packages': packages, 'files': {**files, '/qualified/head.so': 'b' * 64}}
    driver.check_origins(context, actual)
    assert files == {'/qualified/base.so': 'a' * 64}
    for path, digest in (('/foreign/extra.so', 'c' * 64), ('/qualified/head.so', 'c' * 64)):
        rejects(lambda: driver.check_origins(context, {'packages': packages, 'files': {**actual['files'], path: digest}}), path)
    for mutation in (
        lambda m: m[-1]['input_guards'].clear(),
        lambda m: m[-1]['origins'].update(packages={}),
        lambda m: m[-1]['origins']['files'].update({'/qualified/base.so': 'c' * 64}) or
                  m[-1]['input_guards'].update({'/qualified/base.so': 'c' * 64}),
    ):
        changed = copy.deepcopy(records); mutation(changed)
        rejects(lambda: driver.qualified_origins({**context, 'origin_records': changed}))


def preexit_checks(driver, path, original):
    """Differential exit falsifier: genuine validators, synthetic bytes only.

    The old nested traversal is the inventory oracle, never a source of hashes
    for the new exit. Only loaded modules and /proc maps are synthetic; all
    resolution, metadata, SHA streaming, cache advice and closures remain real.
    """
    started = driver.time.monotonic()
    scripts = path.parent
    source = load('_exit_source', scripts / 'qualify_siglip2_substrate_cpu.py')
    reference = load('_exit_reference', scripts / 'export_siglip2_substrate_fit.py')
    exporter = load('_exit_genuine', scripts / 'export_siglip2_genuine_views.py')
    evidence = scripts.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    fit = json.loads((evidence / 'late-dense-v1/native256-fit-manifest-v1.json').read_bytes())
    partition = json.loads((evidence / 'identity-mix-v1/partition.json').read_bytes())
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    body = functions['run'].body
    start = next(i + 1 for i, n in enumerate(body) if any(
        isinstance(v, ast.Constant) and v.value == 'whole-unit RNG/flags/CUDA differs' for v in ast.walk(n)))
    end = next(i for i in range(start, len(body)) if any(
        isinstance(v, ast.Constant) and v.value == 'exit separate evaluator/trainer/reference closure differs'
        for v in ast.walk(body[i])))
    preexit = ast.parse('def preexit(context):\n    args = context["args"]\n    return origins\n').body[0]
    preexit.body[1:1] = body[start:end + 1]
    final_closures = ast.parse('def final_closures(context):\n    args = context["args"]\n').body[0]
    final_closures.body.append(body[end])
    namespace = vars(driver).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[preexit, final_closures], type_ignores=[])), str(path), 'exec'), namespace)

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata = set()
        def frozen(name, names):
            directory = root / name; directory.mkdir()
            for file in names:
                (directory / file).write_bytes((scripts / file).read_bytes()
                    if file == 'extract_siglip2_vision_source.py' else b'# synthetic closure\n')
            code = {file: file_ref(directory / file)['sha256'] for file in names}
            descriptor = write_json(directory / 'execution.json', code)
            metadata.update(str(directory / file) for file in (*names, 'execution.json'))
            return directory, code, descriptor['sha256']
        source_root, code, source_sha = frozen('source', source.FILES)
        ref_root, ref_code, ref_sha = frozen('reference', reference.FILES)
        genuine_root, genuine_code, genuine_sha = frozen('genuine', exporter.FILES)
        eval_root, eval_code, eval_sha = frozen('evaluator', driver.FILES)
        train_root, train_code, train_sha = frozen('trainer', driver.TRAIN_FILES)
        math_root, math_code, math_sha = frozen('math', driver.REFERENCE_PINS)
        namespace.update(REFERENCE_EXECUTION_SHA=math_sha, REFERENCE_PINS=math_code)
        extract = load('extract_siglip2_vision_source', source_root / 'extract_siglip2_vision_source.py')
        dataset = root / 'z-images'; (dataset / 'Img/img').mkdir(parents=True)
        fit['dataset_root'] = str(dataset)
        image_sha = hashlib.sha256(b'image').hexdigest()
        images = []
        for i, row in enumerate(fit['rows']):
            row.update(relative_path=f'Img/img/{i:05}.jpg', image_sha256=image_sha)
            image = dataset / row['relative_path']; image.write_bytes(b'image'); images.append(image)
        part = write_json(root / 'partition.json', partition)
        ast_path = root / 'ImageRows.py'
        ast_path.write_text(ast.unparse(exporter.image_rows_node(scripts / 'train_sop_siglip2_compact.py')))
        metadata.update((part['path'], str(ast_path)))
        prior = {'source_driver': source, 'extract': extract, 'fit': fit, 'images': images[:2],
                 'all_images': images, 'root': source_root, 'code': code,
                 'args': SimpleNamespace(execution_sha256=source_sha), 'own_root': ref_root,
                 'own_code': ref_code, 'export_args': SimpleNamespace(execution_sha256=ref_sha)}
        manifest = exporter.selected_manifest(partition, fit)
        manifest['resolved_paths'] = [str(images[r]) for r in manifest['original_rows']]
        genuine = {'prior': prior, 'reference': reference, 'root': genuine_root, 'code': genuine_code,
                   'args': SimpleNamespace(execution_sha256=genuine_sha), 'selected': manifest,
                   'launch': {'partition': part, 'image_rows': file_ref(ast_path)}}
        packages, loaded = {}, {'extract_siglip2_vision_source': extract}
        for name in ('numpy', 'torch'):
            directory = root / name; directory.mkdir()
            packages[name] = {'root': str(directory)}
            loaded[name] = ModuleType(name)
        origin = root / 'numpy/origin.py'
        origin.write_bytes(bytes(range(256)) * 8192 + b'partial tail!')
        loaded['numpy.origin'] = ModuleType('numpy.origin'); loaded['numpy.origin'].__file__ = str(origin)
        for name, cls_name in (('ops', '_Ops'), ('classes', '_Classes')):
            backing = ModuleType('torch._' + name)
            backing.__file__ = str(root / 'torch' / ('_' + name + '.py'))
            Path(backing.__file__).write_bytes(b'# dynamic module backing\n')
            cls = type(cls_name, (ModuleType,), {'__module__': backing.__name__})
            dynamic = cls('torch.' + name); dynamic.__file__ = '_' + name + '.py'
            setattr(backing, cls_name, cls); setattr(backing, name, dynamic); setattr(loaded['torch'], name, dynamic)
            loaded[backing.__name__] = backing; loaded[dynamic.__name__] = dynamic
        library = root / 'mapped.so'; library.write_bytes(b'')
        payloads = [root / name for name in ('source.pt', 'view.npy', 'endpoint.pt')]
        for file in payloads:
            file.write_bytes(b'original payload')
        maps = f'0-1 r-xp 00000000 00:00 1 {library}\n'
        real_text, real_open = Path.read_text, Path.open
        after_discovery = None
        def maps_text(file, *args, **kwargs):
            if str(file) == '/proc/self/maps':
                if after_discovery is not None:
                    after_discovery()
                return maps
            return real_text(file, *args, **kwargs)
        streamed, opened, ranges = Counter(), Counter(), {}
        fail_read = False
        class Stream:
            def __init__(self, stream, file):
                self.stream, self.file = stream, str(file)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def read(self, size=-1):
                if fail_read and self.file == str(origin) and self.stream.tell():
                    raise OSError('exit mid-read failed')
                raw = self.stream.read(size); streamed[self.file] += len(raw); return raw
            def readinto(self, buffer):
                count = self.stream.readinto(buffer); streamed[self.file] += count; return count
        def recording_open(file, mode='r', *args, **kwargs):
            stream = real_open(file, mode, *args, **kwargs)
            if mode == 'rb':
                opened[str(file)] += 1
                return Stream(stream, file)
            return stream
        real_advice = os.posix_fadvise
        def advice(fd, offset, size, hint):
            file = os.readlink('/proc/self/fd/' + str(fd))
            ranges.setdefault(file, []).append((offset, size, hint))
            real_advice(fd, offset, size, hint)
        def old_exit(context):
            selected = context['selected']
            origins = source.imported_origins(extract, selected['packages'])
            driver.check_origins(context, origins)
            exporter.rehash(selected['genuine'])
            for file, digest in context['guards'].items():
                driver.bound_file({}, file, digest)
            return origins
        with patch.object(Path, 'read_text', maps_text), patch.dict(sys.modules, loaded), redirect_stdout(io.StringIO()):
            expected = source.imported_origins(extract, packages)
            prior['guards'] = {str(file): file_ref(file)['sha256'] for file in (*images, *payloads)}
            prior['guards'].update(expected['files'])
            genuine['guards'] = {file: file_ref(Path(file))['sha256'] for file in metadata}
            guards = {**prior['guards'], **genuine['guards']}
            context = {'selected': {'source_driver': source, 'extract': extract, 'packages': packages,
                                   'exporter': exporter, 'genuine': genuine, 'guards': guards},
                       'guards': guards, 'origin_records': [{'origins': copy.deepcopy(expected), 'input_guards': guards.copy()}],
                       'root': eval_root, 'code': eval_code, 'args': SimpleNamespace(execution_sha256=eval_sha),
                       'spec': {'training': {'root': str(train_root), 'execution_sha256': train_sha, 'code': train_code},
                                'evaluation_reference': {'root': str(math_root)}}}
            context['admission'] = original.FlatAdmission()
            for file in (images[2], *payloads, origin):
                context['admission'].bound_file(guards, file, guards[str(file)])
            inventories = [prior['guards'], genuine['guards'], guards, context['origin_records'][0]['input_guards']]
            before = [value.copy() for value in inventories]
            with patch.object(Path, 'open', recording_open):
                assert old_exit(context) == expected
            assert inventories == before
            assert opened[str(images[2])] == opened[str(origin)] == 3, 'nested baseline must reread bulk inputs'
            bulk = guards.keys() - metadata
            for _ in range(2):
                streamed.clear(); opened.clear(); ranges.clear()
                with patch.object(Path, 'open', recording_open), patch.object(os, 'posix_fadvise', advice):
                    assert namespace['preexit'](context) == expected
                assert inventories == before, 'exit changed the complete guard inventories'
                for file in bulk:
                    assert opened[file] == 1, ('repeated bulk exit open', file, opened[file])
                    assert streamed[file] == Path(file).stat().st_size, ('incomplete exit read', file, streamed[file])
                for file in (origin, library):
                    assert ranges.get(str(file), []) == [(offset, min(1024**2, file.stat().st_size - offset), os.POSIX_FADV_DONTNEED)
                        for offset in range(0, file.stat().st_size, 1024**2)]
            def reject_exit(message):
                rejects(lambda: namespace['preexit'](context), message)
                assert inventories == before, 'failed exit changed guard inventories'
            def flip(file):
                stamp = file.stat()
                with real_open(file, 'r+b') as stream:
                    stream.seek(stamp.st_size - 1); byte = stream.read(1)
                    stream.seek(stamp.st_size - 1); stream.write(bytes([byte[0] ^ 1]))
                os.utime(file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                assert file.stat().st_size == stamp.st_size and file.stat().st_mtime_ns == stamp.st_mtime_ns
                def restore():
                    with real_open(file, 'r+b') as stream:
                        stream.seek(stamp.st_size - 1); stream.write(byte)
                    os.utime(file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                return restore
            # Admission may already have every digest cached. Exit must still
            # reject equal-size/restored-mtime bytes, including after discovery.
            for file in (images[2], *payloads, origin):
                restore = flip(file)
                try:
                    with patch.object(Path, 'open', recording_open):
                        opened.clear()
                        context['admission'].bound_file({}, file, guards[str(file)])
                        assert not opened, 'synthetic admission cache was not warm'
                    reject_exit('file SHA256 differs')
                finally:
                    restore()
                restorers = []
                after_discovery = lambda: restorers.append(flip(file))
                try:
                    reject_exit('file SHA256 differs')
                    assert len(restorers) == 1
                finally:
                    after_discovery = None
                    for restore in restorers:
                        restore()
            # Conflicting input and qualified authorities fail before bulk IO.
            for inventory in (genuine['guards'], guards):
                file = str(payloads[0]); old = inventory.get(file)
                inventory[file] = '0' * 64
                try:
                    with patch.object(Path, 'open', recording_open):
                        opened.clear()
                        rejects(lambda: driver.exit_rehash(context), 'conflicting exit file authority')
                        assert not opened
                finally:
                    if old is None:
                        del inventory[file]
                    else:
                        inventory[file] = old
            records = context['origin_records']
            for fault in ('packages', 'unguarded', 'conflict', 'actual-conflict'):
                changed = copy.deepcopy(records)
                if fault == 'packages':
                    changed[0]['origins']['packages'] = {}
                elif fault == 'unguarded':
                    del changed[0]['input_guards'][str(origin)]
                else:
                    extra = copy.deepcopy(records[0])
                    extra['origins']['files'][str(origin)] = extra['input_guards'][str(origin)] = '0' * 64
                    changed = [*changed, extra] if fault == 'conflict' else [extra]
                context['origin_records'] = changed
                try:
                    with patch.object(Path, 'open', recording_open):
                        opened.clear()
                        rejects(lambda: driver.exit_rehash(context),
                                'packages differ' if fault == 'packages' else
                                'lacks authenticated guard' if fault == 'unguarded' else 'conflicting qualified origin')
                        assert not opened
                finally:
                    context['origin_records'] = records
            unknown = root / 'unqualified.so'; unknown.write_bytes(b'')
            original_maps = maps
            maps += f'0-1 r-xp 00000000 00:00 2 {unknown}\n'
            try:
                with patch.object(Path, 'open', recording_open):
                    opened.clear()
                    reject_exit('outside admitted qualified union')
                    assert not opened
                    # A guard alone cannot qualify an unknown actual library.
                    guards[str(unknown)] = hashlib.sha256(b'').hexdigest()
                    try:
                        rejects(lambda: driver.exit_rehash(context), 'outside admitted qualified union')
                        assert not opened
                    finally:
                        del guards[str(unknown)]
            finally:
                maps = original_maps
            for name, module in (
                ('torch.ops', SimpleNamespace(__file__='_ops.py', __name__='torch.ops')),
                ('torch.classes', SimpleNamespace(__file__='_classes.py', __name__='torch.classes')),
                ('numpy.origin', SimpleNamespace(__file__=str(payloads[0]))),
                ('numpy.origin', SimpleNamespace(__file__=str(root / 'absent.py'))),
            ):
                with patch.dict(sys.modules, {name: module}):
                    reject_exit('loaded native module origin')
            for index, target, message in ((0, images[2], 'resolution'), (2, origin, 'escaped'),
                                           (2, images[3], 'aliases')):
                file = images[index]; held = file.with_suffix('.held'); file.rename(held)
                try:
                    file.symlink_to(target)
                    reject_exit(message)
                finally:
                    file.unlink(); held.rename(file)
            for key in ('images', 'all_images'):
                original = prior[key]; prior[key] = list(reversed(original))
                try:
                    reject_exit('resolution')
                finally:
                    prior[key] = original
            row = fit['rows'][2]; original = row['image_sha256']; row['image_sha256'] = '0' * 64
            try:
                reject_exit('conflicting FIT file authority')
            finally:
                row['image_sha256'] = original
            fit['quality_read'] = True
            try:
                reject_exit('FIT manifest profile')
            finally:
                fit['quality_read'] = False
            original = manifest['resolved_paths'][0]; manifest['resolved_paths'][0] = str(origin)
            try:
                reject_exit('TRAIN mapping changed')
            finally:
                manifest['resolved_paths'][0] = original
            restore = flip(Path(part['path']))
            try:
                reject_exit('JSON SHA256/size differs')
            finally:
                restore()
            ast_bytes = ast_path.read_bytes()
            ast_path.write_bytes(ast_bytes.replace(b'class ImageRows', b'class ImageRowz'))
            try:
                reject_exit('original ImageRows class differs')
            finally:
                ast_path.write_bytes(ast_bytes)
            for owner, key in ((prior, 'code'), (prior, 'own_code'), (genuine, 'code')):
                original = owner[key]; owner[key] = {}
                try:
                    reject_exit('closure differs')
                finally:
                    owner[key] = original
            with patch.object(extract, '__file__', str(origin)):
                reject_exit('extractor loaded origin differs')
            with patch.object(extract.__spec__, 'origin', str(origin)):
                reject_exit('extractor import origin differs')
            # Execute the unchanged final run() predicate independently for
            # each closure mismatch; successful preexit above includes it too.
            for directory, file in ((eval_root, next(iter(eval_code))), (train_root, next(iter(train_code))),
                                    (math_root, next(iter(math_code)))):
                restore = flip(directory / file)
                try:
                    rejects(lambda: namespace['final_closures'](context), 'file SHA256 differs')
                finally:
                    restore()
            def failed_open(file, *args, **kwargs):
                if file == origin:
                    raise OSError('exit open failed')
                return real_open(file, *args, **kwargs)
            with patch.object(Path, 'open', failed_open):
                reject_exit('exit open failed')
            fail_read = True; streamed.clear()
            try:
                with patch.object(Path, 'open', recording_open):
                    reject_exit('exit mid-read failed')
                    assert streamed[str(origin)] == 1024**2
            finally:
                fail_read = False
            def failed_advice(fd, *args):
                if os.readlink('/proc/self/fd/' + str(fd)) == str(origin):
                    raise OSError('exit advice failed')
                real_advice(fd, *args)
            with patch.object(os, 'posix_fadvise', failed_advice):
                reject_exit('exit advice failed')
            assert inventories == before
        assert sum(file.stat().st_size for file in root.rglob('*') if file.is_file()) <= 4 * 1024**2
    assert driver.time.monotonic() - started < 30, 'exit falsifier exceeded 30 seconds'


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return file_ref(path)


def terminal_fixture(driver, context, output, phase, cpu_terminal=None, gain=.004):
    output.mkdir()
    args = context['args']
    authority = file_ref(args.authority) if args.authority.is_file() and json.loads(args.authority.read_text()) == context['spec'] else write_json(output / 'authority.json', context['spec'])
    args.authority, args.authority_sha256 = Path(authority['path']), authority['sha256']
    unit, identity = output.name, hashlib.md5(str(output).encode(), usedforsecurity=False).hexdigest()
    cgroup = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
        'memory.max': str(8 * 1024**3), 'memory.current': '1024', 'memory.peak': '2048',
        'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
        'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n'}}
    prior = None if cpu_terminal is None else write_json(output / 'cpu-terminal.json', cpu_terminal)
    record = {**driver.bind(context), 'schema': driver.SCHEMA, 'phase': phase,
              'resource_policy': driver.policy(phase), 'optimizer_updates': 0, 'optimizer_members': 5,
              'quality_read': phase == 'score', 'numerical_flags': {}, 'authority': authority,
              'output': str(output), 'prerequisite': prior, 'cpu_terminal': cpu_terminal,
              'peak_cuda_allocated_bytes': 0, 'files': {}, 'input_guards': {}, 'origins': {'packages': {}, 'files': {}},
              'wall_seconds': 10., 'process_peak_rss_kib': 1000, 'cgroup_before': cgroup, 'cgroup_after': copy.deepcopy(cgroup),
              'head_facts': dict.fromkeys((driver.label(e) for e in context['spec']['endpoints']), 'a' * 64),
              'train_witnesses': {driver.label(e): {} for e in context['spec']['endpoints']},
              'invocation': {'invocation_id': identity, 'optimize': 0, 'cuda_visible_devices': '',
                             'python': '/qualified/python', 'python_sha256': 'a' * 64, 'python_version': 'fixture',
                             'argv': driver.cli_argv(args.authority, args.authority_sha256, args.execution_sha256,
                                                     phase, output, prior)},
              **dict.fromkeys(('pass', 'engineering_admission_pass', 'strict_independent_head_reload_exact',
                              'train_raw_unit_cpu_packed_exact', 'first_heads_released_before_reload',
                              'rng_flags_preserved', 'exit_rehash_pass'), True),
              **dict.fromkeys(('official_read', 'global_production_goal_met', 'public_latency_measured', 'cuda_initialized'), False)}
    if phase == 'score':
        quality = metric_fixture(driver, context['spec']['stage'], gains=(gain, gain))
        deltas, average = driver.averaged_deltas(quality, context['spec']['stage'], 'selection')
        intervals = {} if context['spec']['stage'] == 'first' else {m: {
            'mean_delta': sum(average[m]) / len(average[m]), 'product_lower95': .001, 'product_upper95': .01,
            'query_lower95': -.001, 'query_upper95': .01} for m in driver.METRICS}
        passed = driver.first_gate(deltas) if context['spec']['stage'] == 'first' else driver.quality_gate(deltas, intervals)[1]
        cost = driver.paired_cost(context['records'], context['spec']['stage'])
        record.update(quality=quality, paired_seed_average_intervals=intervals, quality_pass=passed,
                      mean_deltas={m: sum(average[m]) / len(average[m]) for m in driver.METRICS},
                      cost=cost, cost_policy=driver.COST_POLICY, cost_pass=all(v['pass'] for v in cost.values()),
                      full_panel_raw_unit_packed_replay_exact=True, per_query_replay_exact=True,
                      bootstrap_draws=0 if context['spec']['stage'] == 'first' else 5000, bootstrap_seed=179019,
                      canonical_panel_input_only=True, serving_view_averaging=False,
                      preparation_costs=driver.preparation_costs(context['selected']),
                      decision=('CONTINUE' if context['spec']['stage'] == 'first' else 'GO') if passed else 'KILL')
        for name in driver.file_names(context['spec']):
            path = output / name; path.write_bytes(b'stdlib contract fixture, never native wire evidence')
            record['files'][name] = file_ref(path)['sha256']
    receipt = write_json(output / 'receipt.json', record)
    log = '\n'.join([f'Running as unit: {unit}.service; invocation ID: {identity}', '\tExit status: 0',
        'Finished with result: success', 'Main processes terminated with: code=exited/status=0', '\tSwaps: 0',
        'Memory swap peak: 0B', '\tMaximum resident set size (kbytes): 1000', 'Service runtime: 12.0s',
        'FINAL_CGROUP ' + json.dumps({**cgroup, 'invocation_id': identity})]) + '\n'
    (output / 'unit.log').write_text(log)
    terminal = {'receipt': receipt, 'log': file_ref(output / 'unit.log'), 'unit': unit, 'invocation_id': identity,
                'service_seconds': 12., 'native_peak_rss_kib': 1000, 'both_locks_held': True}
    return terminal, record


def panel_checks(driver, original, initializer, helper, records, selected, root):
    admission = original.FlatAdmission(); admission.init = initializer
    selected = {**selected, 'source_cpu': {**selected['source_cpu'], 'numerical_flags': {},
                'invocation': {'python': '/qualified/python', 'python_sha256': 'a' * 64, 'python_version': 'fixture'}}}
    spec = spec_fixture(driver, 'first')
    # The real gate must reject validation at the entry boundary before any native loader.
    context = {'args': SimpleNamespace(execution_sha256='d' * 64, authority=Path('/fixture/authority.json'),
                                      authority_sha256='a' * 64), 'spec': spec, 'selected': selected,
               'records': {k: records[k] for k in driver.endpoint_order('first')}, 'code': {},
               'admission': admission, 'helper': helper, 'guards': {}, 'required_guards': {},
               'terminals': [], 'origin_records': [], 'base_guards': {}}
    driver.panel_authority(context)
    bad = {**context, 'spec': spec_fixture(driver, 'full', 'validation')}
    bad['spec']['selection_go'] = None
    rejects(lambda: driver.panel_authority(bad), 'selection GO')
    cpu_terminal, cpu = terminal_fixture(driver, context, root / 'first-cpu', 'cpu')
    first_terminal, first = terminal_fixture(driver, context, root / 'first-score', 'score', cpu_terminal)
    driver.check_receipt(context, first, 'score')
    full_spec = spec_fixture(driver, 'full')
    full_spec['endpoints'][:2] = spec['endpoints']
    full_spec['first_selection'] = first_terminal
    context.update(spec=full_spec, records=records)
    driver.panel_authority(context)
    assert context['first_selection']['decision'] == 'CONTINUE'
    rejects(lambda: driver.accept_selection({**context, 'guards': {}, 'base_guards': {'/missing-source': 'a' * 64}},
                                           first_terminal, 'first'), 'complete evaluator input guards')
    for mutation in (
        lambda m: m.update(decision='GO'), lambda m: m.update(quality_pass=False),
        lambda m: m.update(per_query_replay_exact=False),
        lambda m: m.update(paired_seed_average_intervals={'unrequested': {}}),
        lambda m: m.update(canonical_panel_input_only=False),
        lambda m: m['preparation_costs']['shared'].update(whole_service_seconds=0), lambda m: m['quality'][str(driver.SEEDS[0])]['candidate']['per_query_r1'].__setitem__(0, 0),
        lambda m: m['spec']['endpoints'][0]['checkpoint'].update(sha256='f' * 64),
    ):
        changed = copy.deepcopy(first); mutation(changed)
        # Rehashing a mutated receipt cannot legitimize a wrong gate or checkpoint.
        ref = write_json(Path(first_terminal['receipt']['path']), changed)
        altered_context = {**context, 'guards': {}}
        rejects(lambda: driver.accept_selection(altered_context, dict(first_terminal, receipt=ref), 'first'))
    write_json(Path(first_terminal['receipt']['path']), first)
    context['guards'] = {}
    full_cpu_terminal, full_cpu = terminal_fixture(driver, context, root / 'full-cpu', 'cpu')
    full_terminal, full = terminal_fixture(driver, context, root / 'full-score', 'score', full_cpu_terminal)
    validation_spec = copy.deepcopy(full_spec); validation_spec.update(panel='validation', selection_go=full_terminal)
    validation = {**context, 'spec': validation_spec, 'guards': {}}
    driver.panel_authority(validation)
    assert validation['selection_go']['decision'] == 'GO'
    for key in ('decision', 'quality_pass', 'cost_pass', 'per_query_replay_exact'):
        changed = copy.deepcopy(full); changed[key] = 'KILL' if key == 'decision' else False
        ref = write_json(Path(full_terminal['receipt']['path']), changed)
        altered = copy.deepcopy(validation_spec); altered['selection_go']['receipt'] = ref
        rejects(lambda: driver.panel_authority({**validation, 'spec': altered, 'guards': {}}))
    write_json(Path(full_terminal['receipt']['path']), full)
    changed = copy.deepcopy(validation_spec); changed['endpoints'][-1]['checkpoint']['sha256'] = 'f' * 64
    rejects(lambda: driver.panel_authority({**validation, 'spec': changed, 'guards': {}}), 'selection authority/checkpoints')
    log_path = Path(full_terminal['log']['path'])
    log_path.write_text(log_path.read_text().replace('\tExit status: 0', '\tExit status: 1'))
    changed = copy.deepcopy(validation_spec); changed['selection_go']['log'] = file_ref(log_path)
    fresh_admission = original.FlatAdmission(); fresh_admission.init = initializer
    rejects(lambda: driver.panel_authority({**validation, 'spec': changed, 'guards': {}, 'admission': fresh_admission}), 'normal-exit log')
    rejects(lambda: driver.accept_terminal({**context, 'guards': {}, 'required_guards': {'/missing': 'a' * 64}},
                                          full_cpu_terminal, 'cpu'), 'complete evaluator input guards')
    actual = {'per_query_r1': [1, 0], 'per_query_ap': [.5, 0]}
    driver.replay_equal(actual, copy.deepcopy(actual))
    rejects(lambda: driver.replay_equal(actual, {**actual, 'per_query_ap': [.5, .1]}), 'per-query packed quality replay')


def file_checks(driver, root):
    path = root / 'input'; path.write_bytes(b'original')
    ref = file_ref(path); guards = {}
    driver.bound_file(guards, path, ref['sha256'])
    path.write_bytes(b'changed')
    rejects(lambda: driver.bound_file(guards, path, ref['sha256']), 'file SHA256')
    link = root / 'link'; link.symlink_to(path)
    rejects(lambda: driver.bound_file({}, link, file_ref(path)['sha256']), 'canonical file')
    rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate JSON')
    rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite JSON')
    for name in driver.FILES:
        (root / name).write_text('# immutable fixture\n')
    code = {name: file_ref(root / name)['sha256'] for name in driver.FILES}
    ref = write_json(root / 'execution.json', code)
    assert driver.closure(root, ref['sha256'], driver.FILES, {}) == code
    ref = write_json(root / 'execution.json', dict(code, extra='0' * 64))
    rejects(lambda: driver.closure(root, ref['sha256'], driver.FILES, {}), 'exact execution closure')


def endpoint_admission_checks(driver, trainer, original, initializer, helper, path, root):
    """Run the actual authority endpoint loop with real receipts and byte admission."""
    authority = next(n for n in ast.parse(path.read_bytes()).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'authority')
    loop = next(n for n in authority.body if isinstance(n, ast.For) and
                ast.unparse(n.iter) == "spec['endpoints']")
    function = ast.parse('def admit(spec, selected, admission):\n    records, terminals = {}, []\n    return records, terminals\n').body[0]
    function.body.insert(1, loop)
    namespace = {**vars(driver), 'trainer': trainer, 'helper': helper, 'train_root': root, 'required': {}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    shared = root / 'shared'; shared.write_bytes(b'original')
    shared_ref = file_ref(shared)
    spec = spec_fixture(driver, 'first'); spec['training']['execution_sha256'] = 'e' * 64
    fixtures, inventories = [], {}
    for seed, arm in driver.endpoint_order('first'):
        endpoint, record, selected = record_fixture(driver, trainer, seed, arm)
        context = {'args': SimpleNamespace(execution_sha256='d' * 64, authority=root / 'authority.json',
                                          authority_sha256='a' * 64), 'spec': spec, 'selected': selected, 'code': {}}
        output = root / arm
        terminal, cpu = terminal_fixture(driver, context, output, 'cpu')
        checkpoint = output / 'resume.pt'; checkpoint.write_bytes(b'checkpoint')
        endpoint.update(launch=write_json(output / 'launch.json', record['launch']),
                        checkpoint=file_ref(checkpoint), terminal=terminal)
        record.update({k: cpu[k] for k in ('invocation', 'process_peak_rss_kib', 'cgroup_before', 'cgroup_after')})
        record['invocation'].update(cuda_visible_devices='0', cublas_workspace_config=':4096:8',
            argv=[str(root / 'train_siglip2_genuine_views.py'), '--execution-sha256', 'e' * 64,
                  '--authority', endpoint['launch']['path'], '--authority-sha256', endpoint['launch']['sha256'],
                  '--phase', 'train', '--arm', arm, '--seed', str(seed), '--output', str(output)])
        inventory = {str(shared): shared_ref['sha256'], str(checkpoint): endpoint['checkpoint']['sha256']}
        record.update(authority=endpoint['launch'], authority_sha256=endpoint['launch']['sha256'],
                      execution_sha256='e' * 64, checkpoint=endpoint['checkpoint'], input_guards=inventory.copy())
        inventories[seed, arm] = inventory
        with Path(terminal['log']['path']).open('a') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in record['steps']))
        terminal.update(log=file_ref(Path(terminal['log']['path'])),
                        receipt=write_json(Path(terminal['receipt']['path']), record))
        fixtures.append((endpoint, record))
    spec['endpoints'] = [e for e, _ in fixtures]
    selected['guards'] = {}
    selected['source_cpu']['invocation'] = {k: cpu['invocation'][k] for k in ('python', 'python_sha256', 'python_version')}
    admission = original.FlatAdmission(); admission.init = initializer
    assert not admission.verified
    reads, real_open = [], Path.open
    tracked = {str(shared), *(e['checkpoint']['path'] for e in spec['endpoints'])}
    def recording_open(file, mode='r', *args, **kwargs):
        if str(file) in tracked and mode == 'rb':
            reads.append(str(file))
        return real_open(file, mode, *args, **kwargs)
    with patch.object(Path, 'open', recording_open):
        records, terminals = namespace['admit'](spec, selected, admission)
        assert reads.count(str(shared)) == 1, ('duplicate endpoint inventory hash reads', reads)
        assert all(reads.count(e['checkpoint']['path']) == 1 for e in spec['endpoints'])
        assert len(terminals) == 2 and {k: r['input_guards'] for k, r in records.items()} == inventories
        assert all(selected['guards'][p] == h for inventory in inventories.values() for p, h in inventory.items())
        endpoint, record = fixtures[-1]
        changed = copy.deepcopy(record); changed['input_guards'][str(shared)] = 'f' * 64
        endpoint['terminal']['receipt'] = write_json(Path(endpoint['terminal']['receipt']['path']), changed)
        selected['guards'] = {}
        rejects(lambda: namespace['admit'](spec, selected, admission), 'conflicting file SHA256/size authority')
        endpoint['terminal']['receipt'] = write_json(Path(endpoint['terminal']['receipt']['path']), record)
        rejects(lambda: admission.bound_file({}, shared, shared_ref['sha256'], size=9), 'bound file size differs')
        shared.write_bytes(b'longer original')
        selected['guards'] = {}
        rejects(lambda: namespace['admit'](spec, selected, admission), 'conflicting file SHA256/size authority')
        shared.write_bytes(b'original')
        altered = copy.deepcopy(spec); changed = copy.deepcopy(record)
        altered['endpoints'][-1]['checkpoint']['sha256'] = 'f' * 64
        changed['checkpoint'] = altered['endpoints'][-1]['checkpoint']
        del changed['input_guards'][changed['checkpoint']['path']]
        altered['endpoints'][-1]['terminal']['receipt'] = write_json(Path(endpoint['terminal']['receipt']['path']), changed)
        fresh = original.FlatAdmission(); fresh.init = initializer
        selected['guards'] = {}
        rejects(lambda: namespace['admit'](altered, selected, fresh), 'file SHA256 differs')
        endpoint['terminal']['receipt'] = write_json(Path(endpoint['terminal']['receipt']['path']), record)
        selected['guards'] = {}
        namespace['admit'](spec, selected, admission)
        stat = shared.stat()
        shared.write_bytes(b'mutated!'); os.utime(shared, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert shared.stat().st_size == stat.st_size and shared.stat().st_mtime_ns == stat.st_mtime_ns
        before = reads.count(str(shared))
        admission.bound_file(selected['guards'], shared, shared_ref['sha256'])
        assert reads.count(str(shared)) == before
        # The full genuine exit fixture above checks this same fresh boundary
        # with cached authorities; this admission fixture has no source context.
        rejects(lambda: driver.bound_file({}, shared, shared_ref['sha256']), 'file SHA256 differs')
        assert reads.count(str(shared)) == before + 1


def output_wire_checks(driver, root):
    assert callable(getattr(driver, 'check_output', None)), 'exclusive output boundary is missing'
    assert callable(getattr(driver, 'packed_readback', None)), 'packed readback boundary is missing'
    output = root / 'new'
    driver.check_output(output)
    output.mkdir()
    rejects(lambda: driver.check_output(output), 'exclusive canonical output')
    rejects(lambda: driver.check_output(Path('relative')), 'exclusive canonical output')
    symlink = root / 'alias'; symlink.symlink_to(output, target_is_directory=True)
    rejects(lambda: driver.check_output(symlink), 'exclusive canonical output')
    rejects(lambda: driver.check_output(symlink / 'new'), 'exclusive canonical output')
    dangling = root / 'dangling'; dangling.symlink_to(root / 'absent')
    rejects(lambda: driver.check_output(dangling), 'exclusive canonical output')
    wire = root / 'packed.bin'; wire.write_bytes(b'complete packed bytes')
    ref = file_ref(wire)
    driver.packed_readback({}, wire, ref['sha256'], wire.read_bytes())
    wire.write_bytes(b'altered packed bytes')
    rejects(lambda: driver.packed_readback({}, wire, ref['sha256'], b'complete packed bytes'), 'file SHA256')
    rejects(lambda: driver.packed_readback({}, wire, file_ref(wire)['sha256'], b'complete packed bytes'), 'packed wire')


def source_checks(driver, path):
    tree = ast.parse(path.read_bytes())
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # Only historical endpoint inventories/checkpoints use the invocation-local reader.
    cached = [ast.unparse(n) for n in ast.walk(nodes['authority']) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == 'bound_file']
    assert set(cached) == {"admission.bound_file(selected['guards'], path, digest)",
                          "admission.bound_file(selected['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])"}
    assert len(cached) == 2 and "admission = selected['original'].FlatAdmission()" in functions['authority']
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('verified', 'entries', 'json_bytes')
                   for n in ast.walk(nodes['authority']))
    assert "bound_file(context['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])" in functions['load_head']
    assert "bound_file(context['guards'], proof['checkpoint']['path'], proof['checkpoint']['sha256'])" in functions['load_head']
    exit_source = functions['exit_rehash']
    assert 'bound_file({}, path, digest)' in exit_source
    assert 'imported_origins(' not in exit_source and '.rehash(' not in exit_source
    for predicate in ('qualified_origins(context)', 'source.loaded_module_origin(', 'source.fit_rows(',
                      'source.bootstrap(', "genuine['reference'].bootstrap(", 'exporter.closure(',
                      'exporter.selected_manifest(', 'exporter.file_json(', 'exporter.image_rows_node('):
        assert predicate in exit_source, predicate
    assert functions['run'].index('exit_rehash(context)') < functions['run'].index('check_origins(context, origins)') < functions['run'].index("resources = context['helper'].resources") < functions['run'].index("context['helper'].publish")
    assert "head_from('control', tensors=saved['head'])" in functions['load_head']
    assert functions['load_head'].index('trainer.check_payload(saved, ident, 1000)') < functions['load_head'].index("head_from('control', tensors=saved['head'])")
    assert "context['features']['canonical'][:64]" in functions['qualify_heads']
    assert 'cache_rows(' not in functions['qualify_heads']
    assert 'cache_facts(' not in functions['authority'], 'panel values must remain unread until their authorized score'
    assert 'training_features(selected)' in functions['native_start']
    assert "saved['views'] == selected['views']" in functions['load_head']
    assert "original.fingerprint(masks[i]) == row['mask_sha256']" in functions['load_head']
    assert 'original.fingerprint(saved)' in functions['load_head']
    assert 'original.fingerprint(initial)' in functions['load_head']
    assert 'packed_readback(' in functions['score_panel']
    assert 'panel_authority(context)' in functions['authority']
    assert functions['run'].index('authority(args)') < functions['run'].index('native_start(context)') < functions['run'].index('score_panel(context, cpu)')
    assert 'exit_rehash(context)' in functions['run']
    assert 'scoring_math(' in functions['score_panel'] and 'replay_equal(first, second_quality)' in functions['score_panel']
    assert "panel['original_rows']" in functions['score_panel']
    assert 'qualified_origins(context)' in functions['native_start']
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(w in ast.unparse(n) for w in ('torch', 'numpy', 'PIL'))
                   for n in tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not ast.unparse(node.func).endswith(('reset_peak_memory_stats', 'fresh_source', 'fit_centered_pca', 'decode', 'load_initializers'))


def main():
    root = Path(__file__).resolve().parent
    path = root / 'evaluate_siglip2_genuine_views.py'
    assert path.exists(), 'genuine-view evaluator is missing'
    driver = load('_genuine_view_evaluation_test', path)
    trainer_root = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 and sys.argv[1] == '--trainer-root' else root
    trainer = load('_genuine_view_trainer_fixture', trainer_root / 'train_siglip2_genuine_views.py')
    trainer_test = load('_genuine_view_training_test_fixture', trainer_root / 'test_siglip2_genuine_view_training.py')
    original = load('_original_trainer_fixture', root / 'train_siglip2_substrate_adaptation.py')
    initializer = load('_initializer_fixture', root / 'initialize_siglip2_substrate_fit.py')
    helper = load('_held_helpers_fixture', root / 'export_siglip2_substrate_adaptation.py')
    authority_checks(driver); metric_checks(driver); origin_checks(driver)
    preexit_checks(driver, path, original)
    records, selected = endpoint_checks(driver, trainer)
    trainer_test.trainer_metadata_checks(trainer)
    with TemporaryDirectory() as directory:
        panel_checks(driver, original, initializer, helper, records, selected, Path(directory))
    with TemporaryDirectory() as directory:
        file_checks(driver, Path(directory))
    with TemporaryDirectory() as directory:
        endpoint_admission_checks(driver, trainer, original, initializer, helper, path, Path(directory))
    with TemporaryDirectory() as directory:
        output_wire_checks(driver, Path(directory))
    source_checks(driver, path)
    for file in (path, Path(__file__).resolve()):
        compile(file.read_bytes(), str(file), 'exec')
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, '-B', '-S', flag, str(file), '--help'], capture_output=True, text=True)
            assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
    result = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and all('--' + n in result.stdout for n in
        ('execution-sha256', 'authority-sha256', 'phase', 'output', 'prerequisite-sha256'))
    assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
    print('PASS: authority/endpoint-admission/complete-payload/partition/panel/early-stop/terminal/replay/cost/preparation/origins/preexit-union-differential/fresh-tamper/predicates/io-failures/roles/wire/exclusive-output/syntax/help/-O/-OO; native/resource/quality UNRUN')


if __name__ == '__main__':
    old_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}:
            raise AssertionError('native import attempted by stdlib falsifier: ' + name)
        return old_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    try:
        main()
    finally:
        builtins.__import__ = old_import
