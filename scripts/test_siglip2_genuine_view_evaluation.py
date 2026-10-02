#!/usr/bin/env python3
"""Stdlib contract falsifier; synthetic fixtures are never native evidence.

python3 -B -S scripts/test_siglip2_genuine_view_evaluation.py
Optional --trainer-root DIRECTORY tests the separate authentic trainer2 source.
Requires the integrated original helpers and trainer2 check; no native imports.
No Torch/NumPy/native imports, image/cache reads, GPU, SSH or quality runs.
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
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


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


def preexit_checks(driver, path):
    """Execute the actual pre-exit path/guards with only in-memory file IO."""
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    body = functions['run'].body
    start = next(i + 1 for i, n in enumerate(body) if any(
        isinstance(v, ast.Constant) and v.value == 'whole-unit RNG/flags/CUDA differs' for v in ast.walk(n)))
    end = next(i for i in range(start, len(body)) if any(
        isinstance(v, ast.Constant) and v.value == 'exit separate evaluator/trainer/reference closure differs'
        for v in ast.walk(body[i])))
    preexit = ast.parse("def preexit(context):\n    source = context['selected']['source_driver']\n    return origins\n").body[0]
    preexit.body[1:1] = body[start:end]
    events, memory = [], {}

    class MemoryStream(io.BytesIO):
        def fileno(self):
            return 0

    class MemoryPath(str):
        def is_absolute(self):
            return self.startswith('/')
        def resolve(self):
            return self
        def is_file(self):
            return self in memory
        def open(self, mode):
            assert mode == 'rb'
            events.append(('read', str(self)))
            return MemoryStream(memory[self])

    namespace = {'Path': MemoryPath, 'hashlib': hashlib, 're': driver.re,
                 'os': SimpleNamespace(posix_fadvise=lambda *args: None, POSIX_FADV_DONTNEED=0),
                 'json': json, 'time': driver.time, 'UNIT_STARTED': driver.UNIT_STARTED}
    tree = ast.Module(body=[functions[n] for n in (
        'require', 'bound_file', 'qualified_origins', 'check_origins', 'exit_rehash')] + [preexit], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    packages = {'torch': {'root': '/qualified/torch'}}
    for fault in (None, 'foreign', 'conflicting', 'qualified-conflict', 'mutation'):
        events.clear()
        memory.update({'/qualified/base.so': b'original', '/authority.json': b'authority'})
        guards = {p: hashlib.sha256(raw).hexdigest() for p, raw in memory.items()}
        actual = {'packages': packages, 'files': {'/qualified/base.so': guards['/qualified/base.so']}}
        records = [{'origins': copy.deepcopy(actual), 'input_guards': guards.copy()}]
        if fault in ('foreign', 'conflicting'):
            actual['files']['/foreign/extra.so' if fault == 'foreign' else '/qualified/base.so'] = 'c' * 64
        if fault == 'qualified-conflict':
            records.append({'origins': {'packages': packages, 'files': {'/qualified/base.so': 'c' * 64}},
                            'input_guards': {'/qualified/base.so': 'c' * 64}})

        def scan(extract, admitted):
            assert extract == 'fixture' and admitted == packages
            events.append('scan')
            return copy.deepcopy(actual)

        def rehash(genuine):
            assert genuine == 'fixture'
            events.append('rehash')
            if fault == 'mutation':
                changed = b'changed!'
                assert len(changed) == len(memory['/qualified/base.so'])
                memory['/qualified/base.so'] = changed

        context = {'selected': {'packages': packages, 'extract': 'fixture', 'genuine': 'fixture',
                               'source_driver': SimpleNamespace(imported_origins=scan),
                               'exporter': SimpleNamespace(rehash=rehash)},
                   'origin_records': records, 'guards': guards.copy()}
        with redirect_stdout(io.StringIO()):
            if fault is None:
                assert namespace['preexit'](context) == actual
                assert events == ['scan', 'rehash', *[('read', p) for p in guards]], events
            elif fault == 'mutation':
                rejects(lambda: namespace['preexit'](context), 'file SHA256 differs')
                assert events == ['scan', 'rehash', ('read', '/qualified/base.so')], events
            else:
                rejects(lambda: namespace['preexit'](context),
                        'conflicting qualified origin' if fault == 'qualified-conflict' else 'outside admitted qualified union')
                assert events == ['scan'], events
        assert context['guards'] == guards and records[0]['input_guards'] == guards


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
    preexit_checks(driver, path)
    records, selected = endpoint_checks(driver, trainer)
    trainer_test.trainer_metadata_checks(trainer)
    with TemporaryDirectory() as directory:
        panel_checks(driver, original, initializer, helper, records, selected, Path(directory))
    with TemporaryDirectory() as directory:
        file_checks(driver, Path(directory))
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
    print('PASS: authority/complete-payload/partition/panel/early-stop/terminal/replay/cost/preparation/origins/preexit/tamper/roles/wire/exclusive-output/syntax/help/-O/-OO; native/resource/quality UNRUN')


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
