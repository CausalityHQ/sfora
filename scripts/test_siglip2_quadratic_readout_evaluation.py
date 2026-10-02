#!/usr/bin/env python3
"""Narrow stdlib falsifier. Synthetic fixtures are never native evidence.

python3 -B -S scripts/test_siglip2_quadratic_readout_evaluation.py
No Torch/NumPy/native imports, real cache reads, SSH, GPU or quality execution.
Optional --trainer-root REPOSITORY_SCRIPTS checks the separately owned trainer
metadata API. Requires this checkout's committed source evidence and helpers.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import argparse
import builtins
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
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


def terminal(prefix):
    return {'receipt': {'path': prefix + '/receipt.json', 'sha256': 'a' * 64},
            'log': {'path': prefix + '/unit.log', 'sha256': 'b' * 64},
            'unit': prefix.rsplit('/', 1)[-1], 'invocation_id': 'c' * 32,
            'service_seconds': 12., 'native_peak_rss_kib': 1000, 'both_locks_held': True}


def spec_fixture(d, stage='first', panel='selection'):
    return {'schema': d.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
            'training': {'root': '/immutable/quadratic-trainer4', 'execution_sha256': 'e' * 64,
                         'code': {name: 'f' * 64 for name in ('train_siglip2_quadratic_readout.py',
                             'test_siglip2_quadratic_readout.py', 'quadratic_readout.py', 'quadratic_encoder_frames.py')}},
            'evaluation_reference': {'root': d.REFERENCE_ROOT, 'execution_sha256': d.REFERENCE_EXECUTION_SHA},
            'partition': {'path': '/immutable/partition.json', 'sha256': d.PARTITION_SHA},
            'source_selection': {'inventory': {'path': '/immutable/source-selection-inventory.json',
                                               'sha256': d.SOURCE_INVENTORY_SHA},
                                 'terminal': copy.deepcopy(d.SOURCE_SCORE_TERMINAL)},
            'endpoints': [{'seed': seed, 'arm': arm,
                           'launch': {'path': f'/train/{arm}-{seed}/launch.json', 'sha256': 'a' * 64},
                           'checkpoint': {'path': f'/train/{arm}-{seed}/resume.pt', 'sha256': 'b' * 64},
                           'terminal_state_sha256': 'c' * 64, 'terminal': terminal(f'/train/{arm}-{seed}')}
                          for seed, arm in d.endpoint_order(stage)],
            'stage': stage, 'panel': panel,
            'first_selection': None if stage == 'first' else terminal('/first-score'),
            'selection_go': terminal('/full-score') if panel == 'validation' else None,
            'resource_policies': {p: d.policy(p) for p in ('cpu', 'score')},
            'cost_policy': copy.deepcopy(d.COST_POLICY), 'both_locks_held': True,
            'selection_previously_exposed': True, 'validation_previously_exposed': False}


def authority_checks(d):
    args = SimpleNamespace(execution_sha256='d' * 64)
    for stage, panel in (('first', 'selection'), ('full', 'selection'), ('full', 'validation')):
        spec = spec_fixture(d, stage, panel)
        d.check_spec(spec, args)
        for mutation in (
            lambda x: x.update(extra=True), lambda x: x.update(both_locks_held=False),
            lambda x: x.update(selection_previously_exposed=False),
            lambda x: x.update(validation_previously_exposed=True),
            lambda x: x['resource_policies']['cpu'].update(seconds=121),
            lambda x: x['resource_policies']['score'].update(host_bytes=9 * 1024**3),
            lambda x: x['cost_policy'].update(median_update_ratio_max=1.51),
            lambda x: x['training']['code'].pop('quadratic_readout.py'),
            lambda x: x['training']['code'].pop('quadratic_encoder_frames.py'),
            lambda x: x['training']['code'].update({'unexpected.py': 'f' * 64}),
            lambda x: x['training'].update(execution_sha256='future'),
            lambda x: x['training'].update(root='relative'),
            lambda x: x['evaluation_reference'].update(execution_sha256='0' * 64),
            lambda x: x['partition'].update(sha256='0' * 64),
            lambda x: x['source_selection']['inventory'].update(sha256='0' * 64),
            lambda x: x['source_selection']['terminal']['receipt'].update(sha256='0' * 64),
            lambda x: x['endpoints'].reverse(), lambda x: x['endpoints'].pop(),
            lambda x: x['endpoints'][0].update(seed=True),
            lambda x: x['endpoints'][0].update(terminal_state_sha256='future'),
            lambda x: x['endpoints'][0]['terminal'].update(both_locks_held=False),
            lambda x: x['endpoints'][0]['terminal'].update(service_seconds=float('nan')),
        ):
            changed = copy.deepcopy(spec); mutation(changed)
            rejects(lambda: d.check_spec(changed, args))
    for spec in (spec_fixture(d, 'first', 'validation'),
                 dict(spec_fixture(d, 'full', 'validation'), selection_go=None),
                 dict(spec_fixture(d, 'full'), first_selection=None)):
        rejects(lambda: d.check_spec(spec, args))
    assert d.endpoint_order('full') == ((179061, 'control'), (179061, 'candidate'),
                                       (179069, 'candidate'), (179069, 'control'))
    current, prior = spec_fixture(d, 'full', 'validation'), spec_fixture(d, 'full')
    d.check_prior_binding(current, prior, 'full')
    for mutation in (lambda x: x['endpoints'][3]['checkpoint'].update(sha256='0' * 64),
                     lambda x: x['source_selection']['inventory'].update(sha256='0' * 64),
                     lambda x: x['training']['code'].update({'quadratic_readout.py': '0' * 64}),
                     lambda x: x.update(first_selection=terminal('/another-first')),
                     lambda x: x.update(panel='validation')):
        changed = copy.deepcopy(prior); mutation(changed)
        rejects(lambda: d.check_prior_binding(current, changed, 'full'))


def metric(hits, ap, count):
    return {'per_query_r1': [1] * hits + [0] * (count - hits), 'per_query_ap': [ap] * count,
            'recall_at_1': hits / count, 'map_at_r': ap}


def metric_checks(d):
    source = metric(1670, .8057229533585297, 1734)
    quality = {str(s): {'control': metric(1671, .806, 1734),
                        'candidate': metric(1680, .815, 1734)} for s in d.SEEDS}
    costs = {str(s): {'pass': True} for s in d.SEEDS}
    deltas, average = d.averaged_deltas(quality, 'full', 'selection')
    intervals = {m: {'mean_delta': sum(average[m]) / 1734, 'product_lower95': .001,
                     'product_upper95': .02, 'query_lower95': -.001, 'query_upper95': .02} for m in d.METRICS}
    result = d.decide(quality, source, 'full', 'selection', intervals, costs)
    assert result['decision'] == 'GO' and result['source_floor_pass']
    first = {'179061': quality['179061']}
    assert d.decide(first, source, 'first', 'selection', {}, {'179061': {'pass': True}})['decision'] == 'CONTINUE'
    for hits, ap in ((1671, .806), (1680, .805), (1669, .815)):
        changed = copy.deepcopy(first); changed['179061']['candidate'] = metric(hits, ap, 1734)
        assert d.decide(changed, source, 'first', 'selection', {}, {'179061': {'pass': True}})['decision'] == 'KILL'
    weak = {'179061': {'control': metric(1660, .79, 1734), 'candidate': metric(1669, .8, 1734)}}
    assert d.decide(weak, source, 'first', 'selection', {}, {'179061': {'pass': True}})['decision'] == 'KILL'
    assert d.decide(first, source, 'first', 'selection', {}, {'179061': {'pass': False}})['decision'] == 'KILL'
    # A validation floor belongs to validation: it can be higher OR lower than selection.
    validation = {str(s): {'control': metric(1600, .78, 1749),
                          'candidate': metric(1610, .79, 1749)} for s in d.SEEDS}
    vd, va = d.averaged_deltas(validation, 'full', 'validation')
    vi = {m: {**intervals[m], 'mean_delta': sum(va[m]) / 1749} for m in d.METRICS}
    assert d.decide(validation, metric(1590, .77, 1749), 'full', 'validation', vi, costs)['decision'] == 'GO'
    assert d.decide(validation, metric(1620, .8, 1749), 'full', 'validation', vi, costs)['decision'] == 'KILL'
    rejects(lambda: d.decide(validation, source, 'full', 'validation', vi, costs))
    for mutation in (lambda x: x['179061']['candidate']['per_query_r1'].pop(),
                     lambda x: x['179061']['candidate']['per_query_ap'].__setitem__(0, float('nan')),
                     lambda x: x['179061']['candidate']['per_query_r1'].__setitem__(0, .5),
                     lambda x: x['179061']['candidate'].update(recall_at_1=.9),
                     lambda x: x.pop('179069')):
        changed = copy.deepcopy(quality); mutation(changed)
        rejects(lambda: d.decide(changed, source, 'full', 'selection', intervals, costs))
    changed = copy.deepcopy(intervals); changed['per_query_ap']['product_lower95'] = 0
    assert d.decide(quality, source, 'full', 'selection', changed, costs)['decision'] == 'KILL'
    changed = copy.deepcopy(quality)
    changed['179069'] = {'control': metric(1660, .79, 1734), 'candidate': metric(1680, .8, 1734)}
    _, new_average = d.averaged_deltas(changed, 'full', 'selection')
    new_intervals = {m: {**intervals[m], 'mean_delta': sum(new_average[m]) / 1734} for m in d.METRICS}
    result = d.decide(changed, source, 'full', 'selection', new_intervals, costs)
    assert result['each_seed_quality_pass'] and result['decision'] == 'KILL'
    assert result['source_floor_per_seed'] == {'179061': True, '179069': False}
    rejects(lambda: d.decide(first, source, 'first', 'selection', intervals, costs))
    d.replay_equal(source, copy.deepcopy(source))
    for mutation in (lambda x: x['per_query_r1'].__setitem__(0, 0),
                     lambda x: x['per_query_ap'].__setitem__(0, .1),
                     lambda x: x.update(map_at_r=.1)):
        changed = copy.deepcopy(source); mutation(changed)
        rejects(lambda: d.replay_equal(source, changed))


def file_checks(d, root):
    path = root / 'wire'; path.write_bytes(b'original')
    desc = file_ref(path); guards = {}
    assert d.bound_file(guards, path, desc['sha256']) == path
    d.packed_readback(guards, path, desc['sha256'], b'original')
    rejects(lambda: d.packed_readback(guards, path, desc['sha256'], b'other'))
    path.write_bytes(b'mutated!')  # Same length; a cached digest must not pass.
    rejects(lambda: d.bound_file(guards, path, desc['sha256']))
    path.write_bytes(b'original')
    rejects(lambda: d.bound_file({str(path): '0' * 64}, path, desc['sha256']))
    alias = root / 'alias'; alias.symlink_to(path)
    rejects(lambda: d.bound_file({}, alias, desc['sha256']))
    rejects(lambda: d.strict_json('{"x":1,"x":2}'))
    rejects(lambda: d.strict_json('{"x":NaN}'))
    closure = root / 'closure'; closure.mkdir()
    code = {}
    for name in ('train_siglip2_quadratic_readout.py', 'test_siglip2_quadratic_readout.py',
                 'quadratic_readout.py', 'quadratic_encoder_frames.py'):
        member = closure / name; member.write_bytes(name.encode()); code[name] = file_ref(member)['sha256']
    execution = closure / 'execution.json'; execution.write_text(json.dumps(code))
    assert d.closure(closure, file_ref(execution)['sha256'], d.TRAIN_FILES, {}) == code
    for name in code:
        incomplete = dict(code); incomplete.pop(name); execution.write_text(json.dumps(incomplete))
        rejects(lambda: d.closure(closure, file_ref(execution)['sha256'], d.TRAIN_FILES, {}))
    d.check_output(root / 'new-output')
    rejects(lambda: d.check_output(path))


def cost_checks(d):
    identity = {k: 'a' * 64 for k in ('encoder_sha256', 'complement_sha256', 'buffers_sha256',
                 'head_buffers_sha256', 'means_sha256', 'static_sha256', 'warm_members_sha256',
                 'schedule_sha256', 'full_schedule_sha256')}
    step = {'step': 1, 'batch': list(range(64)), 'schedule_sha256': 'a' * 64,
            'feature_rows_sha256': 'b' * 64, 'mask_sha256': 'c' * 64}
    records = {(179061, arm): {'identity': identity.copy(), 'steps': [copy.deepcopy(step)],
                'service_seconds': 100 if arm == 'control' else 150,
                'median_update_seconds': 1 if arm == 'control' else 1.5,
                'training_wall_seconds': 50 if arm == 'control' else 90} for arm in d.ARMS}
    result = d.paired_cost(records, 'first')['179061']
    assert result['pass'] and result['training_wall_ratio'] == 1.8
    changed = copy.deepcopy(records); changed[179061, 'candidate']['service_seconds'] = 150.001
    assert not d.paired_cost(changed, 'first')['179061']['pass']
    changed = copy.deepcopy(records); changed[179061, 'candidate']['median_update_seconds'] = 1.50001
    assert not d.paired_cost(changed, 'first')['179061']['pass']
    for mutation in (lambda x: x[179061, 'candidate']['identity'].update(means_sha256='0' * 64),
                     lambda x: x[179061, 'candidate']['steps'][0]['batch'].reverse(),
                     lambda x: x[179061, 'candidate'].update(service_seconds=float('inf')),
                     lambda x: x.pop((179061, 'control'))):
        changed = copy.deepcopy(records); mutation(changed)
        rejects(lambda: d.paired_cost(changed, 'first'))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return file_ref(path)


def context_fixture(d, original, initializer, helper, root, stage='first'):
    spec = spec_fixture(d, stage)
    records = {}
    for seed, arm in d.endpoint_order(stage):
        identity = {k: 'a' * 64 for k in ('encoder_sha256', 'complement_sha256', 'buffers_sha256',
                    'head_buffers_sha256', 'means_sha256', 'features_sha256', 'static_sha256', 'warm_members_sha256',
                    'schedule_sha256', 'full_schedule_sha256')}
        records[seed, arm] = {'identity': identity, 'steps': [{'step': 1, 'batch': list(range(64)),
            'schedule_sha256': 'a' * 64, 'feature_rows_sha256': 'b' * 64, 'mask_sha256': 'c' * 64}],
            'service_seconds': 100 if arm == 'control' else 120,
            'median_update_seconds': 1 if arm == 'control' else 1.2, 'training_wall_seconds': 90}
    old = {'source_cpu': {'numerical_flags': {}, 'invocation': {'python': '/qualified/python',
                'python_sha256': 'a' * 64, 'python_version': 'fixture'}},
           'launch': {'selected_export': {'service_seconds': 283.636}},
           'export_record': {'extraction_seconds': 250., 'wall_seconds': 280.}}
    admission = original.FlatAdmission(); admission.init = initializer
    return {'args': SimpleNamespace(execution_sha256='d' * 64, authority=root / 'authority.json', authority_sha256='a' * 64),
            'spec': spec, 'records': records, 'selected': {'source': {}, 'selected': old}, 'code': {},
            'source_record': {'quality': {'179061': {'control': metric(1670, .8057229533585297, 1734)}}},
            'admission': admission, 'helper': helper, 'guards': {}, 'required_guards': {},
            'base_guards': {}, 'terminals': [], 'origin_records': []}


def terminal_fixture(d, context, output, phase, cpu_terminal=None):
    output.mkdir()
    args = context['args']
    if not args.authority.is_file() or json.loads(args.authority.read_bytes()) != context['spec']:
        authority = write_json(output / 'authority.json', context['spec'])
        args.authority, args.authority_sha256 = Path(authority['path']), authority['sha256']
    authority = file_ref(args.authority)
    unit, identity = output.name, hashlib.md5(str(output).encode(), usedforsecurity=False).hexdigest()
    cgroup = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
        'memory.max': str(8 * 1024**3), 'memory.current': '1024', 'memory.peak': '2048',
        'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
        'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}}
    prior = None if cpu_terminal is None else write_json(output / 'cpu-terminal.json', cpu_terminal)
    facts, witnesses = {}, {}
    for endpoint in context['spec']['endpoints']:
        key = d.label(endpoint); ident = context['records'][endpoint['seed'], endpoint['arm']]['identity']
        facts[key] = {'payload_sha256': endpoint['terminal_state_sha256'], 'readout_sha256': 'a' * 64,
                      **{k: ident[k] for k in ('encoder_sha256', 'complement_sha256', 'means_sha256', 'features_sha256')}}
        witnesses[key] = {k: {'shape': shape, 'dtype': dtype, 'sha256': 'a' * 64} for k, shape, dtype in
            (('raw', [64, 128], 'torch.float32'), ('unit', [64, 128], 'torch.float32'),
             ('codes', [64, 128], 'torch.int8'), ('inverse_norms', [64], 'torch.float16'))}
    record = {**d.bind(context), 'schema': d.SCHEMA, 'phase': phase, 'resource_policy': d.policy(phase),
              'optimizer_updates': 0, 'optimizer_members': 1, 'certificate': 'updated cached readout composed with qualified immutable encoder',
              'quality_read': phase == 'score', 'numerical_flags': {}, 'authority': authority,
              'output': str(output), 'prerequisite': prior, 'cpu_terminal': cpu_terminal,
              'peak_cuda_allocated_bytes': 0, 'files': {}, 'input_guards': {}, 'origins': {'packages': {}, 'files': {}},
              'wall_seconds': 10., 'process_peak_rss_kib': 1000, 'cgroup_before': cgroup, 'cgroup_after': copy.deepcopy(cgroup),
              'head_facts': facts, 'train_witnesses': witnesses, 'validation_quality_exposed': False,
              'invocation': {'invocation_id': identity, 'optimize': 0, 'cuda_visible_devices': '',
                  'python': '/qualified/python', 'python_sha256': 'a' * 64, 'python_version': 'fixture',
                  'argv': d.cli_argv(args.authority, args.authority_sha256, args.execution_sha256, phase, output, prior)},
              **dict.fromkeys(('pass', 'engineering_admission_pass', 'strict_independent_head_reload_exact',
                  'train_raw_unit_cpu_packed_exact', 'complete_typed_terminal_state_exact', 'first_heads_released_before_reload',
                  'rng_flags_preserved', 'exit_rehash_pass'), True),
              **dict.fromkeys(('official_read', 'global_production_goal_met', 'public_latency_measured',
                  'public_encoder_qualified', 'cuda_initialized'), False)}
    if phase == 'score':
        spec = context['spec']
        quality = {str(s): {'control': metric(1671, .806, 1734), 'candidate': metric(1680, .815, 1734)} for s in d.seeds(spec['stage'])}
        _, average = d.averaged_deltas(quality, spec['stage'], 'selection')
        intervals = {} if spec['stage'] == 'first' else {m: {'mean_delta': sum(average[m]) / 1734,
            'product_lower95': .001, 'product_upper95': .02, 'query_lower95': -.001, 'query_upper95': .02} for m in d.METRICS}
        cost = d.paired_cost(context['records'], spec['stage']); source = context['source_record']['quality']['179061']['control']
        record.update(**d.decide(quality, source, spec['stage'], 'selection', intervals, cost), quality=quality,
            source_quality=source, source_quality_panel='selection', source_selection_receipt=d.SOURCE_INVENTORY['receipt'],
            source_checkpoint=d.SOURCE_INVENTORY['baseline_endpoint']['checkpoint'],
            source_terminal_state_sha256=d.SOURCE_INVENTORY['baseline_endpoint']['terminal_state_sha256'],
            archived_selection_per_query_exact=True, cost=cost, cost_policy=d.COST_POLICY,
            paired_seed_average_intervals=intervals, full_panel_raw_unit_packed_replay_exact=True, per_query_replay_exact=True,
            bootstrap_draws=0 if spec['stage'] == 'first' else 5000, bootstrap_seed=179019,
            panel_images=3449, query_images=1734, gallery_images=1715, panel_products=498,
            canonical_panel_input_only=True, serving_view_averaging=False,
            preparation_costs=d.preparation_costs(context['selected']['selected']))
        panel_facts = {k: {'shape': [3449, 128] if k != 'inverse_norms' else [3449],
                         'dtype': v['dtype'], 'sha256': v['sha256']} for k, v in next(iter(witnesses.values())).items()}
        record['source_panel_facts'] = copy.deepcopy(panel_facts)
        record['panel_replay_facts'] = {k: copy.deepcopy(panel_facts) for k in facts}
        for name in d.file_names(spec) | d.source_file_names():
            path = output / name; path.write_bytes(b'synthetic contract wire, never native evidence')
            record['files'][name] = file_ref(path)['sha256']
    receipt = write_json(output / 'receipt.json', record)
    (output / 'unit.log').write_text('\n'.join([f'Running as unit: {unit}.service; invocation ID: {identity}',
        '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
        '\tSwaps: 0', 'Memory swap peak: 0B', '\tMaximum resident set size (kbytes): 1000', 'Service runtime: 12.0s',
        'FINAL_CGROUP ' + json.dumps({**cgroup, 'invocation_id': identity})]) + '\n')
    return {'receipt': receipt, 'log': file_ref(output / 'unit.log'), 'unit': unit, 'invocation_id': identity,
            'service_seconds': 12., 'native_peak_rss_kib': 1000, 'both_locks_held': True}, record


def selection_checks(d, root):
    sources = Path(__file__).resolve().parent
    original = load('_quadratic_original_check', sources / 'train_siglip2_substrate_adaptation.py')
    initializer = load('_quadratic_initializer_check', sources / 'initialize_siglip2_substrate_fit.py')
    helper = load('_quadratic_helper_check', sources / 'export_siglip2_substrate_adaptation.py')
    context = context_fixture(d, original, initializer, helper, root)
    d.panel_authority(context)
    cpu_terminal, cpu = terminal_fixture(d, context, root / 'first-cpu', 'cpu')
    first_terminal, first = terminal_fixture(d, context, root / 'first-score', 'score', cpu_terminal)
    d.check_receipt(context, first, 'score')
    full_spec = spec_fixture(d, 'full'); full_spec['endpoints'][:2] = context['spec']['endpoints']; full_spec['first_selection'] = first_terminal
    full_context = context_fixture(d, original, initializer, helper, root, 'full')
    full_context['spec'] = full_spec
    d.panel_authority(full_context)
    assert full_context['first_selection']['decision'] == 'CONTINUE'
    for mutation in (lambda x: x.update(decision='GO'), lambda x: x.update(archived_selection_per_query_exact=False),
                     lambda x: x.update(source_floor_pass=False), lambda x: x.update(optimizer_members=5),
                     lambda x: x.update(complete_typed_terminal_state_exact=False),
                     lambda x: x['head_facts']['control-179061'].update(payload_sha256='0' * 64),
                     lambda x: x['train_witnesses']['control-179061'].pop('inverse_norms'),
                     lambda x: x['source_panel_facts'].pop('codes'),
                     lambda x: x['panel_replay_facts']['control-179061']['raw'].update(shape=[1734, 128]),
                     lambda x: x['source_quality']['per_query_ap'].__setitem__(0, .1),
                     lambda x: x['spec']['endpoints'][0]['checkpoint'].update(sha256='0' * 64)):
        changed = copy.deepcopy(first); mutation(changed)
        ref = write_json(Path(first_terminal['receipt']['path']), changed)
        rejects(lambda: d.accept_selection({**full_context, 'guards': {}}, dict(first_terminal, receipt=ref), 'first'))
    write_json(Path(first_terminal['receipt']['path']), first)
    full_context['guards'] = {}
    full_cpu_terminal, _ = terminal_fixture(d, full_context, root / 'full-cpu', 'cpu')
    full_terminal, full = terminal_fixture(d, full_context, root / 'full-score', 'score', full_cpu_terminal)
    validation = {**full_context, 'guards': {}, 'spec': copy.deepcopy(full_spec)}
    validation['spec'].update(panel='validation', selection_go=full_terminal)
    d.panel_authority(validation)
    assert validation['selection_go']['decision'] == 'GO'
    for key in ('quality_pass', 'cost_pass', 'source_floor_pass', 'per_query_replay_exact'):
        changed = copy.deepcopy(full); changed[key] = False
        ref = write_json(Path(full_terminal['receipt']['path']), changed)
        spec = copy.deepcopy(validation['spec']); spec['selection_go']['receipt'] = ref
        rejects(lambda: d.panel_authority({**validation, 'spec': spec, 'guards': {}}))
    write_json(Path(full_terminal['receipt']['path']), full)
    bad = copy.deepcopy(validation['spec']); bad['selection_go'] = None
    rejects(lambda: d.panel_authority({**validation, 'spec': bad}), 'selection GO')
    bad = copy.deepcopy(validation['spec']); bad['endpoints'][-1]['checkpoint']['sha256'] = '0' * 64
    rejects(lambda: d.panel_authority({**validation, 'spec': bad, 'guards': {}}), 'selection authority/checkpoints')
    log = Path(full_terminal['log']['path']); log.write_text(log.read_text().replace('\tExit status: 0', '\tExit status: 1'))
    bad = copy.deepcopy(validation['spec']); bad['selection_go']['log'] = file_ref(log)
    fresh_admission = original.FlatAdmission(); fresh_admission.init = initializer
    rejects(lambda: d.panel_authority({**validation, 'spec': bad, 'guards': {}, 'admission': fresh_admission}), 'normal-exit log')
    missing = {**full_context, 'guards': {}, 'base_guards': {'/missing-source': 'a' * 64}}
    rejects(lambda: d.accept_selection(missing, first_terminal, 'first'), 'complete evaluator input guards')


def source_checks(d):
    root = Path(__file__).resolve().parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    record = json.loads((root / 'genuine-view-v1/evaluation-source-v4/first-score-receipt.json').read_bytes())
    inventory = root / 'quadratic-readout-v1/source-selection-inventory.json'
    assert file_ref(inventory)['sha256'] == d.SOURCE_INVENTORY_SHA
    assert json.loads(inventory.read_bytes()) == d.SOURCE_INVENTORY
    selected = {'selected': {'source': copy.deepcopy(record['source'])}}
    d.check_source_record(record, selected)
    for mutation in (lambda x: x.update(engineering_admission_pass=False),
                     lambda x: x.update(per_query_replay_exact=False),
                     lambda x: x['spec']['endpoints'][0]['checkpoint'].update(sha256='0' * 64),
                     lambda x: x['spec'].update(panel='validation'),
                     lambda x: x['files'].pop('control-179061.raw.npy'),
                     lambda x: x['files'].update({'control-179061.unit.npy': '0' * 64}),
                     lambda x: x['quality']['179061']['control']['per_query_r1'].pop(),
                     lambda x: x['quality']['179061']['control'].update(map_at_r=.5)):
        changed = copy.deepcopy(record); mutation(changed)
        rejects(lambda: d.check_source_record(changed, selected))


def typed_digest(value, consumed=None):
    """Fake tensor bytes keep stdlib boundary tests separate from native math."""
    def typed(item):
        if isinstance(item, dict):
            return ('dict', tuple((typed(k), typed(v)) for k, v in sorted(item.items(), key=lambda x: repr(x[0]))))
        if isinstance(item, (list, tuple)):
            return (type(item).__name__, tuple(typed(v) for v in item))
        if hasattr(item, 'shape') and hasattr(item, 'dtype'):
            return ('fixture-tensor', tuple(item.shape), str(item.dtype), getattr(item, 'value', None))
        return (type(item).__name__, repr(item))
    return hashlib.sha256(repr(typed(value)).encode()).hexdigest()


def saved_checks(d, trainer_root):
    t = load('_quadratic_trainer_metadata_check', trainer_root / 'train_siglip2_quadratic_readout.py')
    fixture_module = load('_quadratic_trainer_fixture_check', trainer_root / 'test_siglip2_quadratic_readout.py')
    fixture_module.ContractTests.driver = t
    fixture = fixture_module.ContractTests()
    original = load('_quadratic_original_check', trainer_root / 'train_siglip2_substrate_adaptation.py')
    frames = load('_quadratic_frames_check', trainer_root / 'quadratic_encoder_frames.py')
    for arm in d.ARMS:
        saved, ident = fixture.fake_payload(arm, 1000)
        ident['device'] = 'cuda'; saved['cuda_rng'] = [fixture.tensor((5000,), 'torch.uint8')]
        features = fixture.tensor((6355, 1152))
        ident['static_sha256'] = typed_digest({k: saved[k] for k in t.STATIC_KEYS})
        ident['complement_sha256'] = typed_digest({k: saved[k] for k in ('head', 'classifier')})
        ident['means_sha256'] = typed_digest(saved['means']); ident['encoder_sha256'] = typed_digest(saved['encoder'])
        ident['buffers_sha256'] = typed_digest(saved['buffers'])
        ident['head_buffers_sha256'] = typed_digest({k: saved['head'][k] for k in ('center', 'preactivation_std')})
        ident['features_sha256'] = typed_digest(features)
        frozen = {k: saved[k] for k in ('encoder', 'config', 'buffers', 'head', 'classifier', 'means', *t.STATIC_KEYS)}
        frozen['features'] = features; ident['frozen_sha256'] = typed_digest(frozen)
        endpoint = {'seed': 179061, 'arm': arm}
        selected = {'original': SimpleNamespace(fingerprint=typed_digest),
                    'initial': copy.deepcopy({k: saved[k] for k in ('head', 'classifier')}),
                    'static_sha256': ident['static_sha256'], 'terminals': {'cpu:control': {'arms': {
                        arm: {'identity': {'means_sha256': ident['means_sha256']}}}}}}
        selected['encoder'], selected['encoder_fingerprint'], selected['encoder_check'] = frames.seal(
            original.fingerprint, saved['encoder'])
        context = {'trainer': t, 'selected': selected, 'records': {(179061, arm): {'identity': copy.deepcopy(ident)}},
                   'partition': copy.deepcopy(saved['partition']), 'features': features,
                   'feature_state_sha256': typed_digest(features),
                   'typed_encoder': copy.deepcopy((saved['config'], saved['buffers']))}
        d.check_saved_metadata(context, saved, endpoint)
        endpoint['terminal_state_sha256'] = typed_digest(saved)
        d.check_complete_payload(context, saved, endpoint)
        assert type(saved['encoder']) is dict and saved['encoder'] == t.owned_encoder(selected).materialize()
        foreign, _, _ = frames.seal(original.fingerprint, saved['encoder'])
        rejects(lambda: d.check_saved_metadata({**context, 'selected': {**selected, 'encoder': foreign}}, saved, endpoint),
                'foreign')
        for key in ('A', 'bank', 'cpu_rng', 'cuda_rng'):
            changed = copy.deepcopy(saved)
            value = changed[key][0] if key == 'cuda_rng' else changed[key]
            value.value = 'same-shape byte mutation'
            rejects(lambda: d.check_complete_payload(context, changed, endpoint), 'complete serialized typed state')
        for key in saved:
            changed = copy.deepcopy(saved); changed.pop(key)
            rejects(lambda: d.check_complete_payload(context, changed, endpoint))
        for mutation in (lambda x: x['means']['linear'].__setattr__('value', 'same-version-mutation'),
                         lambda x: x['head']['primary.weight'].__setattr__('value', 'changed'),
                         lambda x: x['buffers']['embeddings.position_ids'].__setattr__('value', 'changed'),
                         lambda x: x['classifier'].__setattr__('value', 'changed'),
                         lambda x: x['optimizer']['state'][0]['step'].__setattr__('value', 999),
                         lambda x: x['scaler'].update(_growth_tracker=999),
                         lambda x: x['encoder']['inventory'].pop(),
                         lambda x: x['identity'].update(device='cpu'),
                         lambda x: x.update(counter=999),
                         lambda x: x['schedules'].pop('179069'),
                         lambda x: x['views'].update(ordered_input_sha256='0' * 64)):
            changed = copy.deepcopy(saved); mutation(changed)
            rejects(lambda: d.check_saved_metadata(context, changed, endpoint))
        wrong_features = {**context, 'feature_state_sha256': '0' * 64}
        rejects(lambda: d.check_saved_metadata(wrong_features, saved, endpoint))
        # JSON's tuple->list conversion cannot satisfy the typed config boundary.
        config = {**context, 'typed_encoder': ({'changed': ()}, context['typed_encoder'][1])}
        rejects(lambda: d.check_saved_metadata(config, saved, endpoint), 'typed config')
    endpoint_checks(d, t, fixture)
    encoder_checks(d, t, fixture, original, frames)


def encoder_checks(d, t, fixture, original, frames):
    # Execute the original metadata serializer with only its native import removed.
    # No tensors enter this check; production fingerprints remain untouched.
    node = next(n for n in ast.parse(Path(original.__file__).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    node.body = [n for n in node.body if not isinstance(n, ast.Import)]
    namespace = dict(vars(original), torch=SimpleNamespace(Tensor=()))
    exec(compile(ast.Module(body=[node], type_ignores=[]), original.__file__, 'exec'), namespace)
    metadata = SimpleNamespace(fingerprint=namespace['fingerprint'], FlatAdmission=original.FlatAdmission)
    with TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / 'vision.pt'; checkpoint.write_bytes(b'structural fixture only')
        def selected_fixture():
            encoder = fixture.encoder()
            encoder['checkpoint']['path'] = str(checkpoint)
            proof = encoder['source_proof']
            proof['typed_witness'] = {0: ('x', 1), '0': ['x', 1]}
            proof['input_guards'] = {str(checkpoint): encoder['checkpoint']['sha256']}
            prior = dict(expected={n: v['shape'] for n, v in proof['runtime']['vision'].items()},
                         mapping=copy.deepcopy(proof['runtime']['vision']), guards=dict(proof['input_guards']))
            record = dict(source_checkpoint=copy.deepcopy(encoder['checkpoint']), binding=encoder['export_binding'],
                source_runtime=encoder['export_runtime'], strict_independent_reload_exact=True,
                constructor_and_view_rng_preserved=True, caches={'canonical': {'sha256': t.CANONICAL_SHA}})
            selected = dict(source_cpu=proof, export_record=record,
                genuine=dict(reference=SimpleNamespace(binding=lambda prior: copy.deepcopy(encoder['source_binding']))),
                exporter=SimpleNamespace(binding=lambda genuine: copy.deepcopy(encoder['export_binding'])),
                launch=dict(selected_export=encoder['export_terminal']),
                source=dict(source_checkpoint=copy.deepcopy(encoder['checkpoint']), caches=copy.deepcopy(record['caches'])),
                guards=dict(proof['input_guards']))
            context = dict(selected=selected, prior=prior, guards=dict(proof['input_guards']),
                           admission=original.FlatAdmission(), original=metadata, frames=frames)
            context['encoder'] = t.composition(context)
            return context
        selected = selected_fixture()
        extra = Path(temporary) / 'extra'; extra.write_bytes(b'new structural guard')
        selected['selected']['guards'][str(extra)] = file_ref(extra)['sha256']
        capsule, check, fingerprint = (selected[k] for k in ('encoder', 'encoder_check', 'encoder_fingerprint'))
        guards, admission = copy.deepcopy(selected['guards']), copy.deepcopy(vars(selected['admission']))
        d.check_exit_encoder({'trainer': t, 'selected': selected})
        assert t.owned_encoder(selected) is capsule and selected['encoder_check'] is check
        assert selected['encoder_fingerprint'] is fingerprint and selected['guards'] == guards
        assert vars(selected['admission']) == admission
        foreign, _, _ = frames.seal(original.fingerprint, capsule.materialize())
        rejects(lambda: d.check_exit_encoder({'trainer': t, 'selected': {**selected, 'encoder': foreign}}), 'foreign')
        # Equal Python values with different types must still fail the full typed hash.
        selected['selected']['source_cpu']['typed_witness'][0] = ('x', True)
        assert selected['selected']['source_cpu'] == capsule.materialize()['source_proof']
        rejects(lambda: d.check_exit_encoder({'trainer': t, 'selected': selected}), 'exit encoder composition')
        for mutate in (
            lambda c: c['selected']['export_record']['source_runtime']['roles'].pop(),
            lambda c: c['selected']['source_cpu'].update(reload_exact=False),
            lambda c: c['selected']['export_record'].update(strict_independent_reload_exact=False),
            lambda c: c['selected']['export_record']['caches']['canonical'].update(sha256='0' * 64),
            lambda c: c['prior']['expected'].pop('tensor.0'),
            lambda c: c['prior']['mapping']['tensor.0'].update(sha256='0' * 64),
            lambda c: c['guards'].update({str(checkpoint): '0' * 64}),
            lambda c: c['selected']['source_cpu']['input_guards'].update(missing='0' * 64)):
            changed = selected_fixture(); owned = t.owned_encoder(changed); mutate(changed)
            rejects(lambda: d.check_exit_encoder({'trainer': t, 'selected': changed}))
            assert t.owned_encoder(changed) is owned


def endpoint_checks(d, t, fixture):
    for seed, arm in d.ORDER:
        _, ident = fixture.fake_payload(arm, 1000)
        launch, _ = fixture.launch('train', arm, seed)
        control_launch, _ = fixture.launch('train', 'control', 179061)
        cpu_ident = copy.deepcopy(ident)
        cpu_ident['method'] = t.method(launch)
        cpu_ident['seed'], cpu_ident['device'] = 179061, 'cpu'
        ident = {**cpu_ident, 'seed': seed, 'device': 'cuda'}
        steps = [{'step': i, 'batch': list(range(64)), 'schedule_sha256': ident['schedule_sha256'],
                  'feature_rows_sha256': 'b' * 64, 'mask_sha256': 'c' * 64, 'clean_bank_sha256': 'e' * 64,
                  'state_sha256': 'd' * 64, 'ce': 1., 'rank': .125, 'loss': 2., 'scale': 128,
                  'preclip_norm': 1., 'gradient_norms': {'A': 1.}, 'A_updated': True, 'seconds': .001}
                 for i in range(1, 1001)]
        checkpoint = {'path': f'/train/{arm}-{seed}/resume.pt', 'sha256': 'a' * 64}
        record = {'schema': t.SCHEMA, 'phase': 'train', 'seed': seed, 'arm': arm, 'pass': True,
            'quality_read': False, 'strict_reload_exact': True, 'exit_rehash_pass': True,
            'frozen_complement_exact': True, 'sequential_model_ownership': True,
            'optimizer_members': 1, 'native_trainable_tensors': 0, 'trainable_scalars': 4096,
            'certificate': t.RECIPE['certificate'], 'public_encoder_qualified': False,
            'source_factory': 'control', 'canonical_updates_only': True, 'warm_source_updates': 1000,
            'local_initial_counter': 0, 'resource_policy': t.policy('train'), 'launch': launch,
            'invocation': {'optimize': 0, 'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8'},
            'completed_step': 1000, 'identity': ident, 'source': ident['source'], 'numerical_flags': {},
            'peak_cuda_allocated_bytes': 1000, 'steps': steps, 'terminal_state_sha256': 'd' * 64,
            'median_update_seconds': .001, 'training_wall_seconds': 2., 'wall_seconds': 3.,
            'training_state_discarded': False, 'resumed_steps': [], 'checkpoint': checkpoint,
            'input_guards': {checkpoint['path']: checkpoint['sha256']}, 'output': str(Path(checkpoint['path']).parent),
            'training_qualified': True, 'code': {}, 'terminal_cgroups': {}, 'partition_sha256': d.PARTITION_SHA,
            'initial_state_sha256': 'a' * 64, 'initial_raw_unit_packed_sha256': 'b' * 64, 'replay_exact': False}
        selected = {'launch': control_launch, 'source': ident['source'], 'code': {}, 'terminal_cgroups': {},
            'selected': {'source_cpu': {'numerical_flags': {}}}, 'terminals': {'cpu:control': {'arms': {arm: {'identity': cpu_ident}}},
                'mechanics:' + arm: {'initial_state_sha256': 'a' * 64, 'initial_raw_unit_packed_sha256': 'b' * 64,
                                     'steps': copy.deepcopy(steps[:17])}}}
        endpoint = {'seed': seed, 'arm': arm, 'checkpoint': checkpoint, 'terminal_state_sha256': 'd' * 64}
        d.check_endpoint(t, record, endpoint, selected)
        for mutation in (lambda x: x.update(optimizer_members=5), lambda x: x.update(completed_step=999),
                         lambda x: x.update(replay_exact=True), lambda x: x.update(public_encoder_qualified=True),
                         lambda x: x.update(source_factory='candidate'),
                         lambda x: x.update(canonical_updates_only=False), lambda x: x['steps'].pop(),
                         lambda x: x['steps'][0].update(A_updated=False),
                         lambda x: x['identity'].update(means_sha256='0' * 64),
                         lambda x: x['identity']['native_inventory'][0].update(role='trainable'),
                         lambda x: x['identity']['optimizer_serial_groups'][0]['params'].append(1)):
            changed = copy.deepcopy(record); mutation(changed)
            rejects(lambda: d.check_endpoint(t, changed, endpoint, selected))
        if seed == 179061:
            changed = copy.deepcopy(record); changed['steps'][0]['feature_rows_sha256'] = 'f' * 64
            rejects(lambda: d.check_endpoint(t, changed, endpoint, selected), 'first17')


def main():
    path = Path(__file__).with_name('evaluate_siglip2_quadratic_readout.py')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--trainer-root', type=Path, default=path.parent)
    args = parser.parse_args()
    assert path.is_file(), 'quadratic evaluator has not been implemented'
    spec = importlib.util.spec_from_file_location('_quadratic_evaluation_check', path)
    d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    authority_checks(d); metric_checks(d); cost_checks(d); source_checks(d)
    trainer_root = args.trainer_root.resolve()
    trainer_checked = (trainer_root / 'train_siglip2_quadratic_readout.py').is_file()
    if trainer_checked:
        saved_checks(d, trainer_root)
    elif args.trainer_root != path.parent:
        raise AssertionError('requested trainer source is missing')
    with TemporaryDirectory() as temporary:
        file_checks(d, Path(temporary))
    with TemporaryDirectory() as temporary:
        selection_checks(d, Path(temporary))
    for source in (path, Path(__file__)):
        ast.parse(source.read_bytes())
        for flag in ('-O', '-OO'):
            p = subprocess.run([sys.executable, flag, '-B', '-S', str(source), '--help'], capture_output=True, text=True)
            assert p.returncode != 0 and 'optimized mode is forbidden' in p.stderr
    p = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert p.returncode == 0 and '--prerequisite-sha256' in p.stdout
    assert not any(name.split('.')[0] in d.NATIVE for name in sys.modules)
    print('PASS: authority/order/prior-GO/source-floor/per-query-replay/metrics/cost/source-closure/omission/tamper/wire/optimized/help/syntax; '
          + f'separate trainer typed-state/endpoint/encoder-ownership checks={trainer_checked}; native/resource/quality UNRUN')


if __name__ == '__main__':
    old_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}:
            raise AssertionError('native import attempted by stdlib falsifier: ' + name)
        return old_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    main()
