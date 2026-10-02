#!/usr/bin/env python3
"""One bounded stdlib falsifier; synthetic arrays are never native evidence.

Run python3 -B -S scripts/test_siglip2_prototype_residual_evaluation.py.
No native imports, real cache reads, fitting, quality jobs or GPU execution.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def rejects(call, text=None):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        if text:
            assert text in str(error), (text, str(error))
        return
    raise AssertionError('invalid input accepted: ' + str(text))


def terminal(name):
    return {'receipt': {'path': '/' + name + '/receipt.json', 'sha256': 'a' * 64},
            'log': {'path': '/' + name + '/unit.log', 'sha256': 'b' * 64},
            'unit': name, 'invocation_id': hashlib.md5(name.encode()).hexdigest(),
            'service_seconds': 10., 'native_peak_rss_kib': 1000, 'both_locks_held': True}


def fixture(d, panel='selection'):
    return {'schema': d.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
            'training': {'root': '/fitter', 'execution_sha256': 'e' * 64,
                         'code': {name: 'f' * 64 for name in d.TRAIN_FILES}},
            'original_reference': copy.deepcopy(d.ORIGINAL_REFERENCE),
            'original_evaluator': {'root': '/original-evaluator', 'execution_sha256': 'e' * 64,
                                   'code': copy.deepcopy(d.EVALUATOR_PINS)},
            'evaluation_reference': copy.deepcopy(d.EVALUATION_REFERENCE),
            'partition': {'path': '/partition.json', 'sha256': d.PARTITION_SHA},
            'source_selection': {'inventory': {'path': '/inventory.json', 'sha256': d.SOURCE_INVENTORY_SHA},
                                 'terminal': copy.deepcopy(d.SOURCE_SCORE_TERMINAL)},
            'panel': panel, 'endpoints': [
                {'arm': arm, 'launch': {'path': '/' + arm + '/launch.json', 'sha256': 'a' * 64},
                 'terminal': terminal(arm), 'checkpoint': {'path': '/' + arm + '/resume.pt', 'sha256': 'b' * 64},
                 'terminal_state_sha256': 'c' * 64} for arm in d.ARMS],
            'selection_go': terminal('selection-go') if panel == 'validation' else None,
            'resource_policies': {p: d.policy(p) for p in ('cpu', 'score')},
            'cost_policy': copy.deepcopy(d.COST_POLICY), 'both_locks_held': True,
            'selection_previously_exposed': True, 'validation_previously_exposed': False}


def quality(r1, ap):
    return {'recall_at_1': sum(r1) / len(r1), 'map_at_r': sum(ap) / len(ap),
            'per_query_r1': r1, 'per_query_ap': ap}


def check(d):
    args = SimpleNamespace(execution_sha256='d' * 64)
    for panel in ('selection', 'validation'):
        spec = fixture(d, panel); d.check_spec(spec, args)
        for mutation in (
            lambda s: s.update(seed=179069), lambda s: s.update(stage='full'),
            lambda s: s.update(both_locks_held=False), lambda s: s.update(validation_previously_exposed=True),
            lambda s: s['training']['code'].update(extra='f' * 64),
            lambda s: s['original_evaluator']['code'].update(evaluate_siglip2_quadratic_readout='f' * 64),
            lambda s: s['original_reference'].update(execution_sha256='0' * 64),
            lambda s: s['source_selection']['terminal'].update(service_seconds=1.),
            lambda s: s['endpoints'].reverse(), lambda s: s['endpoints'][0].update(seed=179061),
            lambda s: s['endpoints'][0]['terminal'].update(service_seconds=float('nan')),
            lambda s: s['resource_policies']['score'].update(seconds=301),
            lambda s: s['cost_policy'].update(total_fit_core_ratio_max=1.51),
        ):
            bad = copy.deepcopy(spec); mutation(bad); rejects(lambda: d.check_spec(bad, args))
    rejects(lambda: d.check_spec(dict(fixture(d, 'validation'), selection_go=None), args))
    current, prior = fixture(d, 'validation'), fixture(d)
    d.check_prior_binding(current, prior)
    changed = copy.deepcopy(prior); changed['endpoints'][0]['checkpoint']['sha256'] = '0' * 64
    rejects(lambda: d.check_prior_binding(current, changed))

    count = 1734
    source = quality([1.] * (count - 10) + [0.] * 10, [.8] * count)
    linear = quality([1.] * (count - 20) + [0.] * 20, [.79] * count)
    quadratic = copy.deepcopy(source)
    arms = {'linear': linear, 'quadratic': quadratic}
    deltas = d.metric_deltas(arms, source, 'selection')
    intervals = {m: {'mean_delta': sum(deltas['quadratic_minus_linear'][m]) / count,
                     'product_lower95': .001, 'product_upper95': .1,
                     'query_lower95': .001, 'query_upper95': .1} for m in d.METRICS}
    costs = {'linear': {'service_seconds': 10., 'total_fit_core_seconds': 4.},
             'quadratic': {'service_seconds': 15., 'total_fit_core_seconds': 6.}}
    cost = d.paired_cost(costs)
    result = d.decide(arms, source, 'selection', intervals, cost)
    assert result['decision'] == 'GO' and result['selection_go_admits_validation_only']
    assert not result['global_production_goal_met'] and not result['quadratic_source_gain_both']
    assert set(result['deltas']) == {'quadratic_minus_linear', 'quadratic_minus_source', 'linear_minus_source'}
    assert all(x == 0 for x in result['deltas']['quadratic_minus_source']['per_query_r1'])
    assert d.immediate_quality_pass(arms, source, 'selection')
    rejects(lambda: d.replay_equal(source, dict(source, map_at_r=.8000000001)))
    changed = copy.deepcopy(source); changed['per_query_ap'][0] += 1e-12
    rejects(lambda: d.replay_equal(source, changed))
    for changed in (dict(arms, quadratic=linear), dict(arms, quadratic=quality(source['per_query_r1'], [.78] * count))):
        assert not d.immediate_quality_pass(changed, source, 'selection')
        assert d.decide(changed, source, 'selection', {}, cost)['decision'] == 'KILL'
    # Beating a degraded linear arm is insufficient when quadratic is below source.
    below_source = {'linear': quality(linear['per_query_r1'], [.78] * count),
                    'quadratic': quality(source['per_query_r1'], [.79] * count)}
    assert not d.immediate_quality_pass(below_source, source, 'selection')
    assert not d.decide(below_source, source, 'selection', {}, cost)['source_floor_pass']
    bad_ci = copy.deepcopy(intervals); bad_ci['per_query_ap']['product_lower95'] = 0
    assert d.decide(arms, source, 'selection', bad_ci, cost)['decision'] == 'KILL'
    bad_ci = copy.deepcopy(intervals); bad_ci['per_query_ap']['mean_delta'] = .02
    rejects(lambda: d.decide(arms, source, 'selection', bad_ci, cost))
    for key in ('service_seconds', 'total_fit_core_seconds'):
        bad = copy.deepcopy(costs); bad['quadratic'][key] *= 1.000001
        assert not d.paired_cost(bad)['pass']
        assert d.decide(arms, source, 'selection', intervals, d.paired_cost(bad))['decision'] == 'KILL'
        bad['linear'][key] = 0; rejects(lambda: d.paired_cost(bad))
    assert d.policy('cpu')['seconds'] == 120 and d.policy('score')['seconds'] == 300
    threshold = {'linear': quality([1.] * (count - 13) + [0.] * 13, [.79] * count), 'quadratic': quadratic}
    tiny = d.metric_deltas(threshold, source, 'selection')['quadratic_minus_linear']
    tiny_ci = {m: dict(intervals[m], mean_delta=sum(tiny[m]) / count) for m in d.METRICS}
    assert d.decide(threshold, source, 'selection', tiny_ci, cost)['decision'] == 'KILL'
    for bad in (dict(source, per_query_ap=source['per_query_ap'][:-1]),
                dict(source, per_query_r1=[True] * count), dict(source, map_at_r=float('nan'))):
        rejects(lambda: d.metric_deltas(dict(arms, quadratic=bad), source, 'selection'))

    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'bytes'; path.write_bytes(b'original')
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); guards = {}
        d.bound_file(guards, path, digest); path.write_bytes(b'changed!')
        rejects(lambda: d.bound_file(guards, path, digest), 'SHA256')
        rejects(lambda: d.strict_json('{"same":1,"same":2}'), 'duplicate')
        rejects(lambda: d.strict_json('{"bad":NaN}'), 'nonfinite')
    assert not any(name.split('.')[0] in d.NATIVE for name in sys.modules)
    sys.modules['torch'] = SimpleNamespace()
    try:
        rejects(lambda: d.authority(args), 'native imports preceded admission')
    finally:
        del sys.modules['torch']


def rng_binding_check(d, source):
    """Run the actual cross-unit and own-unit guards with synthetic payloads."""
    authority = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'authority')
    guards = [node for node in ast.walk(authority) if isinstance(node, ast.Expr) and
              isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
              node.value.func.id == 'require' and any(
                  isinstance(part, ast.Name) and part.id == 'new_cpu' or
                  isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name) and
                  part.value.id == 'record' and isinstance(part.slice, ast.Constant) and
                  part.slice.value == 'terminal_state_sha256' for part in ast.walk(node.value))]
    assert len(guards) == 2, 'own-unit binding and CPU fitted comparison required'
    function = ast.parse('def compare(accepted, record, endpoint, fitting, new_cpu):\n    pass').body[0]
    function.body = guards
    namespace = {'require': d.require, 'Path': Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 '<actual evaluator admission guards>', 'exec'), namespace)
    compare = namespace['compare']

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def fact(payload):
        output = digest(payload['output_witness'])
        return {'identity': {'source': payload['source'], 'arm': 'linear',
                            'frozen_sha256': digest(payload['source']),
                            'fitted_sha256': digest(payload['A']), 'output_witness_sha256': output},
                'fit_witness': payload['fit_witness'], 'output_witness_sha256': output,
                'terminal_state_sha256': digest(payload)}

    cpu_payload = {'source': {'sha256': 'a' * 64}, 'A': [1., 2.],
                   'fit_witness': {'lambda': .03125}, 'output_witness': {'raw': [3., 4.]},
                   'cpu_rng': [1, 2, 3]}
    fit_payload = copy.deepcopy(cpu_payload); fit_payload['cpu_rng'] = [4, 5, 6]
    cpu, fitted = fact(cpu_payload), fact(fit_payload)
    assert cpu['terminal_state_sha256'] != fitted['terminal_state_sha256']
    assert all(cpu[k] == fitted[k] for k in ('identity', 'fit_witness', 'output_witness_sha256'))
    endpoint = fixture(d)['endpoints'][0]
    endpoint['terminal_state_sha256'] = fitted['terminal_state_sha256']
    launch = {'selected_cpu': terminal('cpu')}
    namespace['launch'] = launch
    record = {**fitted, 'launch': launch, 'authority': endpoint['launch'],
              'authority_sha256': endpoint['launch']['sha256'], 'checkpoint': endpoint['checkpoint'],
              'output': '/linear'}
    fitting, new_cpu = {'launch': launch}, {'arms': {'linear': cpu}}
    compare(copy.deepcopy(record), record, endpoint, fitting, new_cpu)
    for key, value in (('A', [1., 3.]), ('source', {'sha256': 'b' * 64}),
                       ('fit_witness', {'lambda': .0625}), ('output_witness', {'raw': [3., 5.]})):
        changed = copy.deepcopy(fit_payload); changed[key] = value
        bad = {**record, **fact(changed)}
        own_endpoint = dict(endpoint, terminal_state_sha256=bad['terminal_state_sha256'])
        rejects(lambda: compare(copy.deepcopy(bad), bad, own_endpoint, fitting, new_cpu),
                'new CPU qualified exact fitted')
    bad = copy.deepcopy(record); bad['output_witness_sha256'] = '0' * 64
    rejects(lambda: compare(copy.deepcopy(bad), bad, endpoint, fitting, new_cpu), 'new CPU qualified exact fitted')
    bad = dict(record, terminal_state_sha256=cpu['terminal_state_sha256'])
    rejects(lambda: compare(copy.deepcopy(bad), bad, endpoint, fitting, new_cpu), 'original fit endpoint binding')
    bad_endpoint = dict(endpoint, terminal_state_sha256='0' * 64)
    rejects(lambda: compare(copy.deepcopy(record), record, bad_endpoint, fitting, new_cpu),
            'original fit endpoint binding')
    bad = dict(record, checkpoint=dict(record['checkpoint'], sha256='0' * 64))
    rejects(lambda: compare(copy.deepcopy(bad), bad, endpoint, fitting, new_cpu), 'original fit endpoint binding')
    rejects(lambda: compare(dict(record, terminal_state_sha256='0' * 64), record, endpoint, fitting, new_cpu),
            'new CPU qualified exact fitted')


def flow_check(d, source):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    forbidden = {'fresh', 'fit_prototype_residual', 'extract_solver', 'fit_ridge_stitch', 'transform'}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
                   for node in ast.walk(tree)), 'evaluator must never refit or add the ridge intercept'
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree)), 'immutable helpers cannot be rebound'
    head = ast.get_source_segment(source, functions['head_values'])
    assert 'fitter.raw_features(fitting, state, cache)' in head and 'target_mean' not in head
    assert 'normalize' not in head, 'direct FIT must retain original normalization arithmetic'
    authority = ast.get_source_segment(source, functions['authority'])
    assert 'baseline.admit_source_selection(context)' in authority, 'full original source admission is mandatory'
    assert 'fit_terminal_adapter(fitter, fitting)' in authority and (
        "fit_terminal(branch, endpoint['terminal'], 'fit', endpoint['arm'])" in authority), 'fit-only adapter required'
    assert authority.index('fitter.authority(') < authority.index('fit_terminal_adapter(')
    assert "for endpoint in spec['endpoints']:" in authority
    accept = ast.get_source_segment(source, functions['accept_terminal'])
    assert "context['admission'].admit_terminal(record, terminal" in accept, 'CPU/score reader stays original'
    fitter_source = Path(__file__).with_name('fit_siglip2_prototype_residual.py').read_text()
    fitter_tree = ast.parse(fitter_source)
    fitter_authority = next(n for n in fitter_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authority')
    assert "admit_terminal(context, launch['selected_cpu'], 'cpu', 'linear')" in (
        ast.get_source_segment(fitter_source, fitter_authority)), 'fitter CPU authority stays original'
    panel = ast.get_source_segment(source, functions['score_panel'])
    assert panel.index("context['costs']['pass']") < panel.index('baseline.replay_archived_source')
    assert panel.index('baseline.replay_archived_source') < panel.index('baseline.cache_rows') < panel.index('load_head(')
    assert panel.index("context['selection_go']['decision']") < panel.index('baseline.cache_rows')
    assert 'score_saved_wires' in panel and 'replay_equal(first, score_saved_wires' in panel
    assert 'fixed.bootstrap_lower(delta, groups)' in panel and '-fixed.bootstrap_lower(-delta, groups)' in panel
    run = ast.get_source_segment(source, functions['run'])
    assert "args.phase == 'score' and context['costs']['pass']" in run
    assert run.index('authority(args)') < run.index('native_start(context)') < run.index('qualify_heads(context)')
    for name, digest in d.EVALUATOR_PINS.items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    archive = Path(__file__).resolve().parents[1] / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7'
    for name, digest in d.ORIGINAL_PINS.items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((archive / 'execution.json').read_bytes()).hexdigest() == d.ORIGINAL_REFERENCE['execution_sha256']



def terminal_adapter_check(d):
    """Exercise authenticated original terminal predicates, without native work."""
    def load(name, filename):
        path = Path(__file__).with_name(filename).resolve()
        return d.load_bare(name, path, hashlib.sha256(path.read_bytes()).hexdigest())

    original = load('_ms_original_admission', 'train_siglip2_substrate_adaptation.py')
    initializer = load('_ms_original_initializer', 'initialize_siglip2_substrate_fit.py')
    fitter = load('_ms_original_fitter', 'fit_siglip2_prototype_residual.py')
    admission = original.FlatAdmission()
    admission.init = initializer
    original_methods = (original.FlatAdmission.admit_terminal, fitter.admit_terminal)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        guards = {original.__file__: hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()}
        context = {'legacy': {'original': original, 'admission': admission}, 'guards': guards,
                   'code': {Path(fitter.__file__).name: hashlib.sha256(Path(fitter.__file__).read_bytes()).hexdigest()}}
        adapted = d.fit_terminal_adapter(fitter, context)
        reader = adapted.__globals__['_fit_log_terminal']
        unit, identity = 'synthetic-fit', 'a' * 32
        group = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(8 * 1024**3), 'memory.current': '1024', 'memory.peak': '2048',
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}}
        record = {'invocation': {'invocation_id': identity, 'optimize': 0}, 'wall_seconds': 10.,
                  'process_peak_rss_kib': 1000, 'cgroup_before': group, 'cgroup_after': copy.deepcopy(group)}
        footer = dict(copy.deepcopy(group), invocation_id=identity)
        text = '\n'.join([f'Running as unit: {unit}.service; invocation ID: {identity}', '\tExit status: 0',
            'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
            '\tSwaps: 0', 'Memory swap peak: 0B', '\tMaximum resident set size (kbytes): 1000',
            'Service runtime: 3min 745ms', 'FINAL_CGROUP ' + json.dumps(footer)]) + '\n'
        log = root / 'terminal.log'

        def descriptor(value=text, duration=180.745):
            nonlocal admission
            admission = original.FlatAdmission()
            admission.init = initializer
            log.write_text(value)
            return dict(terminal(unit), log={'path': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()},
                        invocation_id=identity, service_seconds=duration)

        desc = descriptor()
        rejects(lambda: admission.admit_terminal(record, desc, 300, {}), 'runtime format')
        assert reader(admission, record, desc, 300, {}) == footer
        for value in ('180.745s', '3min 0.745s', '3min 745ms'):
            desc = descriptor(text.replace('3min 745ms', value))
            assert reader(admission, record, desc, 300, {}) == footer
            if not value.endswith('ms'):
                assert admission.admit_terminal(record, desc, 300, {}) == footer
        for value, duration, message in (
            ('3min 744ms', 180.745, 'numeric binding'),
            ('3min 60s', 240., 'numeric binding'), ('3min 60000ms', 240., 'numeric binding'),
            ('745ms', .745, 'format'), ('3min 0.745ms', 180.000745, 'format'),
            ('3min -1ms', 179.999, 'format'), ('3min 745 ms', 180.745, 'format'),
            ('3min 745ms junk', 180.745, 'format'), (' 3min 745ms', 180.745, 'format')):
            desc = descriptor(text.replace('3min 745ms', value), duration)
            # Keep the earlier service cap predicate satisfied for parser mutants.
            candidate = dict(record, wall_seconds=min(10., duration / 2))
            rejects(lambda: reader(admission, candidate, desc, 300, {}), message)
        for value, message in (
            (text + 'Service runtime: 3min 745ms\n', 'runtime line'),
            (text + 'FINAL_CGROUP ' + json.dumps(footer) + '\n', 'final cgroup footer'),
            (text.replace(identity, 'b' * 32), 'normal-exit log'),
            (text.replace(unit + '.service', 'wrong.service'), 'normal-exit log'),
            (text.replace('\tExit status: 0', '\tExit status: 1'), 'normal-exit log'),
            (text + '\tSwaps: 0\n', 'normal-exit log'),
            (text.replace('"invocation_id": "' + identity, '"invocation_id": "' + 'b' * 32), 'footer'),
            (text.replace('"memory.swap.peak": "0"', '"memory.swap.peak": "1"'), 'memory/swap'),
            (text.replace('"memory.peak": "2048"', '"memory.peak": "1024"'), 'whole-unit peak')):
            desc = descriptor(value)
            rejects(lambda: reader(admission, record, desc, 300, {}), message)
        desc = descriptor(text.replace('3min 745ms', '2min 59999ms'), 179.999)
        assert reader(admission, record, desc, 300, {}) == footer
        desc = descriptor()
        rejects(lambda: reader(admission, record, desc, 180, {}), 'duration/RSS')
        rejects(lambda: reader(admission, record, dict(desc, both_locks_held=False), 300, {}), 'both locks')
        rejects(lambda: reader(admission, record, dict(desc, unit='wrong'), 300, {}), 'normal-exit')
        rejects(lambda: reader(admission, record, dict(desc, log=dict(desc['log'], sha256='0' * 64)), 300, {}), 'SHA256')
        rejects(lambda: reader(admission, record, desc, 300, {str(log): '0' * 64}), 'conflicting')
        bad = copy.deepcopy(record); bad['invocation']['optimize'] = 1
        rejects(lambda: reader(admission, bad, desc, 300, {}), 'invocation')
        # Execute the complete original fitter admission, including final guard rehash.
        old = load('_ms_original_quadratic', 'train_siglip2_quadratic_readout.py')
        def write_json(path, value):
            path.write_text(json.dumps(value))
            return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

        cpu = terminal('new-cpu')
        common = {'schema': fitter.AUTHORITY_SCHEMA, 'execution_sha256': 'a' * 64, 'phase': 'fit',
            'original_reference': copy.deepcopy(fitter.ORIGINAL_REFERENCE),
            'original_cpu': copy.deepcopy(fitter.ORIGINAL_CPU),
            'ridge_solver': {'path': '/solver.py', 'sha256': fitter.SOLVER_SHA},
            'warm_start': {}, 'partition': {'path': '/partition.json', 'sha256': fitter.PARTITION_SHA},
            'recipe': copy.deepcopy(fitter.RECIPE), 'resource_policy': fitter.policy('fit'),
            'both_locks_held': True, 'selected_cpu': cpu}
        flags, source = {'synthetic': True}, {'synthetic_source': 'a' * 64}
        qualified_python = {'python': '/synthetic/python', 'python_sha256': 'a' * 64, 'python_version': 'synthetic'}
        for arm in d.ARMS:
            out = root / arm; out.mkdir()
            launch = dict(common, arm=arm)
            authority = write_json(out / 'authority.json', launch)
            checkpoint = out / 'resume.pt'; checkpoint.write_bytes(b'synthetic checkpoint')
            checkpoint = {'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            desc = descriptor()
            inventory = [{'name': str(i), 'shape': [1], 'dtype': 'torch.float32', 'role': 'frozen'} for i in range(448)]
            ident = {'method': fitter.method(launch), 'source': source, 'arm': arm, 'device': 'cpu',
                'native_inventory': inventory, 'numerical_flags': flags,
                'warm_payload_sha256': 'b78bd945256438bfd24873347e26d62c970bf9d85048663301d4fe1236ac35a9',
                **{k: 'a' * 64 for k in ('frozen_sha256', 'fitted_sha256', 'output_witness_sha256', 'zero_source_sha256')}}
            witness = {'rows': 6355, 'classes': 1008, 'coefficients': 4096, 'arm': arm,
                'feature_energy': 4., 'lambda': .0125, 'stationarity_numerator': 0.,
                'stationarity_denominator': 3., 'normalized_stationarity': 0., 'A_nonzero': True,
                'target': 'raw member-inclusive prototypes', 'intercept': False}
            code = {name: hashlib.sha256(Path(fitter.__file__).with_name(name).read_bytes()).hexdigest()
                    for name in fitter.FILES}
            inputs = {**context['guards'], authority['path']: authority['sha256'], checkpoint['path']: checkpoint['sha256']}
            fitted = {**copy.deepcopy(record), 'schema': fitter.SCHEMA, 'phase': 'fit', 'arm': arm,
                'launch': launch, 'resource_policy': fitter.policy('fit'), 'partition_sha256': fitter.PARTITION_SHA,
                'cuda_initialized': False, 'peak_cuda_allocated_bytes': 0, 'quality_read': False,
                'public_encoder_qualified': False, 'fit_qualified': True,
                **{k: True for k in ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'frozen_complement_exact',
                    'training_only_fit', 'zero_A_source_parity', 'bypass_version_tamper_rejected',
                    'sequential_model_ownership', 'both_locks_held_in_parent_authority', 'independent_refit_exact')},
                'authority_sha256': authority['sha256'], 'authority': authority, 'code': code, 'source': source,
                'execution_sha256': launch['execution_sha256'], 'numerical_flags': flags, 'total_fit_core_seconds': 2.,
                'fit_core_seconds': [1., 1.], 'fit_passes': 2, 'identity': ident, 'fit_witness': witness,
                'output_witness_sha256': ident['output_witness_sha256'], 'terminal_state_sha256': 'a' * 64,
                'checkpoint': checkpoint, 'output': str(out), 'input_guards': inputs}
            fitted['invocation'].update(qualified_python, cuda_visible_devices='',
                argv=fitter.cli(Path(fitter.__file__).parent, authority['path'], authority['sha256'],
                                launch['execution_sha256'], 'fit', arm, out))
            desc['receipt'] = write_json(out / 'receipt.json', fitted)
            def branch():
                return {**context, 'old': old, 'launch': launch, 'source': source, 'code': code,
                    'args': SimpleNamespace(execution_sha256=launch['execution_sha256']),
                    'root': Path(fitter.__file__).parent, 'guards': dict(context['guards']),
                    'required_guards': dict(context['guards']), 'terminals': {}, 'terminal_cgroups': {},
                    'legacy': {**context['legacy'], 'admission': admission, 'invocations': set(),
                        'selected': {'source_cpu': {'numerical_flags': flags, 'invocation': qualified_python}}}}
            rejects(lambda: fitter.admit_terminal(branch(), desc, 'fit', arm), 'runtime format')
            b = branch()
            assert adapted(b, desc, 'fit', arm) == fitted
            assert b['terminals']['fit:' + arm] == fitted and b['terminal_cgroups']['fit:' + arm] == footer
            rejects(lambda: adapted(b, desc, 'fit', arm), 'duplicate new unit invocation')
            for key, value, message in (
                ('source', {'changed': True}, 'source/fitted identity'),
                ('code', dict(code, extra='0' * 64), 'closure terminal'),
                ('quality_read', True, 'deterministic fit terminal'),
                ('input_guards', {}, 'checkpoint binding'),
                ('identity', dict(ident, fitted_sha256='bad'), 'source/fitted identity')):
                bad = dict(fitted, **{key: value})
                bad_desc = dict(desc, receipt=write_json(out / 'mutant.json', bad))
                rejects(lambda: adapted(branch(), bad_desc, 'fit', arm), message)
            bad = copy.deepcopy(fitted)
            bad['input_guards'].pop(str(Path(original.__file__)))
            rejects(lambda: adapted(branch(), dict(desc, receipt=write_json(out / 'mutant.json', bad)), 'fit', arm),
                    'input guards')
            bad = copy.deepcopy(fitted); bad['invocation']['argv'][-1] = '/changed'
            rejects(lambda: adapted(branch(), dict(desc, receipt=write_json(out / 'mutant.json', bad)), 'fit', arm),
                    'authority/source/CLI')
            # Changing a bound input after receipt creation must still fail the original final rehash.
            Path(checkpoint['path']).write_bytes(b'changed checkpoint')
            rejects(lambda: adapted(branch(), desc, 'fit', arm), 'current file SHA256')

        # Reverse the two deliberate substitutions: every remaining AST node must match.
        def definition(path, owner=None):
            tree = ast.parse(Path(path).read_bytes())
            scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body if owner else tree.body
            return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == 'admit_terminal')

        before = definition(original.__file__, 'FlatAdmission')
        after = copy.deepcopy(reader.__terminal_ast__)
        index = next(i for i, n in enumerate(before.body) if isinstance(n, ast.Assign) and
                     isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'match')
        after.body[index:index + 1] = copy.deepcopy(before.body[index:index + 3])
        assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False)
        before = definition(fitter.__file__)
        after = copy.deepcopy(adapted.__terminal_ast__)
        index = next(i for i, n in enumerate(before.body) if isinstance(n, ast.Assign) and
                     isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'final')
        after.body[index] = copy.deepcopy(before.body[index])
        assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False)
        assert original_methods == (original.FlatAdmission.admit_terminal, fitter.admit_terminal)
        assert admission.admit_terminal.__func__ is original_methods[0]
        changed = dict(context, code={Path(fitter.__file__).name: '0' * 64})
        rejects(lambda: d.fit_terminal_adapter(fitter, changed), 'SHA256')
        rejects(lambda: d.fit_terminal_adapter(fitter, dict(context, guards={})), 'original terminal source')
        rejects(lambda: d.terminal_ast(original, d.TERMINAL_SOURCE_SHA, '0' * 64, {}, 'FlatAdmission'), 'body differs')
        alias = SimpleNamespace(**{**vars(original), '__spec__': SimpleNamespace(origin='/wrong/origin.py')})
        rejects(lambda: d.terminal_ast(alias, d.TERMINAL_SOURCE_SHA, d.TERMINAL_AST_SHA, {}, 'FlatAdmission'),
                'source origin')
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


def fitter_check(path):
    """Optional exact API check; reads only the sibling's new stdlib module."""
    spec = importlib.util.spec_from_file_location('_test_prototype_fitter_api', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    witness = {'rows': 6355, 'classes': 1008, 'coefficients': 4096, 'arm': 'linear',
               'feature_energy': 10., 'lambda': .03125, 'stationarity_numerator': .00001,
               'stationarity_denominator': 1., 'normalized_stationarity': .00001, 'A_nonzero': True,
               'target': 'raw member-inclusive prototypes', 'intercept': False}
    module.check_fit_witness(witness, 'linear')
    rejects(lambda: module.check_fit_witness(dict(witness, intercept=True), 'linear'))
    rejects(lambda: module.check_fit_witness(dict(witness, normalized_stationarity=.000011), 'linear'))
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ('prepare_native', 'reload', 'reconstruct'):
        assert not any(isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id in {'fresh', 'fit_prototype_residual', 'extract_solver'} or
            isinstance(node.func, ast.Attribute) and node.func.attr in {'fit_ridge_stitch', 'transform'})
            for node in ast.walk(functions[name])), 'no-refit API contains a solver call'


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--fitter-root', type=Path)
    args = parser.parse_args()
    path = Path(__file__).with_name('evaluate_siglip2_prototype_residual.py')
    assert path.is_file(), 'prototype residual evaluator is not implemented'
    spec = importlib.util.spec_from_file_location('_test_prototype_evaluator', path)
    d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    check(d)
    terminal_adapter_check(d)
    rng_binding_check(d, path.read_text())
    flow_check(d, path.read_text())
    if args.fitter_root is not None:
        fitter_check(args.fitter_root / 'fit_siglip2_prototype_residual.py')
    for name in (path, Path(__file__)):
        ast.parse(name.read_text())
        for option in ('-O', '-OO'):
            run = subprocess.run([sys.executable, '-B', '-S', option, str(name), '--help'], capture_output=True, text=True)
            assert run.returncode != 0 and 'optimized mode is forbidden' in run.stderr
    help_run = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert help_run.returncode == 0 and '--phase {cpu,score}' in help_run.stdout
    print('prototype residual evaluator bounded stdlib check: PASS (native UNRUN)')


if __name__ == '__main__':
    main()
