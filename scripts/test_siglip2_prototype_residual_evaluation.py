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
import mmap
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace


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
    assert d.SCHEMA == 'siglip2-prototype-residual-evaluation-v2'
    assert d.AUTHORITY_SCHEMA == 'siglip2-prototype-residual-evaluation-authority-v2'
    assert d.ARMS == ('linear', 'concat')
    assert d.TRAIN_FILES == {'fit_siglip2_prototype_residual.py', 'test_siglip2_prototype_residual.py',
                             'prototype_residual_readout.py'}
    args = SimpleNamespace(execution_sha256='d' * 64)
    for panel in ('selection', 'validation'):
        spec = fixture(d, panel); d.check_spec(spec, args)
        for mutation in (
            lambda s: s.update(schema='siglip2-prototype-residual-evaluation-authority-v1'),
            lambda s: s['training']['code'].pop('prototype_residual_readout.py'),
            lambda s: s['endpoints'][1].update(arm='quadratic'),
            lambda s: s.update(seed=179069), lambda s: s.update(stage='full'),
            lambda s: s.update(both_locks_held=False), lambda s: s.update(validation_previously_exposed=True),
            lambda s: s['training']['code'].update(extra='f' * 64),
            lambda s: s['original_evaluator']['code'].update(evaluate_siglip2_quadratic_readout='f' * 64),
            lambda s: s['original_reference'].update(execution_sha256='0' * 64),
            lambda s: s['source_selection']['terminal'].update(service_seconds=1.),
            lambda s: s['endpoints'].reverse(), lambda s: s['endpoints'][0].update(seed=179061),
            lambda s: s['endpoints'][0]['terminal'].update(service_seconds=float('nan')),
            lambda s: s['resource_policies']['cpu'].update(seconds=120),
            lambda s: s['resource_policies']['cpu'].update(seconds=300),
            lambda s: s['resource_policies']['score'].update(seconds=300),
            lambda s: s['resource_policies']['score'].update(seconds=301),
            lambda s: s['resource_policies']['cpu'].update(seconds=501),
            lambda s: s['resource_policies']['score'].update(seconds=501),
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
    concat = copy.deepcopy(source)
    arms = {'linear': linear, 'concat': concat}
    deltas = d.metric_deltas(arms, source, 'selection')
    intervals = {m: {'mean_delta': sum(deltas['concat_minus_linear'][m]) / count,
                     'product_lower95': .001, 'product_upper95': .1,
                     'query_lower95': .001, 'query_upper95': .1} for m in d.METRICS}
    costs = {'linear': {'service_seconds': 10., 'total_fit_core_seconds': 4.},
             'concat': {'service_seconds': 15., 'total_fit_core_seconds': 6.}}
    cost = d.paired_cost(costs)
    result = d.decide(arms, source, 'selection', intervals, cost)
    assert result['decision'] == 'GO' and result['selection_go_admits_validation_only']
    assert not result['global_production_goal_met'] and not result['concat_source_gain_both']
    assert set(result['deltas']) == {'concat_minus_linear', 'concat_minus_source', 'linear_minus_source'}
    assert all(x == 0 for x in result['deltas']['concat_minus_source']['per_query_r1'])
    assert d.immediate_quality_pass(arms, source, 'selection')
    rejects(lambda: d.replay_equal(source, dict(source, map_at_r=.8000000001)))
    changed = copy.deepcopy(source); changed['per_query_ap'][0] += 1e-12
    rejects(lambda: d.replay_equal(source, changed))
    for changed in (dict(arms, concat=linear), dict(arms, concat=quality(source['per_query_r1'], [.78] * count))):
        assert not d.immediate_quality_pass(changed, source, 'selection')
        assert d.decide(changed, source, 'selection', {}, cost)['decision'] == 'KILL'
    # Beating a degraded linear arm is insufficient when concat is below source.
    below_source = {'linear': quality(linear['per_query_r1'], [.78] * count),
                    'concat': quality(source['per_query_r1'], [.79] * count)}
    assert not d.immediate_quality_pass(below_source, source, 'selection')
    assert not d.decide(below_source, source, 'selection', {}, cost)['source_floor_pass']
    bad_ci = copy.deepcopy(intervals); bad_ci['per_query_ap']['product_lower95'] = 0
    assert d.decide(arms, source, 'selection', bad_ci, cost)['decision'] == 'KILL'
    bad_ci = copy.deepcopy(intervals); bad_ci['per_query_ap']['mean_delta'] = .02
    rejects(lambda: d.decide(arms, source, 'selection', bad_ci, cost))
    for key in ('service_seconds', 'total_fit_core_seconds'):
        bad = copy.deepcopy(costs); bad['concat'][key] *= 1.000001
        assert not d.paired_cost(bad)['pass']
        assert d.decide(arms, source, 'selection', intervals, d.paired_cost(bad))['decision'] == 'KILL'
        bad['linear'][key] = 0; rejects(lambda: d.paired_cost(bad))
    assert d.policy('cpu') == d.policy('score') == {
        'seconds': 500, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0, 'cuda_visible_devices': ''}
    threshold = {'linear': quality([1.] * (count - 13) + [0.] * 13, [.79] * count), 'concat': concat}
    tiny = d.metric_deltas(threshold, source, 'selection')['concat_minus_linear']
    tiny_ci = {m: dict(intervals[m], mean_delta=sum(tiny[m]) / count) for m in d.METRICS}
    assert d.decide(threshold, source, 'selection', tiny_ci, cost)['decision'] == 'KILL'
    for bad in (dict(source, per_query_ap=source['per_query_ap'][:-1]),
                dict(source, per_query_r1=[True] * count), dict(source, map_at_r=float('nan'))):
        rejects(lambda: d.metric_deltas(dict(arms, concat=bad), source, 'selection'))

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


def encoder_retention_check(d, source):
    """Catch a missing/extra/writable/partial retention or leaked mmap on rejection."""
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    assert hasattr(d, 'retain_encoder'), 'missing owned encoder page retention'
    assert d.ENCODER_CHECKPOINT_BYTES == 1711945083 and d.ENCODER_RETENTION_MAX_BYTES == 2 * 1024**3
    native = copy.deepcopy(next(n for n in functions if n.name == 'native_start'))
    native.body = [n for n in native.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                   isinstance(n.value.func, ast.Name) and n.value.func.id == 'retain_encoder')]
    assert hashlib.sha256(ast.dump(native, include_attributes=False).encode()).hexdigest() == (
        'ff3daccc97a9294e0f9b5aa89232c335851636c92c9ee7459b97381d398dda06')
    page = os.sysconf('SC_PAGESIZE'); size = 2 * page + 1
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'fresh_vision.pt'; path.write_bytes(b'a' * size)
        fact = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        mappings, opened, events = [], [], []

        class Pages:
            def __init__(self, fd, length, access):
                assert Path(os.readlink('/proc/self/fd/' + str(fd))) == path
                assert length == size and access == mmap.ACCESS_READ
                self.real = mmap.mmap(fd, length, access=access); self.offsets = []
                opened.append(fd); mappings.append(self)

            def __getitem__(self, offset):
                self.offsets.append(offset)
                if fail_population and offset == 2 * page:
                    raise OSError('population failed')
                return self.real[offset]

            def close(self):
                self.real.close(); events.append('close')

        fail_population = False
        namespace = {**d.retain_encoder.__globals__, 'ENCODER_CHECKPOINT_BYTES': size,
                     'mmap': SimpleNamespace(mmap=Pages, ACCESS_READ=mmap.ACCESS_READ)}
        retain = FunctionType(d.retain_encoder.__code__, namespace)

        def context(descriptor=fact):
            selected = {}
            def owned(value):
                assert value is selected; events.append('owned')
                return SimpleNamespace(materialize=lambda: {'checkpoint': copy.deepcopy(descriptor)})
            return {'selected': selected, 'trainer': SimpleNamespace(owned_encoder=owned), 'guards': {}}

        c = context(); retain(c)
        kept = c['_encoder_mapping']
        assert mappings == [kept] and kept.offsets == [0, page, 2 * page] and kept.real[-1] == ord('a')
        assert c['guards'] == {str(path): fact['sha256']}
        assert c['encoder_retention'] == {**fact, 'bytes': size, 'max_bytes': 2 * 1024**3,
            'page_bytes': page, 'pages_populated': 3, 'access': 'read-only'}
        rejects(lambda: os.fstat(opened[-1]))
        rejects(lambda: kept.real.__setitem__(0, ord('b')))
        rejects(lambda: retain(c), 'already retained')
        path.write_bytes(b'b' * size)
        rejects(lambda: d.bound_file(c['guards'], path, fact['sha256']), 'SHA256')
        assert not kept.real.closed
        kept.close(); path.write_bytes(b'a' * size)
        for descriptor in (dict(fact, sha256='0' * 64), dict(fact, extra=True),
                           dict(fact, path='fresh_vision.pt'), dict(fact, path=str(path.parent / 'warm.pt'))):
            rejects(lambda: retain(context(descriptor)))
        link = path.parent / 'alias' / 'fresh_vision.pt'; link.parent.mkdir(); link.symlink_to(path)
        rejects(lambda: retain(context(dict(fact, path=str(link)))), 'canonical file')
        for actual_size in (0, size - 1, size + 1):
            path.write_bytes(b'a' * actual_size)
            descriptor = dict(fact, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rejects(lambda: retain(context(descriptor)), 'retention size')
        path.write_bytes(b'a' * size)
        limited = FunctionType(retain.__code__, {**namespace, 'ENCODER_RETENTION_MAX_BYTES': size - 1})
        rejects(lambda: limited(context()), 'retention size')
        assert len(mappings) == 1, 'negative guards created an extra mapping'
        fail_population = True
        failed = context(); rejects(lambda: retain(failed), 'population failed')
        assert mappings[-1].real.closed and '_encoder_mapping' not in failed
        rejects(lambda: os.fstat(opened[-1]))
        fail_population = False
        def no_mapping(fd, length, access):
            opened.append(fd)
            raise OSError('mapping failed')
        failed = context()
        unavailable = FunctionType(retain.__code__, {**namespace,
            'mmap': SimpleNamespace(mmap=no_mapping, ACCESS_READ=mmap.ACCESS_READ)})
        rejects(lambda: unavailable(failed), 'mapping failed')
        assert '_encoder_mapping' not in failed and len(mappings) == 2
        rejects(lambda: os.fstat(opened[-1]))

        # Execute the original admission prefix; stop at the native fitter boundary.
        node = copy.deepcopy(next(n for n in functions if n.name == 'native_start'))
        stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Expr) and
                    isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                    n.value.func.attr == 'prepare_native')
        assert isinstance(node.body[stop - 1], ast.Expr) and node.body[stop - 1].value.func.id == 'retain_encoder'
        node.body = node.body[:stop + 1]
        native_namespace = {**vars(d), 'retain_encoder': retain,
            'os': SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': '1' * 32}),
            'sys': SimpleNamespace(modules={}, flags=SimpleNamespace(optimize=0), executable=str(path), version=sys.version)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<admission prefix>', 'exec'), native_namespace)
        before = {'path': '/sys/fs/cgroup/new.service'}
        c = context(); c['selected'].update(source_driver=SimpleNamespace(cgroup_memory=lambda: before),
            selected={'source_cpu': {'invocation': {'python': str(path), 'python_sha256': fact['sha256'], 'python_version': sys.version}}})
        def stage(name, fail=False):
            def call(*args):
                events.append(name)
                if fail:
                    raise ValueError(name)
            return call
        def prepare(value):
            assert value is c['fitting'] and not c['_encoder_mapping'].real.closed
            events.append('prepare')
        c.update(baseline=SimpleNamespace(qualified_origins=stage('origins')), terminals=[], fitting={},
            fitter=SimpleNamespace(prepare_native=prepare),
            admission=SimpleNamespace(init=SimpleNamespace(admit_cgroup=stage('cgroup'))),
            helper=SimpleNamespace(zero_events=stage('zero')))
        events.clear(); native_namespace['native_start'](c)
        assert events == ['origins', 'cgroup', 'zero', 'owned', 'prepare']; c['_encoder_mapping'].close()
        del c['_encoder_mapping']
        for owner, attribute in ((c['baseline'], 'qualified_origins'), (c['admission'].init, 'admit_cgroup'),
                                 (c['helper'], 'zero_events')):
            original = getattr(owner, attribute); setattr(owner, attribute, stage('admission failed', True))
            events.clear(); rejects(lambda: native_namespace['native_start'](c), 'admission failed')
            assert 'owned' not in events and '_encoder_mapping' not in c
            setattr(owner, attribute, original)

        # Run the actual outer lifecycle with native statements excluded from this stdlib fixture.
        node = copy.deepcopy(next(n for n in functions if n.name == 'run'))
        guarded = next(n for n in node.body if isinstance(n, ast.Try))
        assert not guarded.handlers and guarded.finalbody
        assert all(not isinstance(n, ast.Return) for n in node.body[:node.body.index(guarded)])
        guarded.body = [n for n in guarded.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        args = SimpleNamespace(prerequisite=None, authority=path, authority_sha256=fact['sha256'],
            execution_sha256='f' * 64, phase='cpu', output=path.parent / 'output')
        for failure in (None, 'native', 'heads', 'bootstrap', 'exit', 'resources', 'receipt', 'publish', 'cap'):
            mapping_count = len(mappings)
            c = context(); args.output = path.parent / ('output-' + str(failure))
            c.update(costs={'pass': True}, cpu_terminal=None,
                selected={'source_driver': SimpleNamespace(numerical_flags=lambda: {}),
                          'selected': {'source_cpu': {'invocation': {'python_sha256': fact['sha256']}}}})
            c['trainer'] = SimpleNamespace(owned_encoder=lambda selected: SimpleNamespace(materialize=lambda: {'checkpoint': fact}))
            def observe(name, result=None):
                def call(*values):
                    assert not c['_encoder_mapping'].real.closed
                    events.append(name)
                    if failure == name:
                        raise ValueError(name)
                    return result
                return call
            def start(value):
                retain(value); return observe('native', before)()
            c['resources'] = observe('resources', {})
            c['helper'] = SimpleNamespace(publish=observe('publish'))
            run_namespace = {**vars(d), 'authority': lambda value: c, 'prerequisites': lambda value: None,
                'native_start': start, 'qualify_heads': observe('heads', {}),
                'qualify_bootstrap': observe('bootstrap', {}), 'exit_rehash': observe('exit', {}),
                'bind': lambda value: {}, 'check_receipt': observe('receipt'), 'preparation_costs': lambda value: {},
                'torch': SimpleNamespace(random=SimpleNamespace(get_rng_state=lambda: SimpleNamespace(clone=lambda: 0)),
                    equal=lambda *values: True, cuda=SimpleNamespace(is_initialized=lambda: False)),
                'sys': SimpleNamespace(argv=d.cli_argv(path, fact['sha256'], 'f' * 64, 'cpu', args.output, None),
                    executable=sys.executable, version=sys.version, flags=SimpleNamespace(optimize=0)),
                'os': SimpleNamespace(getpid=os.getpid, environ={'INVOCATION_ID': '1' * 32, 'CUDA_VISIBLE_DEVICES': ''}),
                'time': SimpleNamespace(perf_counter=lambda: d.UNIT_STARTED + (d.policy('cpu')['seconds'] + 1 if failure == 'cap' and 'publish' in events else 1)),
                'print': lambda *values, **kwargs: None}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<run lifecycle>', 'exec'), run_namespace)
            events.clear()
            if failure is None:
                record = run_namespace['run'](args)
                assert record['encoder_retention'] == c['encoder_retention']
                assert events == ['native', 'heads', 'bootstrap', 'exit', 'resources', 'receipt', 'publish', 'close']
            else:
                rejects(lambda: run_namespace['run'](args))
                assert events[-1] == 'close'
            assert len(mappings) == mapping_count + 1 and events.count('close') == 1
            assert mappings[-1].real.closed and '_encoder_mapping' not in c
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


def closure_check(d):
    """Pin mappings must admit exact keys while still authenticating all bytes."""
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        code = {}
        for name, raw in (('fit_siglip2_prototype_residual.py', b'fit source'),
                          ('test_siglip2_prototype_residual.py', b'test source'),
                          ('prototype_residual_readout.py', b'helper source')):
            (root / name).write_bytes(raw)
            code[name] = hashlib.sha256(raw).hexdigest()
        manifest = root / 'execution.json'

        def write_manifest(value):
            raw = json.dumps(value, sort_keys=True).encode()
            manifest.write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()

        digest = write_manifest(code)
        assert code.keys() == d.TRAIN_FILES
        for names in (code, set(code), code.keys()):
            guards = {}
            assert d.closure(root, digest, names, guards) == code
            assert guards == {str(manifest): digest, **{str(root / k): v for k, v in code.items()}}
        rejects(lambda: d.closure(root, '0' * 64, code, {}), 'file SHA256')
        for changed in ({'prototype_residual_readout.py': code['prototype_residual_readout.py']}, dict(code, extra='0' * 64)):
            changed_digest = write_manifest(changed)
            rejects(lambda: d.closure(root, changed_digest, code, {}), 'exact execution closure')
        changed_digest = write_manifest(dict(code, **{'prototype_residual_readout.py': '0' * 64}))
        rejects(lambda: d.closure(root, changed_digest, code, {}), 'file SHA256')
        digest = write_manifest(code)
        (root / 'prototype_residual_readout.py').write_bytes(b'changed source')
        rejects(lambda: d.closure(root, digest, code, {}), 'file SHA256')
    assert not any(name.split('.')[0] in d.NATIVE for name in sys.modules)


def v2_fitter_check(d):
    """A v1 module/payload profile cannot become a new endpoint by relabelling its arm."""
    values = dict(FILES=set(d.TRAIN_FILES), SCHEMA='siglip2-prototype-residual-ridge-v2',
        AUTHORITY_SCHEMA='siglip2-prototype-residual-launch-v2', ARMS=d.ARMS,
        ORIGINAL_REFERENCE=d.ORIGINAL_REFERENCE, ORIGINAL_CODE=d.ORIGINAL_PINS, PARTITION_SHA=d.PARTITION_SHA,
        RECIPE={'feature_width': {'linear': 32, 'concat': 160},
                'basis': {'linear': 'Z32', 'concat': '[Z32,H0raw128]'}, 'intercept': False, 'fit_passes': 2})
    d.check_fitter(SimpleNamespace(**values))
    for mutation in (
        lambda v: v.update(SCHEMA='siglip2-prototype-residual-ridge-v1'),
        lambda v: v.update(AUTHORITY_SCHEMA='siglip2-prototype-residual-launch-v1'),
        lambda v: v['FILES'].remove('prototype_residual_readout.py'),
        lambda v: v['FILES'].add('unbound.py'),
        lambda v: v.update(ARMS=('linear', 'quadratic')),
        lambda v: v['RECIPE']['feature_width'].update(concat=32),
        lambda v: v['RECIPE']['basis'].update(concat='[H0raw128,Z32]'),
        lambda v: v['RECIPE'].update(intercept=True),
    ):
        changed = copy.deepcopy(values); mutation(changed)
        rejects(lambda: d.check_fitter(SimpleNamespace(**changed)), 'prototype fitter profile')


def synthetic_bootstrap_check(d, source):
    """Exercise real dispatch/receipt predicates with stdlib arrays, never native evidence."""
    calls = []

    class Array(list):
        def mean(self):
            return sum(self) / len(self)

        def __neg__(self):
            return Array(-v for v in self)

    def lower(delta, groups):
        calls.append((tuple(delta), tuple(groups)))
        return delta.mean() - (max(delta) - min(delta)) / 8

    node = copy.deepcopy(next(n for n in ast.parse(source).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'paired_intervals'))
    node.body = [n for n in node.body if not isinstance(n, ast.Import)]
    namespace = {**vars(d), 'np': SimpleNamespace(asarray=Array, arange=lambda count: Array(range(count)))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<stdlib interval call schedule>', 'exec'), namespace)
    ticks = iter(range(100))
    qualified = FunctionType(d.qualify_bootstrap.__code__, {**vars(d),
        'paired_intervals': namespace['paired_intervals'], 'time': SimpleNamespace(perf_counter=lambda: next(ticks))})
    # No cache/labels/source_record/partition in context: CPU cannot consult held data.
    context = {'baseline': SimpleNamespace(scoring_math=lambda context: SimpleNamespace(bootstrap_lower=lower))}
    proof = qualified(context)
    d.check_synthetic_bootstrap(proof)
    assert proof['calls'] == len(calls) == 16 and proof['seconds'] == 5
    assert proof['sha256'] == d.json_digest({k: v for k, v in proof.items() if k != 'sha256'})
    for offset, panel in ((0, 'selection'), (8, 'validation')):
        count, products = d.PANELS[panel][1], d.PANELS[panel][3]
        pair, groups = d.synthetic_inputs(panel)
        assert len(groups) == count and len(set(groups)) == products == 498
        assert proof['panels'][panel]['seconds'] == 1
        assert proof['panels'][panel]['inputs_sha256'] == d.json_digest({'deltas': pair, 'product_groups': groups})
        for metric_index, metric in enumerate(d.METRICS):
            for kind_index, expected_groups in enumerate((groups, list(range(count)))):
                index = offset + 4 * metric_index + 2 * kind_index
                assert calls[index] == (tuple(pair[metric]), tuple(expected_groups))
                assert calls[index + 1] == (tuple(-v for v in pair[metric]), tuple(expected_groups))
        fields = proof['panels'][panel]['paired_intervals']
        assert all(fields['per_query_ap'][k] == v / 2 for k, v in fields['per_query_r1'].items())
    for mutation in (
        lambda v: v.update(schema='wrong'), lambda v: v.update(seed=179020),
        lambda v: v.update(draws=4999), lambda v: v.update(calls=8),
        lambda v: v.update(seconds=500), lambda v: v.update(seconds=0),
        lambda v: v.update(seconds=True), lambda v: v['panels'].pop('validation'),
        lambda v: v['panels']['selection'].update(query_images=1749),
        lambda v: v['panels']['selection'].update(products=497),
        lambda v: v['panels']['selection'].update(inputs_sha256='0' * 64),
        lambda v: v['panels']['selection'].update(shared_draws_exact=False),
        lambda v: v['panels']['selection']['paired_intervals']['per_query_ap'].update(product_lower95=.16),
        lambda v: v['panels']['validation']['paired_intervals']['per_query_r1'].update(query_upper95=.25),
        lambda v: v['panels']['validation']['paired_intervals']['per_query_ap'].update(mean_delta=.19),
    ):
        changed = copy.deepcopy(proof); mutation(changed)
        changed['sha256'] = d.json_digest({k: v for k, v in changed.items() if k != 'sha256'})
        rejects(lambda: d.check_synthetic_bootstrap(changed))
    changed = copy.deepcopy(proof); changed['panels']['selection']['seconds'] += .5
    rejects(lambda: d.check_synthetic_bootstrap(changed), 'digest')
    # Timing is part of the proof but never rerun or compared as a fresh head fact on score.
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    run = ast.get_source_segment(source, functions['run'])
    assert "result['synthetic_bootstrap'] = qualify_bootstrap(context) if args.phase == 'cpu' else cpu['synthetic_bootstrap']" in run
    assert 'synthetic_bootstrap' not in ast.get_source_segment(source, functions['qualify_heads'])
    assert "check_synthetic_bootstrap(record['synthetic_bootstrap'])" in ast.get_source_segment(source, functions['check_receipt'])
    # The real native helper remains the pinned implementation, including seed/draw resets.
    bootstrap_source = Path(__file__).with_name('score_inshop_crop_view_pair.py').read_text()
    bootstrap = next(n for n in ast.parse(bootstrap_source).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'bootstrap_lower')
    assert hashlib.sha256(ast.dump(ast.Module(body=[bootstrap], type_ignores=[]),
                                  include_attributes=False).encode()).hexdigest() == (
        '76971d1089c8494e16450d56057a5b2fa2c71e40cf360d4ede0562fc53d9c720')
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


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
    assert 'source_selection_adapter(baseline, fitting)' in authority and 'source_selection(context)' in authority, (
        'full authenticated original source admission is mandatory')
    assert "fitter.admit_terminal(branch, endpoint['terminal'], 'fit', endpoint['arm'])" in authority
    assert authority.index('check_fitter(fitter)') < authority.index('fitter.authority(') < authority.index('fitter.admit_terminal(')
    assert "for endpoint in spec['endpoints']:" in authority
    accept = ast.get_source_segment(source, functions['accept_terminal'])
    assert "context['terminal_reader'](context['admission'], record, terminal" in accept, (
        'fresh prerequisite/validation terminals use the authenticated original reader')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and n.func.attr == 'admit_terminal']
    assert len(calls) == 1 and isinstance(calls[0].func.value, ast.Name) and calls[0].func.value.id == 'fitter', (
        'only fully authenticated v2 fitter terminals delegate directly; original readers keep their adapters')
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
    assert 'paired_intervals(fixed, pair,' in panel
    intervals = ast.get_source_segment(source, functions['paired_intervals'])
    assert 'fixed.bootstrap_lower(delta, groups)' in intervals and '-fixed.bootstrap_lower(-delta, groups)' in intervals
    synthetic = ast.get_source_segment(source, functions['qualify_bootstrap'])
    assert 'paired_intervals(fixed, pair,' in synthetic
    assert all(forbidden not in synthetic for forbidden in ('source_record', 'cache_rows', 'packed_quality', 'replay_archived_source'))
    assert "context['fitter'].prepare_readout(context['fitting'])" in ast.get_source_segment(source, functions['exit_rehash'])
    run = ast.get_source_segment(source, functions['run'])
    assert "args.phase == 'score' and context['costs']['pass']" in run
    assert run.index('authority(args)') < run.index('native_start(context)') < run.index('qualify_heads(context)')
    assert 'resources_adapter(helper, guards)' in authority
    assert "context['resources'](context, args.phase, before)" in run
    for name, digest in d.EVALUATOR_PINS.items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    archive = Path(__file__).resolve().parents[1] / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v7'
    for name, digest in d.ORIGINAL_PINS.items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((archive / 'execution.json').read_bytes()).hexdigest() == d.ORIGINAL_REFERENCE['execution_sha256']



def score_schedule_check(d, source):
    """Execute real scheduling/predicates with stdlib wires; never load native code."""
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    # Assigned f1b09857 CPU qualification and ownership paths stay byte/AST exact.
    assert hashlib.sha256(ast.dump(functions['qualify_heads']).encode()).hexdigest() == (
        '79c91b8c85af593fcc9602b675de57dca0140c4c518e0197adb7c68dfc2f5c38')
    for name, digest in (
        ('qualify_heads', '972dd523c79a82c31094b5726670a545652ea6ef816409c6b00fcd725a1c0432'),
        ('head_values', '86f8e390ac50da999dbbe7dce25ed1c2b368df6e3fba7df3656c2f1a62500c3f'),
        ('load_head', 'a4386afb2a2f1aff904ebd497d2a34085d6229f0fd19d3c1366614300deb3316'),
        ('release_head', '34ec5cfb50ad41be4f3b2e66350bdd9653d23d8def6ccd56b2d1a120f12d62f3')):
        assert hashlib.sha256(ast.get_source_segment(source, functions[name]).encode()).hexdigest() == digest

    # Execute run's actual phase dispatch: score success must not also qualify_heads.
    body = next(n for n in functions['run'].body if isinstance(n, ast.Try)).body
    index = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'result')
    dispatch = ast.parse('def dispatch(args, context, cpu):\n    pass').body[0]
    dispatch.body = copy.deepcopy(body[index:index + 3]) + ast.parse('return facts, result').body
    calls, fresh = [], {'head_facts': {'fresh': True}, 'train_witnesses': {'fresh': True}}

    def qualify(context):
        calls.append('qualify')
        return copy.deepcopy(fresh)

    def score(context, cpu):
        calls.append('score')
        return copy.deepcopy(fresh)

    namespace = {'qualify_heads': qualify, 'score_panel': score, 'require': d.require}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[])), '<actual phase dispatch>', 'exec'), namespace)
    for phase, cost, expected in (('cpu', True, 'qualify'), ('score', False, 'qualify'), ('score', True, 'score')):
        calls.clear()
        facts, result = namespace['dispatch'](SimpleNamespace(phase=phase), {'costs': {'pass': cost}},
                                              None if phase == 'cpu' else copy.deepcopy(fresh))
        assert calls == [expected] and facts == fresh
        if expected == 'qualify':
            assert result['files'] == {} and result['quality_read'] is False
    rejects(lambda: namespace['dispatch'](SimpleNamespace(phase='score'), {'costs': {'pass': True}},
        dict(fresh, train_witnesses={})), 'accepted evaluator CPU witnesses differ')

    class Wire:
        def __init__(self, arm, kind, suffix, load=0):
            self.bits, self.load, self.extra_bits = (arm, kind, suffix), load, ''

        def numpy(self):
            return self

    names = ('raw', 'unit', 'codes', 'inverse_norms')

    def wires(arm, kind, load=0):
        return tuple(Wire(arm, kind, name, load) for name in names)

    def value_facts(context, values):
        return {name: v.bits for name, v in zip(names, values, strict=True)}

    def exercise(fault=None, target=4, component=0, function=d.score_panel, panel_name='selection', cost=True):
        events, active, counts = [], [], {arm: 0 for arm in d.ARMS}
        saved, readbacks, released, trained = {}, set(), [], []
        cpu = {'head_facts': {a: {'payload_sha256': a, 'identity': {'arm': a}, 'readout_sha256': a}
                              for a in d.ARMS},
               'train_witnesses': {a: value_facts(None, wires(a, 'train')) for a in d.ARMS}}
        cpu_before = copy.deepcopy(cpu)
        # Two synthetic rows suffice to falsify scheduling; real panel counts are checked above.
        panels = {name: (2, 1, 1, 1) for name in d.PANELS}
        panel_count, query_count, _, products = panels[panel_name]
        panel = {'original_rows': list(range(panel_count)), 'query': list(range(query_count)),
                 'gallery': list(range(query_count, panel_count)), 'original_class_ids': list(range(products))}
        expected_quality = quality([0.] * query_count, [0.] * query_count)

        def load(context, endpoint):
            assert not active, 'previous source state retained'
            arm = endpoint['arm']; counts[arm] += 1
            assert counts[arm] <= 2, 'redundant independent reload'
            index = sum(counts.values())
            state = {'arm': arm, 'load': index, 'readout': arm}
            active.append(state); events.append(('load', index))
            facts = copy.deepcopy(cpu['head_facts'][arm])
            if fault == 'head' and index == target:
                facts[('payload_sha256', 'identity', 'readout_sha256')[component]] = 'changed'
            return state, facts

        def values(context, state, cache):
            assert active == [state]
            index = state['load']; kind = 'train' if cache == context['features'][:64] else 'panel'
            result = wires(state['arm'], kind, index); events.append((kind, index))
            if kind == 'train':
                trained.append(index)
            if index == target:
                if fault == kind:
                    result[component].bits += ('changed',)
                if fault == kind + '_fingerprint':
                    result[component].extra_bits = 'changed'
                if fault == kind + '_readout':
                    state['readout'] = 'changed'
            return result

        def release(context, state):
            assert active == [state]
            released.append(state['load']); events.append(('release', state['load']))
            state.clear(); active.clear()

        def exact(first, second):
            assert tuple(v.bits for v in first) == tuple(v.bits for v in second), 'wire equality'

        def write(context, key, values):
            saved[key] = values
            return {key + suffix: 'sha' for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}

        def readback(context, key, files, second):
            if fault == 'readback' and key == d.ARMS[component]:
                raise ValueError('saved-wire readback rejected')
            exact(saved[key], second)
            readbacks.add(key); events.append(('readback', key))

        def packed_quality(value, *args, **kwargs):
            arm = value.bits[0]
            if arm != 'source':
                assert not active and released == trained == [1, 2, 3, 4] and readbacks == {
                    'source-179061', *d.ARMS}, 'candidate quality before all four witnesses/readbacks'
            else:
                assert ('archive', panel_name) in events, 'source archive must precede scoring'
            events.append(('quality', arm, value.load))
            if fault == 'second_quality' and value.load == target:
                return dict(expected_quality, map_at_r=.1)
            return copy.deepcopy(expected_quality)

        def saved_quality(context, key, files, fixed, labels, panel, expected):
            assert key in readbacks
            result = packed_quality(Wire(expected[1].bits[0], 'panel', 'unit', -1))
            if fault == 'persisted_quality' and key == d.ARMS[component]:
                result['map_at_r'] = .1
            return result

        baseline = SimpleNamespace(scoring_math=lambda c: SimpleNamespace(packed_quality=packed_quality),
            replay_archived_source=lambda c, f: events.append(('archive', panel_name)),
            cache_rows=lambda c, rows: 'panel', source_values=lambda c, cache: wires('source', 'panel'),
            archived_source_wires=lambda c: wires('source', 'panel'), value_facts=value_facts,
            write_wires=write, readback_wires=readback)
        context = {'spec': fixture(d, panel_name), 'baseline': baseline,
            'selected': {'original': SimpleNamespace(fingerprint=lambda vals: tuple((v.bits, v.extra_bits) for v in vals))},
            'partition': {'panels': {panel_name: panel}}, 'features': list(range(64)),
            'fit': {'class_names': ['product'], 'targets': [0] * panel_count},
            'helper': SimpleNamespace(exact=exact),
            'selection_go': {'decision': 'GO', 'selection_go_admits_validation_only': fault != 'validation'},
            'source_record': {'quality': {'179061': {'control': copy.deepcopy(expected_quality)}}}}
        if fault == 'archive':
            context['source_record']['quality']['179061']['control']['map_at_r'] = .1
        context['costs'] = d.paired_cost({a: {'service_seconds': 1., 'total_fit_core_seconds': 1.} for a in d.ARMS})
        context['costs']['pass'] = cost
        namespace = {**function.__globals__, 'load_head': load, 'head_values': values, 'release_head': release,
            'readout_digest': lambda c, s: s['readout'], 'score_saved_wires': saved_quality,
            'preparation_costs': lambda c: {}, 'gc': SimpleNamespace(collect=lambda: None), 'PANELS': panels,
            '__builtins__': {**vars(__import__('builtins')), '__import__': lambda name, *a, **k: {
                'numpy': SimpleNamespace(), 'torch': SimpleNamespace(device=lambda v: v)}[name]}}
        for name in ('metric_deltas', 'immediate_quality_pass', 'decide'):
            namespace[name] = FunctionType(getattr(d, name).__code__, namespace)
        try:
            result = FunctionType(function.__code__, namespace)(context, cpu)
        except (ValueError, AssertionError):
            if fault not in ('second_quality', 'persisted_quality', 'early_quality'):
                assert not any(e[:2] in (('quality', a) for a in d.ARMS) for e in events), events
            raise
        assert cpu == cpu_before
        assert result['head_facts'] == cpu['head_facts'] and result['head_facts'] is not cpu['head_facts']
        assert result['train_witnesses'] == cpu['train_witnesses'] and result['train_witnesses'] is not cpu['train_witnesses']
        assert all(result['head_facts'][a] is not cpu['head_facts'][a] and
                   result['train_witnesses'][a] is not cpu['train_witnesses'][a] for a in d.ARMS)
        assert [e for e in events if e[0] == 'quality'] == [
            ('quality', 'source', 0), ('quality', 'source', -1),
            ('quality', 'linear', 1), ('quality', 'linear', 2), ('quality', 'linear', -1),
            ('quality', 'concat', 3), ('quality', 'concat', 4), ('quality', 'concat', -1)]
        assert counts == {'linear': 2, 'concat': 2} and not active
        assert result['files'].keys() == d.file_names() and result['decision'] == 'KILL'
        return result

    exercise(); exercise(panel_name='validation')
    for index in range(1, 5):
        for component in range(3):
            rejects(lambda: exercise('head', index, component), 'CPU qualified complete scoring state')
        for component in range(4):
            rejects(lambda: exercise('train', index, component), 'CPU qualified TRAIN scoring witness')
            rejects(lambda: exercise('panel', index, component), 'wire equality')
        for kind in ('train', 'panel'):
            rejects(lambda: exercise(kind + '_readout', index),
                    'readout' if kind == 'train' or index % 2 else 'state/wire bits differ')
            rejects(lambda: exercise(kind + '_fingerprint', index), 'wire bits differ')
    for component in range(2):
        rejects(lambda: exercise('readback', component=component), 'saved-wire readback')
        rejects(lambda: exercise('persisted_quality', component=component), 'per-query packed quality replay')
    for index in (2, 4):
        rejects(lambda: exercise('second_quality', index), 'per-query packed quality replay')
    rejects(lambda: exercise('archive'), 'per-query packed quality replay')
    rejects(lambda: exercise('validation', panel_name='validation'), 'selection GO')
    rejects(lambda: exercise(cost=False), 'cost gate precedes panel access')
    # The observer must fail if candidate quality is moved into the witness loop.
    mutant = copy.deepcopy(functions['score_panel'])
    loops = [n for n in mutant.body if isinstance(n, ast.For)]
    scoring = next(n for n in loops[1].body if isinstance(n, ast.Assign) and
                   isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'first')
    index = next(i for i, n in enumerate(loops[0].body) if isinstance(n, ast.Delete))
    loops[0].body.insert(index, copy.deepcopy(scoring))
    namespace = dict(d.score_panel.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[mutant], type_ignores=[])), '<early quality mutant>', 'exec'), namespace)
    rejects(lambda: exercise('early_quality', function=namespace['score_panel']), 'before all four witnesses')
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


def terminal_adapter_check(d):
    """Exercise authenticated original terminal predicates, without native work."""
    def load(name, filename):
        path = Path(__file__).with_name(filename).resolve()
        return d.load_bare(name, path, hashlib.sha256(path.read_bytes()).hexdigest())

    original = load('_ms_original_admission', 'train_siglip2_substrate_adaptation.py')
    initializer = load('_ms_original_initializer', 'initialize_siglip2_substrate_fit.py')
    baseline = load('_ms_original_baseline', 'evaluate_siglip2_quadratic_readout.py')
    admission = original.FlatAdmission()
    admission.init = initializer
    original_methods = (original.FlatAdmission.admit_terminal, baseline.admit_source_selection)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        guards = {original.__file__: hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()}
        context = {'legacy': {'original': original, 'admission': admission}, 'guards': guards}
        source_selection = d.source_selection_adapter(baseline, context)
        reader = source_selection.__globals__['_original_log_terminal']
        archive = Path(__file__).resolve().parents[1] / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
        for relative, unit_name, duration, log_sha, receipt_sha in (
            ('prototype-residual-ridge-v1/fit-linear-v1', 'sfora-so400-prototype-residual-fit-linear-v1',
             180.745, '0a02d3eb758e7de6a930bc2908937890ff1b26c388cf8d26ed3d72895788dc44',
             '2a861afc762d92018b93eaf4f3b5a566a7be095d71b6da3fabfba6d8c4cada09'),
            ('genuine-view-v1/evaluation-source-v4/first-score', d.SOURCE_SCORE_TERMINAL['unit'],
             120.835, d.SOURCE_SCORE_TERMINAL['log']['sha256'], d.SOURCE_SCORE_TERMINAL['receipt']['sha256'])):
            archived_log = archive / (relative + '.log')
            proof_receipt = {'path': str(archive / (relative + '-receipt.json')), 'sha256': receipt_sha}
            archived_record = d.read_json(proof_receipt, {})
            proof = dict(terminal(unit_name), receipt=proof_receipt,
                log={'path': str(archived_log), 'sha256': log_sha},
                invocation_id=archived_record['invocation']['invocation_id'], service_seconds=duration,
                native_peak_rss_kib=archived_record['process_peak_rss_kib'])
            expected = json.loads(next(line.removeprefix('FINAL_CGROUP ') for line in archived_log.read_text().splitlines()
                                       if line.startswith('FINAL_CGROUP ')))
            for archived_reader in (reader,):
                archived_admission = original.FlatAdmission(); archived_admission.init = initializer
                rejects(lambda: archived_admission.admit_terminal(archived_record, proof, 300, {}), 'runtime format')
                assert archived_reader(archived_admission, archived_record, proof, 300, {}) == expected
                rejects(lambda: archived_reader(archived_admission, archived_record,
                        dict(proof, service_seconds=duration + .001), 300, {}), 'numeric binding')
                rejects(lambda: archived_reader(archived_admission, archived_record,
                        dict(proof, log=dict(proof['log'], sha256='0' * 64)), 300, {}), 'SHA256')
            assert hashlib.sha256(archived_log.read_bytes()).hexdigest() == log_sha
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
            (text.replace('max 0', 'max 1'), 'memory failure events'),
            (text.replace('"memory.max": "8589934592"', '"memory.max": "8589934593"'), 'memory/swap'),
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
        # Reverse only the deliberate substitutions: every remaining AST node must match.
        def definition(path, owner=None, name='admit_terminal'):
            tree = ast.parse(Path(path).read_bytes())
            scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body if owner else tree.body
            return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)

        before = definition(original.__file__, 'FlatAdmission')
        after = copy.deepcopy(reader.__terminal_ast__)
        index = next(i for i, n in enumerate(before.body) if isinstance(n, ast.Assign) and
                     isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'match')
        after.body[index:index + 1] = copy.deepcopy(before.body[index:index + 3])
        assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False)
        for module, adapter, name in ((baseline, source_selection, 'admit_source_selection'),):
            before = definition(module.__file__, name=name)
            after = copy.deepcopy(adapter.__terminal_ast__)
            index = next(i for i, n in enumerate(before.body) if isinstance(n, ast.Assign) and
                         isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'final')
            call = after.body[index].value
            assert call.func.id == '_original_log_terminal'
            call.func = ast.Attribute(value=call.args.pop(0), attr='admit_terminal', ctx=ast.Load())
            assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False), (
                'every original predicate and terminal argument must remain intact')
        for name in ('read_json', 'require', 'SOURCE_INVENTORY', 'check_source_record', 'SOURCE_SCORE_TERMINAL', 'SOURCE_INVENTORY_SHA'):
            assert source_selection.__globals__[name] is getattr(baseline, name)
        assert original_methods == (original.FlatAdmission.admit_terminal, baseline.admit_source_selection)
        assert admission.admit_terminal.__func__ is original_methods[0]
        rejects(lambda: d.source_selection_adapter(baseline, dict(context, guards={})), 'original terminal source')
        rejects(lambda: d.terminal_ast(baseline, d.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'],
            '0' * 64, {}, name='admit_source_selection'), 'body differs')
        rejects(lambda: d.terminal_ast(original, d.TERMINAL_SOURCE_SHA, '0' * 64, {}, 'FlatAdmission'), 'body differs')
        alias = SimpleNamespace(**{**vars(original), '__spec__': SimpleNamespace(origin='/wrong/origin.py')})
        rejects(lambda: d.terminal_ast(alias, d.TERMINAL_SOURCE_SHA, d.TERMINAL_AST_SHA, {}, 'FlatAdmission'),
                'source origin')
        alias = SimpleNamespace(**{**vars(baseline), '__spec__': SimpleNamespace(origin='/wrong/origin.py')})
        rejects(lambda: d.source_selection_adapter(alias, context), 'source origin')
        changed_source = root / 'baseline.py'; changed_source.write_bytes(Path(baseline.__file__).read_bytes() + b'\n')
        alias = SimpleNamespace(**{**vars(baseline), '__file__': str(changed_source),
                                  '__spec__': SimpleNamespace(origin=str(changed_source))})
        rejects(lambda: d.source_selection_adapter(alias, context), 'SHA256')
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


def resources_adapter_check(d):
    """CPU/score500 is prospective; preserve the pinned checker and every resource gate."""
    path = Path(__file__).with_name('export_siglip2_substrate_adaptation.py').resolve()
    helper = d.load_bare('_resources_original_helper', path, d.REFERENCE_PINS[path.name])
    original_globals = dict(vars(helper))
    guards = {}
    adapted = d.resources_adapter(helper, guards)
    node = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'resources')
    assert ast.dump(adapted.__resources_ast__) == ast.dump(node), 'every predicate/report field stays exact'
    assert adapted.__globals__.keys() == {
        'Path', 'zero_events', 'time', 'UNIT_STARTED', 'require', 'resource', 'policy', '__builtins__', 'resources'}
    for name in ('Path', 'zero_events', 'time', 'UNIT_STARTED', 'require', 'resource'):
        assert adapted.__globals__[name] is getattr(helper, name)
    assert adapted.__globals__['policy'] is d.policy and adapted is not helper.resources
    assert guards == {str(path): d.REFERENCE_PINS[path.name]}
    init_path = path.with_name('initialize_siglip2_substrate_fit.py')
    initializer = d.load_bare('_resources_original_initializer', init_path, hashlib.sha256(init_path.read_bytes()).hexdigest())
    before = {'path': '/sys/fs/cgroup/prospective-cpu.service', 'values': {
        'memory.max': str(8 * 1024**3), 'memory.current': '1024', 'memory.peak': '2048',
        'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
        'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}}

    def run(function=adapted, phase='cpu', elapsed=120.103, rss=1000, swap=0, after=None):
        after = copy.deepcopy(before) if after is None else after
        context = {'unit_started': 10., 'selected_context': {
            'source': SimpleNamespace(cgroup_memory=lambda: after), 'initialized': {'init': initializer}}}
        # Only host observations are synthetic; execute the actual checker and cgroup predicates.
        namespace = {**function.__globals__, 'time': SimpleNamespace(perf_counter=lambda: 10. + elapsed),
            'resource': SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=rss)),
            'Path': lambda value: SimpleNamespace(read_text=lambda: f'VmSwap:\t{swap} kB\n')
                    if value == '/proc/self/status' else Path(value)}
        return FunctionType(function.__code__, namespace)(context, phase, before)

    for phase in ('cpu', 'score'):
        result = run(phase=phase, elapsed=499.)
        assert result == {'resource_policy': {'seconds': 500, 'host_bytes': 8 * 1024**3,
                'swap_bytes': 0, 'cuda_visible_devices': ''}, 'wall_seconds': 499.,
            'process_peak_rss_kib': 1000, 'peak_cuda_allocated_bytes': 0,
            'cgroup_before': before, 'cgroup_after': before, 'both_locks_held_in_parent_authority': True,
            'terminal_exit_and_both_locks_require_parent_receipt': True}
        assert run(phase=phase, elapsed=300.)['wall_seconds'] == 300.
        for elapsed in (500., 501.):
            rejects(lambda: run(phase=phase, elapsed=elapsed), 'complete-unit resources')
    rejects(lambda: run(helper.resources), 'complete-unit resources')  # Historical CPU120 still fails.
    assert abs(run()['wall_seconds'] - 120.103) < 1e-12
    original_score = run(helper.resources, phase='score')
    assert original_score['resource_policy'] == dict(d.policy('score'), seconds=300)
    assert run(phase='score') == dict(original_score, resource_policy=d.policy('score'))
    for elapsed in (300., 499.):
        rejects(lambda: run(helper.resources, phase='score', elapsed=elapsed), 'complete-unit resources')
    for rss in (0, 8 * 1024**2 + 1):
        rejects(lambda: run(rss=rss), 'complete-unit resources')
    rejects(lambda: run(swap=1), 'native swap')
    rejects(lambda: run(after=dict(before, path='/sys/fs/cgroup/different.service')), 'complete-unit resources')
    for key, value, message in (
        ('memory.max', str(16 * 1024**3), 'memory/swap caps'),
        ('memory.peak', str(8 * 1024**3 + 1), 'memory/swap caps'),
        ('memory.swap.current', '1', 'memory/swap caps'),
        ('memory.swap.peak', '1', 'memory/swap caps'),
        ('memory.swap.max', '1', 'memory/swap caps'),
        ('memory.events', before['values']['memory.events'].replace('high 0', 'high 1'), 'events must be zero')):
        after = copy.deepcopy(before); after['values'][key] = value
        rejects(lambda: run(after=after), message)
    rejects(lambda: d.terminal_ast(helper, d.REFERENCE_PINS[path.name], '0' * 64, {}, name='resources'), 'body differs')
    alias = SimpleNamespace(**{**vars(helper), '__spec__': SimpleNamespace(origin='/wrong/origin.py')})
    rejects(lambda: d.resources_adapter(alias, {}), 'source origin')
    # A live predicate bypass with intact pinned disk bytes must also be rejected.
    alias = SimpleNamespace(**vars(helper))
    mutant = copy.deepcopy(node)
    guard = next(n for n in mutant.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == 'require')
    guard.value.args[0] = ast.Constant(value=True)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[mutant], type_ignores=[])), str(path), 'exec'), vars(alias))
    rejects(lambda: d.resources_adapter(alias, {}), 'body differs')
    with TemporaryDirectory() as temporary:
        changed = Path(temporary) / path.name
        changed.write_bytes(path.read_bytes().replace(b"peak < 10_000_000_000", b"peak < 20_000_000_000"))
        alias = SimpleNamespace(**{**vars(helper), '__file__': str(changed), '__spec__': SimpleNamespace(origin=str(changed))})
        rejects(lambda: d.resources_adapter(alias, {}), 'SHA256')
    assert vars(helper) == original_globals and helper.policy('cpu')['seconds'] == 120
    assert helper.policy('score')['seconds'] == 300
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)


def fitter_check(d, path):
    """Optional exact API check; reads only the sibling's new stdlib module."""
    spec = importlib.util.spec_from_file_location('_test_prototype_fitter_api', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    d.check_fitter(module)
    witness = {'rows': 6355, 'classes': 1008, 'coefficients': 4096, 'feature_width': 32, 'arm': 'linear',
               'reference_energy': 10., 'reference_scale': .3125, 'lambda_bits': '0000003d',
               'actual_solve_checked': True, 'feature_energy': 10., 'lambda': .03125, 'stationarity_numerator': .00001,
               'stationarity_denominator': 1., 'normalized_stationarity': .00001, 'A_nonzero': True,
               'target': 'raw member-inclusive prototypes', 'intercept': False}
    module.check_fit_witness(witness, 'linear')
    module.check_fit_witness(dict(witness, arm='concat', feature_width=160, coefficients=20480), 'concat')
    rejects(lambda: module.check_fit_witness(dict(witness, arm='concat'), 'concat'))
    rejects(lambda: module.check_fit_witness(dict(witness, intercept=True), 'linear'))
    rejects(lambda: module.check_fit_witness(dict(witness, normalized_stationarity=.000011), 'linear'))
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    # Execute the actual typed payload boundary before any native tensor validation.
    boundary = next(n for n in functions['check_payload'].body if isinstance(n, ast.Expr) and
                    isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'require')
    reject_old = ast.parse('def reject_old(saved, ident, context):\n    pass').body[0]
    reject_old.body = [copy.deepcopy(boundary)]
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reject_old], type_ignores=[])),
                 '<actual typed payload boundary>', 'exec'), namespace)
    v1 = dict.fromkeys(module.PAYLOAD_KEYS); v1['schema'] = 'siglip2-prototype-residual-ridge-v1'
    rejects(lambda: namespace['reject_old'](v1, {}, {}), 'complete new typed payload')
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
    v2_fitter_check(d)
    synthetic_bootstrap_check(d, path.read_text())
    encoder_retention_check(d, path.read_text())
    closure_check(d)
    resources_adapter_check(d)
    terminal_adapter_check(d)
    rng_binding_check(d, path.read_text())
    flow_check(d, path.read_text())
    score_schedule_check(d, path.read_text())
    if args.fitter_root is not None:
        fitter_check(d, args.fitter_root / 'fit_siglip2_prototype_residual.py')
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
