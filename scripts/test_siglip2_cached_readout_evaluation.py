#!/usr/bin/env python3
"""Stdlib-only evaluator falsifier; native/resource/quality gates remain UNRUN.

Run python3 -B -S scripts/test_siglip2_cached_readout_evaluation.py.
Synthetic receipts/tensors are contract fixtures, never native evidence.
No Torch/NumPy/PIL/SSH/GPU/training/quality evaluation or consultation runs.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejects(call, message):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        assert message in str(error), (message, str(error))
        return
    raise AssertionError('invalid contract accepted: ' + message)


def descriptor(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return descriptor(path)


def steps(cached, count=1000):
    return [{'step': i, 'batch': list(range(64)), 'schedule_sha256': 'a' * 64,
             'feature_rows_sha256': 'b' * 64, 'state_sha256': 'c' * 64, 'ce': 1., 'rank': .1,
             'loss': 1.8, 'scale': 128., 'preclip_norm': 1., 'seconds': .001,
             'gradient_norms': dict.fromkeys(cached.PARAMETERS, 0.)} for i in range(1, count + 1)]


def endpoint_fixture(driver, cached, trainer_test, seed=179032, arm='control'):
    launch = trainer_test.launch_fixture(cached, 'train', arm, seed)
    launch['selected_cpu'] = {'receipt': 'actual cpu fixture'}
    launch['selected_mechanics'] = {a: {'receipt': 'actual mechanics ' + a} for a in cached.ARMS}
    default = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
               'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
    groups = [dict(default, lr=1e-4)] * 2
    identity = {'method': cached.method(launch), 'arm': arm, 'seed': seed, 'source': {'checkpoint': {'sha256': cached.SOURCE_SHA}},
                'selected_cpu': launch['selected_cpu'], 'parameter_names': cached.PARAMETERS,
                'numerical_flags': {}, 'optimizer_defaults': default, 'optimizer_groups': groups,
                'optimizer_serial_groups': [dict(groups[0], params=[0, 1, 2, 3]), dict(groups[1], params=[4])],
                **dict.fromkeys(('static_sha256', 'buffers_sha256', 'feature_state_sha256', 'schedule_sha256'), 'a' * 64)}
    endpoint = {'seed': seed, 'arm': arm, 'launch': {'path': '/train/launch.json', 'sha256': 'd' * 64},
                'terminal': {'receipt': {'path': '/train/receipt.json', 'sha256': 'e' * 64}},
                'checkpoint': {'path': '/train/resume.pt', 'sha256': 'f' * 64}, 'terminal_state_sha256': 'c' * 64}
    record = {'schema': cached.SCHEMA, 'phase': 'train', 'arm': arm, 'seed': seed, 'launch': launch,
              'completed_step': 1000, 'optimizer_members': 5, 'source': identity['source'],
              'terminal_cgroups': {}, 'checkpoint': endpoint['checkpoint'], 'terminal_state_sha256': endpoint['terminal_state_sha256'],
              'resource_policy': cached.policy('train'), 'code': {}, 'numerical_flags': {}, 'head_scalars': 188544,
              'trainable_scalars': 445056, 'encoder_updates': 0, 'resumed_steps': [], 'identity': identity,
              'peak_cuda_allocated_bytes': 1000000, 'steps': steps(cached), 'median_update_seconds': .001,
              'training_wall_seconds': 2., 'wall_seconds': 10., 'service_seconds': 12., 'initial_state_sha256': 'g' * 64,
              'affine_reconstruction': {'residual_energy': 1., 'fit_unexplained_energy_fraction': .03},
              **dict.fromkeys(('pass', 'training_qualified', 'strict_reload_exact', 'exit_rehash_pass', 'fixed_cached_views'), True),
              **dict.fromkeys(('quality_read', 'trained_state_reused', 'training_state_discarded', 'image_augmentation', 'replay_exact'), False)}
    context = {'launch': launch, 'source': identity['source'], 'terminal_cgroups': {}, 'code': {},
               'old_cpu': {'numerical_flags': {}, 'state': {'optimizer_defaults': default}},
               'terminals': {'mechanics:' + a: {'initial_state_sha256': record['initial_state_sha256'],
                            'steps': copy.deepcopy(record['steps'][:17])} for a in driver.ARMS}}
    return endpoint, record, context


def endpoint_checks(driver, cached, trainer_test):
    for seed, arm in driver.ORDER:
        endpoint, record, context = endpoint_fixture(driver, cached, trainer_test, seed, arm)
        driver.check_endpoint(cached, record, endpoint, context)
        mutations = [
            (lambda m: m.update(completed_step=999), 'complete TRAIN1000'),
            (lambda m: m.update(optimizer_members=208), 'complete TRAIN1000'),
            (lambda m: m.update(trained_state_reused=True), 'fresh complete TRAIN1000'),
            (lambda m: m.update(strict_reload_exact=False), 'fresh complete TRAIN1000'),
            (lambda m: m.update(peak_cuda_allocated_bytes=10_000_000_000), 'fresh complete TRAIN1000'),
            (lambda m: m.update(quality_read=True), 'fresh complete TRAIN1000'),
            (lambda m: m.update(resumed_steps=[{}]), 'complete TRAIN1000'),
            (lambda m: m['identity'].update(parameter_names=cached.PARAMETERS[:-1]), 'five-member endpoint'),
            (lambda m: m['identity']['optimizer_groups'][0].update(lr=.001), 'five-member endpoint'),
            (lambda m: m['identity']['optimizer_serial_groups'][0].update(params=[0, 1, 2, 4]), 'five-member endpoint'),
            (lambda m: m['steps'].pop(), 'complete update records'),
            (lambda m: m['steps'][-1].update(state_sha256='e' * 64), 'terminal state/schedule'),
            (lambda m: m.update(median_update_seconds=.5), 'terminal state/schedule'),
            (lambda m: m.update(training_wall_seconds=.5), 'terminal state/schedule'),
            (lambda m: m['steps'][20].update(schedule_sha256='d' * 64), 'terminal state/schedule'),
            (lambda m: m['steps'][20].update(batch=[13283] * 64), 'complete finite update'),
            (lambda m: m['affine_reconstruction'].update(residual_energy=-1), 'FIT affine reconstruction')]
        if seed == driver.SEEDS[0]:
            mutations.append((lambda m: m['steps'][0].update(feature_rows_sha256='d' * 64), 'fresh original mechanics replay'))
        for mutation, message in mutations:
            changed = copy.deepcopy(record)
            mutation(changed)
            rejects(lambda: driver.check_endpoint(cached, changed, endpoint, context), message)
    saved, ident = trainer_test.fixture(cached)
    saved['counter'] = 1000
    saved['scaler']['_growth_tracker'] = 1000
    for moments in saved['optimizer']['state'].values():
        moments['step'] = trainer_test.Step(1000)
    cached.check_payload(saved, ident, 1000)
    for mutation, message in (
        (lambda m: m['optimizer']['state'].update({5: m['optimizer']['state'][4]}), 'EXACT five optimizer'),
        (lambda m: m['head'].update(center=trainer_test.Tensor((1024,))), 'complete tensor layout'),
        (lambda m: m.update(cuda_rng=[]), 'optimizer counters/scaler/RNG'),
        (lambda m: m['scaler'].update(_growth_tracker=999), 'optimizer counters/scaler/RNG'),
        (lambda m: m['schedules'].update({'179041': trainer_test.Tensor((100, 64), 'torch.int64')}), 'complete tensor layout')):
        changed = copy.deepcopy(saved)
        mutation(changed)
        rejects(lambda: cached.check_payload(changed, ident, 1000), message)


def metric_checks(driver, score, helper, cached, trainer_test):
    score.export = helper
    quality = {seed: {arm: {'per_query_r1': [0.1 if arm == 'control' else .103] * 6354,
                           'per_query_ap': [.2 if arm == 'control' else .203] * 6354}
                       for arm in driver.ARMS} for seed in driver.SEEDS}
    deltas, average = score.averaged_deltas(quality)
    intervals = {m: {'mean_delta': statistics_mean(average[m]), 'product_lower95': .001,
                    'product_upper95': .01, 'query_lower95': -.001, 'query_upper95': .01} for m in score.METRICS}
    assert score.quality_gate(deltas, intervals) == (True, True)
    for metric in score.METRICS:
        changed = copy.deepcopy(intervals)
        changed[metric]['product_lower95'] = 0
        assert score.quality_gate(deltas, changed) == (True, False)
        changed[metric]['product_lower95'] = float('nan')
        assert score.quality_gate(deltas, changed) == (True, False)
    def gate(r1, ap):
        ds = {s: {'per_query_r1': [r1[i]] * 6354, 'per_query_ap': [ap[i]] * 6354}
              for i, s in enumerate(driver.SEEDS)}
        ci = {m: dict(intervals[m], mean_delta=(statistics_mean(ds[179032][m]) + statistics_mean(ds[179041][m])) / 2)
              for m in score.METRICS}
        return score.quality_gate(ds, ci)
    assert gate((.002, .002), (.002, .002)) == (True, True)
    assert gate((.001999, .001999), (.003, .003)) == (True, False)
    assert gate((0, .006), (.003, .003)) == (False, False)
    assert gate((.003, .003), (-.001, .01)) == (False, False)
    assert gate((.003, .003), (0, .004)) == (True, True)
    bad = copy.deepcopy(quality)
    bad[179032]['candidate']['per_query_ap'][-1] = float('nan')
    rejects(lambda: score.averaged_deltas(bad), '')
    records = {key: endpoint_fixture(driver, cached, trainer_test, *key)[1] for key in driver.ORDER}
    result = driver.paired_cost(records)
    assert set(result) == {'179032', '179041'} and result['179032']['control']['service_seconds'] == 12.
    for key, limit in (('service_seconds', 18.), ('median_update_seconds', .0015)):
        changed = copy.deepcopy(records)
        changed[179032, 'candidate'][key] = limit
        driver.paired_cost(changed)
        changed[179032, 'candidate'][key] = math.nextafter(limit, math.inf)
        rejects(lambda: driver.paired_cost(changed), 'fresh whole-service/median cost')
    changed = copy.deepcopy(records)
    changed[179041, 'candidate']['steps'][-1]['feature_rows_sha256'] = 'f' * 64
    rejects(lambda: driver.paired_cost(changed), 'paired cached inputs')
    changed = copy.deepcopy(records)
    changed[179041, 'candidate']['service_seconds'] = float('inf')
    rejects(lambda: driver.paired_cost(changed), 'finite original endpoint cost')
    changed = copy.deepcopy(records)
    changed.pop((179041, 'control'))
    rejects(lambda: driver.paired_cost(changed), 'all four original endpoint costs')


def statistics_mean(values):
    import statistics
    return statistics.mean(values)


def split_checks(helper):
    products = ['held-' + str(i) for i in range(1993)]
    rows = [{'relative_path': 'Img/img/' + str(i) + '.jpg', 'product': products[i % 1993], 'image_sha256': 'a' * 64}
            for i in range(12599)]
    fit = {'rows': [{'product': 'fit'}], 'class_names': ['fit']}
    frozen = {'fit_manifest': fit['rows'], 'held_manifest': rows, 'query': list(range(6354)), 'gallery': list(range(6354, 12599))}
    helper.validate_split(frozen, fit)
    for mutation, message in (
        (lambda m: m['gallery'].__setitem__(0, 0), 'original TRAIN-held split'),
        (lambda m: m['query'].pop(), 'original TRAIN-held split'),
        (lambda m: m['held_manifest'][0].update(product='fit'), 'original TRAIN-held split'),
        (lambda m: m['held_manifest'][0].update(relative_path='../outside.jpg'), 'held manifest row'),
        (lambda m: m['held_manifest'][0].update(image_sha256='bad'), 'held manifest row')):
        changed = copy.deepcopy(frozen)
        mutation(changed)
        rejects(lambda: helper.validate_split(changed, fit), message)


def authority_checks(driver, cached, trainer_test):
    endpoints = [endpoint_fixture(driver, cached, trainer_test, seed, arm)[0] for seed, arm in driver.ORDER]
    spec = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'a' * 64,
            'training': {'root': '/frozen/trainer2-v2', 'execution_sha256': driver.TRAIN_EXECUTION_SHA},
            'endpoints': endpoints, 'frozen_split': {'path': '/frozen/split.json', 'sha256': 'b' * 64},
            'resource_policies': {p: driver.policy(p) for p in ('cpu', 'export', 'score')},
            'cost_policy': driver.COST_POLICY.copy(), 'both_locks_held': True}
    args = SimpleNamespace(execution_sha256='a' * 64, authority=Path('/frozen/evaluation.json'),
                           authority_sha256='c' * 64, phase='cpu')
    driver.check_spec(spec, args)
    for mutation, message in (
        (lambda m: m.update(extra=True), 'evaluation authority profile'),
        (lambda m: m.update(both_locks_held=False), 'evaluation authority profile'),
        (lambda m: m['cost_policy'].update(whole_service_ratio_max=1.51), 'evaluation authority profile'),
        (lambda m: m['resource_policies']['export'].update(host_bytes=16 * 1024**3), 'evaluation authority profile'),
        (lambda m: m['training'].update(execution_sha256='0' * 64), 'exact corrected training REFERENCE'),
        (lambda m: m['endpoints'].pop(), 'four ordered endpoints'),
        (lambda m: m['endpoints'].reverse(), 'four ordered endpoints'),
        (lambda m: m['endpoints'].__setitem__(3, m['endpoints'][2]), 'four ordered endpoints'),
        (lambda m: m['endpoints'][0].update(partial=True), 'exact endpoint descriptor'),
        (lambda m: m['endpoints'][0].update(terminal_state_sha256='future'), 'exact endpoint descriptor')):
        changed = copy.deepcopy(spec)
        mutation(changed)
        rejects(lambda: driver.check_spec(changed, args), message)
    context = {'args': args, 'spec': spec, 'code': {}, 'selected': {'source': {}, 'old_cpu': {
        'numerical_flags': {}, 'invocation': {'python': '/python', 'python_sha256': 'd' * 64, 'python_version': 'fixture'}}},
        'frozen': {'query': list(range(6354)), 'gallery': list(range(6354, 12599))}}
    record = {**driver.bind(context), 'schema': driver.SCHEMA, 'phase': 'cpu', 'resource_policy': driver.policy('cpu'),
              'optimizer_updates': 0, 'optimizer_members': 5, 'numerical_flags': {},
              'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'output': '/new/cpu',
              'prerequisite': None, 'cuda_initialized': False, 'peak_cuda_allocated_bytes': 0, 'files': {},
              **dict.fromkeys(('pass', 'strict_independent_head_reload_exact', 'source_reload_exact', 'fit_raw_unit_packed_exact',
                              'first_heads_released_before_reload', 'source_head_rng_flags_preserved', 'exit_rehash_pass'), True),
              **dict.fromkeys(('quality_read', 'official_read', 'claim_eligible', 'public_serving_qualified', 'public_latency_measured'), False)}
    record['invocation'] = {'argv': driver.cli_argv(args.authority, args.authority_sha256, args.execution_sha256,
                                                  'cpu', Path(record['output']), None), 'cuda_visible_devices': '',
                            **context['selected']['old_cpu']['invocation']}
    driver.check_receipt(context, record, 'cpu')
    for mutation, message in (
        (lambda m: m.update(optimizer_members=208), 'accepted updated CPU/export'),
        (lambda m: m.update(quality_read=True), 'accepted updated CPU/export'),
        (lambda m: m.update(strict_independent_head_reload_exact=False), 'accepted updated CPU/export'),
        (lambda m: m.update(cuda_initialized=True), 'CPU-only qualification'),
        (lambda m: m['invocation'].update(cuda_visible_devices='0'), 'original evaluator invocation'),
        (lambda m: m['invocation']['argv'].append('--quality-preview'), 'original evaluator invocation')):
        changed = copy.deepcopy(record)
        mutation(changed)
        rejects(lambda: driver.check_receipt(context, changed, 'cpu'), message)
    prior = {'path': '/frozen/updated-cpu-terminal.json', 'sha256': 'e' * 64}
    exported = {**record, 'phase': 'export', 'resource_policy': driver.policy('export'), 'prerequisite': prior,
                'output': '/new/export', 'full_held_independent_raw_unit_packed_exact': True, 'batch': 32,
                'precision': driver.PRECISION, 'cache_shape': [12599, 1152], 'query_images': 6354,
                'gallery_images': 6245, 'held_products': 1993, 'batch_sizes': driver.batch_sizes(context),
                'files': dict.fromkeys(driver.file_names(), 'f' * 64), 'peak_cuda_allocated_bytes': 1000}
    exported['invocation'] = {**record['invocation'], 'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8',
                             'argv': driver.cli_argv(args.authority, args.authority_sha256, args.execution_sha256,
                                                    'export', Path(exported['output']), prior)}
    driver.check_receipt(context, exported, 'export')
    for mutation, message in (
        (lambda m: m.update(full_held_independent_raw_unit_packed_exact=False), 'all four full held export'),
        (lambda m: m['files'].pop('held1152.npy'), 'all four full held export'),
        (lambda m: m['files'].pop('candidate-179041.packed.bin'), 'all four full held export'),
        (lambda m: m.update(cache_shape=[12599, 1024]), 'all four full held export'),
        (lambda m: m.update(precision='half-model'), 'all four full held export'),
        (lambda m: m['batch_sizes']['gallery'].pop(), 'all four full held export'),
        (lambda m: m.update(peak_cuda_allocated_bytes=10_000_000_000), 'all four full held export')):
        changed = copy.deepcopy(exported)
        mutation(changed)
        rejects(lambda: driver.check_receipt(context, changed, 'export'), message)


def terminal_checks(driver, root, original, initializer, helper):
    """Real stdlib terminal reader rejects unauthenticated normal exits/footers."""
    admission = original.FlatAdmission()
    admission.init = initializer
    identity, unit = 'a' * 32, 'synthetic-unit'
    cgroup = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
        'memory.max': str(8 * 1024**3), 'memory.current': '1024', 'memory.peak': '2048',
        'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
        'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n'}}
    record = {'invocation': {'invocation_id': identity, 'optimize': 0}, 'wall_seconds': 10.,
              'process_peak_rss_kib': 1000, 'cgroup_before': cgroup, 'cgroup_after': copy.deepcopy(cgroup)}
    footer = {**copy.deepcopy(cgroup), 'invocation_id': identity}
    log = '\n'.join([f'Running as unit: {unit}.service; invocation ID: {identity}', '\tExit status: 0',
        'Finished with result: success', 'Main processes terminated with: code=exited/status=0', '\tSwaps: 0',
        'Memory swap peak: 0B', '\tMaximum resident set size (kbytes): 1000', 'Service runtime: 12.0s',
        'FINAL_CGROUP ' + json.dumps(footer)]) + '\n'
    path = root / 'terminal.log'
    path.write_text(log)
    terminal = {'receipt': write_json(root / 'receipt.json', record), 'log': descriptor(path), 'unit': unit,
                'invocation_id': identity, 'service_seconds': 12., 'native_peak_rss_kib': 1000, 'both_locks_held': True}
    helper.zero_events(admission.admit_terminal(record, terminal, 120, {}))
    for changed_log, message in (
        (log.replace('\tExit status: 0', '\tExit status: 1'), 'original normal-exit log'),
        (log.replace('Service runtime: 12.0s', 'Service runtime: 11.0s'), 'original service runtime numeric'),
        (log + 'FINAL_CGROUP ' + json.dumps(footer) + '\n', 'original final cgroup footer')):
        path.write_text(changed_log)
        changed = dict(terminal, log=descriptor(path))
        rejects(lambda: original.FlatAdmission().admit_terminal(record, changed, 120, {}), message)
    path.write_text(log)
    changed = dict(terminal, log=descriptor(path), both_locks_held=False)
    rejects(lambda: admission.admit_terminal(record, changed, 120, {}), 'terminal descriptor/both locks')
    changed_cgroup = copy.deepcopy(cgroup)
    changed_cgroup['values']['memory.events'] = 'low 0\nhigh 1\nmax 0\noom 0\noom_kill 0\n'
    rejects(lambda: helper.zero_events(changed_cgroup), 'whole-unit memory events must be zero')
    changed_cgroup['values']['memory.swap.current'] = '1'
    rejects(lambda: initializer.admit_cgroup(changed_cgroup, unit), 'whole-cgroup memory/swap')


def file_checks(driver, root):
    path = root / 'original'
    path.write_bytes(b'original bytes')
    ref = descriptor(path)
    driver.bound_file({}, path, ref['sha256'])
    path.write_bytes(b'changed bytes')
    rejects(lambda: driver.bound_file({}, path, ref['sha256']), 'file SHA256 differs')
    link = root / 'alias'
    link.symlink_to(path)
    rejects(lambda: driver.bound_file({}, link, descriptor(path)['sha256']), 'canonical file')
    rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate JSON')
    rejects(lambda: driver.strict_json('{"a":Infinity}'), 'nonfinite JSON')
    for name in driver.FILES:
        (root / name).write_text('# immutable\n')
    code = {name: descriptor(root / name)['sha256'] for name in driver.FILES}
    ref = write_json(root / 'execution.json', code)
    driver.closure(root, ref['sha256'], driver.FILES, {})
    for mutant in (dict(code, extra='0' * 64), {next(iter(code)): next(iter(code.values()))}):
        ref = write_json(root / 'execution.json', mutant)
        rejects(lambda: driver.closure(root, ref['sha256'], driver.FILES, {}), 'exact execution closure')


def source_checks(driver, path, helper, score):
    tree = ast.parse(path.read_bytes())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert 'reset_peak_memory_stats' not in attributes and 'half' not in attributes
    assert 'exit_rehash' in attributes and 'check_payload' in attributes and 'fingerprint' in attributes
    head = ast.unparse(functions['head_values'])
    assert 'pack_int8_unit_embeddings(unit.cpu())' in head and 'head(cache.float())' in head
    assert 'enabled=False' in head and 'F.normalize(raw, dim=1)' in head
    export = ast.unparse(functions['export_wires'])
    assert 'torch.autocast(device_type=\'cuda\', dtype=torch.float16)' in export
    assert 'F.normalize(pooled.float(), dim=1)' in export and 'model.cuda()' in export
    assert 'head_values(context, head, torch.from_numpy(cache[rows]).cuda())' in export
    assert 'np.array_equal(inverse.view(np.uint16)' in export
    loader = ast.unparse(functions['load_head'])
    assert 'cached.check_payload(saved, ident, 1000)' in loader
    assert 'map_location=\'cpu\'' in loader and 'weights_only=True' in loader
    assert 'cached.fresh' not in loader and 'optimizer.load_state_dict' not in loader
    source = ast.unparse(functions['load_source'])
    assert 'original.load_vision' in source and 'source.model_facts' in source and 'src[\'proof\'][\'sample\']' in source
    run = ast.unparse(functions['run'])
    assert run.index('authority(args)') < run.index('prerequisites(context)') < run.index('native_start(context)') < run.index('score_wires(context, export)')
    auth = ast.unparse(functions['authority'])
    assert auth.index('check_endpoint') < auth.index('paired_cost(records)') < auth.index('helper.validate_split')
    score_src = ast.unparse(functions['score_wires'])
    assert score_src.index('packed.to_bytes()') < score_src.index('fixed.packed_quality')
    assert 'scoring.averaged_deltas(quality)' in score_src and 'scoring.quality_gate(deltas, intervals)' in score_src
    assert driver.COST_POLICY == helper.COST_POLICY
    for name in ('export_siglip2_substrate_adaptation.py', 'score_siglip2_substrate_adaptation.py'):
        assert descriptor(path.parent / name)['sha256'] == driver.PINS[name]
    for name, digest in driver.TRAIN_PINS.items():
        assert descriptor(path.parent / name)['sha256'] == digest
    for name, pin in helper.REFERENCES.items():
        source_path = path.parent / name.removeprefix('reference_')
        assert descriptor(source_path)['sha256'] == driver.PINS[name] == pin['source']
        reference = ast.parse(source_path.read_bytes())
        nodes = [n for n in reference.body if isinstance(n, ast.FunctionDef) and n.name == pin['name']]
        assert len(nodes) == 1
        selected = ast.Module(body=nodes, type_ignores=[])
        assert hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest() == pin['ast']
    scorer = ast.unparse(nodes[0])
    assert 'np.random.default_rng(179019)' in scorer and 'np.empty(5000)' in scorer
    assert 'np.quantile(draws, 0.025)' in scorer
    assert set(driver.file_names()) == {'held1152.npy'} | {a + '-' + str(s) + suffix for s, a in driver.ORDER
                                                        for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}


def main():
    path = Path(__file__).absolute().with_name('evaluate_siglip2_cached_readout.py')
    native = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
    old_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in native:
            raise AssertionError('native import attempted by stdlib falsifier: ' + name)
        return old_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    try:
        driver = load('cached_evaluation', path)
        cached = load('cached_trainer_fixture', path.with_name('train_siglip2_cached_readout.py'))
        trainer_test = load('cached_trainer_test_fixture', path.with_name('test_siglip2_cached_readout.py'))
        helper = load('held_helpers_fixture', path.with_name('export_siglip2_substrate_adaptation.py'))
        score = load('score_helpers_fixture', path.with_name('score_siglip2_substrate_adaptation.py'))
        original = load('original_trainer_fixture', path.with_name('train_siglip2_substrate_adaptation.py'))
        initializer = load('initializer_fixture', path.with_name('initialize_siglip2_substrate_fit.py'))
        authority_checks(driver, cached, trainer_test)
        endpoint_checks(driver, cached, trainer_test)
        metric_checks(driver, score, helper, cached, trainer_test)
        split_checks(helper)
        source_checks(driver, path, helper, score)
        with TemporaryDirectory() as directory:
            file_checks(driver, Path(directory))
        with TemporaryDirectory() as directory:
            terminal_checks(driver, Path(directory), original, initializer, helper)
        assert not native.intersection(n.split('.')[0] for n in sys.modules)
    finally:
        builtins.__import__ = old_import
    for flags in ([], ['-O'], ['-OO']):
        result = subprocess.run([sys.executable, '-B', '-S', *flags, str(path), '--help'], capture_output=True, text=True)
        assert result.returncode == (1 if flags else 0), result.stderr
        assert ('optimized mode is forbidden' in result.stderr) if flags else ('--phase {cpu,export,score}' in result.stdout)
    print('PASS stdlib four-endpoint/five-payload/replay/cost/metric/split/closure/tamper/source checks; help/-O/-OO; native gates UNRUN')


if __name__ == '__main__':
    main()
