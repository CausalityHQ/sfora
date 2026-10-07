#!/usr/bin/env python3
"""Stdlib source-only falsifiers; no native/model-fit/quality qualification."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import statistics
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'evaluate_siglip2_connected_mlp.py'
EVIDENCE = HERE.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'


def rejects(call, text):
    try:
        call()
    except (ValueError, KeyError, TypeError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError('accepted mutant: ' + text)


def metadata_api():
    assert DRIVER.is_file(), 'connected evaluator metadata validator is missing'
    tree = ast.parse(DRIVER.read_text())
    names = {'require', 'sha', 'check_file', 'check_unit', 'check_endpoint',
             'label', 'check_endpoint_binding'}
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign)) or
             isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'__file__': str(DRIVER)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DRIVER), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def nested_fixture(e, seed=179061, arm='candidate'):
    # The actual CPU receipt is a source fixture only: its qualification is NOT
    # passed to training admission. Keep the trainer's real nested payload shape.
    cpu = json.loads((EVIDENCE / 'connected-mlp-cpu-v3/receipt.json').read_text())
    result = copy.deepcopy(cpu['qualifications'][0])
    result['training_updates'] = 128
    result['identity'].update(seed=seed, arm=arm, device='cuda')
    root = Path('/fixture') / (arm + '-' + str(seed))
    result['checkpoint']['path'] = str(root / (root.name + '-terminal.pt'))
    result['parity']['bundle']['path'] = str(root / (root.name + '-bundle/bundle.json'))
    base = result['identity']['base_vision']
    if arm == 'control':
        result['parity']['vision_sha256'] = base['sha256']
    fact = lambda name: {'path': str(root / name), 'sha256': 'a' * 64}
    endpoint = {'seed': seed, 'arm': arm, 'launch': fact('authority.json'),
        'terminal': {'receipt': fact('receipt.json'), 'log': fact('original.log'),
            'unit': root.name, 'invocation_id': hashlib.md5(root.name.encode()).hexdigest(),
            'service_seconds': 500., 'native_peak_rss_kib': 7000000, 'both_locks_held': True},
        'checkpoint': result['checkpoint'], 'terminal_state_sha256': result['terminal_state_sha256'],
        'bundle': result['parity']['bundle'], 'inference_state_sha256': 'b' * 64}
    code = json.loads((EVIDENCE / 'connected-mlp-cpu-v3-freeze/execution.json').read_text())
    manifest = {'schema': 'siglip2-connected-mlp-bundle-v1',
        'endpoint_state_sha256': endpoint['inference_state_sha256'], 'code': code,
        'files': {'vision.pt': base['checkpoint']['sha256']},
        'vision_sha256': result['parity']['vision_sha256'], 'base_vision_sha256': base['sha256'],
        'encoder_identity': result['identity']['encoder_identity'], 'scope': result['identity']['scope']}
    record = {'phase': 'train', 'arm': arm, 'seed': seed, 'authority': endpoint['launch'], 'result': result}
    return endpoint, record, manifest, {'code': code}


def nested_binding():
    e = metadata_api()
    visits = []
    for seed, arm in e.ORDER:
        endpoint, record, manifest, training = nested_fixture(e, seed, arm)
        e.check_endpoint(endpoint)
        e.check_endpoint_binding(endpoint, record, manifest, training)
        visits.append((seed, arm))
    assert visits == [(179061, 'control'), (179061, 'candidate'), (179069, 'candidate'), (179069, 'control')]
    endpoint, record, manifest, training = nested_fixture(e)
    entered = []

    def admit_then_native_and_score(value):
        e.check_endpoint_binding(endpoint, record, value, training)
        entered.append('native/scorer sentinel')

    replaced = {**manifest, 'vision_sha256': manifest['base_vision_sha256']}
    rejects(lambda: admit_then_native_and_score(replaced), 'updated encoder')
    assert entered == [], 'substituted base reached native/scorer'
    legacy = {**record, 'checkpoint': record['result']['checkpoint']}
    del legacy['result']
    rejects(lambda: e.check_endpoint_binding(endpoint, legacy, manifest, training), 'result')
    for key in ('endpoint_state_sha256', 'base_vision_sha256'):
        rejects(lambda key=key: e.check_endpoint_binding(endpoint, record,
            {**manifest, key: '0' * 64}, training), 'endpoint')
    wrong = copy.deepcopy(record)
    wrong['result']['parity']['bundle']['sha256'] = '0' * 64
    rejects(lambda: e.check_endpoint_binding(endpoint, wrong, manifest, training), 'endpoint')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_contract(e, trainer, reference):
    tree = ast.parse(DRIVER.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    source = lambda name: ast.unparse(functions[name])
    actual = ast.parse((HERE / 'train_siglip2_connected_mlp.py').read_text())
    api = {n.name for n in actual.body if isinstance(n, ast.FunctionDef)}
    # Every direct connected trainer call must exist in the actual public API.
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'trainer'}
    assert calls <= api, calls - api
    for name, digest in e.EVALUATOR_PINS.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    for name, digest in {**e.GENUINE_PINS, **e.NEAREST_EVALUATOR['code'], **e.REFERENCE['code']}.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    assert e.FILES == {'evaluate_siglip2_connected_mlp.py', 'test_connected_mlp_evaluation.py'}
    assert e.TRAIN_FILES == trainer.FILES
    freeze = EVIDENCE/'connected-mlp-cpu-v3-freeze'
    source_freeze = json.loads((freeze/'freeze.json').read_text())
    assert e.TRAINING['root'] == source_freeze['source_root']
    assert e.TRAINING['execution_sha256'] == hashlib.sha256((freeze/'execution.json').read_bytes()).hexdigest()
    assert e.TRAINING['code'] == json.loads((freeze/'execution.json').read_text())
    for name,digest in e.TRAINING['code'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    cpu_unit = json.loads((EVIDENCE/'connected-mlp-mechanics-control-179061-v1-freeze'/
        'authority-mechanics-control-179061-v1.json').read_text())['selected_cpu']
    assert e.TRAINING_CPU == cpu_unit
    assert e.TRAINING_CPU['receipt']['sha256'] == hashlib.sha256((EVIDENCE/'connected-mlp-cpu-v3/receipt.json').read_bytes()).hexdigest()
    assert e.policy('cpu')['seconds'] == e.policy('score')['seconds'] == 500
    assert e.policy('export')['seconds'] == 900
    assert e.COST_POLICY['whole_service_ratio_max'] == e.COST_POLICY['total_training_core_ratio_max'] == 1.50
    assert 'candidate_cache' not in e.LAUNCH_KEYS and 'evaluator_reference' in e.LAUNCH_KEYS
    for name in ('require', 'strict_json', 'sha', 'check_file', 'bound_file', 'read_json', 'closure',
                 'merge_guards', 'load_authenticated', 'policy', 'check_unit', 'check_resource_facts',
                 'batch_sizes', 'resources', 'accept_unit'):
        prior = next(n for n in ast.parse((HERE / 'evaluate_siglip2_identity_diversity.py').read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(functions[name], include_attributes=False) == ast.dump(prior, include_attributes=False), name
    assert source('run').index('authority(args)') < source('run').index('native_start(context)')
    admission = source('authority')
    assert admission.index('trainer.authority(targs)') < admission.index("admit_endpoints(context, launch['endpoints'][:2])")
    assert admission.index("first_receipt['decision'] == 'CONTINUE'") < admission.index("admit_endpoints(context, launch['endpoints'][2:])")
    assert "selection['launch']['endpoints'] == launch['endpoints']" in admission
    admission = source('admit_endpoints')
    assert "'mechanics', a, seed" in admission and 'fresh_control' in admission
    assert admission.index('check_endpoint_binding(') < admission.index('paired_cost(')
    assert 'trainer.admit_terminal(t, unit, phase, arm, seed)' in source('admit_training_unit')
    assert 'trainer.check_payload(t, disk, ident, 128)' in source('authenticate_payloads')
    assert 'consumed=pages.consume' in source('authenticate_payloads')
    export = source('native_export')
    assert 'range(2)' in export and "directory / 'train_siglip2_connected_mlp.py'" in export
    assert "portable.load_inference(directory, endpoint['bundle']['sha256'], 'cuda')" in export
    assert export.index('authenticate_payloads(') < export.index('.image_rows(') < export.index('portable.load_inference(')
    assert 'portable.release_inference(state)' in export and 'readback_wires' in export
    assert 'bundle_reads_only' in source('images_outputs') and 'trainer.portable_mutants' in source('images_outputs')
    assert 'encoder_facts' in source('endpoint_facts') and 'serving=True' in source('endpoint_facts')
    assert 'oracle=start == 0 or start + 32 >= len(indices)' in source('export_pass')
    assert 'e.decide(' in source('score_exports') and 'e.immediate_quality_pass(' in source('score_exports')
    assert 'paired_intervals(fixed, average' in source('score_exports')
    assert source('score_exports').index('repeated = native.read_wires(') < source('score_exports').index('quality_after_readiness(')
    assert 'api.exit_rehash(' in source('exit_rehash') and 'trainer.batch_bound_files({},' in source('exit_rehash')
    assert e.NEAREST_EVALUATOR == reference.NEAREST_EVALUATOR
    # No assignments to imported module globals or reconstructed function objects.
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'FunctionType'
                   for n in ast.walk(tree))


def launch_contract(e):
    def unit(number):
        root = '/fixture/unit-' + str(number)
        fact = lambda suffix: {'path': root + suffix, 'sha256': 'a' * 64}
        return {'receipt': fact('/receipt.json'), 'log': fact('/original.log'), 'unit': 'unit-' + str(number),
            'invocation_id': f'{number:032x}', 'service_seconds': 400., 'native_peak_rss_kib': 7000000,
            'both_locks_held': True}
    training = e.TRAINING or {'root': '/fixture/unfrozen', 'execution_sha256': 'a' * 64,
                             'code': dict.fromkeys(e.TRAIN_FILES, 'b' * 64)}
    for stage in ('first', 'full'):
        for phase in ('cpu', 'export', 'score'):
            endpoints = [nested_fixture(e, seed, arm)[0] for seed, arm in e.endpoint_order(stage)]
            launch = {'schema': e.AUTHORITY_SCHEMA, 'execution_sha256': 'c' * 64,
                'training': copy.deepcopy(training), 'nearest_evaluator': e.NEAREST_EVALUATOR,
                'genuine_evaluator': {'root': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4',
                    'execution_sha256': e.GENUINE_EXECUTION_SHA, 'code': e.GENUINE_PINS}, 'reference': e.REFERENCE,
                'evaluator_reference': {'root': '/fixture/evaluator-reference', 'execution_sha256': 'd' * 64, 'code': e.EVALUATOR_PINS},
                'phase': phase, 'arm': 'control' if phase == 'export' else None,
                'seed': 179061 if phase == 'export' else None, 'stage': stage, 'panel': 'selection', 'endpoints': endpoints,
                'selected_cpu': None if phase == 'cpu' else unit(30),
                'exports': {e.label(ep): unit(i+40) for i,ep in enumerate(endpoints)} if phase == 'score' else {},
                'first_selection': unit(50) if stage == 'full' else None, 'selection_go': None,
                'resource_policies': {p: e.policy(p) for p in ('cpu', 'export', 'score')},
                'cost_policy': e.COST_POLICY, 'both_locks_held': True, 'selection_previously_exposed': True,
                'scope': {'path': '/fixture/scope.json', 'sha256': e.SCOPE_SHA256}}
            args = SimpleNamespace(execution_sha256=launch['execution_sha256'], phase=phase,
                arm=launch['arm'], seed=launch['seed'], authority=Path('/fixture/launch'), authority_sha256='f'*64,
                output=Path('/fixture/new-output'))
            if e.TRAINING is None:
                rejects(lambda: e.check_launch(launch,args), 'parent-frozen trainer2')
                continue
            e.check_launch(launch, args)
            for key,value in (('training',{**training,'root':'/foreign'}),('scope',{'path':'relative','sha256':e.SCOPE_SHA256}),
                    ('endpoints',list(reversed(endpoints))),('both_locks_held',False),
                    ('selection_previously_exposed',False),('resource_policies',{})):
                rejects(lambda key=key,value=value: e.check_launch({**launch,key:value},args), '')
            rejects(lambda: e.check_launch({**launch,'candidate_cache':{}},args), 'launch')
            if stage == 'full':
                rejects(lambda: e.check_launch({**launch,'first_selection':None},args), 'continuation')
            if phase != 'cpu' and stage == 'full':
                e.check_launch({**launch,'panel':'validation','selection_go':unit(60)},args)
                rejects(lambda: e.check_launch({**launch,'panel':'validation'},args), 'GO')
            assert vars(e.parser().parse_args(e.cli(args)[1:])) == vars(args)


def paired_decisions(e, reference, math_helper, trainer):
    control,candidate = (nested_fixture(e,arm=a)[1] for a in e.ARMS)
    c,a = control['result']['identity'],candidate['result']['identity']
    for ident in (c,a):
        ident['parameter_names'],ident['parameter_shapes'],_ = trainer.parameter_roles(ident['arm'])
    selected = {'selected_cpu': {'cpu':'shared'}, 'selected_mechanics': {'control':'C061','candidate':'A061'}}
    control['launch'] = copy.deepcopy(selected); candidate['launch'] = copy.deepcopy(selected)
    e.check_paired_initialization(trainer,control,candidate)
    changed = copy.deepcopy(candidate); changed['result']['identity']['validator_identity']['initial_cpu_rng_sha256'] = '0'*64
    rejects(lambda:e.check_paired_initialization(trainer,control,changed), 'same-seed')
    n = e.PANELS['selection'][1]
    def quality(hits,ap):
        r1 = [1]*hits + [0]*(n-hits)
        return {'recall_at_1':statistics.mean(r1),'map_at_r':ap,'per_query_r1':r1,'per_query_ap':[ap]*n}
    source,concat = quality(1660,.80),quality(1673,.8177754035543956)
    pair = {'control':quality(1676,.825),'candidate':quality(1683,.829)}
    for stage in ('first','full'):
        costs = reference.paired_cost({(s,a):{'service_seconds':400.,'total_training_core_seconds':200.}
            for s in e.seeds(stage) for a in e.ARMS},stage)
        quality_by_seed = {str(s):copy.deepcopy(pair) for s in e.seeds(stage)}
        _,average = math_helper.averaged_deltas(quality_by_seed,stage,'selection')
        intervals = {m:{'mean_delta':statistics.mean(values),'product_lower95':.001,'product_upper95':.01,
            'query_lower95':-.001,'query_upper95':.01} for m,values in average.items()} if stage == 'full' else {}
        result = reference.decide(math_helper,quality_by_seed,source,concat,stage,'selection',intervals,costs)
        assert result['decision'] == ('CONTINUE' if stage == 'first' else 'GO')
        quality_by_seed['179061']['candidate'] = copy.deepcopy(pair['control'])
        assert reference.decide(math_helper,quality_by_seed,source,concat,stage,'selection',{},costs)['decision'] == 'KILL'
        if stage == 'first':
            rejects(lambda:reference.decide(math_helper,quality_by_seed,source,concat,stage,'selection',{'CI':0},costs), 'no confidence')
    rows = {(179061,a):{'service_seconds':400.,'total_training_core_seconds':200.} for a in e.ARMS}
    for key in ('service_seconds','total_training_core_seconds'):
        bad = copy.deepcopy(rows); bad[179061,'candidate'][key] *= 1.501
        assert reference.paired_cost(bad,'first')['179061']['pass'] is False
    entered = []
    readiness = dict.fromkeys(e.READINESS,True)
    for key in readiness:
        rejects(lambda key=key:e.quality_after_readiness({**readiness,key:False},lambda:entered.append(1)), 'precede')
    assert not entered


def current_bytes(e):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'source'
        path.write_bytes(b'original')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        e.bound_file({},path,digest)
        saved = path.stat()
        path.write_bytes(b'mutated!')
        import os
        os.utime(path,ns=(saved.st_atime_ns,saved.st_mtime_ns))
        rejects(lambda:e.bound_file({},path,digest), 'SHA256 differs')
        path.unlink(); path.symlink_to(Path(directory)/'target')
        path.resolve().write_bytes(b'original')
        rejects(lambda:e.bound_file({},path,digest), 'canonical')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true', required=True)
    parser.add_argument('--narrow', action='store_true')
    args = parser.parse_args()
    nested_binding()
    if not args.narrow:
        e = module('_connected_eval_source_test', DRIVER)
        trainer = module('_connected_eval_trainer_api', HERE/'train_siglip2_connected_mlp.py')
        reference = module('_connected_eval_reference_api', HERE/'evaluate_siglip2_identity_diversity.py')
        math_helper = module('_connected_eval_math_api', HERE/'evaluate_siglip2_genuine_views.py')
        source_contract(e,trainer,reference)
        launch_contract(e)
        paired_decisions(e,reference,math_helper,trainer)
        current_bytes(e)
        # One affected suite includes the trainer's original terminal/cgroup,
        # current-byte, overlay, optimizer, storage and lifecycle falsifiers.
        saved = sys.argv
        try:
            sys.argv = [str(HERE/'test_siglip2_connected_mlp.py'),'--source-only']
            runpy.run_path(sys.argv[0],run_name='__main__')
        finally:
            sys.argv = saved
    assert not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'sfora', 'transformers'} for n in sys.modules)
    print('PASS connected evaluator source-only' + (' nested binding' if args.narrow else ' affected suite'))


if __name__ == '__main__':
    main()
