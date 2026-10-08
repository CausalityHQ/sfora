#!/usr/bin/env python3
"""Prospective score700/export1500 stdlib falsifiers; no native/model-fit/quality qualification."""
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


def evaluator_score_envelope_inverse(raw):
    """Undo only score700; restore the complete original500/export1500 source."""
    raw = evaluator_original_owner_inverse(raw)
    new = b"return {'seconds': 1500 if phase == 'export' else 700 if phase == 'score' else 500, 'host_bytes':8*1024**3,"
    old = b"return {'seconds': 1500 if phase == 'export' else 500, 'host_bytes':8*1024**3,"
    assert raw.count(new) == 1 and raw.count(old) == 0, 'exact score envelope literal differs'
    raw = raw.replace(new, old, 1)
    assert hashlib.sha256(raw).hexdigest() == \
        'bce0c43bae6d24f50ab8ce7c60f8410abd307697ec81d91825e0a043a20efe08', 'score inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        '7e6ae659a786d3e554ccbd17450e25054b80437b8454476d9edce50ca42c5c44', 'score inverse AST differs'
    return raw


def evaluator_export_envelope_inverse(raw):
    """Undo only export1500; restore the complete original900 production source."""
    raw = evaluator_score_envelope_inverse(raw)
    new = b"return {'seconds': 1500 if phase == 'export' else 500, 'host_bytes':8*1024**3,"
    old = b"return {'seconds': 900 if phase == 'export' else 500, 'host_bytes':8*1024**3,"
    assert raw.count(new) == 1 and raw.count(old) == 0, 'exact export envelope literal differs'
    raw = raw.replace(new, old, 1)
    assert hashlib.sha256(raw).hexdigest() == \
        'ca122f669078e39a3c300246f425d441f4c53356b95a156865dfd33cbe177581', 'export inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        '834109b46372a3f3d283c74b36c8c8d433aac04a250f8b6f883e8677e7cd0b6b', 'export inverse AST differs'
    return raw


def evaluator_repin_inverse(raw):
    """Undo only the authorized v6 literals; preserve every old byte and predicate."""
    raw = evaluator_export_envelope_inverse(raw)
    replacements = (
        (b'mechanics1200 + fresh TRAIN128 (3000s) normal exits', b'mechanics300 + fresh TRAIN128 normal exits', 1),
        (b'sfora-connected-mlp-train-source-v6', b'sfora-connected-mlp-train-source-v3', 2),
        (b'a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c', b'5947257ef8e2656fe9b30e94b13c873a55f571f3ed5990f2988d29a085603a3f', 1),
        (b'79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b', b'55935d5a7e7299d1a11f14617cd5ef07f4232afd32148abf942c74edbc636e99', 1),
        (b'8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25', b'35daffe103bc52fe3fbbc4af33a0348494b9477ce49052db7f5d0cf5be083cc1', 1),
        (b'230a11e4f1334336b6d195f60bc0bd7e', b'a8bd65d910fa47dcac56778046350819', 1),
        (b'cpu-v6-original.log', b'cpu-v3-original.log', 1),
        (b'a5c767cee5a689c5d0e0c29b4e355488e8350286033b77b2199e665d816dae14', b'1cfb38cc9752e2e0feee40fecd13f1ae2b5a2252dea4e8eb6f92fe5fa1305c22', 1),
        (b'6426592', b'6424952', 1),
        (b'sfora-connected-mlp-cpu-v6', b'sfora-connected-mlp-cpu-v3', 2),
        (b'4ecd63f75a09ff1757a9a1bdf1e80c29865c5bfb75483c63f3a3e09bc9aeeaa8', b'5c9bd4a3d5b3160ae82493510fd0755c2fd8e6607e39c2a591257bcf99a9f312', 1),
        (b'433.509', b'430.965', 1),
        (b"trainer.policy('cpu')['seconds'] == 600 and trainer.policy('train')['seconds'] == 3000", b"trainer.policy('cpu')['seconds'] == trainer.policy('train')['seconds'] == 600", 1),
        (b"trainer.policy('mechanics')['seconds'] == 1200 and", b"trainer.policy('mechanics')['seconds'] == 300 and", 1),
        (b'parent-frozen complete CPUv6 UNIT required', b'parent-frozen complete CPUv3 UNIT required', 1),
    )
    for new, old, count in replacements:
        assert raw.count(new) == count, 'exact evaluator repin differs'
        raw = raw.replace(new, old)
    assert hashlib.sha256(raw).hexdigest() == \
        '835b2ffd07271c3f26fb671a9764cd79bddc2e02667d5cbcdaa7923f7d0eedd9', 'evaluator inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        'a7d3f8310700926034eeaccea6f41495e23a1cef1419f9b5a762b4fdb94b859e', 'evaluator inverse AST differs'
    return raw


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


def repin_contract(e, trainer):
    freeze = EVIDENCE/'connected-mlp-cpu-v6-freeze'
    cpu = EVIDENCE/'connected-mlp-cpu-v6'
    frozen = json.loads((freeze/'freeze.json').read_text())
    verified = json.loads((cpu/'verification.json').read_text())
    for name, digest in frozen['files'].items():
        assert hashlib.sha256((freeze/name).read_bytes()).hexdigest() == digest
    assert e.TRAINING == {'root': frozen['source_root'],
        'execution_sha256': frozen['execution_sha256'],
        'code': json.loads((freeze/'execution.json').read_text())}
    assert e.TRAINING_CPU == verified['terminal']
    for key, name in (('receipt', 'receipt.json'), ('log', 'original.log')):
        assert hashlib.sha256((cpu/name).read_bytes()).hexdigest() == e.TRAINING_CPU[key]['sha256']
    receipt = json.loads((cpu/'receipt.json').read_text())
    assert receipt['execution_sha256'] == e.TRAINING['execution_sha256']
    assert receipt['code'] == e.TRAINING['code']
    assert receipt['phase'] == 'cpu' and 'qualifications' in receipt and 'result' not in receipt
    assert verified['pass'] is True and verified['engineering_only'] is True
    assert verified['original_launch_exit'] == 0 and verified['all11_frozen_files_exit_rehashed'] is True
    assert verified['memory_events_zero'] is True and verified['swap_bytes'] == 0
    assert verified['quality_read'] is False and verified['state_reuse_eligible'] is False
    assert verified['source_execution_sha256'] == e.TRAINING['execution_sha256']
    tree = ast.parse(DRIVER.read_text())
    authority = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authority')
    def predicate(message):
        node = next(n for n in authority.body if isinstance(n, ast.Expr) and
            isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and
            n.value.func.id == 'require' and n.value.args[1].value == message)
        return compile(ast.Module(body=[node], type_ignores=[]), str(DRIVER), 'exec')
    phase_gate = predicate('connected trainer/pinned scientific predicates differ')
    namespace = dict(vars(e), trainer=trainer,
        native=SimpleNamespace(REFERENCE=e.REFERENCE), e=e,
        math_helper=SimpleNamespace(ORDER=e.ORDER, METRICS=e.METRICS, PANELS=e.PANELS))
    exec(phase_gate, namespace)
    for phase, seconds in (('cpu', 600), ('mechanics', 1200), ('train', 3000)):
        assert trainer.policy(phase)['seconds'] == seconds
        for bad in (seconds-1, seconds+1, 500 if phase == 'cpu' else 300 if phase == 'mechanics' else 600):
            proxy = SimpleNamespace(**vars(trainer))
            proxy.policy = lambda p: {**trainer.policy(p), **({'seconds': bad} if p == phase else {})}
            rejects(lambda: exec(phase_gate, {**namespace, 'trainer': proxy}), 'scientific predicates')
    cpu_gate = predicate('parent-frozen complete CPUv6 UNIT required; no mechanics/TRAIN qualification inferred')
    def admit_cpu(unit):
        exec(cpu_gate, dict(vars(e), first={'launch': {}}, guards={},
            read_json=lambda *args: {'selected_cpu': unit}))
    admit_cpu(e.TRAINING_CPU)
    old_cpu = json.loads((EVIDENCE/'connected-mlp-mechanics-control-179061-v1-freeze'/
        'authority-mechanics-control-179061-v1.json').read_text())['selected_cpu']
    rejects(lambda: admit_cpu(old_cpu), 'CPUv6 UNIT')
    for key in e.TRAINING_CPU:
        changed = copy.deepcopy(e.TRAINING_CPU)
        changed[key] = old_cpu[key] if old_cpu[key] != changed[key] else False
        rejects(lambda: admit_cpu(changed), 'CPUv6 UNIT')
    raw = DRIVER.read_bytes()
    evaluator_repin_inverse(raw)
    for before, after in ((b"else 700, 'host_bytes'", b"else 701, 'host_bytes'"),
                          (b"'whole_service_ratio_max':1.50", b"'whole_service_ratio_max':1.51"),
                          (b"first_receipt['decision'] == 'CONTINUE'", b"first_receipt['decision'] == 'GO'")):
        assert raw.count(before) == 1
        try:
            evaluator_repin_inverse(raw.replace(before, after))
        except AssertionError:
            pass
        else:
            raise AssertionError('evaluator inverse accepted unrelated mutation')
    print('PASS evaluator v6 origin, v3 CPU rejection, phase tampering, exact source/AST inverse')


def source_contract(e, trainer, reference):
    tree = ast.parse(endpoint_auth_inverse(DRIVER.read_bytes()))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    admitted_base = ast.parse(evaluator_admission_batch_inverse(DRIVER.read_bytes()))
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
    freeze = EVIDENCE/'connected-mlp-cpu-v6-freeze'
    source_freeze = json.loads((freeze/'freeze.json').read_text())
    assert e.TRAINING['root'] == source_freeze['source_root']
    assert e.TRAINING['execution_sha256'] == hashlib.sha256((freeze/'execution.json').read_bytes()).hexdigest()
    assert e.TRAINING['code'] == json.loads((freeze/'execution.json').read_text())
    for name,digest in e.TRAINING['code'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    cpu_unit = json.loads((EVIDENCE/'connected-mlp-cpu-v6/verification.json').read_text())['terminal']
    assert e.TRAINING_CPU == cpu_unit
    assert e.TRAINING_CPU['receipt']['sha256'] == hashlib.sha256((EVIDENCE/'connected-mlp-cpu-v6/receipt.json').read_bytes()).hexdigest()
    assert e.policy('cpu')['seconds'] == 700
    assert e.policy('score')['seconds'] == 700
    assert e.policy('export')['seconds'] == 1500
    assert e.COST_POLICY['whole_service_ratio_max'] == e.COST_POLICY['total_training_core_ratio_max'] == 1.50
    assert 'candidate_cache' not in e.LAUNCH_KEYS and 'evaluator_reference' in e.LAUNCH_KEYS
    original_policy = next(n for n in ast.parse(evaluator_export_envelope_inverse(DRIVER.read_bytes())).body
                           if isinstance(n, ast.FunctionDef) and n.name == 'policy')
    for name in ('require', 'strict_json', 'sha', 'check_file', 'bound_file', 'read_json', 'closure',
                 'merge_guards', 'load_authenticated', 'policy', 'check_unit', 'check_resource_facts',
                 'batch_sizes', 'resources', 'accept_unit'):
        prior = next(n for n in ast.parse((HERE / 'evaluate_siglip2_identity_diversity.py').read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == name)
        current = original_policy if name == 'policy' else next(n for n in admitted_base.body
            if isinstance(n,ast.FunctionDef) and n.name == name) if name == 'accept_unit' else functions[name]
        assert ast.dump(current, include_attributes=False) == ast.dump(prior, include_attributes=False), name
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


def export_envelope_contract(e):
    assert e.policy('export') == {'seconds': 1500, 'host_bytes': 8*1024**3,
                                  'swap_bytes': 0, 'cuda_visible_devices': '0'}
    for phase, seconds in (('cpu', 700), ('score', 700)):
        assert e.policy(phase) == {'seconds': seconds, 'host_bytes': 8*1024**3,
                                   'swap_bytes': 0, 'cuda_visible_devices': ''}
    rejects(lambda: e.policy('train'), 'fixed evaluation phase')
    original = EVIDENCE/'connected-mlp-evaluation-first-cpu-v1'
    unit = json.loads((original/'unit.json').read_bytes())
    raw = (original/'receipt.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == unit['receipt']['sha256']
    cpu = json.loads(raw)
    assert cpu['source_code'][DRIVER.name] == hashlib.sha256(
        evaluator_export_envelope_inverse(DRIVER.read_bytes())).hexdigest()
    assert cpu['launch']['resource_policies']['export']['seconds'] == 900
    args = SimpleNamespace(execution_sha256=cpu['execution_sha256'], phase='cpu', arm=None, seed=None)
    cpu500_resource_facts(cpu)  # CPU500 stays valid; its original900 launch cannot qualify1500.
    rejects(lambda: e.check_launch(cpu['launch'], args), 'launch differs')
    launch = copy.deepcopy(cpu['launch'])
    launch['resource_policies'] = {p: e.policy(p) for p in ('cpu', 'export', 'score')}
    e.check_launch(launch, args)  # Policy-bound launch fixture only, never a new native qualification.
    facts = {'resource_policy': e.policy('export'), 'wall_seconds': 1499.999,
        'process_peak_rss_kib': 8*1024**2, 'peak_cuda_allocated_bytes': 9_999_999_999,
        'cuda_initialized': True}
    e.check_resource_facts(facts, 'export')
    for seconds in (1500, 1501):
        rejects(lambda: e.check_resource_facts({**facts, 'wall_seconds': seconds}, 'export'), 'resources')
    for key, value in (('seconds', 900), ('seconds', 1501), ('host_bytes', 8*1024**3+1),
                       ('swap_bytes', 1), ('cuda_visible_devices', '')):
        bad = {**e.policy('export'), key: value}
        rejects(lambda: e.check_resource_facts({**facts, 'resource_policy': bad}, 'export'), 'resources')
        rejects(lambda: e.check_launch({**launch, 'resource_policies':
            {**launch['resource_policies'], 'export': bad}}, args), 'launch differs')
    for key, value in (('wall_seconds', 0), ('wall_seconds', float('nan')), ('wall_seconds', True),
                       ('process_peak_rss_kib', 0), ('process_peak_rss_kib', 8*1024**2+1),
                       ('peak_cuda_allocated_bytes', 0), ('peak_cuda_allocated_bytes', 10_000_000_000),
                       ('peak_cuda_allocated_bytes', True), ('cuda_initialized', False)):
        rejects(lambda: e.check_resource_facts({**facts, key: value}, 'export'), 'resources')
    for phase, seconds in (('cpu', 700), ('score', 700)):
        hidden = {**facts, 'resource_policy': e.policy(phase), 'wall_seconds': seconds-.001,
                  'peak_cuda_allocated_bytes': 0, 'cuda_initialized': False}
        e.check_resource_facts(hidden, phase)
        for wall in (seconds, seconds+1):
            rejects(lambda: e.check_resource_facts({**hidden, 'wall_seconds': wall}, phase), 'resources')
        rejects(lambda: e.check_resource_facts({**hidden, 'peak_cuda_allocated_bytes': 1}, phase), 'resources')
    for seconds in (500, 701):
        bad = {**e.policy('score'), 'seconds': seconds}
        rejects(lambda: e.check_resource_facts({**hidden, 'resource_policy': bad}, 'score'), 'resources')
        rejects(lambda: e.check_launch({**launch, 'resource_policies':
            {**launch['resource_policies'], 'score': bad}}, args), 'launch differs')
    context = {'args': args, 'launch': launch, 'code': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in e.FILES}, 'training_context': {'source': cpu['source']}}
    for record in (cpu, {**cpu, 'launch': launch}):
        rejects(lambda: e.check_receipt(context, record, 'cpu'), 'complete source/resource evaluator receipt')
    print('PASS prospective score700/export1500 boundaries/caps; historical CPU launch/source binding rejected; fresh ownCPU700 required')


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
            old_freeze = EVIDENCE/'connected-mlp-cpu-v3-freeze'
            old_training = {'root': json.loads((old_freeze/'freeze.json').read_text())['source_root'],
                'execution_sha256': hashlib.sha256((old_freeze/'execution.json').read_bytes()).hexdigest(),
                'code': json.loads((old_freeze/'execution.json').read_text())}
            rejects(lambda: e.check_launch({**launch, 'training': old_training}, args), 'parent-frozen trainer2')
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


# BEGIN FRESH ADMISSION FALSIFIER

def evaluator_admission_batch_inverse(raw):
    """Undo only the added helper/import and the two named admission loops."""
    raw = endpoint_auth_inverse(raw)
    start = raw.index(b'def batch_bound_files(guards, items):\n')
    end = raw.index(b'def read_json(value, guards):\n',start)
    assert hashlib.sha256(raw[start:end]).hexdigest() == \
        '9634b13bb6ae039ae15cb155943f20fc3c50e1f4d56b852a9263ac815b135c9b', 'admission helper differs'
    raw = raw[:start]+raw[end:]
    edits = (
        (b'from concurrent.futures import ThreadPoolExecutor\n',b''),
        (b"    batch_bound_files(guards,archived['input_guards'].items())\n",
         b"    for p,h in archived['input_guards'].items():\n        bound_file(guards,p,h)\n"),
        (b"    batch_bound_files(context['guards'],record['input_guards'].items())\n",
         b"    for p,h in record['input_guards'].items():\n        bound_file(context['guards'],p,h)\n"))
    for new,old in edits:
        assert raw.count(new) == 1, 'named admission edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == \
        '919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69', 'admission inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == \
        '65be54726dcc34df31c15bab3164f695fc5facd26d8587169687064622ccc3aa', 'admission inverse AST differs'
    return raw


def admission_batch_test_inverse(raw):
    raw = endpoint_auth_test_inverse(raw)
    start = raw.index(b'# BEGIN FRESH ADMISSION FALSIFIER\n')
    end = raw.index(b'# BEGIN ORIGINAL OWNER FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    edits = (
        (b'    raw = evaluator_admission_batch_inverse(raw)\n',b''),
        (b'    raw = admission_batch_test_inverse(raw)\n',b''),
        (b'    actual_admission_scan_falsifier()\n',b''),
        (b'    admitted_base = ast.parse(evaluator_admission_batch_inverse(DRIVER.read_bytes()))\n',b''),
        (b"        current = original_policy if name == 'policy' else next(n for n in admitted_base.body\n"
         b"            if isinstance(n,ast.FunctionDef) and n.name == name) if name == 'accept_unit' else functions[name]\n",
         b"        current = original_policy if name == 'policy' else functions[name]\n"),
        (b"    restored_raw = evaluator_admission_batch_inverse(DRIVER.read_bytes())\n",b''),
        (b'    for node in ast.parse(restored_raw).body:\n',b'    for node in ast.parse(DRIVER.read_bytes()).body:\n'),
        (b'            assert ast.get_source_segment(restored_raw.decode(),node) == actual_sources[node.name]\n',
         b'            assert ast.get_source_segment(DRIVER.read_text(),node) == actual_sources[node.name]\n'),
        (b"    current_reader = api({'check_receipt','accept_unit','batch_bound_files'},read_json=read_fixture,bound_file=bound_fixture)\n",
         b"    current_reader = api({'check_receipt','accept_unit'},read_json=read_fixture,bound_file=bound_fixture)\n"))
    for new,old in edits:
        assert raw.count(new) == 1, 'named admission test edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == \
        'df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb', 'admission test inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == \
        '4c808537549df4a0cdd76a9785e1b234b4d90c1377b44db449ebfdc491b3d7e7', 'admission test inverse AST differs'
    return raw


def actual_admission_scan_falsifier():
    """Catch serial scans, skipped occurrences and failed owner publication; real reads."""
    import os
    import threading
    from collections import Counter
    from contextlib import contextmanager
    from unittest.mock import patch

    frozen = EVIDENCE/'connected-mlp-evaluation-full-cpu-v1-freeze'/DRIVER.name
    base = frozen.read_bytes()
    assert hashlib.sha256(base).hexdigest() == '919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69'
    original, candidate = ast.parse(base), ast.parse(DRIVER.read_bytes())
    function = lambda tree, name: next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    helpers = [n for n in candidate.body if isinstance(n,(ast.Import,ast.ImportFrom)) or
        isinstance(n,ast.FunctionDef) and n.name in ('require','sha','bound_file','batch_bound_files')]
    namespace = {}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(DRIVER),'exec'),namespace)
    real_bound, real_open = namespace['bound_file'], Path.open

    def run(statement, items, owner, overlap=False):
        lock, barrier = threading.Lock(), threading.Barrier(2,timeout=1.)
        calls, reads, threads, edges = [], [], set(), ['before']
        active = peak = 0
        overlapped = False

        def bound(guards, path, digest):
            with lock:
                calls.append((str(path),digest))
                threads.add(threading.current_thread())
            if threading.current_thread() is not threading.main_thread():
                assert guards == {} and guards is not owner, 'worker shared owner guards'
            return real_bound(guards,path,digest)

        @contextmanager
        def opened(path, *args, **kwargs):
            nonlocal active, peak, overlapped
            with real_open(path,*args,**kwargs) as stream:
                assert args == ('rb',) and not kwargs
                with lock:
                    reads.append(str(path)); edges.append('read')
                    active += 1; peak = max(peak,active)
                    waits = overlap and len(reads) <= 2
                try:
                    if waits:
                        try:
                            barrier.wait(); overlapped = True
                        except threading.BrokenBarrierError:
                            pass  # Serial RED leaves the barrier after one bounded wait.
                    yield stream
                finally:
                    with lock:
                        active -= 1

        code = compile(ast.Module(body=[statement],type_ignores=[]),str(DRIVER),'exec')
        # Real JSON keys are unique; duplicates exercise the helper per occurrence.
        inputs = SimpleNamespace(items=lambda:iter(items))
        error = None
        # Instrument only this extracted namespace, never a loaded evaluator/trainer.
        with patch.dict(namespace,bound_file=bound), patch.object(Path,'open',opened):
            try:
                exec(code,{**namespace,'archived':{'input_guards':inputs},'guards':owner,
                    'record':{'input_guards':inputs},'context':{'guards':owner}})
                edges.append('after')
            except (ValueError,OSError) as exc:
                error = str(exc)
        assert active == 0 and peak <= 4
        assert all(t is threading.current_thread() or not t.is_alive() for t in threads), 'workers not joined'
        assert edges == ['before'] + ['read']*len(reads) + ([] if error else ['after'])
        return SimpleNamespace(error=error,calls=calls,reads=reads,overlapped=overlapped,peak=peak)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        files = [root/f'member{i}' for i in range(5)]
        for i,path in enumerate(files):
            path.write_bytes(b'member %d\n' % i)
        items = [(str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
        repeated = items + [items[0],items[2]]
        initial = dict([items[3],items[0]])
        for name, source in (('authority','archived'),('accept_unit','record')):
            body = function(original,name).body
            index = next(i for i,n in enumerate(body) if isinstance(n,ast.For) and
                ast.unparse(n.iter) == source + "['input_guards'].items()")
            serial = body[index]
            batched = next(n for n in function(candidate,name).body if isinstance(n,ast.Expr) and
                isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id == 'batch_bound_files')
            serial_owner, owner = dict(initial), dict(initial)
            red = run(serial,repeated,serial_owner)
            green = run(batched,repeated,owner,overlap=True)
            assert red.error is green.error is None
            assert red.calls == repeated and red.reads == [p for p,_ in repeated]
            assert red.peak == 1 and not red.overlapped
            assert green.overlapped and 2 <= green.peak <= 4, name + ' admission scan still serial'
            assert Counter(green.calls) == Counter(repeated)
            assert Counter(green.reads) == Counter(p for p,_ in repeated), 'fresh occurrence missing'
            assert list(owner.items()) == list(serial_owner.items()) == list({**initial,**dict(items)}.items())
            print('PASS actual ' + name + ' loop: old overlap1 / new overlap2..4, exact fresh occurrences')

            def failure(entries, expected, guards=initial, overlap=False):
                owner = dict(guards)
                result = run(batched,entries,owner,overlap)
                assert result.error and expected in result.error, (expected,result.error)
                assert list(owner.items()) == list(guards.items()), 'failed scan published owner guards'
                assert Counter(result.calls) == Counter(entries), 'failed scan did not join every occurrence'
                return result

            victim = files[0]
            stamp, content = victim.stat(), victim.read_bytes()
            victim.write_bytes(b'changed!\n')
            os.utime(victim,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            assert (victim.stat().st_size,victim.stat().st_mtime_ns) == (stamp.st_size,stamp.st_mtime_ns)
            mutated = failure(repeated,'SHA256 differs',overlap=True)
            assert mutated.overlapped and Counter(mutated.reads) == Counter(p for p,_ in repeated)
            victim.write_bytes(content)
            link = root/'link'
            link.symlink_to(files[0])
            for path in (str(link),str(root),str(root/'missing'),str(root/'..'/root.name/files[0].name)):
                failure([items[1],(path,items[0][1])],'canonical FILE/SHA')
            link.unlink()
            for malformed in (('relative',items[0][1]),(items[0][0],'A'*64),(items[0][0],None)):
                failure([items[1],malformed],'canonical FILE/SHA')
            failure([items[1],(items[0][0],'0'*64)],'SHA256 differs')
            failure([items[0],(items[0][0],'0'*64)],'SHA256 differs')
            conflict = {items[0][0]:'0'*64}
            conflicted = failure(repeated,'conflicting FILE authority',conflict)
            assert Counter(conflicted.reads) == Counter(p for p,_ in repeated)
            with patch.object(os,'posix_fadvise',side_effect=OSError('injected read failure')):
                failure(repeated,'injected read failure')
            # Failed candidate bulk publication is atomic; original publishes its prefix.
            faults = [items[0],(items[1][0],'0'*64)]
            prefix = {}
            assert 'SHA256 differs' in run(serial,faults,prefix).error
            assert list(prefix.items()) == [items[0]]
            failure(faults,'SHA256 differs',{})
            # All fresh results precede ordered conflict merging: a later byte error
            # wins over an earlier owner conflict, but both paths still reject.
            assert 'conflicting FILE authority' in run(serial,faults,dict(conflict)).error
            failure(faults,'SHA256 differs',conflict)
    assert not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers'} for n in sys.modules)
    nodes = [n for n in candidate.body if isinstance(n,ast.Assign) or
        isinstance(n,ast.FunctionDef) and n.name == 'check_receipt']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DRIVER),'exec'),namespace)
    first_score = json.loads((EVIDENCE/'connected-mlp-evaluation-first-selection-score-v2/receipt.json').read_bytes())
    code = {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in namespace['FILES']}
    # Exact historical quality remains owned by source-v5. New source/code cannot
    # pass current check_receipt; this bounded candidate deliberately cannot fix it.
    for execution, source_code in (('e'*64,first_score['source_code']),
            (first_score['execution_sha256'],code)):
        context = {'args':SimpleNamespace(execution_sha256=execution),'code':source_code,
            'training_context':{'source':first_score['source']}}
        rejects(lambda:namespace['check_receipt'](context,first_score,'score',stage='first',panel='selection'),
            'complete source/resource evaluator receipt')
    evaluator_admission_batch_inverse(DRIVER.read_bytes())
    print('PASS actual admission scans: mutation/symlink/roles/conflict/duplicates/errors, atomic owner and joined failures')
    print('PASS historical first-selection source/execution rejection retained; candidate full-stage requalification blocked')


# END FRESH ADMISSION FALSIFIER


# BEGIN ORIGINAL OWNER FALSIFIER

def evaluator_original_owner_inverse(raw):
    """Remove only the finite owner composition and restore all current base bytes."""
    raw = evaluator_admission_batch_inverse(raw)
    start = raw.index(b'# BEGIN ORIGINAL EXPORT OWNER\n')
    end = raw.index(b'def check_endpoint(endpoint):', start)
    assert hashlib.sha256(raw[start:end]).hexdigest() == '53cba7ca5e95ac84bcf09a38798df64c04976f6659a9a07b639cc55553fff38b', 'original owner definitions differ'
    raw = raw[:start]+raw[end:]
    replacements = ((b"    snapshots = context.setdefault('helper_snapshots',[])\n    modules = [context[key] for key in ('trainer','evaluator_reference','nearest_evaluator','math','reference','helper','baseline')]\n    if 'original_evaluator' in context:\n        modules.append(context['original_evaluator'])\n    if not snapshots:\n", b"    snapshots = context.setdefault('helper_snapshots',[])\n    if not snapshots:\n"), (b'    if not snapshots:\n        for module in modules:\n            values = dict(vars(module))\n', b"    if not snapshots:\n        for key in ('trainer','evaluator_reference','nearest_evaluator','math','reference','helper','baseline'):\n            module = context[key]\n            values = dict(vars(module))\n"), (b"            snapshots.append((module,Path(module.__file__),module.__spec__,values,functions,literals))\n    require(len(snapshots) == len(modules) and all(snapshot[0] is module\n        for snapshot,module in zip(snapshots,modules,strict=True)), 'complete context-bound helper snapshot inventory required')\n    for module,path,spec,values,functions,literals in snapshots:\n", b'            snapshots.append((module,Path(module.__file__),module.__spec__,values,functions,literals))\n    for module,path,spec,values,functions,literals in snapshots:\n'), (b"    keys = ('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference')\n    original_active = (args.phase == 'score' and launch['stage'] == 'first' and launch['panel'] == 'selection' and\n        launch['first_selection'] is None and launch['selection_go'] is None and launch['exports'] == ORIGINAL_EXPORT_UNITS)\n    roots = [root]+([Path(ORIGINAL_EXPORT_OWNER['root'])] if original_active else [])+[Path(launch[k]['root']) for k in keys]\n    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and\n", b"    keys = ('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference')\n    roots = [root]+[Path(launch[k]['root']) for k in keys]\n    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and\n"), (b"    context['preparation_costs'] = preparation_costs(context)\n    original_guard = load_original_owner(context) if original_active else None\n    guard_helpers(context)\n", b"    context['preparation_costs'] = preparation_costs(context)\n    guard_helpers(context)\n"), (b"    if args.phase == 'score':\n        owner = original_owner_context(context,original_guard) if original_active else None\n        context['export_records'] = {label(e):admit_export(context,owner,e,original_guard) for e in launch['endpoints']}\n    return context,original_guard\n\n", b"    if args.phase == 'score':\n        context['export_records'] = {label(e):accept_unit(context,launch['exports'][label(e)],'export',e['arm'],e['seed'])\n            for e in launch['endpoints']}\n    return context\n\n"), (b"\ndef exit_rehash(context, original_guard):\n    if original_guard is not None:\n        original_guard(context)\n    trainer,t=context['trainer'],context['training_context']\n", b"\ndef exit_rehash(context):\n    trainer,t=context['trainer'],context['training_context']\n"), (b"    for descriptor,names,pins in (({'root':str(context['root']),'execution_sha256':context['args'].execution_sha256},FILES,context['code']),\n        *(((ORIGINAL_EXPORT_OWNER,FILES,ORIGINAL_EXPORT_OWNER['code']),) if 'original_evaluator' in context else ()),\n        (context['launch']['training'],TRAIN_FILES,context['launch']['training']['code']),\n", b"    for descriptor,names,pins in (({'root':str(context['root']),'execution_sha256':context['args'].execution_sha256},FILES,context['code']),\n        (context['launch']['training'],TRAIN_FILES,context['launch']['training']['code']),\n"), (b"    merge_guards(context['guards'],t['legacy']['origins']['files'])\n    if original_guard is not None:\n        original_guard(context)\n    return t['legacy']['origins']\n", b"    merge_guards(context['guards'],t['legacy']['origins']['files'])\n    return t['legacy']['origins']\n"), (b"    require(sys.argv == cli(args), 'fixed canonical CLI order required')\n    context,original_guard=authority(args)\n    context['training_context']['fit_context']['unit_started']=UNIT_STARTED\n", b"    require(sys.argv == cli(args), 'fixed canonical CLI order required')\n    context=authority(args)\n    context['training_context']['fit_context']['unit_started']=UNIT_STARTED\n"), (b"    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'begin','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n    origins=exit_rehash(context,original_guard)\n    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'end','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n", b"    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'begin','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n    origins=exit_rehash(context)\n    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'end','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)\n"))
    for new, old in replacements:
        assert raw.count(new) == 1, 'exact original owner integration edit differs'
        raw = raw.replace(new, old, 1)
    assert hashlib.sha256(raw).hexdigest() == 'f1c95755361e42143d30822c2e2426c92d7683e4a5c58d3cd8d0f144bae3f40d', 'original owner inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == 'd89bef72cf224f98d1148c772cc84ccaf787723088c1462a1a5eea233c65321e', 'original owner inverse AST differs'
    return raw



def original_owner_contract(e):
    """Catch foreign dispatch, broadened UNIT routing and weakened original predicates."""
    assert hasattr(e, 'original_owner_context'), 'missing original-owner admission composition'
    freeze = EVIDENCE/'connected-mlp-evaluation-first-cpu-v2-freeze'
    score_freeze = EVIDENCE/'connected-mlp-evaluation-first-selection-score-v1-freeze'
    source = freeze/DRIVER.name
    original_tree = ast.parse(source.read_bytes())
    evaluator_original_owner_inverse(DRIVER.read_bytes())
    assert e.ORIGINAL_EXPORT_OWNER == {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v3',
        'execution_sha256':hashlib.sha256((freeze/'execution.json').read_bytes()).hexdigest(),
        'code':json.loads((freeze/'execution.json').read_bytes())}
    for name, digest in e.ORIGINAL_EXPORT_OWNER['code'].items():
        assert hashlib.sha256((freeze/name).read_bytes()).hexdigest() == digest
    authority_raw = (score_freeze/'authority-first-selection-score-v1.json').read_bytes()
    assert hashlib.sha256(authority_raw).hexdigest() == e.ORIGINAL_SCORE_AUTHORITY['sha256']
    launch = json.loads(authority_raw)
    records = {}; descriptors = {}
    for name, unit in [('connected-mlp-evaluation-first-cpu-v3', launch['selected_cpu']),
        ('connected-mlp-evaluation-export-control-179061-v2', launch['exports']['control-179061']),
        ('connected-mlp-evaluation-export-candidate-179061-v1', launch['exports']['candidate-179061'])]:
        directory = EVIDENCE/name
        assert json.loads((directory/'unit.json').read_bytes()) == unit
        for key, local in (('receipt','receipt.json'),('log','original.log')):
            assert hashlib.sha256((directory/local).read_bytes()).hexdigest() == unit[key]['sha256']
        record = json.loads((directory/'receipt.json').read_bytes())
        records[unit['receipt']['path']] = record
        descriptors[unit['receipt']['path']] = unit['receipt']['sha256']
        records[record['authority']['path']] = record['launch']
        descriptors[record['authority']['path']] = record['authority']['sha256']
    records[e.ORIGINAL_SCORE_AUTHORITY['path']] = launch
    descriptors[e.ORIGINAL_SCORE_AUTHORITY['path']] = e.ORIGINAL_SCORE_AUTHORITY['sha256']
    cpu = records[launch['selected_cpu']['receipt']['path']]
    exported = records[launch['exports']['control-179061']['receipt']['path']]
    actual_sources = {n.name: ast.get_source_segment(source.read_text(), n) for n in original_tree.body if isinstance(n,ast.FunctionDef)}
    restored_raw = evaluator_admission_batch_inverse(DRIVER.read_bytes())
    for node in ast.parse(restored_raw).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('check_receipt','accept_unit'):
            assert ast.get_source_segment(restored_raw.decode(),node) == actual_sources[node.name]

    def api(names, **seams):
        namespace = {**vars(e), **seams}
        nodes = [n for n in ast.parse(DRIVER.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DRIVER),'exec'),namespace)
        return SimpleNamespace(**namespace)

    def read_fixture(value, guards):
        e.check_file(value)
        e.require(descriptors.get(value['path']) == value['sha256'], 'fixture authority SHA differs')
        e.merge_guards(guards,{value['path']:value['sha256']})
        return copy.deepcopy(records[value['path']])

    def pinned_closure(root, digest, names, guards):
        e.require(root == e.ORIGINAL_EXPORT_OWNER['root'] and digest == e.ORIGINAL_EXPORT_OWNER['execution_sha256'] and
            names == e.FILES, 'fixture source closure differs')
        code = e.closure(freeze.resolve(), digest, names, {})
        e.merge_guards(guards,{str(Path(root)/'execution.json'):digest,**{str(Path(root)/n):h for n,h in code.items()}})
        return code

    modules = {key:module('_owner_test_'+key,HERE/filename) for key,filename in (
        ('trainer','train_siglip2_connected_mlp.py'),('evaluator_reference','evaluate_siglip2_identity_diversity.py'),
        ('nearest_evaluator','evaluate_siglip2_nearest_ranking.py'),('math','evaluate_siglip2_genuine_views.py'),
        ('reference','evaluate_siglip2_prototype_residual.py'),('helper','export_siglip2_substrate_adaptation.py'),
        ('baseline','evaluate_siglip2_quadratic_readout.py'))}
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory)/DRIVER.name; local.write_bytes(source.read_bytes())
        def load_fixture(name, path, digest, guards):
            assert name == '_connected_export_owner_v3' and path == Path(e.ORIGINAL_EXPORT_OWNER['root'])/DRIVER.name
            return e.load_authenticated(name, local, digest, guards)
        loader = api({'load_original_owner'},closure=pinned_closure,read_json=read_fixture,load_authenticated=load_fixture)
        fixture = {**modules,'guards':{str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in modules.values()},'launch':copy.deepcopy(launch)}
        loader.load_original_owner(fixture)
        original = fixture['original_evaluator']
        assert original.check_receipt.__globals__ is original.accept_unit.__globals__ is vars(original)
        assert original.check_receipt.__globals__ is not vars(e)
        assert original.policy('score')['seconds'] == 500 and e.policy('score')['seconds'] == 700
        assert sys.modules.pop('_connected_export_owner_v3') is original
        for filename in e.FILES:
            wrong = dict(e.ORIGINAL_EXPORT_OWNER['code']); wrong[filename] = '0'*64
            bad_loader = api({'load_original_owner'},closure=lambda *args:wrong,
                read_json=read_fixture,load_authenticated=load_fixture)
            rejects(lambda:bad_loader.load_original_owner({'guards':{},'launch':launch}), 'exact2 differs')
        changed = {**e.ORIGINAL_SCORE_AUTHORITY,'sha256':'0'*64}
        bad_loader = api({'load_original_owner'},closure=pinned_closure,read_json=read_fixture,
            load_authenticated=load_fixture,ORIGINAL_SCORE_AUTHORITY=changed)
        rejects(lambda:bad_loader.load_original_owner({'guards':{},'launch':launch}), 'authority SHA')
        assert sys.modules.pop('_connected_export_owner_v3') is not None
        for change in ({'endpoints':list(reversed(launch['endpoints']))}, {'scope':{'path':'/wrong','sha256':e.SCOPE_SHA256}}):
            rejects(lambda:loader.load_original_owner({'guards':{},'launch':{**launch,**change}}), 'endpoint/procedure')
            assert sys.modules.pop('_connected_export_owner_v3') is not None

    # Compile the ORIGINAL reader, replacing only external file/terminal/native-origin
    # seams. Its check_receipt, accept_unit and all pure scientific validators are real.
    known = {}
    for record in records.values():
        if 'input_guards' in record:
            e.merge_guards(known,record['input_guards'])
            e.merge_guards(known,{str(Path(record['output'])/n):h for n,h in record['files'].items()})
    e.merge_guards(known,descriptors)
    def bound_fixture(guards, path, digest):
        if Path(path) == Path(e.ORIGINAL_EXPORT_OWNER['root'])/DRIVER.name:
            e.bound_file({},source.resolve(),digest)
        else:
            e.require(known.get(str(path)) == digest, 'fixture current bytes differ')
        e.merge_guards(guards,{str(path):digest})
        return Path(path)
    spec = importlib.util.spec_from_file_location('_connected_export_owner_v3',Path(e.ORIGINAL_EXPORT_OWNER['root'])/DRIVER.name)
    original = importlib.util.module_from_spec(spec)
    original.__dict__.update(read_json=read_fixture,bound_file=bound_fixture)
    nodes = [n for n in original_tree.body if not isinstance(n,ast.FunctionDef) or n.name not in {'read_json','bound_file'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(spec.origin),'exec'),vars(original))
    sys.modules[original.__name__] = original
    assert original.check_receipt.__globals__ is original.accept_unit.__globals__ is vars(original)
    guard_api = api({'guard_helpers'},bound_file=bound_fixture)
    current_reader = api({'check_receipt','accept_unit','batch_bound_files'},read_json=read_fixture,bound_file=bound_fixture)
    production = api({'load_original_owner','original_owner_context','check_original_owner','admit_export'},
        read_json=read_fixture,closure=pinned_closure,load_authenticated=lambda *args:original,
        guard_helpers=guard_api.guard_helpers,accept_unit=current_reader.accept_unit)
    # Tiny externally supplied origin proof isolates the unchanged exact-four check.
    native_four = sorted(exported['origins']['native_files'])[:4]
    site = Path('/home/riomus/group-learning/.venv/lib/python3.13/site-packages')
    members = {str(Path(p).relative_to(site)): {'sha256':exported['origins']['files'][p]} for p in native_four}
    nearest_training = module('_owner_nearest_training',HERE/'train_siglip2_nearest_ranking.py')
    proof_file = {'path':'/fixture/native-proof','sha256':nearest_training.NATIVE_PROOF_PINS['proof']}
    native_authority = {'path':'/fixture/native-authority','sha256':'a'*64}
    records[proof_file['path']] = {'authority':{'installed_site_root':str(site)},'comparison':{'selected_members':members}}
    records[native_authority['path']] = {'proof':proof_file}
    descriptors.update({proof_file['path']:proof_file['sha256'],native_authority['path']:native_authority['sha256']})
    original_source_origins = {p:h for p,h in exported['origins']['files'].items() if p not in native_four}
    legacy = {'invocations':set(),'admission':object(),'selected':{'source_cpu':{'numerical_flags':cpu['numerical_flags'],
        'invocation':cpu['invocation'],'origins':{'files':original_source_origins}},'packages':cpu['origins']['packages']},
        'warm_record':{'origins':{'files':{}}}}
    nearest_seam = SimpleNamespace(native_source_api=lambda t:SimpleNamespace(audit_origins=lambda *args,**kwargs:None),
        NATIVE_PROOF_PINS=nearest_training.NATIVE_PROOF_PINS)
    def terminal(admission, record, unit, seconds, guards):
        assert admission is legacy['admission'] and seconds == (e if record['source_code'] == code else original).policy(record['phase'])['seconds']
        e.require(unit['invocation_id'] == record['invocation']['invocation_id'], 'fixture terminal invocation differs')
        return record['cgroup_after']
    panel = {'original_rows':[None]*e.PANELS['selection'][0],'query':[],'gallery':[]}
    for image in exported['images']:
        for row in image['rows']:
            panel['original_rows'][row['panel_ordinal']] = row['original_row']
            panel[row['role']].append(row['panel_ordinal'])
    current_launch = copy.deepcopy(launch)
    current_launch.update(execution_sha256='e'*64,selected_cpu={**launch['selected_cpu'],'unit':'fresh-current-CPU'},
        resource_policies={p:e.policy(p) for p in ('cpu','export','score')})
    root = Path('/fixture/current-source')
    code = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in e.FILES}
    current_closure = {str(root/'execution.json'):'e'*64,**{str(root/n):h for n,h in code.items()}}
    common = {**current_closure, '/fixture/shared':'b'*64}
    known['/fixture/shared'] = 'b'*64
    for r in (cpu,exported,records[launch['exports']['candidate-179061']['receipt']['path']]):
        r['input_guards']['/fixture/shared'] = 'b'*64
    context = {**modules,'original_evaluator':original,'original_launch':launch,
        'root':root,'args':SimpleNamespace(execution_sha256='e'*64,phase='score',arm=None,seed=None,
            authority=Path('/fixture/current-launch'),authority_sha256='f'*64,output=Path('/fixture/current-output')),
        'code':code,'launch':current_launch,'guards':{},'common_guards':common,'accepted_units':[],
        'training_context':{'source':cpu['source'],'legacy':legacy,'nearest':nearest_seam,'launch':{'native_authority':native_authority}},
        'cpu':None,'costs':cpu['cost'],'preparation_costs':cpu['preparation_costs'],
        'records':{(ep['seed'],ep['arm']):{'result':{'identity':cpu['payload_facts'][e.label(ep)]['identity']}} for ep in launch['endpoints']},
        'manifests':{(ep['seed'],ep['arm']):cpu['payload_facts'][e.label(ep)] for ep in launch['endpoints']},
        'score_context':{'partition':{'panels':{'selection':panel}}},'terminal_reader':terminal}
    for m in modules.values():
        path = Path(m.__file__); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        context['guards'][str(path)] = digest; known[str(path)] = digest
    context['guards'][str(Path(spec.origin))] = e.ORIGINAL_EXPORT_OWNER['code'][DRIVER.name]
    # Synthetic current-source CPU must independently pass the unchanged CURRENT
    # reader; historical CPU4 or old CPU3 cannot qualify the new execution bytes.
    fresh = copy.deepcopy(cpu)
    fresh.update(execution_sha256='e'*64,source_code=code,output='/fixture/current-cpu-output',
        launch={**current_launch,'phase':'cpu','selected_cpu':None,'exports':{}},
        authority={'path':'/fixture/current-cpu-authority','sha256':'c'*64},authority_sha256='c'*64)
    fresh['resource_policy'] = e.policy('cpu')
    fresh['binding'] = e.binding({'launch':fresh['launch']})
    fresh['invocation']['invocation_id'] = 'f'*32
    fresh['invocation']['argv'] = e.cli(SimpleNamespace(execution_sha256='e'*64,phase='cpu',arm=None,seed=None,
        authority=Path(fresh['authority']['path']),authority_sha256='c'*64,output=Path(fresh['output'])))
    fresh['input_guards'].update(common)
    fresh_unit = {**launch['selected_cpu'],'unit':'fresh-current-CPU','invocation_id':'f'*32,
        'receipt':{'path':fresh['output']+'/receipt.json','sha256':'d'*64}}
    records[fresh_unit['receipt']['path']] = fresh
    records[fresh['authority']['path']] = fresh['launch']
    descriptors.update({fresh_unit['receipt']['path']:'d'*64,fresh['authority']['path']:'c'*64})
    known.update(fresh['input_guards'])
    current_launch['selected_cpu'] = fresh_unit
    context['cpu'] = current_reader.accept_unit(context,fresh_unit,'cpu',panel='selection')
    rejects(lambda:production.load_original_owner(context), 'original selection endpoint/procedure')
    # Exercise the unchanged archived owner after proving actual CPU700 cannot adopt it.
    context.setdefault('helper_snapshots',[])
    archived_context = {**context,'launch':{**current_launch,'resource_policies':
        {**current_launch['resource_policies'],'cpu':launch['resource_policies']['cpu']}}}
    original_guard = production.load_original_owner(archived_context)
    before = dict(common); current_cpu = context['cpu']; accumulated = dict(context['guards'])
    owner = production.original_owner_context(context,original_guard)
    assert context['common_guards'] == before and context['cpu'] is current_cpu
    assert owner.keys() == context.keys()
    specific = {'root','args','code','launch','common_guards','cpu'}
    assert {k for k in owner if owner[k] is not context[k]} == specific
    assert owner['guards'] is context['guards'] and owner['accepted_units'] is context['accepted_units']
    assert all(context['guards'][p] == h for p,h in accumulated.items())
    assert owner['launch']['selected_cpu'] == launch['selected_cpu'] != current_launch['selected_cpu']
    assert set(common)-set(owner['common_guards']) == set(current_closure)
    assert set(owner['common_guards'])-set(common) == {str(Path(e.ORIGINAL_EXPORT_OWNER['root'])/n) for n in (*e.FILES,'execution.json')}
    production.check_original_owner(context,owner,original_guard)
    for snapshots in (context['helper_snapshots'][:-1],context['helper_snapshots'][1:],
        context['helper_snapshots']+[context['helper_snapshots'][-1]],list(reversed(context['helper_snapshots']))):
        changed = {**context,'helper_snapshots':snapshots}
        rejects(lambda:production.admit_export(changed,{**owner,'helper_snapshots':snapshots},launch['endpoints'][0],original_guard), 'snapshot binding')
        rejects(lambda:guard_api.guard_helpers(changed), 'snapshot inventory')
    for key in modules.keys()|{'original_evaluator'}:
        changed = {**context,key:SimpleNamespace()}
        rejects(lambda:production.admit_export(changed,{**owner,key:changed[key]},launch['endpoints'][0],original_guard),
            'snapshot binding' if key == 'original_evaluator' else 'snapshot inventory')
    forged = importlib.util.module_from_spec(spec)
    forged.__dict__.update(vars(original))
    snapshots = list(context['helper_snapshots'])
    old = snapshots[-1]
    snapshots[-1] = (forged,old[1],old[2],dict(vars(forged)),old[4],copy.deepcopy(old[5]))
    changed = {**context,'original_evaluator':forged,'helper_snapshots':snapshots}
    rejects(lambda:production.admit_export(changed,{**owner,'original_evaluator':forged,'helper_snapshots':snapshots},
        launch['endpoints'][0],original_guard), 'snapshot binding')
    for slot,value in ((3,{}),(4,[]),(5,{})):
        snapshots = list(context['helper_snapshots']); row = list(snapshots[-1]); row[slot] = value; snapshots[-1] = tuple(row)
        changed = {**context,'helper_snapshots':snapshots}
        rejects(lambda:production.admit_export(changed,{**owner,'helper_snapshots':snapshots},launch['endpoints'][0],original_guard), 'snapshot binding')
    for key,value in (('root',Path('/foreign')),('args',context['args']),('code',code),('launch',current_launch),
        ('cpu',current_cpu),('common_guards',common),('guards',dict(context['guards'])),('accepted_units',[])):
        rejects(lambda:production.admit_export(context,{**owner,key:value},launch['endpoints'][0],original_guard), 'context binding')
    changed = {**context,'common_guards':{p:h for p,h in common.items() if p != str(root/'execution.json')}}
    rejects(lambda:production.original_owner_context(changed,original_guard), 'three owner closure')
    changed = {**context,'common_guards':{**common,str(Path(spec.origin)):'0'*64}}
    rejects(lambda:production.original_owner_context(changed,original_guard), 'three owner closure')
    saved_original_launch = context['original_launch']
    old_selected = launch['selected_cpu']; context['original_launch'] = {**launch,'selected_cpu':current_launch['selected_cpu']}
    rejects(lambda:production.original_owner_context(context,original_guard), 'input authority changed')
    context['original_launch'] = saved_original_launch
    for ep in launch['endpoints']:
        record = production.admit_export(context,owner,ep,original_guard)
        assert record == records[launch['exports'][e.label(ep)]['receipt']['path']]
    assert context['accepted_units'] == [fresh_unit,old_selected]+[launch['exports'][e.label(ep)] for ep in launch['endpoints']]
    # Execute the real activation predicate and current admission tail. Inactive
    # phases have no original root/module/closure work; full/VAL use current gates.
    authority = next(n for n in ast.parse(endpoint_auth_inverse(DRIVER.read_bytes())).body if isinstance(n,ast.FunctionDef) and n.name == 'authority')
    activation = next(n for n in authority.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'original_active')
    roots_node = next(n for n in authority.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'roots')
    def active(value, phase='score'):
        ns = dict(vars(e), launch=value,args=SimpleNamespace(phase=phase),root=root,
            keys=('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference'))
        exec(compile(ast.Module(body=[activation,roots_node],type_ignores=[]),str(DRIVER),'exec'),ns)
        assert (Path(e.ORIGINAL_EXPORT_OWNER['root']) in ns['roots']) is ns['original_active']
        return ns['original_active']
    assert active(current_launch)
    for phase in ('cpu','export'):
        assert not active(current_launch,phase)
    for key,value in (('stage','full'),('panel','validation'),('first_selection',fresh_unit),('selection_go',fresh_unit),
        ('exports',{}),('exports',{**current_launch['exports'],'control-179061':fresh_unit})):
        assert not active({**current_launch,key:value})
    full = copy.deepcopy(current_launch)
    full.update(stage='full',first_selection={**fresh_unit,'unit':'current-first-score'},
        endpoints=launch['endpoints']+[nested_fixture(e,seed,arm)[0] for seed,arm in e.ORDER[2:]])
    full['exports'] = {e.label(ep):{**fresh_unit,'unit':'current-export-'+e.label(ep)} for ep in full['endpoints']}
    visits = []; decision = ['CONTINUE']
    def current_unit(c,unit,phase,arm=None,seed=None,stage=None,panel=None):
        assert c['launch'] is full and unit not in launch['exports'].values()
        visits.append((phase,unit['unit']))
        if unit == full['first_selection']:
            return {'decision':decision[0],'launch':{'endpoints':full['endpoints'][:2]}}
        if unit == full['selection_go']:
            return {'decision':'GO','selection_go_admits_validation_only':True,'launch':{'endpoints':full['endpoints']}}
        if phase == 'cpu':
            return {'payload_facts':{e.label(ep):{} for ep in full['endpoints']}}
        return {'current_export':e.label({'arm':arm,'seed':seed})}
    tail_start = next(i for i,n in enumerate(authority.body) if isinstance(n,ast.If) and
        ast.unparse(n.test) == "launch['stage'] == 'full'")
    tail_body = copy.deepcopy(authority.body[tail_start:])
    definition = ast.parse('def current_tail(context, launch, args, original_active, original_guard):\n    pass').body[0]
    definition.body = ast.parse("guards = context['guards']").body+tail_body; ast.fix_missing_locations(definition)
    dispatch = api({'admit_export'},accept_unit=current_unit)
    def admit_remaining(c,eps):
        assert visits == [('score','current-first-score')]
        visits.append(('endpoints',tuple(ep['seed'] for ep in eps)))
    ns = {**vars(e),'accept_unit':current_unit,'admit_endpoints':admit_remaining,'admit_export':dispatch.admit_export,
        'original_owner_context':lambda *args: (_ for _ in ()).throw(AssertionError('inactive original CPU admission'))}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(DRIVER),'exec'),ns)
    for panel_name in ('selection','validation'):
        full['panel'] = panel_name
        full['selection_go'] = {**fresh_unit,'unit':'current-selection-GO'} if panel_name == 'validation' else None
        visits.clear(); c = {'launch':full,'args':context['args'],'guards':{}}
        result, guard = ns['current_tail'](c,full,context['args'],False,None)
        assert result is c and guard is None and 'original_evaluator' not in c
        assert visits[:2] == [('score','current-first-score'),('endpoints',(179069,179069))]
        assert visits[-5:][0] == ('cpu','fresh-current-CPU')
        assert len(c['export_records']) == 4
    decision[0] = 'KILL'; visits.clear()
    rejects(lambda:ns['current_tail']({'launch':full,'args':context['args'],'guards':{}},full,context['args'],False,None), 'KILL prohibits')
    assert visits == [('score','current-first-score')]
    for panel_name in ('selection','validation'):
        rejects(lambda:current_reader.check_receipt({**context,'launch':{**current_launch,'stage':'full','panel':panel_name}},
            exported,'export','control',179061), 'source/resource')
    # The real exit entry uses the independently held guard before external work.
    exit_api = api({'exit_rehash'},guard_helpers=guard_api.guard_helpers)
    rejects(lambda:exit_api.exit_rehash({**context,'original_evaluator':forged,'helper_snapshots':snapshots},original_guard), 'snapshot binding')

    # Reader regressions reach actual ORIGINAL check_receipt/accept_unit branches.
    for field,value,text in (('execution_sha256','0'*64,'source/resource'),('source_code',code,'source/resource'),
        ('payload_facts',{},'export/readback'),('preparation_costs',{},'preparation costs'),('cost',{},'endpoint/cost')):
        bad = {**exported,field:value}
        rejects(lambda:original.check_receipt(owner,bad,'export','control',179061), text)
    bad = copy.deepcopy(exported); bad['invocation']['argv'][0] = str(DRIVER)
    rejects(lambda:original.check_receipt(owner,bad,'export','control',179061), 'CLI/binding')
    rejects(lambda:original.check_receipt({**owner,'cpu':{'payload_facts':{'control-179061':{}}}},exported,'export','control',179061), 'export/readback')
    wrong_launch = {**owner['launch'],'selected_cpu':current_launch['selected_cpu']}
    rejects(lambda:original.check_receipt({**owner,'launch':wrong_launch},exported,'export','control',179061), 'complete stage CPU')
    # Relabeling source alone still cannot satisfy original CLI/policy/CPU/common.
    relabeled = {**exported,'execution_sha256':'e'*64,'source_code':code}
    rejects(lambda:current_reader.check_receipt(context,relabeled,'export','control',179061), 'launch differs')
    # Removed frozen closure guard reaches original accept_unit after all metadata.
    legacy['invocations'].clear()
    changed = copy.deepcopy(exported); del changed['input_guards'][str(Path(spec.origin))]
    receipt_path = launch['exports']['control-179061']['receipt']['path']
    records[receipt_path] = changed
    rejects(lambda:original.accept_unit(owner,launch['exports']['control-179061'],'export','control',179061), 'source guards')
    records[receipt_path] = exported
    legacy['invocations'].clear()
    wrong_cpu = copy.deepcopy(current_cpu); wrong_cpu['payload_facts']['control-179061']['fixed_sha256'] = '0'*64
    changed = {**context,'cpu':wrong_cpu}; changed_owner = {**owner,'cpu':owner['cpu']}
    rejects(lambda:production.admit_export(changed,changed_owner,launch['endpoints'][0],original_guard), 'current independently admitted CPU')
    for key,value in (('panel','validation'),('phase','export'),('stage','full')):
        changed = {**context,'launch':{**current_launch,key:value}} if key in ('panel','stage') else {
            **context,'args':SimpleNamespace(**{**vars(context['args']),key:value})}
        rejects(lambda:production.admit_export(changed,owner,launch['endpoints'][0],original_guard), 'first-selection-only')
    wrong_unit = {**launch['exports']['control-179061'],'invocation_id':'0'*32}
    changed = {**context,'launch':{**current_launch,'exports':{**current_launch['exports'],'control-179061':wrong_unit}}}
    rejects(lambda:production.admit_export(changed,owner,launch['endpoints'][0],original_guard), 'source/resource')
    # Live original code/registry/global integrity is checked, independent of dispatch.
    del sys.modules['_connected_export_owner_v3']
    rejects(lambda:production.check_original_owner(context,owner,original_guard), 'live source/global')
    sys.modules['_connected_export_owner_v3'] = original
    saved_policy = original.__dict__['policy']; original.__dict__['policy'] = lambda phase:{}
    rejects(lambda:production.check_original_owner(context,owner,original_guard), 'live source/global')
    original.__dict__['policy'] = saved_policy
    sys.modules.pop('_connected_export_owner_v3')
    print('PASS exact original-owner closure/authority/context, unchanged original readers, CPU/payload/CLI/guard/UNIT/VAL and live snapshot falsifiers')


def original_owner_test_inverse():
    raw = Path(__file__).read_bytes()
    raw = admission_batch_test_inverse(raw)
    start = raw.index(b'# BEGIN ORIGINAL OWNER FALSIFIER\n')
    end = raw.index(b'\n\ndef main():\n',start)+2
    raw = raw[:start]+raw[end:]
    raw = raw.replace(b'    raw = evaluator_original_owner_inverse(raw)\n',b'',1)
    raw = raw.replace(b'    original_owner_contract(e)\n',b'',1)
    raw = raw.replace(b'    original_owner_test_inverse()\n',b'',1)
    assert hashlib.sha256(raw).hexdigest() == '0a2be2dec8f6fa2033e62a749b2805184e00055a76cd55bafc2981bc138785df', 'old test bytes differ'


# END ORIGINAL OWNER FALSIFIER


# BEGIN BOOTSTRAP PREREQUISITE FALSIFIER

def bootstrap_inverse(raw):
    """Exact complete evaluator restoration to 37e1cf50 before historical inverses."""
    raw = evaluator_cpu_envelope_inverse(raw)
    start = raw.index(b'# BEGIN BOOTSTRAP PREREQUISITE READER\n')
    end = raw.index(b'# BEGIN ENDPOINT READER AUTHENTICATION\n',start)
    raw = raw[:start]+raw[end:]
    edits = [
        (b"    context = {'trainer':trainer,'guards':guards,'code':code}\n    bootstrap,endpoint_guard = load_bootstrap_authority(context,targs)\n    t,endpoint_reader = bootstrap()  # Original CPU/gradient, then three fresh batched prerequisites.\n",
         b'    t = trainer.authority(targs)  # Original CPU, actual gradient, CPU600 and mechanics061 normal exits.\n'),
        (b"    context.update({'args':args,'root':root,'guards':guards,'code':code,'launch':launch,'trainer':trainer,",
         b"    context = {'args':args,'root':root,'guards':guards,'code':code,'launch':launch,'trainer':trainer,"),
        (b"'accepted_units':[],'common_guards':common_guards})\n",
         b"'accepted_units':[],'common_guards':common_guards}\n    endpoint_reader,endpoint_guard = load_endpoint_reader(context)\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'bootstrap evaluator inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '9cf3ba0e005fcf1b9c1433359bae09a2540fb6143d6a5c83b8c2b1cfc60802a3', 'bootstrap evaluator inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == 'fd8ed53935be222f520604587c7ac9e707861cc9c57caec188f20516a650a68d', 'bootstrap evaluator inverse AST differs'
    return raw


def bootstrap_test_inverse(raw):
    raw = cpu_envelope_test_inverse(raw)
    start = raw.index(b'# BEGIN BOOTSTRAP PREREQUISITE FALSIFIER\n')
    end = raw.index(b'# BEGIN INITIALIZER DICT FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    for line in (b'    raw = bootstrap_inverse(raw)\n', b'    raw = bootstrap_test_inverse(raw)\n',
                 b'    bootstrap_prerequisite_contract()\n'):
        assert raw.count(line) == 1
        raw = raw.replace(line,b'',1)
    assert hashlib.sha256(raw).hexdigest() == '2e869791c5667558a6c4d5d2459c4fba973a6705c08d5921a16ffac002ddef8d', 'bootstrap test inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == 'ede89414d4efdeec6c97661c28010badfccffa0419fa9a5019ea5477b2028ef8', 'bootstrap test inverse AST differs'
    return raw


def bootstrap_execution_contract(e):
    """Execute both complete bootstrap ASTs; isolate only historical external I/O."""
    from contextlib import ExitStack
    from types import FunctionType
    from unittest.mock import patch
    import os
    import threading
    import time
    names=('train_siglip2_connected_mlp','train_siglip2_substrate_adaptation','fit_siglip2_prototype_residual',
        'train_siglip2_nearest_ranking','train_siglip2_quadratic_readout','train_siglip2_identity_diversity')
    trainer,flat,fitter,nearest,old,training = [module('_bootstrap_'+n,HERE/(n+'.py')) for n in names]
    legacy=original_initializer(flat);init=legacy['admission'].init
    modules=(trainer,flat,fitter,nearest,old,training,init)
    source_guards={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    initializer=json.loads((EVIDENCE/'identity-diversity-v1/cpu-v5/receipt.json').read_bytes())
    cpu=json.loads((EVIDENCE/'connected-mlp-cpu-v6/receipt.json').read_bytes())
    raw=Path(trainer.__file__).read_bytes()
    original=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
    dump=lambda n:ast.dump(n,include_attributes=False)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); output=root/'out'; witness=root/'witness';witness.mkdir()
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
            return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        code=cpu['code']
        for name,digest in code.items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
            write(root/name,(HERE/name).read_bytes())
        execution=write(root/'execution.json',(EVIDENCE/'connected-mlp-cpu-v6-freeze/execution.json').read_bytes())
        launch=json.loads((EVIDENCE/'connected-mlp-train-control-179061-v1-freeze/authority-train-control-179061-v1.json').read_bytes())
        launch['witness']['root']=str(witness)
        for name,digest in launch['witness']['files'].items():
            assert write(witness/name,(HERE/name).read_bytes())['sha256']==digest
        extras=[write(root/('input'+str(i)),str(i).encode()) for i in range(6)]
        records=[];units=[]
        folders=['connected-mlp-cpu-v6','connected-mlp-mechanics-control-179061-v2','connected-mlp-mechanics-candidate-179061-v2']
        for i,folder in enumerate(folders):
            record=json.loads((EVIDENCE/folder/'receipt.json').read_bytes())
            terminal_root=root/str(i); terminal_root.mkdir()
            record['launch']['witness']=copy.deepcopy(launch['witness'])
            result=record['qualifications'][0] if i==0 else record['result']
            result['identity']['method']=trainer.method(record['launch'])
            auth=write(terminal_root/'authority.json',record['launch'])
            record.update(authority=auth,authority_sha256=auth['sha256'])
            record['invocation']['argv']=trainer.cli(root,auth['path'],auth['sha256'],execution['sha256'],
                record['phase'],record['arm'],record['seed'],terminal_root)
            record['input_guards']={str(root/'execution.json'):execution['sha256'],**{str(root/n):h for n,h in code.items()},
                **{str(witness/n):h for n,h in launch['witness']['files'].items()},auth['path']:auth['sha256'],
                **{f['path']:f['sha256'] for f in extras}}
            receipt=write(terminal_root/'receipt.json',record)
            unit=copy.deepcopy(e.TRAINING_CPU if i==0 else launch['selected_mechanics'][record['arm']])
            unit['receipt']=receipt;unit['log']=write(terminal_root/'original.log',(EVIDENCE/folder/'original.log').read_bytes())
            records.append(record);units.append(unit)
        launch['selected_cpu']=units[0];launch['selected_mechanics']=dict(zip(trainer.ARMS,units[1:]))
        auth=write(root/'bootstrap.json',launch)
        args=SimpleNamespace(execution_sha256=execution['sha256'],authority=Path(auth['path']),
            authority_sha256=auth['sha256'],output=output,phase='train',arm='control',seed=179061)
        witness_api=SimpleNamespace(FILES=trainer.WITNESS_FILES,HELPERS=launch['witness']['files'],
            TRAIN_ROOT=Path(launch['original_cpu']['authority']['path']).parent,
            CPU_AUTHORITY_SHA=launch['original_cpu']['authority']['sha256'],CPU_UNIT=launch['original_cpu']['terminal'],
            TRAIN_EXECUTION='historical',TRAIN_CODE={'train_siglip2_identity_diversity.py':source_guards[training.__file__]})
        visited=[];active=[0,0];readers=[];batches=[];lock=threading.Lock();real_open=Path.open
        def observe(p,*a,**kw):
            if str(p) in {f['path'] for f in extras} and (not a or a[0]=='rb'):
                frame=sys._getframe(1)
                while frame and 'self' not in frame.f_locals:frame=frame.f_back
                reader=frame.f_locals.get('self') if frame else None
                with lock:visited.append(str(p));readers.append(reader);active[0]+=1;active[1]=max(active)
                try:time.sleep(.003)
                finally:
                    with lock:active[0]-=1
            return real_open(p,*a,**kw)
        def setup():
            legacy['invocations'].clear();guards=dict(source_guards)
            t={'trainer':training,'guards':guards,'legacy':legacy,'nearest':nearest,'fitter':fitter,'old':old,
                'fit_context':{'legacy':legacy,'guards':guards},'source':cpu['source']}
            context={'trainer':trainer,'guards':dict(source_guards),
                'code':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in e.FILES}}
            return context,t
        def execute(adapted,mutate=None):
            context,t=setup();order=[];owners=[]
            run,guard=e.load_bootstrap_authority(context,args)
            derivative=next(c.cell_contents for c in run.__closure__ if getattr(c.cell_contents,'__name__',None)=='connected_bootstrap_authority')
            template=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
            # Recover the executed adapted AST from the pinned production loader's exact replacements.
            for call in ast.walk(template):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='admit_terminal' and call.lineno in (253,256):
                    if adapted:call.func.id='_bootstrap_terminal'
            if adapted:
                inverse=copy.deepcopy(template)
                for call in ast.walk(inverse):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='_bootstrap_terminal':call.func.id='admit_terminal'
                assert dump(inverse)==dump(original)
                template.name='connected_bootstrap_authority'
                ns=dict(derivative.__globals__)
                exec(compile(ast.Module(body=[template],type_ignores=[]),str(DRIVER),'exec'),ns)
                assert ns[template.name].__code__==derivative.__code__, 'executed bootstrap differs from production derivative'
            else:ns=dict(vars(trainer))
            if mutate:mutate(template)
            def historical(historical_args):
                order.append('historical-authority');assert historical_args.phase=='cpu';return t
            def original_cpu(*a):order.append('historical-cpu');return initializer
            with ExitStack() as historical_patches:
                historical_patches.enter_context(patch.object(training,'authority',historical))
                historical_patches.enter_context(patch.object(training,'admit_terminal',original_cpu))
                def gradient(current):
                    order.append('actual-gradient');assert current is t
                    historical_patches.close()
                def selected(current,unit,phase,arm,seed):
                    order.append((phase,arm,seed));assert order[:3]==['historical-authority','historical-cpu','actual-gradient']
                    return trainer.admit_terminal(current,unit,phase,arm,seed)
                terminal=ns.get('_bootstrap_terminal')
                def prerequisite(*a):
                    order.append(tuple(a[2:]));assert order[:3]==['historical-authority','historical-cpu','actual-gradient']
                    return terminal(*a)
                # These are external historical bootstrap seams only. All three prospective
                # terminal calls execute genuine validators, original FlatAdmission and logs.
                ns.update(HERE=root,closure=lambda path,*a:code if path==root else witness_api.TRAIN_CODE,
                    read_json=lambda *a:copy.deepcopy(launch),load_authenticated=lambda name,*a:witness_api if name=='_connected_admitted_witness' else training,
                    admit_actual_gradient=gradient,admit_terminal=selected)
                if adapted:ns['_bootstrap_terminal']=prerequisite
                exec(compile(ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[])),str(DRIVER),'exec'),ns)
                def profile(frame,event,value):
                    if event=='return' and frame.f_code is trainer.fresh_terminal_reader.__code__:owners.append(value)
                    if frame.f_code is e.batch_terminal_files.__code__ and event in ('call','return'):
                        r=frame.f_locals['reader'];g=frame.f_locals['guards']
                        batches.append((event,(dict(r.entries),set(r.verified),dict(r.json_bytes),dict(g))))
                prior_profile=sys.getprofile();sys.setprofile(profile)
                try:
                    with patch.object(Path,'open',observe):
                        if adapted:
                            # Execute the actual run wrapper too, swapping only its external-I/O fixture closure.
                            cell=lambda value:(lambda:value).__closure__[0]
                            cells=tuple(cell(ns[template.name]) if name=='derivative' else c
                                for name,c in zip(run.__code__.co_freevars,run.__closure__))
                            invoke=FunctionType(run.__code__,run.__globals__,run.__name__,closure=cells)
                            result,returned_reader=invoke()
                            assert returned_reader is not None
                        else:result=ns[template.name](args)
                finally:sys.setprofile(prior_profile)
            assert result is t and order==['historical-authority','historical-cpu','actual-gradient',
                ('cpu','control',179061),('mechanics','control',179061),('mechanics','candidate',179061)]
            assert t['connected_required_guards']==t['guards']
            assert list(t['connected_terminals'].values())==records
            assert legacy['invocations']=={u['invocation_id'] for u in units}
            if adapted:guard(context)
            assert len(owners)==3 and all(type(r) is flat.FlatAdmission for r in owners)
            states=[(dict(r.entries),set(r.verified),dict(r.json_bytes),r.init) for r in owners]
            return context,t,run,guard,derivative,states
        original_context,serial,_,_,_,serial_readers=execute(False)
        assert active[1]==1, 'original bootstrap scan must be serial'
        serial_state=copy.deepcopy((serial['guards'],serial['connected_required_guards'],serial['connected_terminals'],serial['connected_terminal_cgroups'],legacy['invocations']))
        visited.clear();readers.clear();active[:]=[0,0]
        context,t,run,guard,derivative,parallel_readers=execute(True)
        assert serial_readers==parallel_readers
        assert serial_state==(t['guards'],t['connected_required_guards'],t['connected_terminals'],t['connected_terminal_cgroups'],legacy['invocations'])
        assert visited and len(visited)==18 and all(visited.count(f['path'])==3 for f in extras)
        assert 1<active[1]<=4 and active[0]==0 and len({id(r) for r in readers})==18
        assert all(type(r) is flat.FlatAdmission and r is not legacy['admission'] for r in readers)
        # Authenticated production namespace, callbacks, real context and snapshots are independent.
        def denied(text):
            rejects(lambda:guard(context),text)
            rejects(lambda:e.exit_rehash(context,guard),text)
        with patch.dict(derivative.__globals__,{'admit_actual_gradient':lambda *a:None}):denied('derivative binding')
        with patch.dict(derivative.__globals__,{'__builtins__':dict(derivative.__builtins__)}):denied('derivative binding')
        callback=derivative.__globals__['_bootstrap_terminal']
        for fn,text in ((callback,'callback binding'),(derivative,'derivative binding'),
                        (trainer.authority,'authenticated'),(e.load_endpoint_reader,'owned callback'),(e.require,'owned callback')):
            for key,value in (('__code__',fn.__code__.replace(co_name='forged')),('__defaults__',('forged',)),('__kwdefaults__',{'forged':True}),
                              ('__module__','forged'),('__name__','forged'),('__qualname__','forged')):
                old_value=getattr(fn,key);setattr(fn,key,value)
                try:denied(text)
                finally:setattr(fn,key,old_value)
        live=e.source_live_guard;live_code=live.__code__
        live.__code__=live_code.replace(co_name='forged')
        try:denied('owned callback')
        finally:live.__code__=live_code
        with patch.dict(context,training_context={**t}):denied('context/reader binding')
        with patch.dict(context,guards=dict(context['guards'])):denied('owner/arguments')
        with patch.object(args,'seed',179069):denied('owner/arguments')
        with patch.object(flat,'FlatAdmission',type('FlatAdmission',(flat.FlatAdmission,),{})):denied('binding changed')
        with patch.object(e.source_live_guard,'initializer',lambda *a:lambda:None):denied('initializer delegate')
        with patch.object(init,'require',lambda *a:None):
            context['helper_snapshots']=[];denied('binding changed');del context['helper_snapshots']
        guard(context)
        rejects(run,'once-only')
        # Foreign captured builtins cannot become the bootstrap's endpoint-loader authority.
        import builtins
        loader=e.load_endpoint_reader; prior=vars(e)['__builtins__']
        vars(e)['__builtins__']={**vars(builtins),'bool':lambda value:True}
        forged=FunctionType(loader.__code__,vars(e),loader.__name__)
        forged.__module__=loader.__module__;forged.__qualname__=loader.__qualname__
        vars(e)['__builtins__']=prior
        with patch.object(e,'load_endpoint_reader',forged):
            rejects(lambda:e.load_bootstrap_authority(setup()[0],args),'owned callback source')
        def serial_dispatch(node):
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='_bootstrap_terminal':
                    call.func.id='admit_terminal'
        rejects(lambda:execute(True,serial_dispatch),'complete bootstrap prerequisites')
        for phase,arm,seed in (('cpu','control',179061),('train','candidate',179061),('train','control',179069)):
            bad=SimpleNamespace(**{**vars(args),'phase':phase,'arm':arm,'seed':seed})
            rejects(lambda:e.load_bootstrap_authority(context,bad),'fixed evaluator bootstrap')
        # Same real adapter rejects corruption despite restored mtime; no invocation is published.
        p=Path(extras[0]['path']); stat=p.stat();p.write_bytes(b'x');os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        batches.clear();visited.clear();readers.clear()
        rejects(lambda:execute(True),'SHA256')
        assert not legacy['invocations'] and active[0]==0 and len(visited)==6
        assert [v[0] for v in batches]==['call','return'] and batches[0][1]==batches[1][1]
        p.write_bytes(b'0')
        batches.clear();visited.clear()
        with patch.dict(source_guards,{extras[0]['path']:'f'*64}):
            rejects(lambda:execute(True),'stage file authority')
        assert active[0]==0 and len(visited)==6 and batches[0][1]==batches[1][1]
        target=root/'elsewhere';target.write_bytes(b'0');p.unlink();p.symlink_to(target)
        rejects(lambda:execute(True),'canonical');assert not legacy['invocations'];p.unlink();p.write_bytes(b'0')
        # Omitting state or a predicate changes the complete original AST, never an allowed derivative.
        for change in ('predicate','snapshot','order'):
            mutant=copy.deepcopy(original)
            if change=='predicate':mutant.body.pop(0)
            elif change=='snapshot':mutant.body.pop(-2)
            else:mutant.body[-4],mutant.body[-3]=mutant.body[-3],mutant.body[-4]
            assert hashlib.sha256(dump(mutant).encode()).hexdigest()!='44d79c7536e18016f1bc9a85120e74926f81e5980964e573173b51d8c93441b5'
            parse=ast.parse
            def altered(source,*a,**kw):
                tree=parse(source,*a,**kw)
                if source==raw:
                    tree.body=[copy.deepcopy(mutant) if isinstance(n,ast.FunctionDef) and n.name=='authority' else n for n in tree.body]
                return tree
            with patch.object(ast,'parse',altered):
                rejects(lambda:e.load_bootstrap_authority(setup()[0],args),'frozen bootstrap AST')


def bootstrap_routing_contract(e):
    """Execute the real evaluator bootstrap seam, including its right-biased guard merge."""
    original=ast.parse(bootstrap_inverse(DRIVER.read_bytes()));adapted=ast.parse(DRIVER.read_bytes())
    launch={'scope':'scope','endpoints':[{'launch':{'path':'/first'}}]}
    args=SimpleNamespace(authority=Path('/outer'))
    guards={'overlap':'outer','/outer':'excluded'}
    t={'launch':{'scope':'scope'},'guards':{'overlap':'training','/first':'excluded','inner':'retained'},
        'connected_launch':{'selected_cpu':'cpu','selected_mechanics':{'control':'control','candidate':'candidate'}}}
    trainer=SimpleNamespace(authority=lambda args:t);reader=object();guard=object()
    results=[]
    for tree in (original,adapted):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
            ast.unparse(n.targets[0])==('t' if tree is original else 'context'))
        end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and
            isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='admit_endpoints')
        owners=[]
        def bootstrap(context,selected_args):
            owners.append(context)
            def run():context['training_context']=t;return t,reader
            return run,guard
        ns={'trainer':trainer,'targs':object(),'guards':guards,'code':{},'args':args,'root':HERE,
            'launch':launch,'first':launch['endpoints'][0],'require':e.require,'load_bootstrap_authority':bootstrap,
            'load_endpoint_reader':lambda context:(reader,guard),'e':object(),'native':object(),'math_helper':object(),'reference':object()}
        exec(compile(ast.Module(body=node.body[start:end],type_ignores=[]),str(DRIVER),'exec'),ns)
        current=ns['context']
        assert current['common_guards']=={'overlap':'training','inner':'retained'}
        assert ns['endpoint_reader'] is reader and ns['endpoint_guard'] is guard
        if tree is adapted:assert owners==[current] and owners[0] is current
        results.append({k:v for k,v in current.items() if k not in ('evaluator_reference','nearest_evaluator','math','reference')})
    assert results[0]==results[1]


def bootstrap_prerequisite_contract():
    e = module('_bootstrap_evaluator', DRIVER)
    assert hasattr(e, 'load_bootstrap_authority'), 'missing authenticated bootstrap prerequisite reader'
    bootstrap_inverse(DRIVER.read_bytes());bootstrap_test_inverse(Path(__file__).read_bytes())
    bootstrap_routing_contract(e)
    with original_initializer_files():bootstrap_execution_contract(e)
    assert not any(n.split('.')[0] in e.NATIVE for n in sys.modules)
    print('PASS exact bootstrap AST/inverse, three ordered genuine admissions, state/fresh parallel reads and live guards')


# END BOOTSTRAP PREREQUISITE FALSIFIER


# BEGIN INITIALIZER DICT FALSIFIER

def initializer_dict_inverse(raw):
    """Restore exact held 15eb6e0 bytes; keep all earlier inverse assertions."""
    raw = bootstrap_inverse(raw)
    edits = [
        (b"    binding_error = ValueError\n    wrapper_dict,get_attribute = source_live_guard.__dict__,getattr\n    def verify_delegate():\n        if (source_live_guard.__dict__ is not wrapper_dict or\n                get_attribute(source_live_guard,'initializer',None) is not delegate or\n                delegate.__code__ is not delegate_code or delegate.__globals__ is not delegate_globals or\n", b"    binding_error = ValueError\n    def verify_delegate():\n        if (source_live_guard.__dict__.get('initializer') is not delegate or\n                delegate.__code__ is not delegate_code or delegate.__globals__ is not delegate_globals or\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer dictionary inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '04fb4f774b61a28bc7314da051eb106344da4d3ca2a04719d9c1bdfbaf7ca527', 'initializer dictionary inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '57bc2d7ddf61b91e14fca77467bc61a5f262c64a7b6f0f988ebcff3fe37bb63c', 'initializer dictionary inverse AST differs'
    return raw


def initializer_dict_test_inverse(raw):
    """Restore exact held 15eb6e0 bytes; keep all earlier inverse assertions."""
    raw = bootstrap_test_inverse(raw)
    start = raw.index(b'# BEGIN INITIALIZER DICT FALSIFIER\n')
    end = raw.index(b'# BEGIN INITIALIZER DELEGATE FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    edits = [
        (b'    """Restore exact held 6841013 bytes before all unchanged historical inverses."""\n    raw = initializer_dict_inverse(raw)\n    edits = [\n', b'    """Restore exact held 6841013 bytes before all unchanged historical inverses."""\n    edits = [\n'),
        (b'    """Restore exact held 6841013 bytes before all unchanged historical inverses."""\n    raw = initializer_dict_test_inverse(raw)\n    start = raw.index(b\'# BEGIN INITIALIZER DELEGATE FALSIFIER\\n\')\n', b'    """Restore exact held 6841013 bytes before all unchanged historical inverses."""\n    start = raw.index(b\'# BEGIN INITIALIZER DELEGATE FALSIFIER\\n\')\n'),
        (b"            for call in checks:rejects(call,'initializer delegate')\n        wrapper_dict=e.source_live_guard.__dict__\n        class Mask(dict):\n            def get(self,key,default=None):\n                return delegate if key=='initializer' else super().get(key,default)\n        for namespace in (Mask(wrapper_dict),dict(wrapper_dict)):\n            e.source_live_guard.__dict__=namespace\n            try:\n                e.source_live_guard.initializer=lambda context:lambda:None\n                rejected()\n                e.source_live_guard.initializer=delegate\n                rejected()  # Even a replacement dictionary with the genuine value is foreign.\n            finally:e.source_live_guard.__dict__=wrapper_dict\n        del e.source_live_guard.initializer\n        try:rejected()  # A missing attribute must raise the same clear binding error.\n        finally:e.source_live_guard.initializer=delegate\n        with patch.object(e.source_live_guard,'initializer',lambda context:lambda:None):rejected()\n", b"            for call in checks:rejects(call,'initializer delegate')\n        with patch.object(e.source_live_guard,'initializer',lambda context:lambda:None):rejected()\n"),
        (b"                    return super().keys()\n            class MaskingUnit(MutatingUnit):\n                def keys(self):\n                    e.source_live_guard.__dict__=Mask(wrapper_dict)\n                    return super().keys()\n            for unit_type in (MutatingUnit,MaskingUnit):\n                visited.clear()\n                try:rejects(lambda:admit(unit_type(unit),'cpu','control',179061),'initializer delegate')\n                finally:\n                    e.source_live_guard.__dict__=wrapper_dict;e.source_live_guard.initializer=delegate\n                assert visited\n        fresh()();live()\n", b"                    return super().keys()\n            try:rejects(lambda:admit(MutatingUnit(unit),'cpu','control',179061),'initializer delegate')\n            finally:e.source_live_guard.initializer=delegate\n            assert visited\n        fresh()();live()\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer dictionary inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == 'f15d98ddbbe8a585ee0185b79c47a6dd9a3bd434c775889fc0831d20a1b242e0', 'initializer dictionary inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == 'a05d61453bdf7fde7b73b88b7aaf18885e32e6b1e93e00d43d209298971e1144', 'initializer dictionary inverse AST differs'
    return raw


# END INITIALIZER DICT FALSIFIER


# BEGIN INITIALIZER DELEGATE FALSIFIER

def initializer_delegate_inverse(raw):
    """Restore exact held 6841013 bytes before all unchanged historical inverses."""
    raw = initializer_dict_inverse(raw)
    edits = [
        (b"    error,code = ValueError,function.__code__\n    def source_live_guard(module, digest, guards, names=None, class_name=None):\n        if verify_delegate.__code__ is not binding_code:\n            raise error('authenticated initializer delegate checker changed')\n        verify_delegate()\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error('authenticated builtin baseline/source binding changed')\n        return function(canonical,module,digest,guards,names,class_name,None,verify_delegate)\n    def initializer_live_guard(context):\n        if initializer_check.__code__ is not initializer_check_code:\n            raise error('authenticated initializer delegate checker changed')\n        initializer_check()\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error('authenticated builtin baseline/source binding changed')\n", b"    error,code = ValueError,function.__code__\n    def source_live_guard(module, digest, guards, names=None, class_name=None):\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error('authenticated builtin baseline/source binding changed')\n        return function(canonical,module,digest,guards,names,class_name,None)\n    def initializer_live_guard(context):\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error('authenticated builtin baseline/source binding changed')\n"),
        (b"        return function(canonical,t['legacy']['selected']['genuine']['reference'],\n            '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8',\n            t['guards'],{'admit_cgroup','require'},None,context,initializer_check)\n    # Independent cells: changing the delegate's closure cannot move its checker\n    # or the import-time expectations shared by fresh and already returned guards.\n    delegate = initializer_live_guard\n    delegate_code,delegate_globals,delegate_builtins = delegate.__code__,delegate.__globals__,delegate.__builtins__\n    delegate_metadata = (delegate.__module__,delegate.__name__,delegate.__qualname__)\n    delegate_closure = delegate.__closure__\n    binding_error = ValueError\n    def verify_delegate():\n        if (source_live_guard.__dict__.get('initializer') is not delegate or\n                delegate.__code__ is not delegate_code or delegate.__globals__ is not delegate_globals or\n                delegate.__builtins__ is not delegate_builtins or delegate.__defaults__ is not None or\n                delegate.__kwdefaults__ is not None or delegate.__closure__ is not delegate_closure or\n                (delegate.__module__,delegate.__name__,delegate.__qualname__) != delegate_metadata):\n            raise binding_error('authenticated initializer delegate binding changed')\n        for cell,value in delegate_cells:\n            if cell.cell_contents is not value:\n                raise binding_error('authenticated initializer delegate closure changed')\n    initializer_check = verify_delegate\n    binding_code = initializer_check_code = verify_delegate.__code__\n    delegate_cells = tuple((cell,cell.cell_contents) for cell in delegate_closure)\n    source_live_guard.initializer = initializer_live_guard\n    return source_live_guard\n", b"        return function(canonical,t['legacy']['selected']['genuine']['reference'],\n            '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8',\n            t['guards'],{'admit_cgroup','require'},None,context)\n    source_live_guard.initializer = initializer_live_guard\n    return source_live_guard\n"),
        (b'\n@_capture_source_builtins\ndef source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner, delegate_guard):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = {key:value for key,value in baseline}\n', b'\n@_capture_source_builtins\ndef source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = {key:value for key,value in baseline}\n'),
        (b"    namespaces = [module.__dict__,canonical['globals']()]\n    error = canonical['ValueError']\n    delegate_guard_code = delegate_guard.__code__\n    def builtin_guard():\n        if delegate_guard.__code__ is not delegate_guard_code:\n            raise error('authenticated initializer delegate checker changed')\n        delegate_guard()\n        # No global/builtin calls: even all/any/type/ValueError may have changed.\n        if _SOURCE_BUILTINS is not baseline:\n", b"    namespaces = [module.__dict__,canonical['globals']()]\n    error = canonical['ValueError']\n    def builtin_guard():\n        # No global/builtin calls: even all/any/type/ValueError may have changed.\n        if _SOURCE_BUILTINS is not baseline:\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer delegate inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '12b9d3bcdf2af64aaf1b169344a3a2bfffd9a891eede14124b0f430659af8a36', 'initializer delegate inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '805de9c064ac6ad6344140f9a6ebdd508c7fce3b64c4738b598ca6f101737c7e', 'initializer delegate inverse AST differs'
    return raw


def initializer_delegate_test_inverse(raw):
    """Restore exact held 6841013 bytes before all unchanged historical inverses."""
    raw = initializer_dict_test_inverse(raw)
    start = raw.index(b'# BEGIN INITIALIZER DELEGATE FALSIFIER\n')
    end = raw.index(b'# BEGIN INITIALIZER ORIGIN FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    edits = [
        (b'def initializer_origin_inverse(raw):\n    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""\n    raw = initializer_delegate_inverse(raw)\n    edits = [\n        (b\'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        return function(canonical,module,digest,guards,names,class_name,None)\\n    def initializer_live_guard(context):\\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        t = context[\\\'training_context\\\']\\n        return function(canonical,t[\\\'legacy\\\'][\\\'selected\\\'][\\\'genuine\\\'][\\\'reference\\\'],\\n            \\\'163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8\\\',\\n            t[\\\'guards\\\'],{\\\'admit_cgroup\\\',\\\'require\\\'},None,context)\\n    source_live_guard.initializer = initializer_live_guard\\n    return source_live_guard\\n\\n\\n@_capture_source_builtins\\ndef source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner):\\n    """Independent lexical source/runtime binding, including genuine class methods."""\\n    canonical = {key:value for key,value in baseline}\\n\', b\'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        return function(canonical,module,digest,guards,names,class_name)\\n    return source_live_guard\\n\\n\\n@_capture_source_builtins\\ndef source_live_guard(baseline, module, digest, guards, names=None, class_name=None):\\n    """Independent lexical source/runtime binding, including genuine class methods."""\\n    canonical = {key:value for key,value in baseline}\\n\'),\n', b'def initializer_origin_inverse(raw):\n    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""\n    edits = [\n        (b\'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        return function(canonical,module,digest,guards,names,class_name,None)\\n    def initializer_live_guard(context):\\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        t = context[\\\'training_context\\\']\\n        return function(canonical,t[\\\'legacy\\\'][\\\'selected\\\'][\\\'genuine\\\'][\\\'reference\\\'],\\n            \\\'163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8\\\',\\n            t[\\\'guards\\\'],{\\\'admit_cgroup\\\',\\\'require\\\'},None,context)\\n    source_live_guard.initializer = initializer_live_guard\\n    return source_live_guard\\n\\n\\n@_capture_source_builtins\\ndef source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner):\\n    """Independent lexical source/runtime binding, including genuine class methods."""\\n    canonical = {key:value for key,value in baseline}\\n\', b\'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\\n            raise error(\\\'authenticated builtin baseline/source binding changed\\\')\\n        return function(canonical,module,digest,guards,names,class_name)\\n    return source_live_guard\\n\\n\\n@_capture_source_builtins\\ndef source_live_guard(baseline, module, digest, guards, names=None, class_name=None):\\n    """Independent lexical source/runtime binding, including genuine class methods."""\\n    canonical = {key:value for key,value in baseline}\\n\'),\n'),
        (b'def initializer_origin_test_inverse(raw):\n    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""\n    raw = initializer_delegate_test_inverse(raw)\n    start = raw.index(b\'# BEGIN INITIALIZER ORIGIN FALSIFIER\\n\')\n    end = raw.index(b\'# BEGIN BUILTIN BASELINE FALSIFIER\\n\',start)\n', b'def initializer_origin_test_inverse(raw):\n    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""\n    start = raw.index(b\'# BEGIN INITIALIZER ORIGIN FALSIFIER\\n\')\n    end = raw.index(b\'# BEGIN BUILTIN BASELINE FALSIFIER\\n\',start)\n'),
        (b'            builtin_endpoint_finally_contract(admit,unit)\n            initializer_endpoint_finally_contract(admit,unit,legacy)\n            initializer_delegate_contract(e,context,admit,guard,unit,legacy)\n        # A source or callback mutation after loader capture is rejected at call and exit.\n        saved=trainer.check_terminal.__code__;trainer.check_terminal.__code__=(lambda *args:None).__code__\n', b'            builtin_endpoint_finally_contract(admit,unit)\n            initializer_endpoint_finally_contract(admit,unit,legacy)\n        # A source or callback mutation after loader capture is rejected at call and exit.\n        saved=trainer.check_terminal.__code__;trainer.check_terminal.__code__=(lambda *args:None).__code__\n'),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer delegate inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == 'aa0f564a83b446085e05960e438fffa8d5a8b8ace8de2c726f14ebb382835d96', 'initializer delegate inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '3d587521ba567d4c3aeb728a1384855bc5757e7cfe790c3f196ff0a6d8555018', 'initializer delegate inverse AST differs'
    return raw



def initializer_delegate_contract(e,context=None,admit=None,endpoint_guard=None,unit=None,legacy=None):
    """Independent import-time binding rejects delegate replacement and metadata/closure mutation."""
    from types import FunctionType
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'ordinary.py';path.write_text('def valid(v):\n    return all(v)\n')
        ordinary=module('_delegate_ordinary',path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        live=e.source_live_guard(ordinary,digest,{})
        delegate=e.source_live_guard.initializer
        fresh=lambda:e.source_live_guard(ordinary,digest,{})
        checks=[fresh,live]
        if context is not None:
            checks.append(e.source_live_guard.initializer(context))
            checks.extend((lambda:e.load_endpoint_reader(context),lambda:endpoint_guard(context),
                lambda:admit(unit,'cpu','control',179061),lambda:e.exit_rehash(context,endpoint_guard)))
        def rejected():
            for call in checks:rejects(call,'initializer delegate')
        wrapper_dict=e.source_live_guard.__dict__
        class Mask(dict):
            def get(self,key,default=None):
                return delegate if key=='initializer' else super().get(key,default)
        for namespace in (Mask(wrapper_dict),dict(wrapper_dict)):
            e.source_live_guard.__dict__=namespace
            try:
                e.source_live_guard.initializer=lambda context:lambda:None
                rejected()
                e.source_live_guard.initializer=delegate
                rejected()  # Even a replacement dictionary with the genuine value is foreign.
            finally:e.source_live_guard.__dict__=wrapper_dict
        del e.source_live_guard.initializer
        try:rejected()  # A missing attribute must raise the same clear binding error.
        finally:e.source_live_guard.initializer=delegate
        with patch.object(e.source_live_guard,'initializer',lambda context:lambda:None):rejected()
        # Same bytecode with a foreign globals dictionary is still a replacement.
        clone=FunctionType(delegate.__code__,dict(delegate.__globals__),delegate.__name__,delegate.__defaults__,delegate.__closure__)
        with patch.object(e.source_live_guard,'initializer',clone):rejected()
        original_code=delegate.__code__
        # Code.replace preserves the freevar count while changing the code identity.
        code=original_code.replace(co_consts=original_code.co_consts+('changed',))
        mutations=(('__code__',code),('__defaults__',(None,)),('__kwdefaults__',{'forged':True}),
            ('__name__','forged'),('__qualname__','forged'),('__module__','forged'))
        for key,value in mutations:
            saved=getattr(delegate,key);setattr(delegate,key,value)
            try:rejected()
            finally:setattr(delegate,key,saved)
        for cell in delegate.__closure__:
            saved=cell.cell_contents;cell.cell_contents=object()
            try:rejected()
            finally:cell.cell_contents=saved
        checker=dict(zip(delegate.__code__.co_freevars,(c.cell_contents for c in delegate.__closure__)))['initializer_check']
        saved=checker.__code__;checker.__code__=saved.replace(co_consts=saved.co_consts+('changed',))
        try:rejected()
        finally:checker.__code__=saved
        # Context and public snapshots cannot authorize a new delegate.
        if context is not None:
            with patch.object(e.source_live_guard,'initializer',lambda context:lambda:None),\
                    patch.dict(context,{'helper_snapshots':[], 'training_context':dict(context['training_context'])}):
                rejected()
            visited=[]
            class MutatingUnit(dict):
                def keys(self):
                    visited.append(True);e.source_live_guard.initializer=lambda context:lambda:None
                    return super().keys()
            class MaskingUnit(MutatingUnit):
                def keys(self):
                    e.source_live_guard.__dict__=Mask(wrapper_dict)
                    return super().keys()
            for unit_type in (MutatingUnit,MaskingUnit):
                visited.clear()
                try:rejects(lambda:admit(unit_type(unit),'cpu','control',179061),'initializer delegate')
                finally:
                    e.source_live_guard.__dict__=wrapper_dict;e.source_live_guard.initializer=delegate
                assert visited
        fresh()();live()
    print('PASS initializer delegate import/fresh/cached/call/finally/exit identity and metadata binding')


# END INITIALIZER DELEGATE FALSIFIER


# BEGIN INITIALIZER ORIGIN FALSIFIER

def initializer_origin_inverse(raw):
    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""
    raw = initializer_delegate_inverse(raw)
    edits = [
        (b'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error(\'authenticated builtin baseline/source binding changed\')\n        return function(canonical,module,digest,guards,names,class_name,None)\n    def initializer_live_guard(context):\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error(\'authenticated builtin baseline/source binding changed\')\n        t = context[\'training_context\']\n        return function(canonical,t[\'legacy\'][\'selected\'][\'genuine\'][\'reference\'],\n            \'163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8\',\n            t[\'guards\'],{\'admit_cgroup\',\'require\'},None,context)\n    source_live_guard.initializer = initializer_live_guard\n    return source_live_guard\n\n\n@_capture_source_builtins\ndef source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = {key:value for key,value in baseline}\n', b'        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error(\'authenticated builtin baseline/source binding changed\')\n        return function(canonical,module,digest,guards,names,class_name)\n    return source_live_guard\n\n\n@_capture_source_builtins\ndef source_live_guard(baseline, module, digest, guards, names=None, class_name=None):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = {key:value for key,value in baseline}\n'),
        (b"        'authenticated module/builtins required')\n    path = Path(module.__file__); spec = module.__spec__\n    if initializer_owner is None:\n        def registry_guard():\n            return sys.modules.get(module.__name__) is module\n    else:\n        # The original exporter never registers this one module. Its actual\n        # holder objects, captured independently of mutable snapshots, own it.\n        from importlib.machinery import SourceFileLoader\n        t = initializer_owner['training_context']; legacy = t['legacy']\n        selected = legacy['selected']; genuine = selected['genuine']; admission = legacy['admission']\n        fit_context = t['fit_context']; original = legacy['original']; Flat = original.FlatAdmission\n        require(module.__name__ == '_genuine_fit_reference' and\n            path == Path('/home/riomus/runs/sfora-native256-fit-export-source-v1/export_siglip2_substrate_fit.py') and\n            spec is not None and type(spec.loader) is SourceFileLoader and module.__loader__ is spec.loader and\n            spec.loader.name == module.__name__ and spec.loader.path == str(path) and\n            type(admission) is Flat, 'authenticated initializer original source/loader differs')\n        loader_name,loader_path = spec.loader.name,spec.loader.path\n        def registry_guard():\n            require(initializer_owner['training_context'] is t and t['legacy'] is legacy and\n                t['fit_context'] is fit_context and fit_context['legacy'] is legacy and\n                legacy['selected'] is selected and selected['genuine'] is genuine and\n                genuine['reference'] is module and legacy['admission'] is admission and admission.init is module and\n                legacy['original'] is original and original.FlatAdmission is Flat and type(admission) is Flat,\n                'authenticated initializer owner binding changed')\n            require(t['guards'] is guards and guards.get(str(path)) == digest,\n                'authenticated initializer source digest changed')\n            require(module.__loader__ is spec.loader and vars(spec.loader) == {'name':loader_name,'path':loader_path},\n                'authenticated initializer loader changed')\n            return module.__name__ not in sys.modules and all(value is not module for value in sys.modules.values())\n    require(spec is not None and spec.loader is not None and spec.name == module.__name__ and\n        Path(spec.origin) == path and registry_guard(),\n        'authenticated module registry/origin differs')\n    raw = bound_file(guards,path,digest).read_bytes()\n    require(hashlib.sha256(raw).hexdigest() == digest, 'authenticated source changed before compilation')\n    tree = ast.parse(raw,filename=str(path)); compiled = compile(raw,str(path),'exec',dont_inherit=True)\n    if initializer_owner is not None:\n        # This pinned initializer has only stdlib imports and constant assignments.\n        expected_globals = {}\n        nodes = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))]\n        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',dont_inherit=True),expected_globals)\n        expected_names = set(expected_globals) | {n.name for n in tree.body if isinstance(n,ast.FunctionDef)} | {\n            '__name__','__doc__','__package__','__loader__','__spec__','__file__','__cached__'}\n        require(vars(module).keys() == expected_names and all(type(vars(module)[k]) is type(v) and\n            vars(module)[k] == v for k,v in expected_globals.items()), 'authenticated initializer source globals differ')\n    values = dict(vars(module)); literals = {k:copy.deepcopy(v) for k,v in values.items()\n        if k != '__builtins__' and isinstance(v,(dict,list,tuple,set,frozenset))}\n", b"        'authenticated module/builtins required')\n    path = Path(module.__file__); spec = module.__spec__\n    require(spec is not None and spec.loader is not None and spec.name == module.__name__ and\n        Path(spec.origin) == path and sys.modules.get(module.__name__) is module,\n        'authenticated module registry/origin differs')\n    raw = bound_file(guards,path,digest).read_bytes()\n    require(hashlib.sha256(raw).hexdigest() == digest, 'authenticated source changed before compilation')\n    tree = ast.parse(raw,filename=str(path)); compiled = compile(raw,str(path),'exec',dont_inherit=True)\n    values = dict(vars(module)); literals = {k:copy.deepcopy(v) for k,v in values.items()\n        if k != '__builtins__' and isinstance(v,(dict,list,tuple,set,frozenset))}\n"),
        (b'    def guard():\n        builtin_guard()\n        require(module.__name__ == name and registry_guard() and module.__spec__ is spec and\n            spec.name == name and spec.loader is loader and Path(spec.origin) == Path(module.__file__) == path and\n            vars(module).keys() == values.keys() and all(vars(module)[k] is v for k,v in values.items()) and\n', b'    def guard():\n        builtin_guard()\n        require(module.__name__ == name and sys.modules.get(name) is module and module.__spec__ is spec and\n            spec.name == name and spec.loader is loader and Path(spec.origin) == Path(module.__file__) == path and\n            vars(module).keys() == values.keys() and all(vars(module)[k] is v for k,v in values.items()) and\n'),
        (b"        for m,names in ((t['trainer'],{'check_steps','check_ranking_bank','ranking_membership',\n                                      'ranking_bank','json_sha256','require'}),\n                       (t['fitter'],None),(t['old'],{'zero_events','require'}))) + (source_live_guard.initializer(context),)\n    node = next(n for n in ast.parse(Path(trainer.__file__).read_bytes()).body\n        if isinstance(n,ast.FunctionDef) and n.name == 'admit_terminal')\n", b"        for m,names in ((t['trainer'],{'check_steps','check_ranking_bank','ranking_membership',\n                                      'ranking_bank','json_sha256','require'}),\n                       (t['fitter'],None),(t['old'],{'zero_events','require'}),\n                       (initializer,{'admit_cgroup','require'})))\n    node = next(n for n in ast.parse(Path(trainer.__file__).read_bytes()).body\n        if isinstance(n,ast.FunctionDef) and n.name == 'admit_terminal')\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer origin inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '2390c60fe5e87e82ab122c5c0101476b378792541bc454470c37d6c7f0410d40', 'initializer origin inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == 'c40c939619cff5ddf2a7666ee975592b2fc8caf6f71d9f28af0e3993a9c81aa4', 'initializer origin inverse AST differs'
    return raw


def initializer_origin_test_inverse(raw):
    """Restore exact b9bc24f7 bytes; no normalization or historical hash changes."""
    raw = initializer_delegate_test_inverse(raw)
    start = raw.index(b'# BEGIN INITIALIZER ORIGIN FALSIFIER\n')
    end = raw.index(b'# BEGIN BUILTIN BASELINE FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    edits = [
        (b'def builtin_baseline_inverse(raw):\n    """Restore the held repair exactly before applying its unchanged inverses."""\n    raw = initializer_origin_inverse(raw)\n    edits = [\n        (b"# BEGIN ENDPOINT READER AUTHENTICATION\\n# Capture the interpreter bindings at evaluator import, before helper admission.\\n_SOURCE_BUILTINS = tuple(vars(__import__(\'builtins\')).items())\\n\\nFIRST_SELECTION_OWNER = {\'root\':\'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5\',\\n", b"# BEGIN ENDPOINT READER AUTHENTICATION\\n# Capture the interpreter bindings at evaluator import, before helper admission.\\n_SOURCE_BUILTINS = vars(__import__(\'builtins\')).copy()\\n\\nFIRST_SELECTION_OWNER = {\'root\':\'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5\',\\n"),\n', b'def builtin_baseline_inverse(raw):\n    """Restore the held repair exactly before applying its unchanged inverses."""\n    edits = [\n        (b"# BEGIN ENDPOINT READER AUTHENTICATION\\n# Capture the interpreter bindings at evaluator import, before helper admission.\\n_SOURCE_BUILTINS = tuple(vars(__import__(\'builtins\')).items())\\n\\nFIRST_SELECTION_OWNER = {\'root\':\'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5\',\\n", b"# BEGIN ENDPOINT READER AUTHENTICATION\\n# Capture the interpreter bindings at evaluator import, before helper admission.\\n_SOURCE_BUILTINS = vars(__import__(\'builtins\')).copy()\\n\\nFIRST_SELECTION_OWNER = {\'root\':\'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5\',\\n"),\n'),
        (b"\ndef builtin_baseline_test_inverse(raw):\n    raw = initializer_origin_test_inverse(raw)\n    start = raw.index(b'# BEGIN BUILTIN BASELINE FALSIFIER\\n')\n    end = raw.index(b'# BEGIN TRANSITIVE AUTHENTICATION FALSIFIER\\n',start)\n", b"\ndef builtin_baseline_test_inverse(raw):\n    start = raw.index(b'# BEGIN BUILTIN BASELINE FALSIFIER\\n')\n    end = raw.index(b'# BEGIN TRANSITIVE AUTHENTICATION FALSIFIER\\n',start)\n"),
        (b'\n\n@original_initializer_files()\ndef endpoint_derivative_contract(e):\n    """Real frozen callbacks/class/terminal log and actual CPU metadata; tiny I/O inventory."""\n', b'\n\ndef endpoint_derivative_contract(e):\n    """Real frozen callbacks/class/terminal log and actual CPU metadata; tiny I/O inventory."""\n'),
        (b"    modules={n:module('_endpoint_deployed_'+n,HERE/(n+'.py')) for n in names}\n    trainer,flat,fitter,init,nearest,old,training=modules.values()\n    legacy=original_initializer(flat);init=legacy['admission'].init\n    modules['export_siglip2_substrate_fit']=init\n    accepted=json.loads((EVIDENCE/'connected-mlp-cpu-v6/receipt.json').read_bytes())\n    initializer=json.loads((EVIDENCE/'identity-diversity-v1/cpu-v5/receipt.json').read_bytes())\n", b"    modules={n:module('_endpoint_deployed_'+n,HERE/(n+'.py')) for n in names}\n    trainer,flat,fitter,init,nearest,old,training=modules.values()\n    accepted=json.loads((EVIDENCE/'connected-mlp-cpu-v6/receipt.json').read_bytes())\n    initializer=json.loads((EVIDENCE/'identity-diversity-v1/cpu-v5/receipt.json').read_bytes())\n"),
        (b"        def is_file(p,*a,**kw): return real_is_file(aliases.get(str(p),p),*a,**kw)\n        def resolve_file(p,*a,**kw): return p if str(p) in aliases else real_resolve(p,*a,**kw)\n        reader=legacy['admission']\n        guards=dict(source_guards)\n        t={'trainer':training,'guards':guards,'legacy':legacy,'nearest':nearest,'fitter':fitter,'old':old,\n", b"        def is_file(p,*a,**kw): return real_is_file(aliases.get(str(p),p),*a,**kw)\n        def resolve_file(p,*a,**kw): return p if str(p) in aliases else real_resolve(p,*a,**kw)\n        reader=flat.FlatAdmission();reader.init=init\n        legacy={'original':flat,'admission':reader,'selected':{'genuine':{'reference':init}},'invocations':set()}\n        guards=dict(source_guards)\n        t={'trainer':training,'guards':guards,'legacy':legacy,'nearest':nearest,'fitter':fitter,'old':old,\n"),
        (b"            assert admit(unit,'cpu','control',179061)==record\n            builtin_endpoint_finally_contract(admit,unit)\n            initializer_endpoint_finally_contract(admit,unit,legacy)\n        # A source or callback mutation after loader capture is rejected at call and exit.\n        saved=trainer.check_terminal.__code__;trainer.check_terminal.__code__=(lambda *args:None).__code__\n", b"            assert admit(unit,'cpu','control',179061)==record\n            builtin_endpoint_finally_contract(admit,unit)\n        # A source or callback mutation after loader capture is rejected at call and exit.\n        saved=trainer.check_terminal.__code__;trainer.check_terminal.__code__=(lambda *args:None).__code__\n"),
        (b"def endpoint_authentication_contract():\n    e=module('_endpoint_complete_evaluator',DRIVER)\n    initializer_constructor_contract(e)\n    builtin_baseline_contract(e)\n    builtin_namespace_contract(e)\n", b"def endpoint_authentication_contract():\n    e=module('_endpoint_complete_evaluator',DRIVER)\n    builtin_baseline_contract(e)\n    builtin_namespace_contract(e)\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'initializer origin inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '28d0297cebd6a7bab395a44bc60a4aa1bcaacb9160a8093044a54fa270b104d9', 'initializer origin inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == 'acf40797489143b179c5742377c73737996d4c27b5fe1e4f647725a2e0fdb009', 'initializer origin inverse AST differs'
    return raw


from contextlib import contextmanager


@contextmanager
def original_initializer_files():
    """Map only absent remote FILE reads; preserve loader, code and module globals."""
    from importlib.machinery import SourceFileLoader
    from unittest.mock import patch
    path=Path('/home/riomus/runs/sfora-native256-fit-export-source-v1/export_siglip2_substrate_fit.py')
    with tempfile.TemporaryDirectory() as directory:
        local=Path(directory)/path.name;local.write_bytes((HERE/path.name).read_bytes())
        methods={key:getattr(Path,key) for key in ('open','stat','resolve','is_file')}
        get_data=SourceFileLoader.get_data
        def data(loader,name):
            if name==str(path):return local.read_bytes()
            if loader.path==str(path):raise FileNotFoundError(name)  # No remote bytecode cache.
            return get_data(loader,name)
        def open_file(p,*a,**kw):return methods['open'](local if p==path else p,*a,**kw)
        def stat_file(p,*a,**kw):return methods['stat'](local if p==path else p,*a,**kw)
        def resolve_file(p,*a,**kw):return p if p==path else methods['resolve'](p,*a,**kw)
        def is_file(p,*a,**kw):return methods['is_file'](local if p==path else p,*a,**kw)
        with patch.object(SourceFileLoader,'get_data',data),patch.object(Path,'open',open_file),\
                patch.object(Path,'stat',stat_file),patch.object(Path,'resolve',resolve_file),\
                patch.object(Path,'is_file',is_file):
            yield local


def original_initializer(flat):
    """Execute the original exporter constructor/return and quadratic holder binding."""
    exporter=module('_initializer_exporter',HERE/'export_siglip2_genuine_views.py')
    tree=ast.parse(Path(exporter.__file__).read_bytes())
    authority=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authority')
    start=next(i for i,n in enumerate(authority.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='spec')
    # Consecutive original spec/module/exec/origin check, and original context return.
    constructor=copy.deepcopy(authority);constructor.name='constructor';constructor.decorator_list=[]
    constructor.body=copy.deepcopy(authority.body[start:start+4]+[authority.body[-1]])
    ns={**vars(exporter),'ref_root':exporter.ORIGINAL_REF_ROOT,'root':HERE,'code':{},'launch':{},
        'guards':{},'ref_code':{},'prior':{},'selected':{}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[constructor],type_ignores=[])),exporter.__file__,'exec'),ns)
    genuine=ns['constructor'](SimpleNamespace())
    assert '_genuine_fit_reference' not in sys.modules
    quadratic=module('_initializer_quadratic',HERE/'train_siglip2_quadratic_readout.py')
    authority=next(n for n in ast.parse(Path(quadratic.__file__).read_bytes()).body
        if isinstance(n,ast.FunctionDef) and n.name=='authority')
    start=next(i for i,n in enumerate(authority.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='admission')
    ns={'selected':{'original':flat,'genuine':genuine}}
    exec(compile(ast.Module(body=authority.body[start:start+2],type_ignores=[]),quadratic.__file__,'exec'),ns)
    return {'original':flat,'selected':ns['selected'],'admission':ns['admission'],'invocations':set()}


def initializer_constructor_contract(e):
    """All four actual constructors: three registered modules, one owned unregistered initializer."""
    trainer=module('_origin_connected',HERE/'train_siglip2_connected_mlp.py')
    nearest=module('_origin_nearest',HERE/'train_siglip2_nearest_ranking.py')
    flat=module('_origin_flat',HERE/'train_siglip2_substrate_adaptation.py')
    guards={}
    def load(loader,name,filename):
        sys.modules.pop(name,None)
        path=HERE/filename
        return loader.load_authenticated(name,path,hashlib.sha256(path.read_bytes()).hexdigest(),guards)
    training=load(trainer,'_connected_original_trainer','train_siglip2_identity_diversity.py')
    fitter=load(nearest,'_compact_fitter','fit_siglip2_prototype_residual.py')
    old=load(fitter,'_prototype_original','train_siglip2_quadratic_readout.py')
    with original_initializer_files() as local:
        legacy=original_initializer(flat);init=legacy['admission'].init
        assert init is legacy['selected']['genuine']['reference']
        for m in (training,fitter,old):
            assert sys.modules[m.__name__] is m
            e.source_live_guard(m,guards[m.__file__],guards)()
        guards.update({m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (trainer,nearest,flat,init)})
        rejects(lambda:e.source_live_guard(init,guards[init.__file__],guards,names={'admit_cgroup','require'}),
            'authenticated module registry/origin differs')
        t={'trainer':training,'fitter':fitter,'old':old,'nearest':nearest,'legacy':legacy,
            'fit_context':{'legacy':legacy},'guards':guards}
        context={'trainer':trainer,'training_context':t,'guards':dict(guards),
            'code':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in e.FILES}}
        admit,guard=e.load_endpoint_reader(context)
        guard(context)
        from unittest.mock import patch
        from types import ModuleType
        import builtins
        def reject_boundaries(text='authenticated',fresh=True):
            for call in (lambda:guard(context),lambda:admit({},'cpu','control',179061),
                         lambda:e.exit_rehash(context,guard)):
                rejects(call,text)
            if fresh:rejects(lambda:e.load_endpoint_reader(context),text)
        # Other unregistered helpers and any foreign/None/self registry entry reject.
        for m in (training,fitter,old):
            saved=sys.modules.pop(m.__name__)
            try:reject_boundaries()
            finally:sys.modules[m.__name__]=saved
        for key,value in ((init.__name__,object()),(init.__name__,None),(init.__name__,init),('_foreign_initializer_alias',init)):
            assert key not in sys.modules
            sys.modules[key]=value
            try:reject_boundaries()
            finally:del sys.modules[key]
        spec=init.__spec__;loader=spec.loader
        for obj,key,value in ((init,'__name__','foreign'),(init,'__file__',str(local)),
                (init,'__spec__',None),(init,'__spec__',copy.copy(spec)),(init,'__loader__',object()),
                (spec,'name','foreign'),(spec,'origin',str(local)),(spec,'loader',None),
                (loader,'name','foreign'),(loader,'path',str(local))):
            with patch.object(obj,key,value):
                reject_boundaries(fresh=not(obj is init and key=='__spec__' and value is not None))
        for fn in (init.require,init.admit_cgroup):
            for key,value in (('__code__',(lambda *args:None).__code__),('__defaults__',(True,)),
                              ('__kwdefaults__',{'forged':True}),('__module__','foreign')):
                saved=getattr(fn,key);setattr(fn,key,value)
                try:reject_boundaries()
                finally:setattr(fn,key,saved)
        with patch.object(init,'require',lambda *args:None):reject_boundaries()
        with patch.dict(vars(init),{'all':lambda values:True}):reject_boundaries('builtin')
        with patch.dict(vars(init),{'Path':lambda p:p}):reject_boundaries()
        with patch.dict(init.STARTUP_POLICY,{'host_bytes':16*1024**3}):reject_boundaries()
        with patch.dict(vars(init),{'extra_global':True}):reject_boundaries()
        saved_all=builtins.all;error_type=ValueError;baseline=e._SOURCE_BUILTINS
        try:
            builtins.all=lambda values:True
            e._SOURCE_BUILTINS=tuple({**dict(baseline),'all':builtins.all}.items())
            for call in (lambda:guard(context),lambda:admit({},'cpu','control',179061),
                         lambda:e.exit_rehash(context,guard),lambda:e.load_endpoint_reader(context)):
                try:call()
                except error_type as error:assert 'builtin' in str(error)
                else:raise AssertionError('initializer accepted poisoned builtin/baseline')
        finally:
            builtins.all=saved_all;e._SOURCE_BUILTINS=baseline
        # Mutable holder/snapshot pairs cannot replace independently captured identities.
        genuine=legacy['selected']['genuine'];admission=legacy['admission']
        forged=ModuleType(init.__name__);vars(forged).update(vars(init))
        with patch.dict(genuine,{'reference':forged}),patch.object(admission,'init',forged),\
                patch.dict(context,{'helper_snapshots':[(forged,dict(vars(forged)))]}):
            reject_boundaries('binding changed',fresh=False)
            rejects(lambda:e.load_endpoint_reader(context),'authenticated')
        replacement=flat.FlatAdmission();replacement.init=init
        for holder,key,value in ((legacy,'admission',replacement),(legacy,'selected',dict(legacy['selected'])),
                (legacy['selected'],'genuine',dict(genuine)),(t,'legacy',dict(legacy)),
                (t,'fit_context',dict(t['fit_context'])),(context,'training_context',dict(t))):
            with patch.dict(holder,{key:value}),patch.dict(context,{'helper_snapshots':[value]}):
                reject_boundaries('binding changed',fresh=False)
        # Cached guards reread bytes even when the caller replaces its mutable digest.
        raw=local.read_bytes();digest=guards[init.__file__]
        with patch.dict(guards,{init.__file__:'0'*64}):
            reject_boundaries('source digest')
        local.write_bytes(raw+b'\n# changed original source\n')
        try:
            reject_boundaries('SHA256')
            with patch.dict(guards,{init.__file__:hashlib.sha256(local.read_bytes()).hexdigest()}):
                reject_boundaries('source digest')
        finally:local.write_bytes(raw)
        assert guards[init.__file__]==digest
        guard(context)
        assert init.__name__ not in sys.modules
        assert not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers'} for n in sys.modules)
    print('PASS original all-four constructors; finite initializer initial/cached/call/exit tamper rejection')


def initializer_endpoint_finally_contract(admit,unit,legacy):
    """Replace the actual holder during genuine check_unit; the exit guard must catch it."""
    original=legacy['admission'];replacement=legacy['original'].FlatAdmission();replacement.init=original.init
    visited=[]
    class MutatingUnit(dict):
        def keys(self):
            visited.append(True);legacy['admission']=replacement
            return super().keys()
    try:
        rejects(lambda:admit(MutatingUnit(unit),'cpu','control',179061),'initializer owner binding')
    finally:legacy['admission']=original
    assert visited
    print('PASS actual initializer holder replacement after preguard rejected by admission finally')


# END INITIALIZER ORIGIN FALSIFIER


# BEGIN BUILTIN BASELINE FALSIFIER

def builtin_baseline_inverse(raw):
    """Restore the held repair exactly before applying its unchanged inverses."""
    raw = initializer_origin_inverse(raw)
    edits = [
        (b"# BEGIN ENDPOINT READER AUTHENTICATION\n# Capture the interpreter bindings at evaluator import, before helper admission.\n_SOURCE_BUILTINS = tuple(vars(__import__('builtins')).items())\n\nFIRST_SELECTION_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5',\n", b"# BEGIN ENDPOINT READER AUTHENTICATION\n# Capture the interpreter bindings at evaluator import, before helper admission.\n_SOURCE_BUILTINS = vars(__import__('builtins')).copy()\n\nFIRST_SELECTION_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5',\n"),
        (b'\n\ndef _capture_source_builtins(function):\n    """Keep the import-time baseline out of the mutable helper-admission globals."""\n    canonical = _SOURCE_BUILTINS\n    error,code = ValueError,function.__code__\n    def source_live_guard(module, digest, guards, names=None, class_name=None):\n        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:\n            raise error(\'authenticated builtin baseline/source binding changed\')\n        return function(canonical,module,digest,guards,names,class_name)\n    return source_live_guard\n\n\n@_capture_source_builtins\ndef source_live_guard(baseline, module, digest, guards, names=None, class_name=None):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = {key:value for key,value in baseline}\n    builtin_namespace = canonical[\'__import__\'](\'builtins\').__dict__\n    namespaces = [module.__dict__,canonical[\'globals\']()]\n', b'\n\ndef source_live_guard(module, digest, guards, names=None, class_name=None):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = _SOURCE_BUILTINS.copy()\n    builtin_namespace = canonical[\'__import__\'](\'builtins\').__dict__\n    namespaces = [module.__dict__,canonical[\'globals\']()]\n'),
        (b"    def builtin_guard():\n        # No global/builtin calls: even all/any/type/ValueError may have changed.\n        if _SOURCE_BUILTINS is not baseline:\n            raise error('authenticated builtin baseline binding changed')\n        for key,value in canonical.items():\n            if builtin_namespace.get(key) is not value:\n", b'    def builtin_guard():\n        # No global/builtin calls: even all/any/type/ValueError may have changed.\n        for key,value in canonical.items():\n            if builtin_namespace.get(key) is not value:\n'),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'builtin baseline source inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '08220b8dcb4ee5ad6b5cda7ecc9f587c3ef90f6ba52046a4af6ddadf4471fe0a', 'builtin baseline source inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '695567e05996a71b56ebb8d345277ebb3737ae8178055229d8c0aebc50d9c996', 'builtin baseline source inverse AST differs'
    return raw


def builtin_baseline_test_inverse(raw):
    raw = initializer_origin_test_inverse(raw)
    start = raw.index(b'# BEGIN BUILTIN BASELINE FALSIFIER\n')
    end = raw.index(b'# BEGIN TRANSITIVE AUTHENTICATION FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    for added in (b'    raw = builtin_baseline_inverse(raw)\n', b'    raw = builtin_baseline_test_inverse(raw)\n', b'    builtin_baseline_contract(e)\n'):
        assert raw.count(added) == 1, 'builtin baseline test inverse edit differs'
        raw = raw.replace(added,b'',1)
    assert hashlib.sha256(raw).hexdigest() == '42f95d16a86e4d7f13e7872597520bfc0c82ee5c6d66d9af859482ae6d036016', 'builtin baseline test inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '5566cbae69e80179d4b641819b65ae743134d53b542c3d2a3eb2eb9fa05fdb60', 'builtin baseline test inverse AST differs'
    return raw


def builtin_baseline_contract(e, case=None):
    """The import-time truth cannot be jointly poisoned with canonical all."""
    import builtins
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'genuine.py';path.write_text('def valid(v):\n    return all(v)\n')
        genuine=module('_builtin_baseline_genuine',path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        cases=('poison_before','rebind_before','poison_after','rebind_after')
        for selected in cases if case is None else (case,):
            baseline=e._SOURCE_BUILTINS;saved=builtins.all;changed=False
            values=dict(baseline)
            replacement=lambda values:True
            guard=e.source_live_guard(genuine,digest,{})
            guard();assert genuine.valid([False]) is False
            try:
                if selected.startswith('poison'):
                    try:
                        baseline['all']=replacement
                        changed=True
                    except TypeError:
                        assert values['all'] is saved
                        if selected=='poison_after':
                            guard()
                            continue
                else:
                    e._SOURCE_BUILTINS={**values,'all':replacement}
                if selected.endswith('before'):
                    builtins.all=replacement
                    assert genuine.valid([False]) is True
                    rejects(lambda:e.source_live_guard(genuine,digest,{}),'builtin')
                else:
                    # The canonical builtin is unchanged: the baseline alone is guarded.
                    rejects(guard,'builtin')
            finally:
                builtins.all=saved
                if changed:
                    baseline['all']=saved
                e._SOURCE_BUILTINS=baseline
            guard();assert genuine.valid([False]) is False
        from types import FunctionType
        core=next(cell.cell_contents for cell in e.source_live_guard.__closure__
            if type(cell.cell_contents) is FunctionType)
        saved=core.__code__
        try:
            core.__code__=(lambda *args:None).__code__
            rejects(lambda:e.source_live_guard(genuine,digest,{}),'builtin baseline/source')
        finally:
            core.__code__=saved
        e.source_live_guard(genuine,digest,{})()
    print('PASS immutable import-time builtin baseline',case or 'all cases')


# END BUILTIN BASELINE FALSIFIER


# BEGIN TRANSITIVE AUTHENTICATION FALSIFIER

def transitive_auth_inverse(raw):
    """Undo only this finite binding repair; retain every earlier inverse hash."""
    raw = builtin_baseline_inverse(raw)
    edits = [
        (b"\n# BEGIN ENDPOINT READER AUTHENTICATION\n# Capture the interpreter bindings at evaluator import, before helper admission.\n_SOURCE_BUILTINS = vars(__import__('builtins')).copy()\n\nFIRST_SELECTION_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5',\n    'execution_sha256':'a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a',\n", b"\n# BEGIN ENDPOINT READER AUTHENTICATION\nFIRST_SELECTION_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5',\n    'execution_sha256':'a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a',\n"),
        (b'def source_live_guard(module, digest, guards, names=None, class_name=None):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    canonical = _SOURCE_BUILTINS.copy()\n    builtin_namespace = canonical[\'__import__\'](\'builtins\').__dict__\n    namespaces = [module.__dict__,canonical[\'globals\']()]\n    error = canonical[\'ValueError\']\n    def builtin_guard():\n        # No global/builtin calls: even all/any/type/ValueError may have changed.\n        for key,value in canonical.items():\n            if builtin_namespace.get(key) is not value:\n                raise error(\'authenticated builtin binding changed: \'+key)\n            if key not in (\'__name__\',\'__doc__\',\'__package__\',\'__loader__\',\'__spec__\'):\n                for namespace in namespaces:\n                    if key in namespace:\n                        raise error(\'authenticated builtin shadow: \'+key)\n    def require(condition, message):\n        if not condition:\n            raise error(message)\n    builtin_guard()\n    import builtins\n    from types import ModuleType\n', b'def source_live_guard(module, digest, guards, names=None, class_name=None):\n    """Independent lexical source/runtime binding, including genuine class methods."""\n    import builtins\n    from types import ModuleType\n'),
        (b"                    'unexpected authenticated function decorator')\n                wrapper,fn = fn,fn.__wrapped__\n                namespaces.append(wrapper.__globals__)\n                template = contextmanager(fn)\n                require(wrapper.__code__ is template.__code__ and wrapper.__globals__ is template.__globals__ and\n", b"                    'unexpected authenticated function decorator')\n                wrapper,fn = fn,fn.__wrapped__\n                template = contextmanager(fn)\n                require(wrapper.__code__ is template.__code__ and wrapper.__globals__ is template.__globals__ and\n"),
        (b'                qualified,wrapper,wrapper.__code__ if wrapper is not None else None))\n    def guard():\n        builtin_guard()\n        require(module.__name__ == name and sys.modules.get(name) is module and module.__spec__ is spec and\n            spec.name == name and spec.loader is loader and Path(spec.origin) == Path(module.__file__) == path and\n', b'                qualified,wrapper,wrapper.__code__ if wrapper is not None else None))\n    def guard():\n        require(module.__name__ == name and sys.modules.get(name) is module and module.__spec__ is spec and\n            spec.name == name and spec.loader is loader and Path(spec.origin) == Path(module.__file__) == path and\n'),
        (b"    initializer = t['legacy']['selected']['genuine']['reference']\n    terminal_guards = tuple(source_live_guard(m,t['guards'][m.__file__],t['guards'],names=names)\n        for m,names in ((t['trainer'],{'check_steps','check_ranking_bank','ranking_membership',\n                                      'ranking_bank','json_sha256','require'}),\n                       (t['fitter'],None),(t['old'],{'zero_events','require'}),\n                       (initializer,{'admit_cgroup','require'})))\n    node = next(n for n in ast.parse(Path(trainer.__file__).read_bytes()).body\n", b"    initializer = t['legacy']['selected']['genuine']['reference']\n    terminal_guards = tuple(source_live_guard(m,t['guards'][m.__file__],t['guards'],names=names)\n        for m,names in ((t['fitter'],None),(t['old'],{'zero_events','require'}),\n                       (initializer,{'admit_cgroup','require'})))\n    node = next(n for n in ast.parse(Path(trainer.__file__).read_bytes()).body\n"),
        (b"    dependencies = {key:t[key] for key in ('trainer','nearest','fitter','old','fit_context','legacy')}\n    def guard(current):\n        source_guard()\n        if (any(fn.__code__ is not c or fn.__defaults__ is not None or fn.__kwdefaults__ is not None or\n                fn.__builtins__ is not vars(builtins) or (fn.__module__,fn.__name__,fn.__qualname__) != (__name__,name,name)\n", b"    dependencies = {key:t[key] for key in ('trainer','nearest','fitter','old','fit_context','legacy')}\n    def guard(current):\n        if (any(fn.__code__ is not c or fn.__defaults__ is not None or fn.__kwdefaults__ is not None or\n                fn.__builtins__ is not vars(builtins) or (fn.__module__,fn.__name__,fn.__qualname__) != (__name__,name,name)\n"),
        (b"            t['legacy']['admission'].init is initializer,\n            'endpoint context/source binding changed')\n        flat_guard(); nearest_guard()\n        for check in terminal_guards:\n            check()\n", b"            t['legacy']['admission'].init is initializer,\n            'endpoint context/source binding changed')\n        source_guard(); flat_guard(); nearest_guard()\n        for check in terminal_guards:\n            check()\n"),
        (b"    used = False\n    def guard(current):\n        live_guard()\n        actual = current['helper_snapshots']\n        require(FIRST_SELECTION_OWNER == fact and FIRST_SELECTION_UNIT == unit and current['first_evaluator'] is original and\n", b"    used = False\n    def guard(current):\n        actual = current['helper_snapshots']\n        require(FIRST_SELECTION_OWNER == fact and FIRST_SELECTION_UNIT == unit and current['first_evaluator'] is original and\n"),
        (b"                tuple(a[4]) == b[4] and a[5] == b[5] for a,b in zip(actual,authenticated,strict=True)),\n            'first-selection owner/snapshot binding changed')\n        guard_helpers(current)\n        require(closure(fact['root'],fact['execution_sha256'],FILES,{}) == fact['code'],\n            'first-selection fresh owner closure differs')\n", b"                tuple(a[4]) == b[4] and a[5] == b[5] for a,b in zip(actual,authenticated,strict=True)),\n            'first-selection owner/snapshot binding changed')\n        live_guard(); guard_helpers(current)\n        require(closure(fact['root'],fact['execution_sha256'],FILES,{}) == fact['code'],\n            'first-selection fresh owner closure differs')\n"),
    ]
    for new,old in edits:
        assert raw.count(new) == 1, 'transitive source inverse edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '9cf68f903f297ff7c1ae79b27e82658a799037291db4d9b6dfa2ec08dfa2188b', 'transitive source inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '0bba63a200792f67a205391af1ba7b00398c6ae10bf809816755765835a069a0', 'transitive source inverse AST differs'
    return raw


def transitive_auth_test_inverse(raw):
    raw = builtin_baseline_test_inverse(raw)
    start = raw.index(b'# BEGIN TRANSITIVE AUTHENTICATION FALSIFIER\n')
    end = raw.index(b'# BEGIN ENDPOINT AUTHENTICATION FALSIFIER\n',start)
    raw = raw[:start]+raw[end:]
    for added in (b'    raw = transitive_auth_test_inverse(raw)\n', b'    raw = transitive_auth_inverse(raw)\n', b'        transitive_steps_contract(e,trainer,training,context,admit,guard,unit,record)\n', b'    builtin_namespace_contract(e)\n', b'            builtin_endpoint_finally_contract(admit,unit)\n'):
        assert raw.count(added) == 1, 'transitive test inverse edit differs'
        raw = raw.replace(added,b'',1)
    assert hashlib.sha256(raw).hexdigest() == '9c86b29dd2fb396f5c48ffd4539d15dbf5ad48645fcd937dbb5633b77a952297', 'transitive test inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == '100d2c1c6b01b628bb2a3739357f30da34a1b3294c343efa870bcdf321d820ed', 'transitive test inverse AST differs'
    return raw


def builtin_namespace_contract(e, case=None):
    """Real UNIT/length predicates cannot acquire builtin shadows or changed builtins."""
    import builtins
    trainer=module('_transitive_builtin_trainer',HERE/'train_siglip2_connected_mlp.py')
    digest=hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest()
    unit=copy.deepcopy(e.TRAINING_CPU);unit['service_seconds']=-1
    rejects(lambda:trainer.check_unit(unit),'resources')
    cases=('shadow_all','shadow_len','owned_all','builtin_all','builtin_len','builtin_any','builtin_type','builtin_ValueError')
    for selected in cases if case is None else (case,):
        if selected=='shadow_all':
            trainer.all=lambda values:True
            try:
                trainer.check_unit(unit)  # Demonstrate the actual predicate bypass.
                rejects(lambda:e.source_live_guard(trainer,digest,{}),'builtin')
            finally:
                del trainer.all
        elif selected=='shadow_len':
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'length.py';path.write_text('def valid(value):\n    return len(value) == 1\n')
                genuine=module('_transitive_length',path)
                assert genuine.valid([]) is False
                genuine.len=lambda value:1
                try:
                    assert genuine.valid([]) is True
                    rejects(lambda:e.source_live_guard(genuine,hashlib.sha256(path.read_bytes()).hexdigest(),{}),'builtin')
                finally:
                    del genuine.len
        elif selected=='owned_all':
            e.all=lambda values:True
            try:
                rejects(lambda:e.source_live_guard(trainer,digest,{}),'builtin')
            finally:
                del e.all
        else:
            guard=e.source_live_guard(trainer,digest,{})
            guard()
            key=selected.removeprefix('builtin_');old=vars(builtins)[key]
            error_type=ValueError
            try:
                vars(builtins)[key]=lambda *args:True
                failures=[]
                for boundary in ('before','after'):
                    try:
                        guard()
                    except error_type as error:
                        assert 'builtin' in str(error), str(error)
                    else:
                        failures.append(boundary)
                    if boundary=='before' and key=='all':
                        trainer.check_unit(unit)
                assert not failures, ('accepted changed builtin',key,failures)
            finally:
                vars(builtins)[key]=old
            guard()
        rejects(lambda:trainer.check_unit(unit),'resources')
    print('PASS builtin namespace/control primitive rejection',case or 'all cases')


def transitive_steps_contract(e, trainer, training, context, admit, guard, unit, record):
    """Rank100 must not pass through changed downstream validator objects or code."""
    t=context['training_context']
    trainer.check_terminal(t,record,'cpu','control',179061)
    bad=copy.deepcopy(record);bad['qualifications'][0]['steps'][0]['rank']=100
    rejects(lambda:trainer.check_terminal(t,bad,'cpu','control',179061),'objective arithmetic')
    actual=training.check_steps;saved=actual.__code__
    try:
        actual.__code__=(lambda *args:None).__code__
        trainer.check_terminal(t,bad,'cpu','control',179061)
        rejects(lambda:guard(context),'authenticated')
        rejects(lambda:admit(unit,'cpu','control',179061),'authenticated')
        rejects(lambda:e.exit_rehash(context,guard),'authenticated')
        rejects(lambda:e.load_endpoint_reader(context),'authenticated')
    finally:
        actual.__code__=saved
    try:
        training.check_steps=lambda *args:None
        trainer.check_terminal(t,bad,'cpu','control',179061)
        rejects(lambda:guard(context),'authenticated')
        rejects(lambda:admit(unit,'cpu','control',179061),'authenticated')
        rejects(lambda:e.exit_rehash(context,guard),'authenticated')
        rejects(lambda:e.load_endpoint_reader(context),'authenticated')
    finally:
        training.check_steps=actual
    for name in ('check_ranking_bank','ranking_membership','ranking_bank','json_sha256','require'):
        fn=getattr(training,name);saved=fn.__code__
        try:
            fn.__code__=(lambda *args:None).__code__
            rejects(lambda:guard(context),'authenticated')
            rejects(lambda:e.load_endpoint_reader(context),'authenticated')
        finally:
            fn.__code__=saved
    guard(context)
    rejects(lambda:trainer.check_terminal(t,bad,'cpu','control',179061),'objective arithmetic')
    import builtins
    error_type=ValueError
    for key in ('all','any','type','ValueError'):
        saved=vars(builtins)[key]
        try:
            vars(builtins)[key]=lambda *args:True
            for check in (lambda:guard(context),lambda:admit(unit,'cpu','control',179061),
                          lambda:e.exit_rehash(context,guard)):
                try:
                    check()
                except error_type as error:
                    assert 'builtin' in str(error), str(error)
                else:
                    raise AssertionError('endpoint accepted changed builtin: '+key)
        finally:
            vars(builtins)[key]=saved
    print('PASS genuine transitive check_steps code/object capture/call/exit rejection')


def builtin_endpoint_finally_contract(admit, unit):
    """Mutate all inside genuine check_unit, after the admission guard has passed."""
    import builtins
    saved=builtins.all;visited=[]
    class MutatingUnit(dict):
        def keys(self):
            visited.append(True)
            builtins.all=lambda values:True
            return super().keys()
    bad=MutatingUnit(unit);bad['service_seconds']=-1
    try:
        rejects(lambda:admit(bad,'cpu','control',179061),'builtin binding')
    finally:
        builtins.all=saved
    assert visited, 'actual check_unit boundary was not reached'
    print('PASS actual UNIT mutation after preguard rejected by admission finally')


# END TRANSITIVE AUTHENTICATION FALSIFIER


# BEGIN ENDPOINT AUTHENTICATION FALSIFIER

def endpoint_auth_test_inverse(raw):
    raw = transitive_auth_test_inverse(raw)
    start = raw.index(b'# BEGIN ENDPOINT AUTHENTICATION FALSIFIER\n')
    end = raw.index(b'def main():\n',start)
    raw = raw[:start]+raw[end:]
    edits = [(b'    """Undo only the added helper/import and the two named admission loops."""\n    raw = endpoint_auth_inverse(raw)\n', b'    """Undo only the added helper/import and the two named admission loops."""\n'), (b'def admission_batch_test_inverse(raw):\n    raw = endpoint_auth_test_inverse(raw)\n', b'def admission_batch_test_inverse(raw):\n'), (b'    tree = ast.parse(endpoint_auth_inverse(DRIVER.read_bytes()))\n    functions =', b'    tree = ast.parse(DRIVER.read_text())\n    functions ='), (b"    authority = next(n for n in ast.parse(endpoint_auth_inverse(DRIVER.read_bytes())).body if isinstance(n,ast.FunctionDef) and n.name == 'authority')\n", b"    authority = next(n for n in ast.parse(DRIVER.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name == 'authority')\n"), (b'    endpoint_authentication_contract()\n    actual_admission_scan_falsifier()\n    e =', b'    actual_admission_scan_falsifier()\n    e ='), (b"            serial = body[index]\n            batched = next(n for n in function(candidate,name).body if isinstance(n,ast.Expr) and\n                isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id == 'batch_bound_files')\n", b'            serial, batched = body[index], function(candidate,name).body[index]\n')]
    for new,old in edits:
        assert raw.count(new) == 1, 'endpoint test integration edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == '3aa5aa8be23c569e9e7f161edfb6c7daf0b293839bb156fcaf6ff876e2292529', 'endpoint original test bytes differ'
    return raw


def endpoint_auth_inverse(raw):
    """Exact inverse to the assigned base; every previous inverse remains active."""
    raw = transitive_auth_inverse(raw)
    start = raw.index(b'# BEGIN ENDPOINT READER AUTHENTICATION\n')
    end = raw.index(b'# BEGIN ORIGINAL EXPORT OWNER\n',start)
    raw = raw[:start]+raw[end:]
    edits = [(b'import ast\n', b''), (b"    if 'original_evaluator' in context:\n        modules.append(context['original_evaluator'])\n    if 'first_evaluator' in context:\n        modules.append(context['first_evaluator'])\n", b"    if 'original_evaluator' in context:\n        modules.append(context['original_evaluator'])\n"), (b'def admit_training_unit(context, unit, phase, arm, seed, endpoint_reader):', b'def admit_training_unit(context, unit, phase, arm, seed):'), (b'    record = endpoint_reader(unit,phase,arm,seed)', b'    record = trainer.admit_terminal(t,unit,phase,arm,seed)'), (b'def admit_endpoints(context, endpoints, endpoint_reader):', b'def admit_endpoints(context, endpoints):'), (b"admit_training_unit(context,el['selected_mechanics'][a],'mechanics',a,seed,endpoint_reader)", b"admit_training_unit(context,el['selected_mechanics'][a],'mechanics',a,seed)"), (b"admit_training_unit(context,endpoint['terminal'],'train',arm,seed,endpoint_reader)", b"admit_training_unit(context,endpoint['terminal'],'train',arm,seed)"), (b"    if launch['stage'] == 'full':\n        roots.append(Path(FIRST_SELECTION_OWNER['root']))\n    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and\n", b'    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and\n'), (b"    endpoint_reader,endpoint_guard = load_endpoint_reader(context)\n    admit_endpoints(context,launch['endpoints'][:2],endpoint_reader)", b"    admit_endpoints(context,launch['endpoints'][:2])"), (b"    original_guard = load_original_owner(context) if original_active else None\n    first_reader,first_guard = load_first_selection(context) if launch['stage'] == 'full' else (None,None)\n    guard_helpers(context)\n    if launch['stage'] == 'full':\n        first_receipt = first_reader(context,launch['first_selection'],'score',stage='first',panel='selection')", b"    original_guard = load_original_owner(context) if original_active else None\n    guard_helpers(context)\n    if launch['stage'] == 'full':\n        first_receipt = accept_unit(context,launch['first_selection'],'score',stage='first',panel='selection')"), (b"        admit_endpoints(context,launch['endpoints'][2:],endpoint_reader)", b"        admit_endpoints(context,launch['endpoints'][2:])"), (b'    original_guard = admission_exit_guard(endpoint_guard,first_guard,original_guard)\n    return context,original_guard\n', b'    return context,original_guard\n')]
    for new,old in edits:
        assert raw.count(new) == 1, 'endpoint integration edit differs'
        raw = raw.replace(new,old,1)
    assert hashlib.sha256(raw).hexdigest() == 'bb966f62b6ff609a6a3c3247ad5e625d6a0526824f8c5242ef1ce1b2d0bf54cf', 'endpoint production inverse differs'
    return raw


def endpoint_reader_state_contract():
    """The staged path must retain actual FlatAdmission state and reject cached mutations."""
    e = module('_endpoint_state_evaluator', DRIVER)
    assert hasattr(e, 'batch_terminal_files'), 'missing genuine reader-state staging'
    flat_module = module('_endpoint_flat_source', HERE/'train_siglip2_substrate_adaptation.py')
    Flat = flat_module.FlatAdmission
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)/'a.json'; p.write_bytes(b'{"x":1}')
        q = Path(directory)/'b'; q.write_bytes(b'hello')
        items = [(str(p),hashlib.sha256(p.read_bytes()).hexdigest()),
                 (str(q),hashlib.sha256(q.read_bytes()).hexdigest())]
        serial,parallel = Flat(),Flat(); sg,pg = {},{}
        for reader,guards in ((serial,sg),(parallel,pg)):
            assert reader.read_json(p,items[0][1],guards) == {'x':1}
        repeated = items+[items[1]]
        for path,digest in repeated: serial.bound_file(sg,path,digest)
        e.batch_terminal_files(Flat,parallel,pg,repeated)
        assert (serial.entries,serial.verified,serial.json_bytes,sg) == (parallel.entries,parallel.verified,parallel.json_bytes,pg)
        saved = copy.deepcopy((parallel.entries,parallel.verified,parallel.json_bytes,pg))
        import os
        times=q.stat(); q.write_bytes(b'jello'); os.utime(q,ns=(times.st_atime_ns,times.st_mtime_ns))
        rejects(lambda:e.batch_terminal_files(Flat,parallel,pg,repeated),'SHA256')
        assert saved == (parallel.entries,parallel.verified,parallel.json_bytes,pg)
        q.write_bytes(b'hello'); parallel.entries[str(q)] = (items[1][1],999)
        saved = copy.deepcopy((parallel.entries,parallel.verified,parallel.json_bytes,pg))
        rejects(lambda:e.batch_terminal_files(Flat,parallel,pg,repeated),'SHA256/size')
        assert saved == (parallel.entries,parallel.verified,parallel.json_bytes,pg)
    assert hasattr(e, 'load_endpoint_reader'), 'missing authenticated endpoint reader'
    assert hasattr(e, 'load_first_selection'), 'missing original first-CONTINUE owner'


def authenticated_source_contract(e):
    """Reject live forgeries both before capture and after an independent capture."""
    from types import FunctionType
    from functools import wraps
    from unittest.mock import patch
    targets = [('trainer','train_siglip2_connected_mlp.py',None),
        ('flat','train_siglip2_substrate_adaptation.py','FlatAdmission'),
        ('fitter','fit_siglip2_prototype_residual.py',None),
        ('first',str(EVIDENCE/'connected-mlp-evaluation-full-cpu-v1-freeze'/DRIVER.name),None)]
    with tempfile.TemporaryDirectory() as directory:
        for key,filename,cls in targets:
            source = Path(filename) if Path(filename).is_absolute() else HERE/filename
            path = Path(directory)/Path(filename).name; raw = source.read_bytes(); path.write_bytes(raw)
            m = module('_endpoint_auth_'+key,path); digest = hashlib.sha256(raw).hexdigest()
            guard = e.source_live_guard(m,digest,{},class_name=cls)
            f = m.FlatAdmission.bound_file if cls else m.require
            guard()
            for attribute,value in (('__module__','forged'),('__name__','forged'),('__qualname__','forged'),
                                    ('__defaults__',('forged',)),('__kwdefaults__',{'forged':True}),('__code__',(lambda:None).__code__)):
                old = getattr(f,attribute)
                setattr(f,attribute,value)
                rejects(guard,'authenticated')
                rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'authenticated')
                setattr(f,attribute,old)
            saved = sys.modules[m.__name__]
            sys.modules[m.__name__] = SimpleNamespace(**vars(m)); rejects(guard,'binding changed')
            sys.modules[m.__name__] = saved
            with patch.object(m.__spec__,'origin','/forged'):
                rejects(guard,'binding changed')
            with patch.object(m.__spec__,'loader',object()):
                rejects(guard,'binding changed')
            with patch.object(m,'require',lambda *args:None):
                rejects(guard,'binding changed')
            literal = next((v for k,v in vars(m).items() if k != '__builtins__' and isinstance(v,dict) and v),None)
            if literal is not None:
                literal['forged'] = True; rejects(guard,'binding changed'); del literal['forged']
            path.write_bytes(raw+b'\n'); rejects(guard,'SHA256'); path.write_bytes(raw)
            if cls:
                Flat = m.FlatAdmission
                Flat.__qualname__ = 'forged'
                rejects(guard,'class'); Flat.__qualname__ = 'FlatAdmission'
                with patch.object(Flat,'bound_file',lambda *args:Path('/forged')):
                    rejects(guard,'class')
                with patch.object(Flat,'__new__',staticmethod(lambda cls:object.__new__(cls)),create=True):
                    rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'class inventory')
                with patch.object(Flat,'__getattribute__',lambda self,key:object.__getattribute__(self,key),create=True):
                    rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'class inventory')
                with patch.object(Flat,'SCHEMA',property(Flat.SCHEMA.fget,lambda self,value:None)):
                    rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'property descriptor')
                with patch.object(Flat,'register',staticmethod(Flat.register)):
                    rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'class inventory')
                with patch.object(m,'FlatAdmission',type('FlatAdmission',(Flat,),{})):
                    rejects(guard,'binding changed')
                # A new function carrying copied globals is not the source function.
                copied = FunctionType(f.__code__,dict(vars(m)),f.__name__,f.__defaults__)
                with patch.object(Flat,'bound_file',copied):
                    rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'source function')
            if key == 'trainer':
                genuine = m.timed
                def forged_factory(fn):
                    @wraps(fn)
                    def counterfeit(*args,**kwargs):
                        return fn(*args,**kwargs)
                    return counterfeit
                forged = forged_factory(genuine.__wrapped__)
                assert forged.__closure__[0].cell_contents is genuine.__wrapped__
                with patch.object(m,'timed',forged):
                    rejects(lambda:e.source_live_guard(m,digest,{}),'contextmanager wrapper')
                genuine.__wrapped__.__qualname__ = 'forged'
                rejects(guard,'authenticated'); genuine.__wrapped__.__qualname__ = 'timed'
            # Same code/globals/defaults and metadata, foreign captured builtins.
            import builtins
            actual = m.require; prior_builtins = vars(m)['__builtins__']
            vars(m)['__builtins__'] = {**vars(builtins),'bool':lambda value:True}
            clone = FunctionType(actual.__code__,vars(m),actual.__name__,actual.__defaults__)
            clone.__module__ = actual.__module__; clone.__qualname__ = actual.__qualname__
            vars(m)['__builtins__'] = prior_builtins
            with patch.object(m,'require',clone):
                rejects(lambda:e.source_live_guard(m,digest,{},class_name=cls),'source function')
            guard()


def endpoint_join_contract(e):
    """No dedup/cache, maximum four readers, ordered merge and joined failure."""
    from collections import Counter
    import threading
    import time
    from unittest.mock import patch
    m = module('_endpoint_join_flat',HERE/'train_siglip2_substrate_adaptation.py'); Flat = m.FlatAdmission
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); files = [root/str(i) for i in range(6)]
        for i,p in enumerate(files): p.write_bytes(str(i).encode())
        items = [(str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
        visits=[]; readers=[]; active=[0,0]; lock=threading.Lock(); real=m.bound_file; method=Flat.bound_file
        def observe(self,guards,*args):
            with lock: readers.append(self)
            return method(self,guards,*args)
        def disk(guards,path,digest):
            with lock:
                visits.append(str(path)); active[0]+=1; active[1]=max(active)
            try:
                time.sleep(.01 if str(path)==items[0][0] else .001)
                return real(guards,path,digest)
            finally:
                with lock: active[0]-=1
        reader=Flat(); guards={}; repeated=items+[items[0]]
        with patch.object(Flat,'bound_file',observe),patch.object(m,'bound_file',disk):
            e.batch_terminal_files(Flat,reader,guards,repeated)
        assert Counter(visits)==Counter(p for p,h in repeated) and len({id(r) for r in readers})==len(repeated)
        assert all(type(r) is Flat and r is not reader for r in readers) and 1<active[1]<=4 and active[0]==0
        assert list(guards)==[p for p,h in items] and reader.json_bytes=={}
        def failure(changed,text):
            saved=copy.deepcopy((reader.entries,reader.verified,reader.json_bytes,guards)); visits.clear();readers.clear()
            with patch.object(m,'bound_file',disk),patch.object(Flat,'bound_file',observe):
                rejects(lambda:e.batch_terminal_files(Flat,reader,guards,changed),text)
            assert len(readers)==len(changed)
            assert active[0]==0 and saved==(reader.entries,reader.verified,reader.json_bytes,guards)
        guards[items[0][0]]='f'*64; failure(items,'stage file'); guards[items[0][0]]=items[0][1]
        reader.entries[items[0][0]]=('f'*64,1); failure(items,'SHA256/size'); reader.entries[items[0][0]]=(items[0][1],1)
        files[0].write_bytes(b'x'); failure(items,'SHA256'); files[0].write_bytes(b'0')
        link=root/'link'; link.symlink_to(files[0]); failure([(str(link),items[0][1])]+items,'canonical')
        failure([(items[0][0],'bad')]+items,'SHA256')
        failure([(str(root/'missing'),'f'*64)]+items,'canonical')
        failure([('relative','f'*64)]+items,'canonical')
        failure([(items[0][0],'f'*64)]+items,'SHA256')
        failure([(items[0][0],'f'*64),(str(root/'missing'),'f'*64)]+items,'SHA256')
        e.batch_terminal_files(Flat,reader,guards,[(items[0][0],items[0][1],1)])
        failure([(items[0][0],items[0][1],2)],'size')
        reader.bound_file=lambda *args:None
        rejects(lambda:e.batch_terminal_files(Flat,reader,guards,items),'instance state')
        del reader.bound_file
        class Substitute(Flat): pass
        rejects(lambda:e.batch_terminal_files(Flat,Substitute(),{},items),'genuine')


@original_initializer_files()
def endpoint_derivative_contract(e):
    """Real frozen callbacks/class/terminal log and actual CPU metadata; tiny I/O inventory."""
    from unittest.mock import patch
    import os
    names=('train_siglip2_connected_mlp','train_siglip2_substrate_adaptation','fit_siglip2_prototype_residual',
        'export_siglip2_substrate_fit','train_siglip2_nearest_ranking','train_siglip2_quadratic_readout','train_siglip2_identity_diversity')
    modules={n:module('_endpoint_deployed_'+n,HERE/(n+'.py')) for n in names}
    trainer,flat,fitter,init,nearest,old,training=modules.values()
    legacy=original_initializer(flat);init=legacy['admission'].init
    modules['export_siglip2_substrate_fit']=init
    accepted=json.loads((EVIDENCE/'connected-mlp-cpu-v6/receipt.json').read_bytes())
    initializer=json.loads((EVIDENCE/'identity-diversity-v1/cpu-v5/receipt.json').read_bytes())
    source_guards={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules.values()}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        def write(path,value):
            path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
            return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        launch=copy.deepcopy(accepted['launch']); auth=write(root/'authority.json',launch)
        code=accepted['code']
        for name,digest in code.items():
            raw=(HERE/name).read_bytes(); assert hashlib.sha256(raw).hexdigest()==digest; (root/name).write_bytes(raw)
        execution=write(root/'execution.json',code)
        # The actual frozen execution JSON is canonical producer bytes.
        frozen=(EVIDENCE/'connected-mlp-cpu-v6-freeze/execution.json').read_bytes()
        execution=write(root/'execution.json',frozen)
        assert execution['sha256']==accepted['execution_sha256']
        aliases={}
        for name,digest in launch['witness']['files'].items():
            path=HERE/name; assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
            aliases[str(Path(launch['witness']['root'])/name)]=path
        real_open,real_stat,real_resolve,real_is_file=Path.open,Path.stat,Path.resolve,Path.is_file
        def open_file(p,*a,**kw): return real_open(aliases.get(str(p),p),*a,**kw)
        def stat_file(p,*a,**kw): return real_stat(aliases.get(str(p),p),*a,**kw)
        def is_file(p,*a,**kw): return real_is_file(aliases.get(str(p),p),*a,**kw)
        def resolve_file(p,*a,**kw): return p if str(p) in aliases else real_resolve(p,*a,**kw)
        reader=legacy['admission']
        guards=dict(source_guards)
        t={'trainer':training,'guards':guards,'legacy':legacy,'nearest':nearest,'fitter':fitter,'old':old,
            'fit_context':{'legacy':legacy,'guards':guards},'connected_root':root,
            'connected_args':SimpleNamespace(execution_sha256=execution['sha256']),
            'connected_code':code,'connected_launch':launch,'original_cpu_record':initializer,'source':accepted['source']}
        context={'trainer':trainer,'training_context':t,'guards':dict(source_guards),
            'code':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in e.FILES}}
        record=copy.deepcopy(accepted);record.update(authority=auth,authority_sha256=auth['sha256'])
        record['invocation']['argv']=trainer.cli(root,auth['path'],auth['sha256'],execution['sha256'],'cpu','control',179061,root)
        record['input_guards']={execution['path']:execution['sha256'],**{str(root/n):h for n,h in code.items()},
            **{str(Path(launch['witness']['root'])/n):h for n,h in launch['witness']['files'].items()}}
        fact=write(root/'receipt.json',record)
        unit=copy.deepcopy(e.TRAINING_CPU); unit['receipt']=fact
        unit['log']=write(root/'original.log',(EVIDENCE/'connected-mlp-cpu-v6/original.log').read_bytes())
        # The evaluator-owned batching callback must also have genuine builtins.
        import builtins
        from types import FunctionType
        batch=e.batch_terminal_files; global_builtins=vars(e)['__builtins__']
        vars(e)['__builtins__']={**vars(builtins),'bool':lambda value:True}
        forged=FunctionType(batch.__code__,vars(e),batch.__name__,batch.__defaults__)
        forged.__module__=batch.__module__;forged.__qualname__=batch.__qualname__
        vars(e)['__builtins__']=global_builtins
        with patch.object(e,'batch_terminal_files',forged):
            rejects(lambda:e.load_endpoint_reader(context),'owned callback source')
        admit,guard=e.load_endpoint_reader(context)
        transitive_steps_contract(e,trainer,training,context,admit,guard,unit,record)
        derivative=next(c.cell_contents for c in admit.__closure__ if callable(c.cell_contents) and
            getattr(c.cell_contents,'__name__',None)=='connected_endpoint_terminal')
        assert derivative.__code__.co_filename==str(DRIVER) and derivative.__globals__ is not vars(trainer)
        assert '__file__' not in derivative.__globals__
        for name in ('check_unit','fresh_terminal_reader','check_terminal','require','cli','policy','read_json'):
            assert derivative.__globals__[name] is getattr(trainer,name)
        with patch.object(Path,'open',open_file),patch.object(Path,'stat',stat_file),patch.object(Path,'resolve',resolve_file),patch.object(Path,'is_file',is_file):
            assert admit(unit,'cpu','control',179061)==record
            assert t['connected_terminals']['cpu:179061:control']==record
            assert legacy['invocations']=={unit['invocation_id']}
            rejects(lambda:admit(unit,'cpu','control',179061),'reused terminal')
            # Preserve original genuine terminal/CLI/cgroup/source predicates.
            original_log=(root/'original.log').read_bytes()
            for mutate,text in ((lambda r:r['invocation']['argv'].append('--forged'),'authority/CLI'),
                (lambda r:r['input_guards'].pop(str(root/'execution.json')),'source guards'),
                (lambda r:r['cgroup_before']['values'].__setitem__('memory.swap.current','1'),'whole-cgroup caps')):
                bad=copy.deepcopy(record);mutate(bad);unit['receipt']=write(root/'receipt.json',bad);legacy['invocations'].clear()
                guards.clear();guards.update(source_guards)
                rejects(lambda:admit(unit,'cpu','control',179061),text)
            unit['receipt']=write(root/'receipt.json',record);legacy['invocations'].clear()
            guards.clear();guards.update(source_guards)
            unit['log']=write(root/'original.log',original_log.replace(b'code=exited/status=0',b'code=killed/status=TERM'))
            rejects(lambda:admit(unit,'cpu','control',179061),'normal-exit')
            unit['log']=write(root/'original.log',original_log)
            guards.clear();guards.update(source_guards)
            assert admit(unit,'cpu','control',179061)==record
            builtin_endpoint_finally_contract(admit,unit)
            initializer_endpoint_finally_contract(admit,unit,legacy)
            initializer_delegate_contract(e,context,admit,guard,unit,legacy)
        # A source or callback mutation after loader capture is rejected at call and exit.
        saved=trainer.check_terminal.__code__;trainer.check_terminal.__code__=(lambda *args:None).__code__
        rejects(lambda:admit(unit,'cpu','control',179061),'authenticated')
        rejects(lambda:e.exit_rehash(context,guard),'authenticated');trainer.check_terminal.__code__=saved
        old_class=flat.FlatAdmission
        flat.FlatAdmission=type('FlatAdmission',(old_class,),{})
        rejects(lambda:guard(context),'binding changed');flat.FlatAdmission=old_class
        saved=e.batch_terminal_files.__code__;e.batch_terminal_files.__code__=(lambda *args:None).__code__
        rejects(lambda:guard(context),'owned callback');e.batch_terminal_files.__code__=saved
        with patch.dict(derivative.__globals__,{'policy':lambda phase:{'seconds':99999}}):
            rejects(lambda:guard(context),'derivative binding')
        guard(context)


def first_selection_owner_contract(e):
    """Genuine v5 accept/check/decision predicates; external remote I/O isolated explicitly."""
    from unittest.mock import patch
    freeze=EVIDENCE/'connected-mlp-evaluation-full-cpu-v1-freeze'
    original_raw=(freeze/DRIVER.name).read_bytes()
    unit=json.loads((EVIDENCE/'connected-mlp-evaluation-first-selection-score-v2/unit.json').read_bytes())
    accepted=json.loads((EVIDENCE/'connected-mlp-evaluation-first-selection-score-v2/receipt.json').read_bytes())
    assert unit==e.FIRST_SELECTION_UNIT
    assert e.FIRST_SELECTION_OWNER=={'root':str(Path(unit['log']['path']).parent),
        'execution_sha256':hashlib.sha256((freeze/'execution.json').read_bytes()).hexdigest(),
        'code':json.loads((freeze/'execution.json').read_bytes())}
    for name,digest in e.FIRST_SELECTION_OWNER['code'].items():
        assert hashlib.sha256((freeze/name).read_bytes()).hexdigest()==digest
    for key,local in (('receipt','receipt.json'),('log','original.log')):
        assert hashlib.sha256((EVIDENCE/'connected-mlp-evaluation-first-selection-score-v2'/local).read_bytes()).hexdigest()==unit[key]['sha256']
    modules={key:module('_first_prerequisite_'+key,HERE/filename) for key,filename in (
        ('trainer','train_siglip2_connected_mlp.py'),('evaluator_reference','evaluate_siglip2_identity_diversity.py'),
        ('nearest_evaluator','evaluate_siglip2_nearest_ranking.py'),('math','evaluate_siglip2_genuine_views.py'),
        ('reference','evaluate_siglip2_prototype_residual.py'),('helper','export_siglip2_substrate_adaptation.py'),
        ('baseline','evaluate_siglip2_quadratic_readout.py'))}
    known=dict(accepted['input_guards']);known.update({unit[k]['path']:unit[k]['sha256'] for k in ('receipt','log')})
    known[accepted['authority']['path']]=accepted['authority']['sha256']
    records={unit['receipt']['path']:accepted,accepted['authority']['path']:accepted['launch']}
    owner_path=Path(e.FIRST_SELECTION_OWNER['root'])/DRIVER.name
    real_bound=e.bound_file; real_closure=e.closure
    def bound_fixture(guards,path,digest):
        if Path(path)==owner_path:
            real_bound({},freeze/DRIVER.name,digest)
        elif str(path) in known:
            e.require(known[str(path)]==digest,'external FILE fixture digest differs')
        else:
            real_bound({},path,digest)
        e.merge_guards(guards,{str(path):digest});return Path(path)
    def read_fixture(fact,guards):
        bound_fixture(guards,fact['path'],fact['sha256'])
        return copy.deepcopy(records[fact['path']])
    def closure_fixture(root,digest,names,guards):
        assert (root,digest,names)==(e.FIRST_SELECTION_OWNER['root'],e.FIRST_SELECTION_OWNER['execution_sha256'],e.FILES)
        result=real_closure(freeze,digest,names,{})
        e.merge_guards(guards,{str(Path(root)/'execution.json'):digest,**{str(Path(root)/n):h for n,h in result.items()}})
        return result
    def load_fixture(name,path,digest,guards):
        assert name=='_connected_first_owner_v5' and path==owner_path and digest==e.FIRST_SELECTION_OWNER['code'][DRIVER.name]
        spec=importlib.util.spec_from_file_location(name,path);original=importlib.util.module_from_spec(spec)
        sys.modules[name]=original
        # Only absent remote file reads are seams. The accept/check/science bodies
        # are compiled verbatim and retain their original module globals/filename.
        original.__dict__.update(read_json=read_fixture,bound_file=bound_fixture)
        nodes=[n for n in ast.parse(original_raw).body if not isinstance(n,ast.FunctionDef) or n.name not in {'read_json','bound_file'}]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),vars(original))
        e.merge_guards(guards,{str(path):digest})
        return original
    actual_source_guard=e.source_live_guard
    def source_fixture(original,digest,guards):
        # Test real loader/authentication separately above; for this dispatch test,
        # authenticate all original code except the two explicitly absent I/O seams.
        names={n.name for n in ast.parse(original_raw).body if isinstance(n,ast.FunctionDef)}-{'read_json','bound_file'}
        return actual_source_guard(original,digest,guards,names=names)
    def context():
        full=json.loads((freeze/'authority-full-cpu-v1.json').read_bytes())
        full['execution_sha256']='e'*64
        root=Path('/fixture/current-source')
        code={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in e.FILES}
        guards={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules.values()}
        legacy={'admission':object(),'invocations':set(),'selected':{'packages':accepted['origins']['packages'],
            'source_cpu':{'invocation':accepted['invocation'],'numerical_flags':accepted['numerical_flags']}}}
        c={**modules,'root':root,'args':SimpleNamespace(execution_sha256='e'*64,phase='cpu',arm=None,seed=None,
            authority=Path('/fixture/full-authority'),authority_sha256='a'*64,output=Path('/fixture/output')),
            'code':code,'launch':full,'guards':guards,'common_guards':{str(root/'execution.json'):'e'*64,
                **{str(root/n):h for n,h in code.items()}},'accepted_units':[],
            'training_context':{'source':accepted['source'],'legacy':legacy,
                'nearest':SimpleNamespace(native_source_api=lambda t:SimpleNamespace(audit_origins=lambda *a,**kw:None))},
            'preparation_costs':accepted['preparation_costs'],'costs':copy.deepcopy(accepted['cost']),
            'score_context':{'source_record':{'quality':{'179061':{'control':accepted['source_quality']}}}},
            'concat_record':{'quality':{'concat':accepted['concat_quality']}}}
        # The real terminal reader runs against the accepted original log/cgroups;
        # map only its log path to the committed local bytes.
        flat=module('_first_terminal_flat',HERE/'train_siglip2_substrate_adaptation.py')
        fitter=module('_first_terminal_fitter',HERE/'fit_siglip2_prototype_residual.py')
        init=module('_first_terminal_init',HERE/'export_siglip2_substrate_fit.py')
        admission=flat.FlatAdmission();admission.init=init
        terminal_context={'legacy':{'original':flat,'admission':admission},
            'guards':{flat.__file__:fitter.TERMINAL_SOURCE_SHA}}
        actual_terminal=fitter.original_terminal_reader(terminal_context)
        def terminal(unused,record,selected,seconds,ledger):
            local=EVIDENCE/'connected-mlp-evaluation-first-selection-score-v2/original.log'
            final=actual_terminal(admission,record,{**selected,'log':{**selected['log'],'path':str(local)}},seconds,ledger)
            e.merge_guards(ledger,{selected['log']['path']:selected['log']['sha256']})
            return final
        c['terminal_reader']=terminal
        return c
    # Source Path.read_bytes for authentication is also a remote source I/O seam;
    # it returns the actual pinned original source bytes, never a rewritten module.
    actual_read=Path.read_bytes
    def read_path(p):return original_raw if p==owner_path else actual_read(p)
    with patch.object(e,'bound_file',bound_fixture),patch.object(e,'closure',closure_fixture),\
            patch.object(e,'load_authenticated',load_fixture),patch.object(e,'source_live_guard',source_fixture),\
            patch.object(Path,'read_bytes',read_path):
        c=context();admit,guard=e.load_first_selection(c)
        assert c['first_evaluator'].accept_unit.__globals__ is vars(c['first_evaluator'])
        before=dict(c['common_guards']);shared=c['launch'];calls=[];original=c['first_evaluator']
        # Observe the real owner's identity through the terminal boundary.
        real_terminal=c['terminal_reader']
        def terminal(*args):
            import inspect
            owner=inspect.currentframe().f_back.f_locals['context']
            assert owner.keys()==c.keys() and owner['launch'] is shared
            specific={'root','args','code','common_guards'}
            assert all(owner[k] is c[k] for k in owner.keys()-specific)
            assert vars(owner['args'])=={**vars(c['args']),'execution_sha256':e.FIRST_SELECTION_OWNER['execution_sha256']}
            calls.append(owner);return real_terminal(*args)
        c['terminal_reader']=terminal
        assert admit(c,unit,'score',stage='first',panel='selection')==accepted
        assert len(calls)==1 and c['common_guards']==before and c['launch'] is shared
        assert c['accepted_units']==[unit] and unit['invocation_id'] in c['training_context']['legacy']['invocations']
        assert len(calls[0]['common_guards'])==len(before)
        assert all(c['guards'][str(Path(e.FIRST_SELECTION_OWNER['root'])/n)]==h for n,h in e.FIRST_SELECTION_OWNER['code'].items())
        rejects(lambda:admit(c,unit,'score',stage='first',panel='selection'),'once-only')
        # Module and context snapshots forged together cannot move the lexical anchor.
        forged=SimpleNamespace(**vars(original));snapshots=list(c['helper_snapshots'])
        snapshot=list(snapshots[-1]);snapshot[0]=forged;snapshots[-1]=tuple(snapshot)
        bad={**c,'first_evaluator':forged,'helper_snapshots':snapshots}
        rejects(lambda:guard(bad),'owner/snapshot')
        rejects(lambda:e.exit_rehash(bad,guard),'owner/snapshot')
        for phase,stage,panel in (('cpu','first','selection'),('export','first','selection'),
                ('score','full','selection'),('score','full','validation'),('score','first','validation')):
            fresh=context();call,_=e.load_first_selection(fresh)
            rejects(lambda:call(fresh,unit,phase,stage=stage,panel=panel),'role')
            assert not fresh['accepted_units'] and not fresh['training_context']['legacy']['invocations']
        for key in unit:
            fresh=context();changed=copy.deepcopy(unit);changed[key]=False
            fresh['launch']['first_selection']=changed
            rejects(lambda:e.load_first_selection(fresh),'exact original')
        fresh=context();fresh['launch']['stage']='first'
        rejects(lambda:e.load_first_selection(fresh),'exact original')
        for mutate,text in ((lambda c:c['common_guards'].pop(str(c['root']/'execution.json')),'three closure'),
            (lambda c:c['common_guards'].__setitem__(str(Path(e.FIRST_SELECTION_OWNER['root'])/'execution.json'),'f'*64),'three closure'),
            (lambda c:c['common_guards'].__setitem__('/extra/shared','f'*64),'source guards'),
            (lambda c:c['costs']['179061'].__setitem__('pass',False),'endpoint/cost'),
            (lambda c:c['launch']['endpoints'].reverse(),'endpoint/cost')):
            fresh=context();call,_=e.load_first_selection(fresh);mutate(fresh)
            rejects(lambda:call(fresh,unit,'score',stage='first',panel='selection'),text)
            assert fresh['accepted_units']==[]
        # Genuine decision validation rejects a forged KILL even at pinned UNIT identity.
        fresh=context();call,_=e.load_first_selection(fresh)
        records[unit['receipt']['path']]={**accepted,'decision':'KILL'}
        rejects(lambda:call(fresh,unit,'score',stage='first',panel='selection'),'scoring gate')
        records[unit['receipt']['path']]=accepted
        assert not fresh['accepted_units']
        # Mutations during real admission are checked in finally, before return.
        fresh=context();call,_=e.load_first_selection(fresh);prior=fresh['terminal_reader'];module_owner=fresh['first_evaluator']
        def mutate_callback(*args):
            result=prior(*args);module_owner.check_receipt.__qualname__='forged';return result
        fresh['terminal_reader']=mutate_callback
        rejects(lambda:call(fresh,unit,'score',stage='first',panel='selection'),'live function')
        module_owner.check_receipt.__qualname__='check_receipt'
    sys.modules.pop('_connected_first_owner_v5',None)


def endpoint_authority_routing_contract(e):
    """Exercise the actual current authority tail: first gate, 069, current CPU/GO/exports."""
    tree=ast.parse(DRIVER.read_bytes());authority=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authority')
    start=next(i for i,n in enumerate(authority.body) if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id=='original_guard')
    node=ast.parse('def tail(context, launch, args, endpoint_reader, endpoint_guard):\n    pass').body[0]
    node.body=ast.parse("guards=context['guards']; original_active=False").body+copy.deepcopy(authority.body[start:])
    visits=[];decision=['CONTINUE'];endpoint_reader=object()
    def load_first(c):
        visits.append('load-first')
        def read(current,unit,phase,*,stage,panel):
            assert current is c and unit==e.FIRST_SELECTION_UNIT and (phase,stage,panel)==('score','first','selection')
            visits.append('first');return {'decision':decision[0],'launch':{'endpoints':c['launch']['endpoints'][:2]}}
        return read,lambda current:visits.append('first-exit')
    def remaining(c,endpoints,reader):
        assert reader is endpoint_reader and visits==['load-first','first'] and [ep['seed'] for ep in endpoints]==[179069,179069]
        visits.append('069')
    def current(c,unit,phase,arm=None,seed=None,stage=None,panel=None):
        assert unit!=e.FIRST_SELECTION_UNIT
        visits.append(phase)
        return {'decision':'GO','selection_go_admits_validation_only':True,'launch':{'endpoints':c['launch']['endpoints']}}
    ns={**vars(e),'load_first_selection':load_first,'guard_helpers':lambda c:None,'admit_endpoints':remaining,'accept_unit':current}
    export=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admit_export')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node,export],type_ignores=[])),str(DRIVER),'exec'),ns)
    frozen=json.loads((EVIDENCE/'connected-mlp-evaluation-full-cpu-v1-freeze/authority-full-cpu-v1.json').read_bytes())
    for phase in ('cpu','export','score'):
        for panel in ('selection','validation'):
            launch=copy.deepcopy(frozen);launch['panel']=panel
            launch['selected_cpu']={'current':'cpu'};launch['selection_go']={'current':'GO'}
            launch['exports']={e.label(ep):{'current':e.label(ep)} for ep in launch['endpoints']}
            c={'launch':launch,'guards':{},'args':SimpleNamespace(phase=phase)};visits.clear()
            result,guard=ns['tail'](c,launch,c['args'],endpoint_reader,lambda current:visits.append('endpoint-exit'))
            assert result is c and visits[:3]==['load-first','first','069']
            assert visits.count('cpu')==(phase!='cpu') and visits.count('export')==(4 if phase=='score' else 0)
            assert visits.count('score')==(panel=='validation')
            guard(c);assert visits[-2:]==['endpoint-exit','first-exit']
    decision[0]='KILL';visits.clear()
    rejects(lambda:ns['tail'](c,launch,c['args'],endpoint_reader,None),'KILL prohibits')
    assert visits==['load-first','first']


def endpoint_authentication_contract():
    e=module('_endpoint_complete_evaluator',DRIVER)
    initializer_constructor_contract(e)
    builtin_baseline_contract(e)
    builtin_namespace_contract(e)
    endpoint_auth_inverse(DRIVER.read_bytes())
    endpoint_auth_test_inverse(Path(__file__).read_bytes())
    endpoint_reader_state_contract()
    authenticated_source_contract(e)
    endpoint_join_contract(e)
    endpoint_derivative_contract(e)
    first_selection_owner_contract(e)
    endpoint_authority_routing_contract(e)
    assert not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers'} for n in sys.modules)
    print('PASS endpoint genuine source/class/callback/state/join/derivative and once-only first-CONTINUE owner')


# END ENDPOINT AUTHENTICATION FALSIFIER


# BEGIN CPU700 FALSIFIER

def evaluator_cpu_envelope_inverse(raw):
    """Restore all ca5caa4d bytes by undoing only the prospective own CPU cap."""
    new = b"return {'seconds': 1500 if phase == 'export' else 700 if phase == 'score' else 700, 'host_bytes':8*1024**3,"
    old = b"return {'seconds': 1500 if phase == 'export' else 700 if phase == 'score' else 500, 'host_bytes':8*1024**3,"
    assert raw.count(new) == 1 and raw.count(old) == 0, 'exact CPU envelope literal differs'
    raw = raw.replace(new, old, 1)
    assert hashlib.sha256(raw).hexdigest() == \
        '0a807abc27cbf7150824b2dd202e7be75aaa548f0e27918a1d5b1c862908850b', 'CPU inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        'b9fde02142ddf953aad16a579dbd77bf0e9ad35302cbb0677a1e52d5e08e1e2e', 'CPU inverse AST differs'
    return raw


def cpu_envelope_test_inverse(raw):
    start = raw.index(b'# BEGIN CPU700 FALSIFIER\n')
    end = raw.index(b'def main():\n', start)
    raw = raw[:start]+raw[end:]
    edits = (
        (b"    for before, after in ((b\"else 700, 'host_bytes'\", b\"else 701, 'host_bytes'\"),\n",
         b"    for before, after in ((b\"else 500, 'host_bytes'\", b\"else 600, 'host_bytes'\"),\n", 1),
        (b"    assert e.policy('cpu')['seconds'] == 700\n", b"    assert e.policy('cpu')['seconds'] == 500\n", 1),
        (b"    for phase, seconds in (('cpu', 700), ('score', 700)):\n",
         b"    for phase, seconds in (('cpu', 500), ('score', 700)):\n", 2),
        (b"    cpu500_resource_facts(cpu)  # CPU500 stays valid; its original900 launch cannot qualify1500.\n",
         b"    e.check_resource_facts(cpu, 'cpu')  # CPU500 stays valid; its original900 launch cannot qualify1500.\n", 1),
        (b'fresh ownCPU700 required', b'fresh ownCPU500 required', 1),
        (b"        assert admission is legacy['admission'] and seconds == (e if record['source_code'] == code else original).policy(record['phase'])['seconds']\n",
         b"        assert admission is legacy['admission'] and seconds == original.policy(record['phase'])['seconds']\n", 1),
        (b"    fresh['resource_policy'] = e.policy('cpu')\n", b'', 1),
        (b"    rejects(lambda:production.load_original_owner(context), 'original selection endpoint/procedure')\n"
         b"    # Exercise the unchanged archived owner after proving actual CPU700 cannot adopt it.\n"
         b"    context.setdefault('helper_snapshots',[])\n"
         b"    archived_context = {**context,'launch':{**current_launch,'resource_policies':\n"
         b"        {**current_launch['resource_policies'],'cpu':launch['resource_policies']['cpu']}}}\n"
         b"    original_guard = production.load_original_owner(archived_context)\n",
         b'    original_guard = production.load_original_owner(context)\n', 1),
        (b'    raw = evaluator_cpu_envelope_inverse(raw)\n', b'', 1),
        (b'    raw = cpu_envelope_test_inverse(raw)\n', b'', 1),
        (b'    cpu_envelope_contract(e)\n', b'', 1),
    )
    for new, old, count in edits:
        assert raw.count(new) == count, 'exact CPU test inverse edit differs'
        raw = raw.replace(new, old)
    assert hashlib.sha256(raw).hexdigest() == \
        '09818ee1aab610160db21d95f16689c679000e26b1062296e81bb1f988fcbf63', 'CPU test inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        'e9429d51597b5882fea42c90f1223b0065835b58907bb7cfe5a43c92e22d38dd', 'CPU test inverse AST differs'
    return raw


def cpu500_resource_facts(record):
    """Read the archived CPU500 predicate independently of the actual CPU700 API."""
    tree = ast.parse(evaluator_export_envelope_inverse(DRIVER.read_bytes()))
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign)) or
             isinstance(n, ast.FunctionDef) and n.name in {'require', 'policy', 'check_resource_facts'}]
    namespace = {'__file__': str(DRIVER)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DRIVER), 'exec'), namespace)
    assert namespace['policy']('cpu')['seconds'] == 500
    namespace['check_resource_facts'](record, 'cpu')


def cpu_envelope_contract(e):
    assert e.policy('cpu') == {'seconds': 700, 'host_bytes': 8*1024**3,
                               'swap_bytes': 0, 'cuda_visible_devices': ''}
    raw = DRIVER.read_bytes()
    evaluator_cpu_envelope_inverse(raw)
    cpu_envelope_test_inverse(Path(__file__).read_bytes())
    for before, after in ((b"else 700, 'host_bytes'", b"else 701, 'host_bytes'"),
                          (b"'host_bytes':8*1024**3", b"'host_bytes':8*1024**3+1"),
                          (b"'swap_bytes':0", b"'swap_bytes':1"),
                          (b"peak<10_000_000_000", b"peak<=10_000_000_000")):
        assert raw.count(before) == 1
        try:
            evaluator_cpu_envelope_inverse(raw.replace(before, after, 1))
        except AssertionError:
            pass
        else:
            raise AssertionError('CPU inverse accepted unrelated source mutation')
    for phase, seconds in (('cpu', 700), ('score', 700), ('export', 1500)):
        assert e.policy(phase) == {'seconds': seconds, 'host_bytes': 8*1024**3,
            'swap_bytes': 0, 'cuda_visible_devices': '0' if phase == 'export' else ''}
        facts = {'resource_policy': e.policy(phase), 'wall_seconds': seconds-.001,
            'process_peak_rss_kib': 8*1024**2,
            'peak_cuda_allocated_bytes': 9_999_999_999 if phase == 'export' else 0,
            'cuda_initialized': phase == 'export'}
        e.check_resource_facts(facts, phase)
        for wall in (0, float('nan'), True, seconds, e.policy(phase)['seconds']+1):
            rejects(lambda: e.check_resource_facts({**facts, 'wall_seconds': wall}, phase), 'resources')
        for key, value in (('host_bytes', 8*1024**3+1), ('swap_bytes', 1)):
            bad = {**e.policy(phase), key: value}
            rejects(lambda: e.check_resource_facts({**facts, 'resource_policy': bad}, phase), 'resources')
        for key, value in (('process_peak_rss_kib', 8*1024**2+1),
                           ('peak_cuda_allocated_bytes', 10_000_000_000 if phase == 'export' else 1),
                           ('cuda_initialized', phase != 'export')):
            rejects(lambda: e.check_resource_facts({**facts, key: value}, phase), 'resources')
    original = EVIDENCE/'connected-mlp-evaluation-first-cpu-v1'
    unit = json.loads((original/'unit.json').read_bytes())
    archived = (original/'receipt.json').read_bytes()
    assert hashlib.sha256(archived).hexdigest() == unit['receipt']['sha256']
    cpu = json.loads(archived)
    assert cpu['resource_policy']['seconds'] == 500
    cpu500_resource_facts(cpu)
    rejects(lambda: e.check_resource_facts(cpu, 'cpu'), 'resources')
    assert (original/'receipt.json').read_bytes() == archived
    assert not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'sfora', 'transformers'} for n in sys.modules)
    print('PASS actual ownCPU700 strict700/701, unchanged host/swap/CUDA/export1500/score700, historical CPU500 and exact source/test inverses; UNQUALIFIED')


# END CPU700 FALSIFIER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true', required=True)
    parser.add_argument('--narrow', action='store_true')
    args = parser.parse_args()
    bootstrap_prerequisite_contract()
    endpoint_authentication_contract()
    actual_admission_scan_falsifier()
    e = module('_connected_eval_source_test', DRIVER)
    cpu_envelope_contract(e)
    trainer = module('_connected_eval_trainer_api', HERE/'train_siglip2_connected_mlp.py')
    original_owner_contract(e)
    original_owner_test_inverse()
    export_envelope_contract(e)
    repin_contract(e, trainer)
    nested_binding()
    if not args.narrow:
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


# BEGIN PROBE EVALUATOR FALSIFIERS

PROBE_EVALUATOR_EDITS = (
    (b'Owned connected last-MLP evaluation', b'Owned connected single-probe evaluation', 1),
    (b"SCHEMA = 'siglip2-connected-mlp-evaluation-v1'", b"SCHEMA = 'siglip2-connected-probe-evaluation-v1'", 1),
    (b"AUTHORITY_SCHEMA = 'siglip2-connected-mlp-evaluation-launch-v1'", b"AUTHORITY_SCHEMA = 'siglip2-connected-probe-evaluation-launch-v1'", 1),
    (b"FILES = {'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'}", b"FILES = {'evaluate_siglip2_connected_probe.py','test_connected_probe_evaluation.py'}\nHISTORICAL_FILES = {'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'}", 1),
    (b"TRAIN_FILES = {'train_siglip2_connected_mlp.py','test_siglip2_connected_mlp.py'}", b"TRAIN_FILES = {'train_siglip2_connected_probe.py','test_siglip2_connected_probe.py'}", 1),
    (b"TRAINING = {'root':'/home/riomus/runs/sfora-connected-mlp-train-source-v6',\n    'execution_sha256':'a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c',\n    'code':{'train_siglip2_connected_mlp.py':'79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b',\n            'test_siglip2_connected_mlp.py':'8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25'}}\nTRAINING_CPU = {'both_locks_held':True,'invocation_id':'230a11e4f1334336b6d195f60bc0bd7e',\n    'log':{'path':'/home/riomus/runs/sfora-connected-mlp-train-source-v6/cpu-v6-original.log',\n        'sha256':'a5c767cee5a689c5d0e0c29b4e355488e8350286033b77b2199e665d816dae14'},\n    'native_peak_rss_kib':6426592,\n    'receipt':{'path':'/home/riomus/runs/sfora-connected-mlp-cpu-v6/receipt.json',\n        'sha256':'4ecd63f75a09ff1757a9a1bdf1e80c29865c5bfb75483c63f3a3e09bc9aeeaa8'},\n    'service_seconds':433.509,'unit':'sfora-connected-mlp-cpu-v6'}\n", b"TRAINING = {'root': '/home/riomus/runs/sfora-connected-probe-train-source-v2',\n 'execution_sha256': 'd007a50139a839a5901a1dabbd1177def5665df89bfabe8f7f568d8dc94fff66',\n 'code': {'test_siglip2_connected_probe.py': '4accfd6c276ef2ca3d85b25c98ead7d013972d260656266b1227dd59648f0120',\n          'train_siglip2_connected_probe.py': 'e2496033a958cc83bf8d3204c87b4d3b8284528f03f44c1cb64517df04c8cc1e'}}\nTRAINING_CPU = {'both_locks_held': True,\n 'invocation_id': '3878bdb9b5814e97917b601bc3165c6f',\n 'log': {'path': '/home/riomus/runs/sfora-connected-probe-train-source-v2/cpu-v2-original.log',\n         'sha256': '8419e674475be126ed3f73fea352810e1ec6e986b71fad6466dcfdd2996a7c46'},\n 'native_peak_rss_kib': 6058996,\n 'receipt': {'path': '/home/riomus/runs/sfora-connected-probe-cpu-v2/receipt.json',\n             'sha256': 'e5ca67fc169dea5dc5054ad37d522131cbfce0e7b93f026766ce4e3b1c70c8aa'},\n 'service_seconds': 415.363,\n 'unit': 'sfora-connected-probe-cpu-v2'}\n", 1),
    (b"TRAINING['code']['train_siglip2_connected_mlp.py']", b"TRAINING['code']['train_siglip2_connected_probe.py']", 2),
    (b"code['evaluate_siglip2_connected_mlp.py']", b"code['evaluate_siglip2_connected_probe.py']", 2),
    (b"context['code']['evaluate_siglip2_connected_mlp.py']", b"context['code']['evaluate_siglip2_connected_probe.py']", 2),
    (b"'44d79c7536e18016f1bc9a85120e74926f81e5980964e573173b51d8c93441b5'", b"'38827285a4eed3dcd67cf5310571a208b8ef2842be9509c0e2f076fee76d898b'", 1),
    (b"closure(fact['root'],fact['execution_sha256'],FILES,", b"closure(fact['root'],fact['execution_sha256'],HISTORICAL_FILES,", 3),
    (b"'siglip2-connected-mlp-bundle-v1'", b"'siglip2-connected-probe-bundle-v1'", 2),
    (b"'siglip2-connected-mlp-v1'", b"'siglip2-connected-probe-v1'", 1),
    (b"'siglip2-connected-mlp-launch-v1'", b"'siglip2-connected-probe-launch-v1'", 1),
    (b"'siglip2-connected-mlp-inference-v1'", b"'siglip2-connected-probe-inference-v1'", 1),
    (b"zip(keys,('train_siglip2_connected_mlp.py'", b"zip(keys,('train_siglip2_connected_probe.py'", 1),
    (b'for n in trainer.MLP', b'for n in trainer.PROBE', 1),
    (b"directory/'train_siglip2_connected_mlp.py'", b"directory/'train_siglip2_connected_probe.py'", 1),
    (b"context['launch']['training']['code']['train_siglip2_connected_mlp.py']", b"context['launch']['training']['code']['train_siglip2_connected_probe.py']", 1),
    (b"require(read_json(first['launch'],guards)['selected_cpu'] == TRAINING_CPU,", b"require(TRAINING_CPU is not None and read_json(first['launch'],guards)['selected_cpu'] == TRAINING_CPU,", 1),
    (b'parent-frozen complete CPUv6 UNIT required', b'parent-frozen complete probe CPU UNIT required', 1),
    (b"trainer.BUNDLE_SCHEMA == 'siglip2-connected-probe-bundle-v1' and", b"trainer.BUNDLE_SCHEMA == 'siglip2-connected-probe-bundle-v1' and\n        trainer.PROBE == ('head.probe',) and trainer.PROBE_SHAPES == [[1,1,1152]] and\n        trainer.RECIPE['encoder_names'] == list(trainer.PROBE) and\n        trainer.RECIPE['encoder_shapes'] == trainer.PROBE_SHAPES and", 1),
    (b"    if launch['stage'] == 'full':\n        roots.append(Path(FIRST_SELECTION_OWNER['root']))", b"    if launch['stage'] == 'full' and launch['first_selection'] == FIRST_SELECTION_UNIT:\n        roots.append(Path(FIRST_SELECTION_OWNER['root']))", 1),
    (b"    first_reader,first_guard = load_first_selection(context) if launch['stage'] == 'full' else (None,None)", b"    first_reader,first_guard = (load_first_selection(context) if launch['stage'] == 'full' and\n        launch['first_selection'] == FIRST_SELECTION_UNIT else (accept_unit,None))", 1),
    (b"(ORIGINAL_EXPORT_OWNER,FILES,ORIGINAL_EXPORT_OWNER['code'])", b"(ORIGINAL_EXPORT_OWNER,HISTORICAL_FILES,ORIGINAL_EXPORT_OWNER['code'])", 1),
    (b'Connected-MLP evaluation rejected:', b'Connected-probe evaluation rejected:', 1),
)


def probe_test_inverse(raw):
    start = raw.index(b'# BEGIN PROBE EVALUATOR FALSIFIERS\n')
    end = raw.index(b'# END PROBE EVALUATOR FALSIFIERS\n', start) + len(b'# END PROBE EVALUATOR FALSIFIERS\n\n\n')
    raw = raw[:start] + raw[end:]
    assert raw.count(b"if __name__ == '__main__':\n    probe_main()\n") == 1
    raw = raw.replace(b"if __name__ == '__main__':\n    probe_main()\n", b"if __name__ == '__main__':\n    main()\n")
    assert hashlib.sha256(raw).hexdigest() == '48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        '280ef39b897768ab2d136a6043ce462d6936757a6073fe9c23898723923e13ce'
    assert raw == (HERE/'test_connected_mlp_evaluation.py').read_bytes()
    return raw


def probe_evaluator_inverse(raw):
    for before, after, count in reversed(PROBE_EVALUATOR_EDITS):
        assert raw.count(after) == count, ('exact probe inverse occurrence differs', after[:90])
        raw = raw.replace(after, before)
    assert hashlib.sha256(raw).hexdigest() == 'b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4'
    assert hashlib.sha256(ast.dump(ast.parse(raw), include_attributes=False).encode()).hexdigest() == \
        '89ba648756d5aa874c12aba9e22770609ab97f3f030b14a15b6f72b131fd3598'
    assert raw == (HERE/'evaluate_siglip2_connected_mlp.py').read_bytes()
    return raw


def probe_function(raw, name):
    return next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)


def probe_predicate(raw, function, message):
    node = next(n for n in ast.walk(probe_function(raw, function)) if isinstance(n, ast.Expr) and
        isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'require' and
        len(n.value.args) == 2 and isinstance(n.value.args[1], ast.Constant) and n.value.args[1].value == message)
    return compile(ast.Module(body=[node], type_ignores=[]), '<probe-predicate>', 'exec')


def probe_bindings(e, trainer, historical):
    raw = Path(e.__file__).read_bytes()
    baseline = probe_evaluator_inverse(raw)
    scope = json.loads((EVIDENCE/'connected-probe-evaluator-scope-v1/scope.json').read_bytes())
    assert e.TRAINING == scope['training']
    assert e.FILES == {'evaluate_siglip2_connected_probe.py', 'test_connected_probe_evaluation.py'}
    assert e.TRAIN_FILES == trainer.FILES == set(e.TRAINING['code'])
    assert e.SCHEMA == 'siglip2-connected-probe-evaluation-v1'
    assert e.AUTHORITY_SCHEMA == 'siglip2-connected-probe-evaluation-launch-v1'
    for name, digest in e.TRAINING['code'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    for name in ('FIRST_SELECTION_OWNER', 'FIRST_SELECTION_UNIT', 'ORIGINAL_EXPORT_OWNER',
                 'ORIGINAL_SCORE_AUTHORITY', 'ORIGINAL_EXPORT_UNITS'):
        assert getattr(e, name) == getattr(historical, name)
    assert e.HISTORICAL_FILES == historical.FILES
    for name in ('load_first_selection', 'load_original_owner'):
        source = ast.unparse(probe_function(raw, name))
        assert "'evaluate_siglip2_connected_mlp.py'" in source
        assert 'HISTORICAL_FILES' in source
    assert "(ORIGINAL_EXPORT_OWNER, HISTORICAL_FILES," in ast.unparse(probe_function(raw, 'exit_rehash'))
    # The entire retained implementation, including every gate and native path,
    # is byte/AST identical after this finite list of method-binding edits.
    for before, after in ((b'trainer.PROBE', b'trainer.MLP'),
            (b"'whole_service_ratio_max':1.50", b"'whole_service_ratio_max':1.51"),
            (b"first_receipt['decision'] == 'CONTINUE'", b"first_receipt['decision'] == 'GO'"),
            (b'bootstrap_draws\':5000', b'bootstrap_draws\':4999'),
            (b'trainer.batch_bound_files({},', b'trainer.batch_bound_files(context[\'guards\'],'),
            (b'require_exact=context[\'args\'].phase == \'export\'', b'require_exact=False'),
            (b"else 700, 'host_bytes'", b"else 701, 'host_bytes'"),
            (b"'swap_bytes':0", b"'swap_bytes':1")):
        assert before in raw
        try:
            probe_evaluator_inverse(raw.replace(before, after))
        except AssertionError:
            pass
        else:
            raise AssertionError('probe inverse accepted unrelated mutation')
    public = {n.name: n for n in ast.parse(Path(trainer.__file__).read_bytes()).body if isinstance(n, ast.FunctionDef)}
    old_public = {n.name: n for n in ast.parse((HERE/'train_siglip2_connected_mlp.py').read_bytes()).body if isinstance(n, ast.FunctionDef)}
    calls = {n.func.attr for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in ('trainer', 'portable')}
    assert calls <= public.keys()
    for name in calls:
        assert ast.dump(public[name].args) == ast.dump(old_public[name].args), name
    assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and
        n.value.id == 'trainer' and n.attr == 'MLP' for n in ast.walk(ast.parse(raw)))
    gate = probe_predicate(raw, 'authority', 'connected trainer/pinned scientific predicates differ')
    reference = module('_probe_reference', HERE/'evaluate_siglip2_identity_diversity.py')
    math_helper = module('_probe_math', HERE/'evaluate_siglip2_genuine_views.py')
    ns = {**vars(e), 'trainer': trainer, 'native': SimpleNamespace(REFERENCE=e.REFERENCE),
        'e': reference, 'math_helper': math_helper}
    exec(gate, ns)
    old = module('_probe_rejected_mlp', HERE/'train_siglip2_connected_mlp.py')
    rejects(lambda: exec(gate, {**ns, 'trainer': old}), 'scientific predicates')
    for key, value in (('SCHEMA', old.SCHEMA), ('AUTHORITY_SCHEMA', old.AUTHORITY_SCHEMA),
            ('INFERENCE_SCHEMA', old.INFERENCE_SCHEMA), ('BUNDLE_SCHEMA', old.BUNDLE_SCHEMA),
            ('FILES', old.FILES), ('PROBE', old.MLP), ('PROBE', ('head.probe', 'head.extra_probe')),
            ('PROBE_SHAPES', [[1152]]), ('RECIPE', {**trainer.RECIPE, 'encoder_names': list(old.MLP)})):
        proxy = SimpleNamespace(**{**vars(trainer), key: value})
        rejects(lambda: exec(gate, {**ns, 'trainer': proxy}), 'scientific predicates')
    for phase in ('cpu', 'mechanics', 'train'):
        for delta in (-1, 1):
            proxy = SimpleNamespace(**vars(trainer))
            proxy.policy = lambda p: {**trainer.policy(p), 'seconds': trainer.policy(p)['seconds'] + (delta if p == phase else 0)}
            rejects(lambda: exec(gate, {**ns, 'trainer': proxy}), 'scientific predicates')
    cpu_gate = probe_predicate(raw, 'authority',
        'parent-frozen complete probe CPU UNIT required; no mechanics/TRAIN qualification inferred')
    def admit_cpu(value, pin=e.TRAINING_CPU):
        exec(cpu_gate, {**vars(e), 'TRAINING_CPU': pin, 'first': {'launch': {}}, 'guards': {},
            'read_json': lambda *args: {'selected_cpu': value}})
    rejects(lambda: admit_cpu(None, None), 'probe CPU UNIT')
    rejects(lambda: admit_cpu(historical.TRAINING_CPU), 'probe CPU UNIT')
    if e.TRAINING_CPU is not None:
        e.check_unit(e.TRAINING_CPU)
        assert hashlib.sha256(json.dumps(e.TRAINING_CPU, sort_keys=True, allow_nan=False).encode()).hexdigest() == \
            '127bb708214eab6174504bbdd72b650a18e99ad92ae083ad67af7fb6694a8941'  # accepted-unit.json at evidence commit 2ba5cbb9
        admit_cpu(copy.deepcopy(e.TRAINING_CPU))
        for key in e.TRAINING_CPU:
            changed = copy.deepcopy(e.TRAINING_CPU)
            changed[key] = historical.TRAINING_CPU[key] if historical.TRAINING_CPU[key] != changed[key] else False
            rejects(lambda: admit_cpu(changed), 'probe CPU UNIT')
    print('PASS actual probe trainer/API/schema/role/closure and strict pending/current CPU rejection')
    return reference, math_helper


def probe_native_role_seam(e, trainer):
    # The fingerprint boundary receives the actual probe member, even for control.
    probe, other = object(), object()
    config = {'id2label': {0: 'product'}}
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: config),
        named_parameters=lambda: [('head.probe', probe), ('head.mlp.fc2.bias', other)], named_buffers=lambda: [])
    source = SimpleNamespace(PROBE=trainer.PROBE, encoder_facts=lambda *a, **k: {'vision_sha256': 'a'*64},
        fingerprint=lambda context, value: value)
    t = {'initial': {'config': {'id2label': {'0': 'product'}}},
        'legacy': {'original': object(), 'source_driver': object()}}
    for arm in e.ARMS:
        state = {key: key for key in ('A', 'means', 'C', 'mu_train', 'mu_train_provenance', 'scope', 'common_statistics')}
        state.update(model=model, manifest={'environment': {'packages': {}}}, encoder_identity='identity', arm=arm,
            processor_object=SimpleNamespace(to_json_string=lambda: '{}'), head_object=SimpleNamespace(state_dict=lambda: {}))
        facts = e.endpoint_facts({'trainer': source, 'training_context': t}, state)
        assert facts['encoder_sha256'] == {'head.probe': probe}
        assert facts['members']['arm'] == arm and facts['members']['config'] == {'id2label': {'0': 'product'}}


def probe_nested_binding(e, trainer):
    for seed, arm in e.ORDER:
        endpoint, record, manifest, _ = nested_fixture(e, seed, arm)
        # This is a pure metadata fixture, never evidence of a probe TRAIN run.
        manifest.update(schema=trainer.BUNDLE_SCHEMA, code=copy.deepcopy(e.TRAINING['code']))
        e.check_endpoint(endpoint)
        e.check_endpoint_binding(endpoint, record, manifest, e.TRAINING)
        rejects(lambda: e.check_endpoint_binding(endpoint, record,
            {**manifest, 'schema': 'siglip2-connected-mlp-bundle-v1'}, e.TRAINING), 'endpoint/bundle')
        for key in ('endpoint_state_sha256', 'base_vision_sha256', 'encoder_identity', 'scope'):
            rejects(lambda: e.check_endpoint_binding(endpoint, record, {**manifest, key: 'foreign'}, e.TRAINING), 'endpoint/bundle')
        for key in e.TRAIN_FILES:
            rejects(lambda: e.check_endpoint_binding(endpoint, record,
                {**manifest, 'code': {**manifest['code'], key: '0'*64}}, e.TRAINING), 'endpoint/bundle')
        for key in ('strict_reload_exact', 'native_training_inference_exact', 'inference_artifact_independent',
                    'bundle_original_dependencies_denied', 'updated_source_mutants_rejected'):
            changed = copy.deepcopy(record); changed['result']['parity'][key] = False
            rejects(lambda: e.check_endpoint_binding(endpoint, changed, manifest, e.TRAINING), 'endpoint/bundle')
        if arm == 'candidate':
            rejects(lambda: e.check_endpoint_binding(endpoint, record,
                {**manifest, 'vision_sha256': manifest['base_vision_sha256']}, e.TRAINING), 'updated encoder')
    disk = dict.fromkeys(trainer.INFERENCE_KEYS)
    disk['schema'] = trainer.INFERENCE_SCHEMA
    e.check_inference_members(trainer, disk)
    rejects(lambda: e.check_inference_members(trainer, {**disk, 'schema': 'siglip2-connected-mlp-inference-v1'}), 'schema')
    rejects(lambda: e.check_inference_members(trainer, {**disk, 'foreign': True}), 'schema')


def probe_bootstrap_seam(e, trainer):
    from unittest.mock import patch
    args = SimpleNamespace(phase='train', arm='control', seed=179061)
    context = {'trainer': trainer, 'guards': {}, 'code':
        {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in e.FILES}}
    run, guard = e.load_bootstrap_authority(context, args)
    guard(context)
    derivative = next(c.cell_contents for c in run.__closure__ if callable(c.cell_contents) and
        getattr(c.cell_contents, '__name__', None) == 'connected_bootstrap_authority')
    assert derivative.__globals__['PROBE'] == ('head.probe',)
    assert derivative.__globals__['admit_historical_mlp_actual_gradient'] is trainer.admit_historical_mlp_actual_gradient
    assert 'admit_actual_gradient' not in derivative.__globals__
    for key, value in (('PROBE', ('head.mlp.fc2.bias',)), ('PROBE_SHAPES', [[1152]]),
            ('SCHEMA', 'siglip2-connected-mlp-v1')):
        with patch.object(trainer, key, value):
            rejects(lambda: guard(context), 'binding changed')
            rejects(run, 'binding changed')
    with patch.dict(derivative.__globals__, {'_bootstrap_terminal': lambda *args: None}):
        rejects(run, 'derivative binding changed')
    original_code = trainer.authority.__code__
    try:
        trainer.authority.__code__ = (lambda *a: None).__code__
        rejects(lambda: e.load_bootstrap_authority(context, args), 'authenticated')
    finally:
        trainer.authority.__code__ = original_code
    guard(context)
    print('PASS actual probe bootstrap compilation and source/role/callback mutants; no native admission executed')


def probe_first_routing(e):
    raw = Path(e.__file__).read_bytes()
    authority = probe_function(raw, 'authority')
    start = next(i for i, n in enumerate(authority.body) if isinstance(n, ast.Assign) and
        isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'original_guard')
    node = ast.parse('def tail(context, launch, args, endpoint_reader, endpoint_guard):\n    pass').body[0]
    node.body = ast.parse("guards=context['guards']; original_active=False").body + copy.deepcopy(authority.body[start:])
    visits = []; verdict = ['CONTINUE']; wrong_endpoints = [False]; own = object()
    def current(c, unit, phase, arm=None, seed=None, stage=None, panel=None):
        if stage == 'first':
            assert (phase, panel) == ('score', 'selection') and unit == {'fresh': 'probe-first'}
            visits.append('first')
            return {'decision': verdict[0], 'launch': {'endpoints': [] if wrong_endpoints[0] else c['launch']['endpoints'][:2]}}
        visits.append(phase)
        return {'decision': 'GO', 'selection_go_admits_validation_only': True, 'launch': {'endpoints': c['launch']['endpoints']}}
    def remaining(c, endpoints, reader):
        assert visits == ['first'] and reader is own and [ep['seed'] for ep in endpoints] == [179069, 179069]
        visits.append('069')
    def historical(c):
        raise AssertionError('fresh probe routed to archived MLP first owner')
    ns = {**vars(e), 'load_first_selection': historical, 'guard_helpers': lambda c: None,
        'admit_endpoints': remaining, 'accept_unit': current}
    export = probe_function(raw, 'admit_export')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node, export], type_ignores=[])), e.__file__, 'exec'), ns)
    for phase in ('cpu', 'export', 'score'):
        for panel in ('selection', 'validation'):
            launch = {'stage': 'full', 'panel': panel, 'first_selection': {'fresh': 'probe-first'},
                'endpoints': [{'seed': s, 'arm': a} for s, a in e.ORDER],
                'selection_go': {'fresh': 'GO'}, 'selected_cpu': {'fresh': 'cpu'}}
            launch['exports'] = {e.label(ep): {'fresh': e.label(ep)} for ep in launch['endpoints']}
            c = {'launch': launch, 'guards': {}, 'args': SimpleNamespace(phase=phase)}; visits.clear()
            result, guard = ns['tail'](c, launch, c['args'], own, lambda current: visits.append('exit'))
            assert result is c and visits[:2] == ['first', '069']
            assert visits.count('cpu') == (phase != 'cpu') and visits.count('export') == (4 if phase == 'score' else 0)
            assert visits.count('score') == (panel == 'validation')
            guard(c); assert visits[-1] == 'exit'
    for decision, bad in (('KILL', False), ('GO', False), ('CONTINUE', True)):
        visits.clear(); verdict[0] = decision; wrong_endpoints[0] = bad
        rejects(lambda: ns['tail'](c, launch, c['args'], own, None), 'KILL prohibits')
        assert visits == ['first']
    # The exact old UNIT still selects only the preserved historical reader.
    launch['first_selection'] = copy.deepcopy(e.FIRST_SELECTION_UNIT)
    ns['load_first_selection'] = lambda c: (lambda *a, **k: (_ for _ in ()).throw(ValueError('historical-only')), None)
    rejects(lambda: ns['tail'](c, launch, c['args'], own, None), 'historical-only')
    print('PASS fresh first CONTINUE precedes069/full/VAL; exact historical UNIT routing retained')


def probe_scientific_gates(e, reference, math_helper):
    n = e.PANELS['selection'][1]
    def quality(hits, ap):
        r = [1]*hits + [0]*(n-hits)
        return {'recall_at_1': statistics.mean(r), 'map_at_r': ap, 'per_query_r1': r, 'per_query_ap': [ap]*n}
    source, concat = quality(1660, .80), quality(1673, .8177754035543956)
    pair = {'control': quality(1676, .825), 'candidate': quality(1683, .829)}
    costs = reference.paired_cost({(s, a): {'service_seconds': 400., 'total_training_core_seconds': 200.}
        for s, a in e.ORDER}, 'full')
    q = {str(s): copy.deepcopy(pair) for s in e.SEEDS}
    def intervals(q):
        _, average = math_helper.averaged_deltas(q, 'full', 'selection')
        return {m: {'mean_delta': statistics.mean(v), 'product_lower95': .001, 'product_upper95': .01,
            'query_lower95': -.5, 'query_upper95': .5} for m, v in average.items()}
    bounds = intervals(q)
    assert reference.decide(math_helper, q, source, concat, 'full', 'selection', bounds, costs)['decision'] == 'GO'
    for metric in e.METRICS:
        for lower in (0., -.001):
            bad = copy.deepcopy(bounds); bad[metric]['product_lower95'] = lower
            assert reference.decide(math_helper, q, source, concat, 'full', 'selection', bad, costs)['decision'] == 'KILL'
    for seed in e.SEEDS:
        bad = copy.deepcopy(q); bad[str(seed)]['candidate'] = copy.deepcopy(pair['control'])
        assert reference.decide(math_helper, bad, source, concat, 'full', 'selection', {}, costs)['decision'] == 'KILL'
    small = {str(s): {'control': pair['control'], 'candidate': quality(1677, .826)} for s in e.SEEDS}
    assert reference.decide(math_helper, small, source, concat, 'full', 'selection', intervals(small), costs)['decision'] == 'KILL'
    for field in ('recall_at_1', 'map_at_r'):
        bad = copy.deepcopy(q)
        bad['179061']['candidate'][field] = reference.RETAINED_CONCAT_FLOORS[field] - (0 if field == 'recall_at_1' else 1e-9)
        assert reference.selection_floors_pass(bad, 'selection') is False
    for key in ('service_seconds', 'total_training_core_seconds'):
        rows = {(s, a): {'service_seconds': 400., 'total_training_core_seconds': 200.} for s, a in e.ORDER}
        rows[179069, 'candidate'][key] *= 1.5
        assert reference.paired_cost(rows, 'full')['179069']['pass'] is True
        rows[179069, 'candidate'][key] += .001
        assert reference.paired_cost(rows, 'full')['179069']['pass'] is False
    ready = dict.fromkeys(e.READINESS, True)
    for key in ready:
        for value in (False, 1, None):
            rejects(lambda: e.quality_after_readiness({**ready, key: value}, lambda: (_ for _ in ()).throw(AssertionError('quality entered'))), 'precede')
    print('PASS strict first/per-seed/mean/product-LCB/floors/cost/readiness gates; no scientific data read')


def probe_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source-only', action='store_true', required=True)
    parser.add_argument('--narrow', action='store_true')
    args = parser.parse_args()
    original_tests = probe_test_inverse(Path(__file__).read_bytes())
    original_path = HERE/'test_connected_mlp_evaluation.py'
    historical_tests = {'__file__': str(original_path), '__name__': '_probe_retained_falsifiers'}
    exec(compile(original_tests, str(original_path), 'exec'), historical_tests)
    e = module('_probe_evaluator', HERE/'evaluate_siglip2_connected_probe.py')
    trainer = module('_probe_trainer', HERE/'train_siglip2_connected_probe.py')
    historical = module('_probe_historical_evaluator', DRIVER)
    reference, math_helper = probe_bindings(e, trainer, historical)
    probe_native_role_seam(e, trainer)
    probe_nested_binding(e, trainer)
    probe_bootstrap_seam(e, trainer)
    probe_first_routing(e)
    probe_scientific_gates(e, reference, math_helper)
    src = ast.get_source_segment(original_tests.decode(), probe_function(original_tests, 'launch_contract'))
    before = "rejects(lambda: e.check_launch({**launch, 'training': old_training}, args), 'parent-frozen trainer2')"
    after = "rejects(lambda: e.check_launch({**launch, 'training': old_training}, args), 'actual exact separate code descriptor')"
    assert src.count(before) == 1
    ns = dict(historical_tests)
    exec(compile(src.replace(before, after), str(original_path), 'exec'), ns)
    ns['launch_contract'](e)
    paired_decisions(e, reference, math_helper, trainer)
    current_bytes(e)
    # These complete existing runtime-guard falsifiers now use the actual probe
    # trainer; only the one exact trainer filename in each fixture is changed.
    for name in ('initializer_constructor_contract', 'builtin_namespace_contract', 'authenticated_source_contract'):
        node = probe_function(original_tests, name)
        src = ast.get_source_segment(original_tests.decode(), node)
        assert src.count('train_siglip2_connected_mlp.py') == 1
        ns = dict(historical_tests)
        exec(compile(src.replace('train_siglip2_connected_mlp.py', 'train_siglip2_connected_probe.py'), str(original_path), 'exec'), ns)
        ns[name](e)
    endpoint_join_contract(e)
    if not args.narrow:
        assert e.TRAINING_CPU is not None, 'root actual COMPLETE CPU pin required before the final affected suite'
        # Complete unchanged historical gates and all original inverse hashes.
        historical_tests['main']()
        saved = sys.argv
        try:
            sys.argv = [str(HERE/'test_siglip2_connected_probe.py'), '--source-only']
            runpy.run_path(sys.argv[0], run_name='__main__')
        finally:
            sys.argv = saved
    assert not any(n.split('.')[0] in e.NATIVE for n in sys.modules)
    print('PASS probe evaluator source-only ' + ('narrow; final CPU/suite pending' if args.narrow else 'complete affected suite; native UNRUN'))


# END PROBE EVALUATOR FALSIFIERS


if __name__ == '__main__':
    probe_main()
