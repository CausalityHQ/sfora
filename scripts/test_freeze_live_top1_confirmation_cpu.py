#!/usr/bin/env python3
"""Stdlib source-only gate with synthetic metadata or parent-supplied inputs.

Run without arguments for synthetic receipt/command fixtures; use --inputs
CANONICAL_JSON for the parent's actual evidence once it exists. All generated
artifacts and adversarial copies stay temporary. No checkpoint loads or launches.
"""

import argparse
import copy
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import importlib

freeze = None


def fixture(root):
    """Synthetic receipt metadata only; never an original endpoint or admission.

    Every fixture FILE hash is computed from bytes written here. No prospective
    native artifact hash is invented or reused as a real frozen descriptor.
    """
    copies = {}
    sequence = 0

    def sha(tag):
        return freeze.digest(('synthetic-fixture:' + tag).encode())

    def store(remote, payload):
        nonlocal sequence
        sequence += 1
        raw = payload if isinstance(payload, bytes) else (json.dumps(payload, sort_keys=True) + '\n').encode()
        p = root / (str(sequence) + '.metadata')
        p.write_bytes(raw)
        fact = {'path': remote, 'sha256': freeze.digest(raw)}
        copies[remote] = str(p)
        return fact

    script_root = Path(__file__).resolve().parent
    execution = dict(freeze.EVALUATOR_PINS)
    for name, expected in execution.items():
        raw = (script_root / name).read_bytes()
        assert freeze.digest(raw) == expected
        store(freeze.ROOT + '/' + name, raw)
    execution_fact = store(freeze.ROOT + '/execution.json', execution)
    evaluator = freeze.load_module(freeze.ROOT + '/evaluate_siglip2_compact_ranking.py',
                                  (script_root / 'evaluate_siglip2_compact_ranking.py').read_bytes(), '_fixture_eval')
    trainer = freeze.load_module(script_root / 'train_siglip2_compact_ranking.py',
                                (script_root / 'train_siglip2_compact_ranking.py').read_bytes(), '_fixture_train')
    math = freeze.load_module(script_root / 'evaluate_siglip2_genuine_views.py',
                             (script_root / 'evaluate_siglip2_genuine_views.py').read_bytes(), '_fixture_math')
    cvalues = {'memory.max': str(8 * 1024**3), 'memory.peak': str(2 * 1024**2),
               'memory.swap.current': '0', 'memory.swap.max': '0', 'memory.swap.peak': '0',
               'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}

    def terminal(name, output, log_path, record):
        identity = sha(name)[:32]
        group = {'path': '/sys/fs/cgroup/system.slice/' + name + '.service', 'values': cvalues}
        record.update(cgroup_before=group, cgroup_after=group, wall_seconds=90.0,
                      process_peak_rss_kib=512)
        record['invocation']['invocation_id'] = identity
        final = {**group, 'invocation_id': identity, 'command_exit_status': 0}
        stop = {**group, 'invocation_id': identity, 'exit_code': 'exited', 'exit_status': '0', 'service_result': 'success'}
        log = '\n'.join([f'Running as unit: {name}.service; invocation ID: {identity}',
                         '\tExit status: 0', 'Finished with result: success',
                         'Main processes terminated with: code=exited/status=0', '\tSwaps: 0',
                         'Memory swap peak: 0B', '\tMaximum resident set size (kbytes): 512',
                         'Service runtime: 100s', 'FINAL_CGROUP ' + json.dumps(final),
                         'STOP_CGROUP ' + json.dumps(stop)]) + '\n'
        receipt = store(output + '/receipt.json', record)
        logfile = store(log_path, log.encode())
        unit = {'receipt': receipt, 'log': logfile, 'unit': name, 'invocation_id': identity,
                'service_seconds': 100.0, 'native_peak_rss_kib': 512, 'both_locks_held': True}
        vf = store('/tmp/synthetic-verification-' + name + '.json', {
            'phase': record['phase'], 'arm': record['arm'], 'unit': name, 'quality_read': record['quality_read'],
            'rss_kib': 512, 'host_peak_bytes': 2 * 1024**2, 'pass': True, 'invocation_id': identity,
            'service_seconds': 100.0, 'receipt_sha256': receipt['sha256'], 'log_sha256': logfile['sha256']})
        local = copies.pop(vf['path'])
        vf['path'] = local
        return unit, vf

    def prerequisite(name):
        return {'receipt': {'path': '/tmp/synthetic-' + name + '/receipt.json', 'sha256': sha(name + ':receipt')},
                'log': {'path': '/tmp/synthetic-' + name + '/log', 'sha256': sha(name + ':log')},
                'unit': name, 'invocation_id': sha(name)[:32], 'service_seconds': 100.0,
                'native_peak_rss_kib': 512, 'both_locks_held': True}

    cpu = prerequisite('synthetic-train-cpu')
    mechanics = {arm: prerequisite('synthetic-mechanics-' + arm) for arm in evaluator.ARMS}
    endpoints, verifications, records = [], [], {}
    for seed, arm in evaluator.ORDER:
        key = arm + '-' + str(seed)
        output = '/tmp/synthetic-live-top1-' + key
        launch = {'schema': trainer.AUTHORITY_SCHEMA, 'execution_sha256': evaluator.TRAINING['execution_sha256'],
                  'nearest': trainer.NEAREST, 'fitter': trainer.FITTER, 'accepted': trainer.ACCEPTED,
                  'readout': trainer.READOUT, 'recipe': trainer.RECIPE, 'native_authority': {
                      'path': '/tmp/synthetic-native-authority.json', 'sha256': sha('native-authority')},
                  'resource_policy': trainer.policy('train'), 'both_locks_held': True,
                  'phase': 'train', 'seed': seed, 'arm': arm, 'selected_cpu': cpu, 'selected_mechanics': mechanics}
        launch_fact = store(evaluator.TRAINING['root'] + '/synthetic-authority-' + key + '.json', launch)
        # Opaque synthetic artifact descriptors; no checkpoint/bundle is created or read.
        checkpoint = {'path': output + '/resume.pt', 'sha256': sha(key + ':checkpoint')}
        bundle = {'path': output + '/bundle/bundle.json', 'sha256': sha(key + ':bundle')}
        identity = {'seed': seed, 'arm': arm, 'source': {'synthetic_fixture': True},
                    'method': trainer.method(launch), 'numerical_flags': {'synthetic_fixture': True},
                    'parameter_names': ['A', 'C'], 'parameter_shapes': [[128, 160], [128, 1152]]}
        for k in ('static_sha256', 'initial_A_sha256', 'initial_C_sha256', 'mu_train_sha256',
                  'mu_train_provenance_sha256', 'initial_cpu_rng_sha256', 'initial_cuda_rng_sha256'):
            identity[k] = sha(str(seed) + ':' + k)
        record = {'schema': trainer.SCHEMA, 'phase': 'train', 'seed': seed, 'arm': arm, 'launch': launch,
                  'authority': launch_fact, 'authority_sha256': launch_fact['sha256'],
                  'code': evaluator.TRAINING['code'], 'execution_sha256': evaluator.TRAINING['execution_sha256'],
                  'completed_step': 128, 'checkpoint': checkpoint, 'bundle': bundle,
                  'terminal_state_sha256': sha(key + ':terminal'), 'inference_state_sha256': sha(key + ':inference'),
                  'output': output, 'resource_policy': trainer.policy('train'), 'quality_read': False,
                  'training_state_discarded': False, 'resumed_steps': [], 'C_exact_zero': False,
                  'C_trainable': True, 'residual_nonzero_witness': True, 'current_C_sha256': sha(key + ':currentC'),
                  'identity': identity, 'source': identity['source'], 'numerical_flags': identity['numerical_flags'],
                  'total_training_core_seconds': 50.0, 'initial_raw_unit_packed_sha256': sha(str(seed) + ':initial'),
                  'steps': [{'step': n, 'arm': arm, 'batch': [n]} for n in range(1, 129)],
                  'invocation': {'argv': trainer.cli(evaluator.TRAINING['root'], launch_fact['path'],
                                                   launch_fact['sha256'], evaluator.TRAINING['execution_sha256'],
                                                   'train', arm, seed, output),
                                 'optimize': 0, 'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8'}}
        for k in ('initial_A_sha256', 'initial_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256'):
            record[k] = identity[k]
        for k in ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'sequential_model_ownership',
                  'forward_oracle_exact', 'native_training_inference_exact', 'inference_artifact_independent',
                  'bundle_original_dependencies_denied', 'both_locks_held_in_parent_authority',
                  'exact_four_native_membership', 'source_substitution_rejected',
                  'omitted_C_mutant_rejected', 'wrong_mu_mutant_rejected'):
            record[k] = True
        record['input_guards'] = {f['path']: f['sha256'] for f in (launch_fact, checkpoint, bundle)}
        record['input_guards'].update({evaluator.TRAINING['root'] + '/' + n: h for n, h in
                                     {**evaluator.TRAINING['code'], 'execution.json': evaluator.TRAINING['execution_sha256']}.items()})
        unit, vf = terminal('synthetic-live-top1-' + key, output, evaluator.TRAINING['root'] + '/' + key + '.log', record)
        endpoints.append({'seed': seed, 'arm': arm, 'launch': launch_fact, 'terminal': unit,
                          'checkpoint': checkpoint, 'bundle': bundle, 'terminal_state_sha256': record['terminal_state_sha256'],
                          'inference_state_sha256': record['inference_state_sha256']})
        verifications.append(vf)
        records[seed, arm] = {**record, 'service_seconds': 100.0}
    first_cpu = {'schema': evaluator.AUTHORITY_SCHEMA, 'execution_sha256': execution_fact['sha256'],
                 'training': evaluator.TRAINING, 'nearest_evaluator': evaluator.NEAREST_EVALUATOR,
                 'genuine_evaluator': {'root': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4',
                     'execution_sha256': evaluator.GENUINE_EXECUTION_SHA, 'code': evaluator.GENUINE_PINS},
                 'reference': evaluator.REFERENCE, 'phase': 'cpu', 'arm': None, 'seed': None, 'stage': 'first',
                 'panel': 'selection', 'endpoints': endpoints[:2], 'selected_cpu': None, 'exports': {},
                 'first_selection': None, 'selection_go': None, 'resource_policies': {
                     p: evaluator.policy(p) for p in ('cpu', 'export', 'score')},
                 'cost_policy': evaluator.COST_POLICY, 'both_locks_held': True, 'selection_previously_exposed': True}
    authority = store(freeze.ROOT + '/authority-first-cpu-v1.json', first_cpu)
    score_launch = {**first_cpu, 'phase': 'score', 'selected_cpu': prerequisite('synthetic-eval-first-cpu'),
                    'exports': {evaluator.label(e): prerequisite('synthetic-export-' + evaluator.label(e)) for e in endpoints[:2]}}
    score_authority = store(freeze.ROOT + '/synthetic-authority-first-score.json', score_launch)
    def quality(zeros, ap):
        return {'recall_at_1': (1734 - zeros) / 1734, 'map_at_r': ap,
                'per_query_r1': [0] * zeros + [1] * (1734 - zeros), 'per_query_ap': [ap] * 1734}
    scores = {'179061': {'control': quality(60, .82), 'candidate': quality(50, .83)}}
    source_quality, concat_quality = quality(100, .8), quality(61, .8177754035543956)
    costs = evaluator.paired_cost({k: v for k, v in records.items() if k[0] == 179061}, 'first')
    first = {**evaluator.decide(math, scores, source_quality, concat_quality, 'first', 'selection', {}, costs),
             'schema': evaluator.SCHEMA, 'phase': 'score', 'arm': None, 'seed': None, 'stage': 'first', 'panel': 'selection',
             'execution_sha256': execution_fact['sha256'], 'source_code': execution, 'source': {'synthetic_fixture': True},
             'cost_policy': evaluator.COST_POLICY, 'resource_policy': evaluator.policy('score'),
             'peak_cuda_allocated_bytes': 0, 'cuda_initialized': False, 'authority': score_authority,
             'authority_sha256': score_authority['sha256'], 'launch': score_launch,
             'numerical_flags': {'synthetic_fixture': True}, 'binding': evaluator.binding({'launch': score_launch}),
             'output': '/home/riomus/runs/sfora-so400-live-top1-evaluation-first-selection-score-v1',
             'invocation': {'argv': evaluator.cli(SimpleNamespace(execution_sha256=execution_fact['sha256'],
                 authority=Path(score_authority['path']), authority_sha256=score_authority['sha256'],
                 phase='score', seed=None, arm=None, output=Path('/home/riomus/runs/sfora-so400-live-top1-evaluation-first-selection-score-v1'))),
                 'optimize': 0, 'cuda_visible_devices': '', 'cublas_workspace_config': ':4096:8'},
             'cost': costs, 'quality': scores, 'source_quality': source_quality, 'concat_quality': concat_quality,
             'paired_seed_average_intervals': {}, 'readiness': dict.fromkeys(evaluator.READINESS, True),
             'quality_read': True, 'files': {}, 'bootstrap_seed': 179019, 'bootstrap_draws': 0}
    for k in ('pass', 'engineering_admission_pass', 'integrity_pass', 'resources_pass', 'exit_rehash_pass',
              'sequential_model_ownership', 'rng_flags_preserved', 'both_locks_held_in_parent_authority',
              'source_archived_perquery_exact', 'concat_archived_perquery_exact',
              'all_export_wires_readback_before_quality', 'persisted_wire_scoring_replay_exact'):
        first[k] = True
    for k in ('official_read', 'global_production_goal_met', 'public_latency_measured', 'product_go'):
        first[k] = False
    first['input_guards'] = {freeze.ROOT + '/' + n: h for n, h in {**execution, 'execution.json': execution_fact['sha256']}.items()}
    for e in endpoints[:2]:
        for f in (e['launch'], e['terminal']['receipt'], e['terminal']['log'], e['checkpoint'], e['bundle']):
            first['input_guards'][f['path']] = f['sha256']
    first_unit, first_vf = terminal('sfora-so400-live-top1-evaluation-first-selection-score-v1', first['output'],
                                   freeze.ROOT + '/first-selection-score-v1.log', first)
    verification_path = Path(first_vf['path'])
    verification = json.loads(verification_path.read_bytes())
    for key in ('phase', 'arm', 'unit', 'quality_read', 'rss_kib'):
        del verification[key]
    verification.update(decision='CONTINUE', cost=costs, deltas=first['mean_deltas'],
                        quality={arm: {k: scores['179061'][arm][k] for k in ('recall_at_1', 'map_at_r')}
                                 for arm in evaluator.ARMS})
    raw = (json.dumps(verification) + '\n').encode()
    verification_path.write_bytes(raw)
    first_vf['sha256'] = freeze.digest(raw)
    guards = {freeze.ROOT + '/' + n: h for n, h in execution.items()}
    guards[execution_fact['path']] = execution_fact['sha256']
    guards[authority['path']] = authority['sha256']
    block = "sha256sum -c <<'HASHES'\n" + ''.join(h + '  ' + p + '\n' for p, h in sorted(guards.items())) + 'HASHES\n'
    argv = ['/home/riomus/group-learning/.venv/bin/python', '-B', freeze.ROOT + '/evaluate_siglip2_compact_ranking.py',
            '--execution-sha256', execution_fact['sha256'], '--authority', authority['path'],
            '--authority-sha256', authority['sha256'], '--phase', 'cpu', '--output', '/home/riomus/runs/' + freeze.FIRST_UNIT]
    command = store(freeze.ROOT + '/first-cpu-v1-command.sh',
                    ('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=\n' + block + shlex.join(argv) + '\n' + block).encode())
    command_path = command['path']
    launcher = store(freeze.ROOT + '/first-cpu-v1-launch.sh', (
        '#!/bin/bash\nset -euo pipefail\n' + 'test ! -e /home/riomus/runs/' + freeze.FIRST_UNIT + '\n' +
        'test -f ' + command_path + '\n' + "printf '%s  %s\\n' " + command['sha256'] + ' ' + command_path + ' | sha256sum -c\n' +
        'systemd-run --unit=' + freeze.FIRST_UNIT + ' /bin/bash ' + command_path + '\n').encode())
    templates = {'authority': authority, 'command': command, 'launch': launcher}
    first_selection = {'terminal': first_unit, 'verification': first_vf}
    return {'schema': freeze.SCHEMA, 'evaluator': {'path': freeze.ROOT + '/evaluate_siglip2_compact_ranking.py',
                'sha256': execution['evaluate_siglip2_compact_ranking.py']}, 'execution': execution_fact,
            'first_cpu': templates, 'first_selection': first_selection, 'endpoints': endpoints,
            'verifications': verifications, 'copies': copies, 'frozen': {
                'evaluator': {'root': freeze.ROOT, 'execution_sha256': execution_fact['sha256'], 'code': execution},
                'first_cpu': copy.deepcopy(templates), 'first_selection': copy.deepcopy(first_selection)}}


class ContractCLI(unittest.TestCase):
    def test_exact_cli_is_available_without_native_imports(self):
        script = Path(__file__).with_name('freeze_live_top1_confirmation_cpu.py')
        process = subprocess.run([sys.executable, '-B', str(script), '--help'],
                                 capture_output=True, timeout=5)
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        self.assertIn(b'--inputs', process.stdout)
        self.assertIn(b'--output', process.stdout)


class FixtureGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global freeze
        assert Path(__file__).with_name('freeze_live_top1_confirmation_cpu.py').is_file(), 'live-top1 freezer is not implemented'
        freeze = importlib.import_module('freeze_live_top1_confirmation_cpu')
        cls.fixture = TemporaryDirectory(prefix='sfora-live-top1-static-fixture-')
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.original = (freeze.strict_json(freeze.read_bytes(INPUTS)) if INPUTS else
                        fixture(Path(cls.fixture.name)))

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='sfora-confirmation-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.value = copy.deepcopy(self.original)
        self.inputs = self.root / 'inputs.json'
        self.output = self.root / 'generated'

    def write_inputs(self):
        self.inputs.write_text(json.dumps(self.value, sort_keys=True, allow_nan=False) + '\n')

    def reject(self, message=None):
        self.write_inputs()
        with self.assertRaises((ValueError, KeyError, TypeError)) as caught:
            freeze.generate(self.inputs, self.output)
        if message:
            self.assertIn(message, str(caught.exception))
        self.assertFalse(self.output.exists())

    def replace_metadata(self, fact, value, tag):
        p = self.root / (tag + '.json')
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        p.write_bytes(raw)
        self.value['copies'][fact['path']] = str(p)
        fact['sha256'] = freeze.digest(raw)

    def mutate_second_receipt(self, action):
        endpoint = self.value['endpoints'][3]
        fact = endpoint['terminal']['receipt']
        record = freeze.strict_json(freeze.read_bytes(self.value['copies'][fact['path']]))
        action(record)
        self.replace_metadata(fact, record, 'changed-receipt')
        verification_fact = self.value['verifications'][3]
        verification = freeze.strict_json(freeze.read_bytes(verification_fact['path']))
        verification['receipt_sha256'] = fact['sha256']
        raw = (json.dumps(verification, sort_keys=True) + '\n').encode()
        p = self.root / 'changed-verification.json'
        p.write_bytes(raw)
        verification_fact.update(path=str(p), sha256=freeze.digest(raw))

    def mutate_first_receipt(self, action):
        first = self.value['first_selection']
        fact = first['terminal']['receipt']
        record = json.loads(Path(self.value['copies'][fact['path']]).read_bytes())
        action(record)
        self.replace_metadata(fact, record, 'first-mutant')
        vf = first['verification']
        verification = json.loads(Path(vf['path']).read_bytes())
        verification['receipt_sha256'] = fact['sha256']
        raw = (json.dumps(verification) + '\n').encode()
        p = self.root / 'first-verification.json'
        p.write_bytes(raw)
        vf.update(path=str(p), sha256=freeze.digest(raw))
        self.value['frozen']['first_selection'] = copy.deepcopy(first)

    def test_first_flags_match_original_training_pair(self):
        self.mutate_first_receipt(lambda record: record.update(numerical_flags={'foreign': True}))
        self.reject('original first CONTINUE/source/pair binding differs')

    def test_original_first_candidate_thresholds_are_not_relaxed(self):
        for boundary in ('matched-r1', 'concat-r1', 'matched-ap', 'concat-ap', 'source-floor'):
            with self.subTest(boundary=boundary):
                self.value = copy.deepcopy(self.original)
                def change(record):
                    pair = record['quality']['179061']
                    candidate = pair['candidate']
                    if boundary in ('matched-r1', 'concat-r1'):
                        other = pair['control'] if boundary == 'matched-r1' else record['concat_quality']
                        for field in ('recall_at_1', 'per_query_r1'):
                            candidate[field] = copy.deepcopy(other[field])
                    elif boundary == 'source-floor':
                        record['source_quality'] = copy.deepcopy(candidate)
                        record['source_quality'].update(map_at_r=.999, per_query_ap=[.999] * 1734)
                    else:
                        other = pair['control'] if boundary == 'matched-ap' else record['concat_quality']
                        ap = other['map_at_r'] - .001
                        candidate.update(map_at_r=ap, per_query_ap=[ap] * 1734)
                self.mutate_first_receipt(change)
                self.reject()

    def test_first_original_verification_summaries_are_bound(self):
        for field in ('decision', 'cost', 'deltas', 'quality', 'pass', 'receipt_sha256', 'log_sha256'):
            with self.subTest(field=field):
                self.value = copy.deepcopy(self.original)
                fact = self.value['first_selection']['verification']
                record = json.loads(Path(fact['path']).read_bytes())
                record[field] = False
                p = self.root / 'verification-mutant.json'
                raw = (json.dumps(record) + '\n').encode()
                p.write_bytes(raw)
                fact.update(path=str(p), sha256=freeze.digest(raw))
                self.value['frozen']['first_selection'] = copy.deepcopy(self.value['first_selection'])
                self.reject()

    def test_actual_cli_freeze_preserves_historical_paths_policies_and_bindings(self):
        self.write_inputs()
        script = Path(freeze.__file__).resolve()
        argv = [sys.executable, '-B', str(script), '--inputs', str(self.inputs), '--output', str(self.output)]
        process = subprocess.run(argv, text=True, capture_output=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        receipt = json.loads(process.stdout)
        self.assertEqual(receipt['generator_argv'], argv)
        self.assertTrue(receipt['source_only'])
        self.assertFalse(receipt['native_pass'])
        self.assertFalse(receipt['helper_in_scientific_closure'])
        self.assertEqual(receipt['inputs']['sha256'], freeze.digest(self.inputs.read_bytes()))
        self.assertEqual(receipt['helper']['sha256'], freeze.digest(script.read_bytes()))
        authority = json.loads((self.output / 'authority-full-cpu-v1.json').read_bytes())
        original = freeze.strict_json(freeze.read_bytes(self.value['copies'][self.value['first_cpu']['authority']['path']]))
        expected = copy.deepcopy(original)
        expected.update(stage='full', endpoints=self.value['endpoints'], first_selection=self.value['first_selection']['terminal'])
        self.assertEqual(authority, expected)
        self.assertEqual(list(receipt['outputs']), sorted(['authority-full-cpu-v1.json', 'full-cpu-v1-command.sh', 'full-cpu-v1-launch.sh']))
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), sorted(receipt['outputs']))
        for name, fact in receipt['outputs'].items():
            self.assertEqual(fact['sha256'], freeze.digest((self.output / name).read_bytes()))
        old_command = freeze.read_bytes(self.value['copies'][self.value['first_cpu']['command']['path']]).decode()
        command = (self.output / 'full-cpu-v1-command.sh').read_text()
        old_blocks, old_guards = freeze.hash_blocks(old_command)
        new_blocks, new_guards = freeze.hash_blocks(command)
        # The first-authority row remains the historical prerequisite, not full authority.
        for block in new_blocks:
            self.assertTrue(block.startswith(old_blocks[0]))
        for path, sha in old_guards.items():
            self.assertEqual(new_guards[path], sha)
        for endpoint in self.value['endpoints']:
            for fact in (endpoint['launch'], endpoint['checkpoint'], endpoint['bundle'],
                         endpoint['terminal']['receipt'], endpoint['terminal']['log']):
                self.assertEqual(new_guards[fact['path']], fact['sha256'])
        for key in ('receipt', 'log'):
            fact = self.value['first_selection']['terminal'][key]
            self.assertEqual(new_guards[fact['path']], fact['sha256'])
        new_authority = freeze.ROOT + '/authority-full-cpu-v1.json'
        self.assertEqual(new_guards[new_authority], receipt['outputs']['authority-full-cpu-v1.json']['sha256'])
        self.assertEqual(receipt['native_argv'][6], new_authority)
        self.assertEqual(receipt['native_argv'][8], new_guards[new_authority])
        self.assertEqual(receipt['native_argv'][-2:], ['--output', freeze.OUTPUT])
        self.assertEqual(command.splitlines()[2], old_command.splitlines()[2])
        old_launch = freeze.read_bytes(self.value['copies'][self.value['first_cpu']['launch']['path']]).decode()
        launcher = (self.output / 'full-cpu-v1-launch.sh').read_text()
        reversed_launch = launcher.replace(freeze.OUTPUT, '/home/riomus/runs/' + freeze.FIRST_UNIT)
        reversed_launch = reversed_launch.replace('--unit=' + freeze.UNIT, '--unit=' + freeze.FIRST_UNIT)
        reversed_launch = reversed_launch.replace(freeze.ROOT + '/full-cpu-v1-command.sh', freeze.ROOT + '/first-cpu-v1-command.sh')
        reversed_launch = reversed_launch.replace(receipt['outputs']['full-cpu-v1-command.sh']['sha256'], self.value['first_cpu']['command']['sha256'])
        self.assertEqual(reversed_launch, old_launch)
        self.assertTrue(all(c['pass'] for c in receipt['costs'].values()))
        ledger = receipt['source_correspondence']
        self.assertTrue(ledger['inverse_exact'])
        self.assertEqual(ledger['historical_helper']['sha256'], freeze.digest(Path(freeze.historical.__file__).read_bytes()))
        self.assertEqual(ledger['unchanged'], ['require', 'strict_json', 'digest', 'local_file', 'read_bytes',
                                             'file_fact', 'add_guard', 'load_module', 'cgroup', 'terminal_log', 'hash_blocks'])

        for name, mapping in ledger['adapted'].items():
            inverse = inspect.getsource(getattr(freeze, name))
            for edit in reversed(mapping['inverse_edits']):
                inverse = inverse[:edit['start']] + edit['old'] + inverse[edit['end']:]
            self.assertEqual(inverse, inspect.getsource(getattr(freeze.historical, name)))

    def test_frozen_descriptors_are_exact_and_required(self):
        for kind in ('missing', 'extra', 'source', 'template', 'first', 'old-method'):
            with self.subTest(kind=kind):
                self.value = copy.deepcopy(self.original)
                if kind == 'missing':
                    del self.value['frozen']
                elif kind == 'extra':
                    self.value['frozen']['future'] = None
                elif kind == 'source':
                    self.value['frozen']['evaluator']['code']['evaluate_siglip2_compact_ranking.py'] = '0' * 64
                elif kind == 'template':
                    self.value['frozen']['first_cpu']['command']['sha256'] = '0' * 64
                elif kind == 'first':
                    self.value['frozen']['first_selection']['terminal']['invocation_id'] = '0' * 32
                else:
                    self.value['schema'] = 'fullfeature-confirmation-cpu-inputs-v1'
                self.reject()

    def test_both_arms_require_trainable_changed_nonzero_residual(self):
        for arm in ('control', 'candidate'):
            for field in ('C_exact_zero', 'C_trainable', 'residual_nonzero_witness', 'current_C_sha256'):
                with self.subTest(arm=arm, field=field):
                    self.value = copy.deepcopy(self.original)
                    index = 3 if arm == 'control' else 2
                    self.value['endpoints'][3], self.value['endpoints'][index] = self.value['endpoints'][index], self.value['endpoints'][3]
                    self.value['verifications'][3], self.value['verifications'][index] = self.value['verifications'][index], self.value['verifications'][3]
                    def change(record):
                        record[field] = record['initial_C_sha256'] if field == 'current_C_sha256' else field == 'C_exact_zero'
                    self.mutate_second_receipt(change)
                    self.value['endpoints'][3], self.value['endpoints'][index] = self.value['endpoints'][index], self.value['endpoints'][3]
                    self.value['verifications'][3], self.value['verifications'][index] = self.value['verifications'][index], self.value['verifications'][3]
                    self.reject()

    def test_first_continue_receipt_checked_by_original_evaluator(self):
        for field in ('decision', 'quality_pass', 'concat_floor_pass', 'readiness', 'schema', 'authority_sha256', 'bootstrap_draws'):
            with self.subTest(field=field):
                self.value = copy.deepcopy(self.original)
                first = self.value['first_selection']
                fact = first['terminal']['receipt']
                record = json.loads(Path(self.value['copies'][fact['path']]).read_bytes())
                if field == 'readiness':
                    record[field]['wire_readbacks'] = False
                else:
                    record[field] = 1 if field == 'bootstrap_draws' else False
                self.replace_metadata(fact, record, 'first-mutant')
                vf = first['verification']
                verification = json.loads(Path(vf['path']).read_bytes())
                verification['receipt_sha256'] = fact['sha256']
                raw = (json.dumps(verification) + '\n').encode()
                p = self.root / 'first-verification.json'
                p.write_bytes(raw)
                vf.update(path=str(p), sha256=freeze.digest(raw))
                self.value['frozen']['first_selection'] = copy.deepcopy(first)
                self.reject()

    def test_missing_extra_input_keys_and_copy_roles(self):
        for mutation in ('missing-key', 'extra-key', 'missing-copy', 'extra-copy'):
            with self.subTest(mutation=mutation):
                self.value = copy.deepcopy(self.original)
                if mutation == 'missing-key':
                    del self.value['execution']
                elif mutation == 'extra-key':
                    self.value['future'] = None
                elif mutation == 'missing-copy':
                    del self.value['copies'][self.value['endpoints'][3]['terminal']['log']['path']]
                else:
                    self.value['copies']['/tmp/unrequested.json'] = str(self.inputs)
                self.reject()

    def test_endpoint_and_verification_order_missing_extra_and_unchanged061(self):
        for mutation in ('swap', 'missing', 'extra', 'first061', 'verification-order', 'unit-reuse', 'role'):
            with self.subTest(mutation=mutation):
                self.value = copy.deepcopy(self.original)
                if mutation == 'swap':
                    self.value['endpoints'][2:] = self.value['endpoints'][2:][::-1]
                elif mutation == 'missing':
                    self.value['endpoints'].pop()
                elif mutation == 'extra':
                    self.value['endpoints'].append(copy.deepcopy(self.value['endpoints'][3]))
                elif mutation == 'first061':
                    self.value['endpoints'][0]['terminal']['service_seconds'] += 1
                elif mutation == 'verification-order':
                    self.value['verifications'][2:] = self.value['verifications'][2:][::-1]
                elif mutation == 'unit-reuse':
                    self.value['endpoints'][3]['terminal']['invocation_id'] = self.value['endpoints'][2]['terminal']['invocation_id']
                else:
                    self.value['endpoints'][3]['bundle']['path'] = '/tmp/wrong/bundle.json'
                self.reject()

    def test_hash_tamper_frozen_source_templates_first_continue_and_verification(self):
        for mutation in ('evaluator', 'execution', 'authority', 'command', 'launch', 'CONTINUE', 'verification'):
            with self.subTest(mutation=mutation):
                self.value = copy.deepcopy(self.original)
                if mutation in ('evaluator', 'execution'):
                    self.value[mutation]['sha256'] = '0' * 64
                elif mutation in ('authority', 'command', 'launch'):
                    self.value['first_cpu'][mutation]['sha256'] = '0' * 64
                elif mutation == 'CONTINUE':
                    self.value['first_selection']['terminal']['receipt']['sha256'] = '0' * 64
                else:
                    self.value['verifications'][3]['sha256'] = '0' * 64
                self.reject()
        self.value = copy.deepcopy(self.original)
        path = self.value['endpoints'][3]['terminal']['log']['path']
        p = self.root / 'tampered.log'
        p.write_bytes(freeze.read_bytes(self.value['copies'][path]) + b'changed\n')
        self.value['copies'][path] = str(p)
        self.reject('SHA256 differs')

    def test_conflicting_path_hash_and_checkpoint_reuse(self):
        self.value['endpoints'][3]['launch'] = copy.deepcopy(self.value['endpoints'][2]['launch'])
        self.value['endpoints'][3]['launch']['sha256'] = '0' * 64
        self.reject()
        guards = {'/tmp/a.json': '1' * 64}
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            freeze.add_guard(guards, {'path': '/tmp/a.json', 'sha256': '2' * 64})
        self.value = copy.deepcopy(self.original)
        self.value['endpoints'][3]['checkpoint']['sha256'] = self.value['endpoints'][2]['checkpoint']['sha256']
        self.reject('distinct actual checkpoint')

    def test_receipt_proof_guard_argv_and_prerequisite_mutations(self):
        for mutation in ('proof', 'guard', 'argv', 'mechanics', 'completed', 'state'):
            with self.subTest(mutation=mutation):
                self.value = copy.deepcopy(self.original)
                def change(record):
                    if mutation == 'proof':
                        record['strict_reload_exact'] = False
                    elif mutation == 'guard':
                        record['input_guards'][record['bundle']['path']] = '0' * 64
                    elif mutation == 'argv':
                        record['invocation']['argv'][-1] = '/tmp/foreign-output'
                    elif mutation == 'mechanics':
                        record['launch']['selected_mechanics']['candidate']['service_seconds'] += 1
                    elif mutation == 'completed':
                        record['completed_step'] = 127
                    else:
                        record['terminal_state_sha256'] = '0' * 64
                self.mutate_second_receipt(change)
                self.reject()

    def test_actual_core_cost_over_limit_rejected_after_rehash(self):
        def change(record):
            record['total_training_core_seconds'] = 1.0
        self.mutate_second_receipt(change)
        self.reject('paired whole/core costs exceed')

    def test_whole_cost_boundary_and_missing_costs_use_frozen_api(self):
        evaluator = freeze.load_module('frozen-evaluator', freeze.read_bytes(self.original['copies'][self.original['evaluator']['path']]), '_fixture_evaluator')
        records = {key: {'service_seconds': 100.0, 'total_training_core_seconds': 50.0} for key in evaluator.ORDER}
        for seed in evaluator.SEEDS:
            records[seed, 'candidate'] = {'service_seconds': 150.0, 'total_training_core_seconds': 75.0}
        self.assertTrue(all(v['pass'] for v in evaluator.paired_cost(records, 'full').values()))
        records[179069, 'candidate']['service_seconds'] = 150.000001
        self.assertFalse(evaluator.paired_cost(records, 'full')['179069']['pass'])
        del records[179069, 'control']
        with self.assertRaises(ValueError):
            evaluator.paired_cost(records, 'full')

    def test_rehashed_log_footer_runtime_and_invocation_mismatches(self):
        for mutation in ('footer', 'runtime', 'invocation'):
            with self.subTest(mutation=mutation):
                self.value = copy.deepcopy(self.original)
                unit = self.value['endpoints'][3]['terminal']
                raw = freeze.read_bytes(self.value['copies'][unit['log']['path']])
                if mutation == 'footer':
                    raw = raw.replace(b'STOP_CGROUP ', b'MISSING_CGROUP ')
                elif mutation == 'runtime':
                    raw = raw.replace(b'Service runtime: ', b'Service runtime: 1')
                else:
                    raw = raw.replace(unit['invocation_id'].encode(), b'0' * 32)
                p = self.root / 'changed.log'
                p.write_bytes(raw)
                self.value['copies'][unit['log']['path']] = str(p)
                unit['log']['sha256'] = freeze.digest(raw)
                vf = self.value['verifications'][3]
                v = freeze.strict_json(freeze.read_bytes(vf['path']))
                v['log_sha256'] = unit['log']['sha256']
                p = self.root / 'changed-log-verification.json'
                raw = (json.dumps(v) + '\n').encode()
                p.write_bytes(raw)
                vf.update(path=str(p), sha256=freeze.digest(raw))
                self.reject()

    def test_exclusive_canonical_output_inputs_and_exact_cli(self):
        self.output.mkdir()
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            freeze.generate(self.inputs, self.output)
        self.assertEqual(list(self.output.iterdir()), [])
        script = str(Path(freeze.__file__).resolve())
        for args in ([], ['--input', str(self.inputs), '--output', str(self.output)],
                     ['--inputs', str(self.inputs), '--output', str(self.output), '--stage', 'full']):
            process = subprocess.run([sys.executable, '-B', script, *args], capture_output=True, timeout=5)
            self.assertNotEqual(process.returncode, 0)
        alias = self.root / 'alias.json'
        alias.symlink_to(self.inputs)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            freeze.local_file(str(alias))

    def test_duplicate_nonfinite_and_unsafe_paths(self):
        for raw in ('{"schema":1,"schema":2}', '{"number":NaN}', '{"number":Infinity}'):
            with self.assertRaises(ValueError):
                freeze.strict_json(raw)
        for path in ('relative', '/tmp/../a', '/tmp/a\ncommand', '/tmp//a', '/tmp/$(id)'):
            with self.assertRaises(ValueError):
                freeze.file_fact({'path': path, 'sha256': 'a' * 64})
        self.assertFalse(any(name.split('.')[0] in freeze.NATIVE for name in sys.modules))

    def test_native_import_is_rejected_before_metadata_admission(self):
        sys.modules['torch'] = object()
        try:
            self.reject('native imports forbidden')
        finally:
            del sys.modules['torch']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--inputs')
    args = parser.parse_args()
    INPUTS = args.inputs
    unittest.main(argv=[sys.argv[0]], verbosity=2)
