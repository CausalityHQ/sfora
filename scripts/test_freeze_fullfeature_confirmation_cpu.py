#!/usr/bin/env python3
"""One stdlib fixture gate against parent-supplied actual confirmation inputs.

Run with --inputs CANONICAL_JSON. All generated artifacts and adversarial copies
stay in a temporary directory. No checkpoint loads or launcher execution.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import freeze_fullfeature_confirmation_cpu as freeze


class FixtureGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = freeze.strict_json(freeze.read_bytes(INPUTS))

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
        reversed_launch = reversed_launch.replace(receipt['outputs']['full-cpu-v1-command.sh']['sha256'], freeze.PINS['first-cpu-v1-command.sh'])
        self.assertEqual(reversed_launch, old_launch)
        self.assertTrue(all(c['pass'] for c in receipt['costs'].values()))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args()
    INPUTS = freeze.local_file(args.inputs)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
