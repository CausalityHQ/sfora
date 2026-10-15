#!/usr/bin/env python3
"""Source-only chunk500 falsifiers; stdlib and tiny synthetic files only."""
import ast
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_siglip2_substrate_full as fixtures
import test_stage_siglip2_substrate_full as stage_fixtures

HERE = Path(__file__).resolve().parent


class Chunk500(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert (HERE / 'train_siglip2_substrate_full_v2.py').is_file(), 'v2 trainer is missing'
        with patch('builtins.__import__', fixtures.stdlib_import):
            cls.full = importlib.import_module('train_siglip2_substrate_full_v2')
            cls.score = importlib.import_module('score_siglip2_substrate_official_v2')
            cls.stage = importlib.import_module('stage_siglip2_substrate_full_v2')
            cls.exact = importlib.import_module('falsify_siglip2_substrate_full_v2')

    def chain(self, root, count=4, arm='large', chunk=500, *, source=None, code=None, staged=False):
        full = self.full
        for name in ('large', 'so400'):
            (root / name).mkdir(exist_ok=True)
        source = source or {'receipt': {'path': '/source/receipt.json', 'sha256': '1' * 64},
                            'checkpoint': {'path': '/source/resume.pt', 'sha256': '2' * 64}}
        code = code or {'trainer': '3' * 64, 'partition_parser': full.PARSER_SHA}
        identity = {'schema': full.SCHEMA, 'arm': arm, 'source': source, 'code': code,
                    'full_rows_sha256': '4' * 64, 'schedule_sha256': '5' * 64}
        previous = terminal = checkpoint = None
        for index in range(1, count + 1):
            end = index * chunk
            folder = root / arm / f'chunk-{end:04d}'
            folder.mkdir()
            (folder / 'resume.pt').write_text(str(end))
            row = {'schema': full.SCHEMA, 'chunk_length': chunk, 'pass': True, 'quality_read': False,
                   'arm': arm, 'width': full.WIDTHS[arm], 'seed': full.SEED, 'completed_step': end,
                   'chunk_start': end - chunk, 'source': source, 'code': code, 'identity': identity,
                   'resource_policy': full.POLICY, 'reload_exact': True, 'exit_rehash_pass': True,
                   'steps': [{'step': step} for step in range(end - chunk + 1, end + 1)],
                   'checkpoint': full.descriptor(folder / 'resume.pt'), 'previous': previous,
                   'previous_terminal': terminal, 'previous_checkpoint_sha256': checkpoint}
            helper = fixtures.Contract()
            unit = self.stage.target('train', arm, end)[0] if staged else f'v2-{arm}-{end}'
            with patch.object(fixtures, 'full', full):
                _, terminal_value, log = helper.terminal_fixture(
                    folder, row, unit, f'{end + (10000 if arm == "so400" else 0):032x}')
            if staged:
                destination = self.stage.ROOT / f'{unit}.log'
                log.rename(destination)
                terminal_value['log'] = full.descriptor(destination)
                (folder / 'terminal.json').write_text(json.dumps(terminal_value))
            previous = full.descriptor(folder / 'receipt.json')
            terminal = full.descriptor(folder / 'terminal.json')
            checkpoint = row['checkpoint']['sha256']
        return previous, terminal, source, code

    def test_only_four_units_complete_and_first_has_no_previous(self):
        for count in (3, 4, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                previous, terminal, source, code = self.chain(root, count)
                self.chain(root, count, 'so400')
                if count == 4:
                    self.assertEqual(self.full.inspect_layout(root, complete=True), {'large': 2000, 'so400': 2000})
                    row = self.full.check_chain(previous, 2000, 'large', source, code, terminal)
                    self.assertEqual(row['completed_step'], 2000)
                else:
                    with self.assertRaises(ValueError):
                        self.full.inspect_layout(root, complete=True)
                    with self.assertRaises(ValueError):
                        self.full.check_chain(previous, 2000, 'large', source, code, terminal)
                first = json.loads((root / 'large/chunk-0500/receipt.json').read_text())
                for key in ('previous', 'previous_terminal', 'previous_checkpoint_sha256'):
                    self.assertIsNone(first[key])
                self.assertIsNone(self.full.check_chain(None, 0, 'large', source, code))

    def test_mixed_v1_receipt_or_chunk_length_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, _, source, code = self.chain(Path(tmp), 1)
            row = self.full.read_json(ref)
            self.full.validate_receipt(row, 500, 'large', source, code)
            for change in ({'schema': 'siglip2-substrate-full-v1'}, {'chunk_length': 100},
                           {'chunk_length': None}, {'chunk_start': 400}, {'previous_checkpoint_sha256': 'x'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.full.validate_receipt({**row, **change}, 500, 'large', source, code)
            with self.assertRaises(ValueError):
                fixtures.full.validate_receipt(row, 500, 'large', source, code)

    def test_chunk_constant_drives_parser_layout_chain_and_stage(self):
        self.assertEqual(list(self.full.parser()._option_string_actions['--chunk-end'].choices), [500, 1000, 1500, 2000])
        for end in (100, 250, 2100):
            with self.assertRaises(AssertionError):
                self.stage.target('train', 'large', end)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.stage.chunks()
        self.assertEqual(out.getvalue(), '500 1000 1500 2000\n')
        # An alternate divisor exposes literal-500 and literal-100 boundary regressions.
        with patch.object(self.full, 'CHUNK', 250), patch.object(self.stage, 'CHUNK', 250):
            self.assertEqual(list(self.full.parser()._option_string_actions['--chunk-end'].choices),
                             [250, 500, 750, 1000, 1250, 1500, 1750, 2000])
            with tempfile.TemporaryDirectory() as tmp:
                ref, terminal, source, code = self.chain(Path(tmp), 8, chunk=250)
                self.chain(Path(tmp), 8, 'so400', chunk=250)
                self.assertEqual(self.full.inspect_layout(Path(tmp), complete=True)['large'], 2000)
                self.full.check_chain(ref, 2000, 'large', source, code, terminal)
            self.stage.target('train', 'large', 250)
        raw = (HERE / 'train_siglip2_substrate_full_v2.py').read_text().replace('CHUNK = 500', 'CHUNK = 333')
        with self.assertRaises(AssertionError):
            exec(compile(raw, '<nondivisor>', 'exec'), {'__file__': str(HERE / 'train_siglip2_substrate_full_v2.py')})

    def test_every_chunk_expression_uses_the_constant(self):
        # Evaluate every boundary expression, including run() before native execution is allowed.
        # Replacing any CHUNK occurrence with the old literal must change its result.
        for name in ('train_siglip2_substrate_full_v2.py', 'stage_siglip2_substrate_full_v2.py'):
            tree = ast.parse((HERE / name).read_text())
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            tested = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (99, 100):
                    owner = parents.get(node)
                    while owner is not None and not isinstance(owner, ast.FunctionDef):
                        owner = parents.get(owner)
                    self.assertEqual(node.value, 100)
                    self.assertIn(owner.name if owner else None, ('validate_source', 'source_state'))
                if not isinstance(node, ast.Name) or node.id != 'CHUNK' or not isinstance(node.ctx, ast.Load):
                    continue
                expr = node
                while isinstance(parents.get(expr), ast.BinOp) or (
                    isinstance(parents.get(expr), ast.Call) and
                    isinstance(parents[expr].func, ast.Name) and parents[expr].func.id == 'range'
                ):
                    expr = parents[expr]
                if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod) and
                        isinstance(expr.left, ast.Name) and expr.left.id == 'TOTAL_UPDATES'):
                    continue  # Both divisors satisfy the assertion; nondivisor tested separately.
                mutant = copy.deepcopy(expr)
                offset = next(i for i, n in enumerate(ast.walk(expr)) if n is node)
                target = list(ast.walk(mutant))[offset]
                target.id = 'OLD_CHUNK'
                differences = []
                for end in (500, 600, 1000):
                    env = {'CHUNK': 500, 'TOTAL_UPDATES': 2000, 'end': end,
                           'args': SimpleNamespace(chunk_end=end), 'chunks': list(range(5)),
                           'row': {'chunk_length': 500}, 'len': len, 'range': range}
                    actual = eval(compile(ast.Expression(expr), '<boundary>', 'eval'), env)
                    changed = eval(compile(ast.Expression(mutant), '<literal-100-mutant>', 'eval'),
                                   {**env, 'OLD_CHUNK': 100})
                    if isinstance(actual, (range, map)):
                        actual, changed = list(actual), list(changed)
                    differences.append(actual != changed)
                self.assertTrue(any(differences), (name, node.lineno, ast.unparse(expr)))
                tested += 1
            self.assertGreater(tested, 5, name)

    def test_training_math_and_integrity_guards_preserved(self):
        unchanged = ('schedule', 'pixels', 'update', 'source_state', 'initialize_full', 'validate_scaler',
                     'check_payload', 'restore_full', 'save', 'service_runtime', 'admit_held_terminal',
                     'admit_terminal', 'install_backstop', 'native_start', 'resources', 'admit_source')
        for name in unchanged:
            self.assertEqual(fixtures.dump(fixtures.function(HERE / 'train_siglip2_substrate_full.py', name)),
                             fixtures.dump(fixtures.function(HERE / 'train_siglip2_substrate_full_v2.py', name)), name)
        for name in ('lifetime', 'environment', 'launch_line'):
            self.assertEqual(fixtures.dump(fixtures.function(HERE / 'stage_siglip2_substrate_full.py', name)),
                             fixtures.dump(fixtures.function(HERE / 'stage_siglip2_substrate_full_v2.py', name)), name)
        for name in ('admit_pair', 'check_score_layout', 'quality_gate', 'paired', 'validate_source_pin'):
            self.assertEqual(fixtures.dump(fixtures.function(HERE / 'score_siglip2_substrate_official.py', name)),
                             fixtures.dump(fixtures.function(HERE / 'score_siglip2_substrate_official_v2.py', name)), name)

    def test_driver_uses_staged_boundaries_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run_siglip2_substrate_full_v2.sh'
            driver.write_bytes((HERE / driver.name).read_bytes())
            tool = root / 'stage_siglip2_substrate_full_v2.py'
            tool.write_text("import json, sys\nfrom pathlib import Path\n"
                            "args = sys.argv[1:]\n"
                            "with (Path(__file__).parent / 'calls').open('a') as stream: stream.write(json.dumps(args)+'\\n')\n"
                            "if args[0] == 'chunks': print('500 1000 1500 2000')\n"
                            "elif args[0] == 'environment': print('fixture')\n"
                            "elif args[0] == 'launch': print('exit 0')\n")
            result = subprocess.run(['bash', str(driver)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            units = [('train', arm, str(end)) for arm in ('large', 'so400') for end in (500, 1000, 1500, 2000)]
            units += [('score', 'large'), ('score', 'so400')]
            self.assertEqual([call[1:] for call in calls if call[0] == 'launch'], list(map(list, units)))
            self.assertEqual([call[1:] for call in calls if call[0] == 'admit'], list(map(list, units)))
            self.assertEqual(calls[0], ['lifetime'])
            self.assertEqual(calls[-1], ['decision'])

    def test_staged_launch_chaining_and_both_complete_before_score(self):
        self.enterContext(patch.object(stage_fixtures, 'stage', self.stage))
        helper = stage_fixtures.Staging()
        self.addCleanup(helper.doCleanups)
        helper.setUp()
        helper.test_staged_closure_and_create_only()
        with self.assertRaises(AssertionError):
            self.stage.arguments(helper.full, helper.score, helper.code, 'train', 'so400', 500)
        script, first = self.stage.arguments(helper.full, helper.score, helper.code, 'train', 'large', 500)
        self.assertEqual(script, 'train_siglip2_substrate_full_v2.py')
        self.assertNotIn('--previous', first)
        code = {'trainer': helper.code[self.stage.TRAINER], 'partition_parser': self.full.PARSER_SHA}
        ref, terminal, _, _ = self.chain(self.stage.OUTPUT, 1, source=helper.sources['large'], code=code, staged=True)
        _, second = self.stage.arguments(helper.full, helper.score, helper.code, 'train', 'large', 1000)
        self.assertEqual(second[second.index('--previous') + 1], ref['path'])
        self.assertEqual(second[second.index('--previous-terminal-sha256') + 1], terminal['sha256'])
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.stage.arguments(helper.full, helper.score, helper.code, 'score', 'large')
        # Fill the remaining units with genuine chain/terminal validation, never native execution.
        for arm in ('large', 'so400'):
            if arm == 'large':
                import shutil
                shutil.rmtree(self.stage.OUTPUT / arm)
            self.chain(self.stage.OUTPUT, 4, arm, source=helper.sources[arm], code=code, staged=True)
        script, args = self.stage.arguments(helper.full, helper.score, helper.code, 'score', 'large')
        self.assertEqual(script, 'score_siglip2_substrate_official_v2.py')
        self.assertIn('--large-terminal-sha256', args)
        self.assertIn('--so400-terminal-sha256', args)
        self.assertFalse((self.stage.OUTPUT / 'official').exists())

    def test_falsifier_collects_all_five_archived_receipts_and_one_v2_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_code = {'trainer': fixtures.full.sha(Path(fixtures.full.__file__)),
                        'partition_parser': fixtures.full.PARSER_SHA}
            old, _, _ = fixtures.Contract().chain(root, 5, code=old_code)
            folder = Path(old['path']).parent
            terminal = fixtures.full.descriptor(folder / 'terminal.json')
            _, rows, _ = self.exact.admitted_endpoint(fixtures.full, folder, terminal['sha256'])
            self.assertEqual([row['step'] for row in rows], list(range(1, 501)))
            with self.assertRaises(ValueError):
                self.exact.admitted_endpoint(fixtures.full, folder, '0' * 64)
        with tempfile.TemporaryDirectory() as tmp:
            code = {'trainer': self.full.sha(Path(self.full.__file__)), 'partition_parser': self.full.PARSER_SHA}
            ref, terminal, _, _ = self.chain(Path(tmp), 1, code=code)
            _, rows, _ = self.exact.admitted_endpoint(self.full, Path(ref['path']).parent, terminal['sha256'])
            self.assertEqual([row['step'] for row in rows], list(range(1, 501)))

    def evidence(self):
        rows = [{key: (step if key == 'step' else 'same') for key in self.exact.STEP_FIELDS}
                for step in range(1, 501)]
        receipt = {'schema': fixtures.full.SCHEMA, 'terminal_state_sha256': 'a' * 64,
                   'identity': {'schema': fixtures.full.SCHEMA, 'code': {'trainer': 'old'},
                                'arm': 'large', 'config': {'hidden_size': 1024}}}
        new_receipt = copy.deepcopy(receipt)
        new_receipt.update(schema=self.full.SCHEMA, chunk_length=500, terminal_state_sha256='d' * 64)
        new_receipt['identity'].update(schema=self.full.SCHEMA, code={'trainer': 'new'})
        fingerprints = {key: 'b' * 64 for key in self.exact.STATE_FIELDS}
        return rows, receipt, new_receipt, fingerprints

    def test_exactness_identical_passes_changed_step_or_fingerprint_fails(self):
        rows, receipt, new_receipt, fingerprints = self.evidence()
        new_rows = copy.deepcopy(rows)
        rows[0]['state_sha256'], new_rows[0]['state_sha256'] = 'old-metadata-hash', 'new-metadata-hash'
        result = self.exact.compare_evidence(rows, new_rows, receipt, new_receipt, fingerprints, fingerprints)
        self.assertEqual(result, {'raw_terminal_hash_equal': False})
        same_raw_hash = {**new_receipt, 'terminal_state_sha256': receipt['terminal_state_sha256']}
        self.assertEqual(self.exact.compare_evidence(rows, new_rows, receipt, same_raw_hash, fingerprints, fingerprints),
                         {'raw_terminal_hash_equal': True})
        for key in self.exact.STEP_FIELDS:
            changed = copy.deepcopy(rows)
            changed[222][key] = 'changed'
            with self.subTest(step_field=key), self.assertRaises(ValueError):
                self.exact.compare_evidence(rows, changed, receipt, new_receipt, fingerprints, fingerprints)
        for key in self.exact.STATE_FIELDS:
            with self.subTest(state_field=key), self.assertRaises(ValueError):
                self.exact.compare_evidence(rows, rows, receipt, new_receipt, fingerprints, {**fingerprints, key: 'c' * 64})
        with self.assertRaises(ValueError):
            self.exact.compare_evidence(rows[:-1], rows, receipt, new_receipt, fingerprints, fingerprints)

    def test_exactness_rejects_other_identity_changes_and_wrong_schema_or_length(self):
        rows, receipt, new_receipt, fingerprints = self.evidence()
        for key, value in (('arm', 'so400'), ('config', {'hidden_size': 1152}), ('extra', 'unexpected')):
            changed = copy.deepcopy(new_receipt)
            changed['identity'][key] = value
            with self.subTest(identity=key), self.assertRaises(ValueError):
                self.exact.compare_evidence(rows, rows, receipt, changed, fingerprints, fingerprints)
        changed = copy.deepcopy(new_receipt)
        del changed['identity']['config']
        with self.assertRaises(ValueError):
            self.exact.compare_evidence(rows, rows, receipt, changed, fingerprints, fingerprints)
        for old, new in ((new_receipt, receipt), (receipt, {**new_receipt, 'schema': receipt['schema']}),
                         (receipt, {**new_receipt, 'chunk_length': 100}),
                         (receipt, {k: v for k, v in new_receipt.items() if k != 'chunk_length'})):
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                self.exact.compare_evidence(rows, rows, old, new, fingerprints, fingerprints)

    def test_exactness_payload_inventory_is_exhaustive_on_each_endpoint(self):
        tree = ast.parse((HERE / 'train_siglip2_substrate_adaptation.py').read_text())
        keys = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign) and
                                     any(isinstance(t, ast.Name) and t.id == 'PAYLOAD_KEYS' for t in node.targets)))
        pilot = SimpleNamespace(PAYLOAD_KEYS=keys)
        disk = dict.fromkeys(keys)
        self.exact.validate_payload_fields(disk, pilot)
        for endpoint in ('v1', 'v2'):
            for key in keys:
                with self.subTest(endpoint=endpoint, missing=key), self.assertRaises(ValueError):
                    self.exact.validate_payload_fields({k: v for k, v in disk.items() if k != key}, pilot)
            with self.subTest(endpoint=endpoint, extra=True), self.assertRaises(ValueError):
                self.exact.validate_payload_fields({**disk, 'extra_numerical_component': None}, pilot)
        for declared in (keys | {'extra_numerical_component'}, keys - {'bank'}, keys - {'identity'}):
            with self.subTest(pilot_keys=declared), self.assertRaises(ValueError):
                self.exact.validate_payload_fields(dict.fromkeys(declared), SimpleNamespace(PAYLOAD_KEYS=declared))


if __name__ == '__main__':
    unittest.main()
