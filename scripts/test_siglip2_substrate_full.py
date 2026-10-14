#!/usr/bin/env python3
"""Stdlib-only falsifiers. No torch/numpy/model/image execution or qualification."""
import ast
import builtins
import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from decimal import Decimal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Checks require unoptimized Python')

ROOT = Path(__file__).resolve().parent
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'torchvision', 'safetensors', 'sfora'}
original_import = builtins.__import__


def stdlib_import(name, *args, **kwargs):
    if name.split('.')[0] in NATIVE:
        raise AssertionError('forbidden native import: ' + name)
    return original_import(name, *args, **kwargs)


with patch('builtins.__import__', stdlib_import):
    import train_siglip2_substrate_full as full
    import score_siglip2_substrate_official as score


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


class Contract(unittest.TestCase):
    def terminal_fixture(self, folder, record, unit='full-large-100', inv='1' * 32):
        value = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(full.POLICY['host_bytes']), 'memory.current': '1000000', 'memory.peak': '2000000',
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'max 0\noom 0\noom_kill 0\n'}}
        record.update(invocation={'argv': [str(ROOT / 'train_siglip2_substrate_full.py')],
                                 'python': str(Path(sys.executable).resolve()), 'python_sha256': 'a' * 64,
                                 'python_version': sys.version, 'optimize': 0, 'pid': 1,
                                 'invocation_id': inv, 'cuda_visible_devices': '0'},
                      cgroup_before=copy.deepcopy(value), cgroup_after=copy.deepcopy(value),
                      wall_seconds=170, process_peak_rss_kib=90, peak_cuda_allocated_bytes=1000000)
        receipt = folder / 'receipt.json'
        full.publish(receipt, record)
        log = folder / 'unit.log'
        lines = [f'Running as unit: {unit}.service; invocation ID: {inv}', '\tMaximum resident set size (kbytes): 100',
                 '\tSwaps: 0', '\tExit status: 0', 'FINAL_CGROUP ' + json.dumps({**value, 'invocation_id': inv}),
                 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
                 'Service runtime: 3min 81ms', 'Memory swap peak: 0B']
        log.write_text('\n'.join(lines) + '\n')
        terminal = {'receipt': full.descriptor(receipt), 'log': full.descriptor(log), 'unit': unit, 'invocation_id': inv,
                    'service_seconds': 180.081, 'native_peak_rss_kib': 100, 'both_locks_held': True}
        full.publish(folder / 'terminal.json', terminal)
        return record, terminal, log

    def terminal_ref(self, receipt):
        return full.descriptor(Path(receipt['path']).parent / 'terminal.json')

    def test_held_v2_runtime_grammar(self):
        for text, value in (('3min 81ms', '180.081'), ('2min 45.123s', '165.123'), ('46.907s', '46.907'), ('999ms', '0.999'),
                            ('5min', '300'), ('59min 59.999s', '3599.999'), ('0.5s', '0.5'), ('1min 0.004s', '60.004')):
            self.assertEqual(full.service_runtime(text), Decimal(value), text)
        for text in ('', ' ', '3min81ms', '3min  81ms', ' 81ms', '81ms ', '3min ', '0min 5s', '03min 5s', '60min', '1h 2min',
                     '3min 60s', '3min 61.5s', '1000ms', '081ms', '0ms', '0s', '0.000s', '-5s', '5', '5sec', '3min 81ms 2s',
                     '3min 5s 81ms', '1.5min', '3min 81us', '٣min 5s', '5.s', '.5s', '1e3ms', '1.0000s', '1.0000000000000001s',
                     '59min 59.' + '9' * 30 + 's', '45.1234s', '3min\t81ms', '3min 81ms\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                full.service_runtime(text)

    def test_original_terminal_admission_rejects_mutations(self):
        with TemporaryDirectory() as temporary:
            record = {'schema': full.SCHEMA, 'pass': True, 'exit_rehash_pass': True, 'resource_policy': full.POLICY}
            record, terminal, log = self.terminal_fixture(Path(temporary), record)
            self.assertEqual(full.admit_terminal(record, terminal)['invocation_id'], '1' * 32)
            original = log.read_text()
            for text in (original + 'Service runtime: 3min 81ms\n', original + 'watchdog: stopped\n',
                         original.replace('invocation ID: ' + '1' * 32, 'invocation ID: ' + '2' * 32),
                         original.replace('\tSwaps: 0', '\tSwaps: 1'), original.replace('Memory swap peak: 0B', 'Memory swap peak: 1B'),
                         original.replace('"memory.swap.peak": "0"', '"memory.swap.peak": "1"'),
                         original.replace('oom 0', 'oom 1'), original.replace('max 0', 'max 1'), original.replace('oom_kill 0', 'oom_kill 1'),
                         original.replace('\tExit status: 0', '\tExit status: 1'), original + 'Running as unit: other.service; invocation ID: ' + '2' * 32 + '\n',
                         original.replace('Service runtime: 3min 81ms', 'Service runtime: 3min 081ms')):
                log.write_text(text)
                with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                    full.admit_terminal(record, {**terminal, 'log': full.descriptor(log)})
            log.write_text(original)
            for edit in ({'both_locks_held': False}, {'invocation_id': '2' * 32}, {'service_seconds': 180.81}, {'native_peak_rss_kib': 101}):
                with self.assertRaises(ValueError):
                    full.admit_terminal(record, {**terminal, **edit})
            # Parent CLI uses the same admission and publishes create-only.
            (Path(temporary) / 'terminal.json').unlink()
            argv = ['--receipt', terminal['receipt']['path'], '--receipt-sha256', terminal['receipt']['sha256'],
                    '--log', str(log), '--log-sha256', terminal['log']['sha256'], '--both-locks-held']
            with contextlib.redirect_stdout(io.StringIO()):
                full.admit_terminal_cli(argv)
                with self.assertRaises(FileExistsError):
                    full.admit_terminal_cli(argv)
            self.assertEqual(full.read_json(full.descriptor(Path(temporary) / 'terminal.json')), terminal)

    def test_invocation_requires_original_systemd_id(self):
        with patch.dict(os.environ, {'INVOCATION_ID': 'c' * 32, 'CUDA_VISIBLE_DEVICES': '0'}):
            value = full.invocation()
            self.assertEqual(value['invocation_id'], 'c' * 32)
            self.assertEqual(value['argv'], sys.argv)
            self.assertEqual(value['python_sha256'], full.sha(Path(sys.executable).resolve()))
        with patch.dict(os.environ, {'INVOCATION_ID': ''}), self.assertRaises(ValueError):
            full.invocation()
    def test_schedule_exact_incumbent_ast(self):
        old = function(ROOT / 'pe_large_optimization.py', 'schedule')
        new = function(ROOT / 'train_siglip2_substrate_full.py', 'schedule')
        class Seed(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if dump(node) == dump(ast.parse('pair.SEED', mode='eval').body):
                    return ast.Name(id='SEED', ctx=ast.Load())
                return self.generic_visit(node)
        old = Seed().visit(old)
        old.body = [n for n in old.body if not (isinstance(n, ast.Assert) and 'coverage.schedule' in ast.unparse(n))]
        new.body = [n for n in new.body if not isinstance(n, ast.Import)]
        self.assertEqual(dump(old), dump(new))
        self.assertEqual((full.TOTAL_UPDATES, full.SEED), (2000, 179032))

    def test_classifier_retention_and_new_mean_exact_ast(self):
        old = function(ROOT / 'pe_large_coverage.py', 'inputs')
        new = function(ROOT / 'train_siglip2_substrate_full.py', 'classifier_from_source')
        old_loop = next(n for n in ast.walk(old) if isinstance(n, ast.For) and ast.unparse(n.target) == '(i, name)')
        new_loop = next(n for n in ast.walk(new) if isinstance(n, ast.For))
        self.assertEqual(dump(old_loop), dump(new_loop))
        # Compare the sorted product -> contiguous target map, not just the loop.
        for name in ('names', 'classes', 'target', 'classifier'):
            old_node = next(n for n in ast.walk(old) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
            new_node = next(n for n in ast.walk(new) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
            self.assertEqual(dump(old_node), dump(new_node))

    def test_update_preserves_pilot_math(self):
        source = (ROOT / 'train_siglip2_substrate_adaptation.py').read_text()
        old = ast.get_source_segment(source, function(ROOT / 'train_siglip2_substrate_adaptation.py', 'update'))
        old = old.replace('    import torch\n', "    import torch\n    pilot = context['pilot']\n", 1)
        old = old.replace('pixels, rgb = augmented_pixels(context, ref, state, batch, step)',
                          "batch_pixels, rgb = pixels(context, state['processor'], identity['full_rows'], batch, step)")
        old = old.replace('fingerprint(pixels)', 'pilot.fingerprint(batch_pixels)')
        old = old.replace('pixels[offset:offset + 16]', 'batch_pixels[offset:offset + 16]')
        old = old.replace('rank = valid_rank(', 'rank = pilot.valid_rank(')
        old = old.replace("    require(scaler.state_dict()['_growth_tracker'] == state['counter'], 'scaler successful-update counter differs')",
                          "    validate_scaler(scaler.state_dict(), state['counter'])")
        old = old.replace('    integrity(state, identity)', '    pilot.integrity(state, identity)')
        old = old.replace("fingerprint(payload(state, identity, flags), state['frozen_cache'])",
                          "pilot.fingerprint(payload(context, state, identity, flags), state['frozen_cache'])")
        self.assertEqual(dump(ast.parse(old).body[0]), dump(function(ROOT / 'train_siglip2_substrate_full.py', 'update')))

    def test_corrected_rank_matches_incumbent(self):
        old = function(ROOT / 'pe_native_valid_anchor.py', 'valid_rank')
        pilot = function(ROOT / 'train_siglip2_substrate_adaptation.py', 'valid_rank')
        text = ast.unparse(old).replace('def valid_rank(', 'def valid_rank(ref, ')
        text = text.replace('driver.pair.smoke.member_bank_rank_loss', 'ref.member_bank_rank_loss')
        self.assertEqual(dump(ast.parse(text).body[0]), dump(pilot))

    def test_width_arm_mismatches(self):
        for arm, width, count in (('large', 1024, 400), ('so400', 1152, 448)):
            full.validate_layout(arm, width, (128, width), count)
            for values in ((arm, 1024 if width == 1152 else 1152, (128, width), count),
                           (arm, width, (128, 1024 if width == 1152 else 1152), count),
                           (arm, width, (128, width), 400 if count == 448 else 448)):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    full.validate_layout(*values)

    def test_scaler_final_step_rollover(self):
        for step in (0, 100, 1900, 1999, 2000):
            value = {'growth_interval': 2000, 'growth_factor': 2., 'backoff_factor': .5,
                     '_growth_tracker': step % 2000, 'scale': 128 * 2 ** (step // 2000)}
            full.validate_scaler(value, step)
            for key, wrong in (('_growth_tracker', step % 2000 + 1), ('growth_interval', 100), ('scale', 64)):
                with self.assertRaisesRegex(ValueError, 'scaler'):
                    full.validate_scaler({**value, key: wrong}, step)

    def test_source_requires_same_arm_successful_seed032_train100(self):
        checkpoint = {'path': '/source/resume.pt', 'sha256': '1' * 64}
        record = {'schema': 'siglip2-substrate-adaptation-v1', 'phase': 'train', 'arm': 'so400',
                  'seed': 179032, 'width': 1152, 'completed_step': 100, 'checkpoint': checkpoint,
                  'identity': {'arm': 'so400', 'seed': 179032, 'config': {'hidden_size': 1152}},
                  'pass': True, 'training_qualified': True, 'quality_read': False, 'training_state_discarded': False,
                  'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
                  'exit_rehash_pass': True, 'resource_policy': full.POLICY,
                  'code': {'train_siglip2_substrate_adaptation.py': full.PILOT_SHA}}
        full.validate_source(record, 'so400', checkpoint)
        for key, wrong in (('arm', 'large'), ('width', 1024), ('seed', 179041), ('completed_step', 17),
                           ('phase', 'mechanics'), ('pass', False), ('training_qualified', False),
                           ('quality_read', True), ('training_state_discarded', True), ('exit_rehash_pass', False),
                           ('checkpoint', {}), ('resource_policy', {}),
                           ('identity', {**record['identity'], 'arm': 'large'})):
            with self.assertRaises(ValueError):
                full.validate_source({**record, key: wrong}, 'so400', checkpoint)

    def test_complete_resume_metadata_rejects_missing_state_and_wrong_width(self):
        identity = {'config': {'hidden_size': 1152}, 'optimizer_defaults': {}, 'optimizer_serial_groups': [],
                    'parameter_names': list(range(208)), 'full_rows': [] , 'full_rows_sha256': full.json_sha([])}
        disk = {'schema': full.SCHEMA, 'identity': identity, 'counter': 2000, 'seed': full.SEED,
                'numerical_flags': {}, 'config': identity['config'], 'optimizer_defaults': {},
                'optimizer': {'param_groups': [], 'state': {i: {'step': 2000, 'exp_avg': 0, 'exp_avg_sq': 0} for i in range(208)}},
                'cuda_rng': [b'cuda'], 'cpu_rng': b'cpu', 'head': {'weight': SimpleNamespace(shape=(128, 1152))},
                'vision': dict.fromkeys(range(448)), 'buffers': {}, 'pca': {}, 'positive': [],
                'classifier': SimpleNamespace(shape=(3997, 128)), 'bank': SimpleNamespace(shape=(25882, 128)),
                'target': SimpleNamespace(shape=(25882,)), 'schedules': {str(full.SEED): SimpleNamespace(shape=(2000, 64))},
                'scaler': {'growth_interval': 2000, 'growth_factor': 2., 'backoff_factor': .5, '_growth_tracker': 0, 'scale': 256}}
        pilot_ast = ast.parse((ROOT / 'train_siglip2_substrate_adaptation.py').read_text())
        keys = ast.literal_eval(next(n.value for n in pilot_ast.body if isinstance(n, ast.Assign) and
                                     any(isinstance(t, ast.Name) and t.id == 'PAYLOAD_KEYS' for t in n.targets)))
        context = {'pilot': SimpleNamespace(PAYLOAD_KEYS=keys)}
        record = {'arm': 'so400', 'completed_step': 2000, 'identity': full.canonical(identity)}
        full.check_payload(context, disk, record, {})
        for key in ('cpu_rng', 'bank', 'optimizer', 'buffers', 'pca'):
            with self.assertRaises(ValueError):
                full.check_payload(context, {k: v for k, v in disk.items() if k != key}, record, {})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            full.check_payload(context, {**disk, 'head': {'weight': SimpleNamespace(shape=(128, 1024))}}, record, {})
        for change in ({'counter': 1999}, {'scaler': {**disk['scaler'], '_growth_tracker': 2000}},
                       {'classifier': SimpleNamespace(shape=(2004, 128))}, {'bank': SimpleNamespace(shape=(13283, 128))}):
            with self.assertRaises(ValueError):
                full.check_payload(context, {**disk, **change}, record, {})
        changed = copy.deepcopy(disk)
        changed['optimizer']['state'][100]['step'] = 1999
        with self.assertRaises(ValueError):
            full.check_payload(context, changed, record, {})

    def chain(self, root, count=2, arm='large'):
        source, code = {'checkpoint': {'path': '/source/resume.pt', 'sha256': '1' * 64},
                        'receipt': {'path': '/source/receipt.json', 'sha256': '2' * 64}}, {'trainer': '3' * 64}
        for name in full.WIDTHS:
            (root / name).mkdir(exist_ok=True)
        previous = checkpoint = previous_terminal = None
        identity = {'schema': full.SCHEMA, 'arm': arm, 'source': source, 'code': code,
                    'full_rows_sha256': 'rows', 'schedule_sha256': 'schedule'}
        for end in range(100, count * 100 + 1, 100):
            folder = root / arm / f'chunk-{end:04d}'
            folder.mkdir()
            path = folder / 'resume.pt'
            path.write_bytes(str(end).encode())
            row = {'schema': full.SCHEMA, 'pass': True, 'quality_read': False, 'arm': arm, 'width': full.WIDTHS[arm],
                   'seed': full.SEED, 'completed_step': end, 'chunk_start': end - 100, 'source': source, 'code': code,
                   'resource_policy': full.POLICY, 'reload_exact': True, 'exit_rehash_pass': True, 'identity': identity,
                   'steps': [{'step': i} for i in range(end - 99, end + 1)], 'checkpoint': full.descriptor(path),
                   'previous': previous, 'previous_terminal': previous_terminal, 'previous_checkpoint_sha256': checkpoint}
            receipt = folder / 'receipt.json'
            self.terminal_fixture(folder, row, f'full-{arm}-{end}', f'{end + (10000 if arm == "so400" else 0):032x}')
            previous, checkpoint = full.descriptor(receipt), row['checkpoint']['sha256']
            previous_terminal = self.terminal_ref(previous)
        return previous, source, code

    def test_complete_twenty_chunk_chain_and_no_native_imports(self):
        with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import):
            ref, source, code = self.chain(Path(temporary), 20)
            row = full.check_chain(ref, 2000, 'large', source, code, self.terminal_ref(ref))
            self.assertEqual(row['completed_step'], 2000)
            self.assertIsNone(full.check_chain(None, 0, 'large', source, code))

    def test_chain_rejects_metadata_and_checkpoint_tamper(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref, source, code = self.chain(root)
            for args in ((ref, 200, 'so400', source, code), (ref, 100, 'large', source, code),
                         (ref, 200, 'large', {**source, 'receipt': {}}, code), (ref, 200, 'large', source, {}),
                         (None, 200, 'large', source, code), (ref, 0, 'large', source, code)):
                with self.assertRaises(ValueError):
                    full.check_chain(*args, self.terminal_ref(ref))
            first = root / 'large/chunk-0100/receipt.json'
            original = first.read_bytes()
            first.write_bytes(original.replace(b'"pass": true', b'"pass": false'))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                full.check_chain(ref, 200, 'large', source, code, self.terminal_ref(ref))
            first.write_bytes(original)
            (root / 'large/chunk-0200/resume.pt').write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                full.check_chain(ref, 200, 'large', source, code, self.terminal_ref(ref))

    def test_chain_rejects_rehashed_wrong_link_and_missing_update(self):
        with TemporaryDirectory() as temporary:
            ref, source, code = self.chain(Path(temporary))
            row = full.read_json(ref)
            terminal_path = Path(ref['path']).parent / 'terminal.json'
            terminal = full.read_json(full.descriptor(terminal_path))
            for edit in ({'previous_checkpoint_sha256': 'f' * 64}, {'previous': None},
                         {'steps': row['steps'][1:]}, {'identity': {**row['identity'], 'arm': 'so400'}}):
                Path(ref['path']).write_text(json.dumps({**row, **edit}))
                changed = full.descriptor(Path(ref['path']))
                terminal_path.write_text(json.dumps({**terminal, 'receipt': changed}))
                with self.assertRaises(ValueError):
                    full.check_chain(changed, 200, 'large', source, code, self.terminal_ref(changed))

    def test_receipt_rejects_resource_reload_exit_quality_and_start_mutants(self):
        with TemporaryDirectory() as temporary:
            ref, source, code = self.chain(Path(temporary), 1)
            row = full.read_json(ref)
            full.validate_receipt(row, 100, 'large', source, code)
            for key, wrong in (('resource_policy', {**full.POLICY, 'swap_bytes': 1}), ('reload_exact', False),
                               ('exit_rehash_pass', False), ('quality_read', True), ('chunk_start', 1)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    full.validate_receipt({**row, key: wrong}, 100, 'large', source, code)

    def test_pinned_matched_pilot_receipts_and_execution(self):
        for arm in ('large', 'so400'):
            source = {'receipt': copy.deepcopy(score.PILOT_RECEIPTS[arm]),
                      'checkpoint': {'path': str(Path(score.PILOT_RECEIPTS[arm]['path']).parent / 'resume.pt'), 'sha256': 'f' * 64}}
            record = {'schema': 'siglip2-substrate-adaptation-v1', 'phase': 'train', 'arm': arm, 'seed': full.SEED,
                      'width': full.WIDTHS[arm], 'completed_step': 100, 'checkpoint': source['checkpoint'],
                      'identity': {'arm': arm, 'seed': full.SEED, 'config': {'hidden_size': full.WIDTHS[arm]}},
                      'pass': True, 'training_qualified': True, 'quality_read': False, 'training_state_discarded': False,
                      'strict_independent_whole_head_buffers_raw_packed_reload_exact': True, 'exit_rehash_pass': True,
                      'resource_policy': full.POLICY, 'code': {'train_siglip2_substrate_adaptation.py': full.PILOT_SHA},
                      'execution_sha256': score.PILOT_EXECUTION_SHA}
            score.validate_source_pin(source, arm, record)
            for bad in ({**source['receipt'], 'sha256': '0' * 64}, {**source['receipt'], 'path': '/other/receipt.json'},
                        score.PILOT_RECEIPTS['large' if arm == 'so400' else 'so400']):
                with self.assertRaisesRegex(ValueError, 'exact matched'):
                    score.validate_source_pin({**source, 'receipt': bad}, arm, record)
            with self.assertRaisesRegex(ValueError, 'execution'):
                score.validate_source_pin(source, arm, {**record, 'execution_sha256': '0' * 64})

    def test_both_chains_gate_before_any_native_or_official_work(self):
        for incomplete in ('large', 'so400'):
            with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import), patch.object(full, 'install_backstop'):
                root = Path(temporary)
                refs = {arm: self.chain(root, 19 if arm == incomplete else 20, arm)[0] for arm in ('large', 'so400')}
                args = SimpleNamespace(arm='large', run_root=root, output=root / 'official/large', control_terminal_sha256=None,
                                       large_terminal_sha256=self.terminal_ref(refs['large'])['sha256'],
                                       so400_terminal_sha256=self.terminal_ref(refs['so400'])['sha256'])
                with patch.object(score, 'export_role') as export, patch.object(full, 'native_start') as native:
                    with self.assertRaisesRegex(ValueError, 'both complete'):
                        score.run(args)
                    export.assert_not_called()
                    native.assert_not_called()
                    self.assertFalse((root / 'official').exists())

    def test_pair_admits_all_forty_original_terminals(self):
        with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import):
            root = Path(temporary)
            refs = {arm: self.chain(root, 20, arm)[0] for arm in ('large', 'so400')}
            shas = {arm: self.terminal_ref(ref)['sha256'] for arm, ref in refs.items()}
            reader = full.read_json
            def read(ref):
                return {} if ref['path'] == '/source/receipt.json' else reader(ref)
            with (patch.object(full, 'read_json', side_effect=read), patch.object(score, 'validate_source_pin') as pinned,
                  patch.object(full, 'admit_terminal', wraps=full.admit_terminal) as admit):
                records, terminals = score.admit_pair(root, shas, {'trainer': '3' * 64})
                self.assertEqual(admit.call_count, 40)
                self.assertEqual([call.args[1] for call in pinned.call_args_list], ['large', 'so400'])
                self.assertEqual(set(records), {'large', 'so400'})
                self.assertEqual({arm: v['sha256'] for arm, v in terminals.items()}, shas)
                # An intact checkpoint/receipt cannot excuse a failed original terminal.
                terminal_path = Path(terminals['so400']['path'])
                terminal = reader(terminals['so400'])
                terminal_path.write_text(json.dumps({**terminal, 'both_locks_held': False}))
                with self.assertRaisesRegex(ValueError, 'locks'):
                    score.admit_pair(root, {**shas, 'so400': full.sha(terminal_path)}, {'trainer': '3' * 64})

    def test_fixed_directories_reject_retry_missing_and_orphan(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.chain(root, 20, 'large')
            self.chain(root, 20, 'so400')
            self.assertEqual(full.inspect_layout(root, complete=True), {'large': 2000, 'so400': 2000})
            for name in ('chunk-2100', 'chunk-0200-retry', 'orphan'):
                extra = root / 'large' / name
                extra.mkdir()
                with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                    full.inspect_layout(root, complete=True)
                extra.rmdir()
            path = root / 'so400/chunk-0100'
            path.rename(root / 'so400/missing')
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                full.inspect_layout(root, complete=True)
            (root / 'so400/missing').rename(path)
            terminal = path / 'terminal.json'
            data = terminal.read_bytes()
            terminal.unlink()
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                full.inspect_layout(root, complete=True)
            terminal.write_bytes(data)
            (root / 'large/chunk-0100/orphan').mkdir()
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                full.inspect_layout(root, complete=True)

    def test_official_output_is_create_only_and_no_orphan_siblings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'official/large'
            score.check_score_layout(root, 'large', output)
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                score.check_score_layout(root, 'large', output)
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                score.check_score_layout(root, 'so400', root / 'official/so400')
            for name in ('receipt.json', 'terminal.json', 'query.pt', 'gallery.pt'):
                (output / name).write_bytes(b'fixture')
            score.check_score_layout(root, 'so400', root / 'official/so400')
            (root / 'official/retry').mkdir()
            with self.assertRaisesRegex(ValueError, 'INVALID/HOLD'):
                score.check_score_layout(root, 'so400', root / 'official/so400')

    def test_training_attempt_reserved_before_source_failure(self):
        with TemporaryDirectory() as temporary, patch.object(full, 'install_backstop'), patch.object(full, 'invocation', return_value={}):
            root = Path(temporary)
            for arm in full.WIDTHS:
                (root / arm).mkdir()
            output = root / 'large/chunk-0100'
            args = full.parser().parse_args(['--arm', 'large', '--source-resume', str(root / 'missing.pt'),
                '--source-sha256', 'a' * 64, '--source-receipt-sha256', 'b' * 64, '--chunk-end', '100', '--output', str(output)])
            with self.assertRaises(ValueError):
                full.run(args)
            self.assertTrue(output.is_dir())
            with self.assertRaisesRegex(ValueError, 'exclusive canonical output'):
                full.run(args)
            with self.assertRaisesRegex(ValueError, 'orphan'):
                full.inspect_layout(root)

    def test_control_rejects_arm_protocol_code_rows_schedule_mutants(self):
        protocol, code = {'protocol': 'official'}, {'code': 'pinned'}
        candidate = {'arm': 'so400', 'source': 'so400-source', 'identity': {'full_rows_sha256': 'rows', 'schedule_sha256': 'schedule'}}
        control = {'schema': score.SCHEMA, 'pass': True, 'arm': 'large', 'completed_step': 2000, 'seed': full.SEED,
                   'protocol': protocol, 'code': code, 'selection': False, 'quality_read': True, 'exit_rehash_pass': True,
                   'training': {**candidate, 'arm': 'large', 'source': 'large-source'},
                   'quality': {metric: [0.5] * 14218 for metric in score.METRICS}}
        score.validate_control(control, protocol, code, candidate)
        for change in ({'arm': 'so400'}, {'protocol': {}}, {'code': {}}, {'completed_step': 100}, {'selection': True}):
            with self.assertRaises(ValueError):
                score.validate_control({**control, **change}, protocol, code, candidate)
        for key in ('full_rows_sha256', 'schedule_sha256'):
            bad = copy.deepcopy(control)
            bad['training']['identity'][key] = 'different'
            with self.assertRaisesRegex(ValueError, 'data/schedule'):
                score.validate_control(bad, protocol, code, candidate)
        bad = copy.deepcopy(control)
        bad['quality']['per_query_r1'].pop()
        with self.assertRaisesRegex(ValueError, 'metric values'):
            score.validate_control(bad, protocol, code, candidate)

    def test_paired_candidate_minus_control_and_product_groups(self):
        # Tiny vector stand-in; never imports or executes NumPy.
        class Array(list):
            @property
            def shape(self): return (len(self),)
            def __ge__(self, other): return Array(v >= other for v in self)
            def __le__(self, other): return Array(v <= other for v in self)
            def __and__(self, other): return Array(a and b for a, b in zip(self, other, strict=True))
            def __sub__(self, other): return Array(a - b for a, b in zip(self, other, strict=True))
            def __neg__(self): return Array(-v for v in self)
            def mean(self): return sum(self) / len(self)
            def all(self): return all(self)
        fake = SimpleNamespace(asarray=Array, isfinite=lambda a: Array(math.isfinite(v) for v in a))
        def imports(name, *args, **kwargs):
            return fake if name == 'numpy' else stdlib_import(name, *args, **kwargs)
        seen = []
        def lower(delta, groups):
            seen.append((tuple(delta), tuple(groups)))
            return min(delta)
        labels = ['product-' + str(i // 3) for i in range(14218)]
        control = {'quality': {metric: [0.2] * 14218 for metric in score.METRICS}}
        quality = {metric: [0.8] * 14218 for metric in score.METRICS}
        with patch('builtins.__import__', imports):
            result = score.paired(quality, control, labels, SimpleNamespace(bootstrap_lower=lower))
        self.assertEqual(result['decision'], 'GO')
        for metric in score.METRICS:
            self.assertAlmostEqual(result['candidate_minus_control'][metric]['mean_delta'], 0.6)
        self.assertEqual(len(seen), 4)
        for index, (delta, groups) in enumerate(seen):
            self.assertEqual(groups, tuple(labels))
            self.assertAlmostEqual(delta[0], 0.6 if index % 2 == 0 else -0.6)

    def test_frozen_scoring_sources_and_final_only(self):
        for name, (digest, fn) in score.REFERENCES.items():
            self.assertEqual(full.sha(ROOT / name), digest)
            function(ROOT / name, fn)
        run = ast.unparse(function(ROOT / 'score_siglip2_substrate_official.py', 'admit_pair'))
        self.assertIn("record['completed_step'] == full.TOTAL_UPDATES", run)
        self.assertIn("full.check_chain(terminal['receipt'], full.TOTAL_UPDATES", run)
        r1 = {'mean_delta': .001, 'product_lower95': .0001, 'product_upper95': .002}
        ap = {'mean_delta': -.001, 'product_lower95': -.002, 'product_upper95': 0.}
        self.assertTrue(score.quality_gate({'per_query_r1': r1, 'per_query_ap': ap}))
        for key in ('mean_delta', 'product_lower95'):
            self.assertFalse(score.quality_gate({'per_query_r1': {**r1, key: 0.}, 'per_query_ap': ap}))

    def test_scorer_rejects_intermediate_before_native_admission(self):
        with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import), patch.object(full, 'install_backstop'):
            root = Path(temporary)
            self.chain(root, 1)
            args = SimpleNamespace(arm='large', run_root=root, large_terminal_sha256='a' * 64, so400_terminal_sha256='b' * 64,
                                   output=root / 'official/large', control_terminal_sha256=None)
            with self.assertRaisesRegex(ValueError, 'both complete'):
                score.run(args)

    def test_cli_requires_hashes_and_fixed_chunks(self):
        args = ['--arm', 'so400', '--source-resume', '/tmp/resume.pt', '--source-sha256', 'a' * 64,
                '--source-receipt-sha256', 'b' * 64, '--chunk-end', '2000', '--output', '/tmp/out']
        self.assertEqual(full.parser().parse_args(args).chunk_end, 2000)
        with contextlib.redirect_stderr(io.StringIO()):
            for changed in (args[:6] + args[8:], [*args[:9], '2001', *args[10:]],
                            ['--arm', 'full', *args[2:]]):
                with self.assertRaises(SystemExit):
                    full.parser().parse_args(changed)
        for name in ('train_siglip2_substrate_full.py', 'score_siglip2_substrate_official.py'):
            path = ROOT / name
            proc = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('--arm {large,so400}', proc.stdout)
            for mode in ('-O', '-OO'):
                proc = subprocess.run([sys.executable, '-B', '-S', mode, str(path), '--help'], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn('unoptimized', proc.stderr)
        self.assertFalse(NATIVE.intersection(sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
