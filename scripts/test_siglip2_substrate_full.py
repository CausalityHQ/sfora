#!/usr/bin/env python3
"""Stdlib-only falsifiers. No torch/numpy/model/image execution or qualification."""
import ast
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
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

    def chain(self, root, count=2):
        source, code = {'checkpoint': {'path': '/source/resume.pt', 'sha256': '1' * 64},
                        'receipt': {'path': '/source/receipt.json', 'sha256': '2' * 64}}, {'trainer': '3' * 64}
        previous = checkpoint = None
        identity = {'schema': full.SCHEMA, 'arm': 'large', 'source': source, 'code': code}
        for end in range(100, count * 100 + 1, 100):
            folder = root / str(end)
            folder.mkdir()
            path = folder / 'resume.pt'
            path.write_bytes(str(end).encode())
            row = {'schema': full.SCHEMA, 'pass': True, 'quality_read': False, 'arm': 'large', 'width': 1024,
                   'seed': full.SEED, 'completed_step': end, 'chunk_start': end - 100, 'source': source, 'code': code,
                   'resource_policy': full.POLICY, 'reload_exact': True, 'exit_rehash_pass': True, 'identity': identity,
                   'steps': [{'step': i} for i in range(end - 99, end + 1)], 'checkpoint': full.descriptor(path),
                   'previous': previous, 'previous_checkpoint_sha256': checkpoint}
            receipt = folder / 'receipt.json'
            full.publish(receipt, row)
            previous, checkpoint = full.descriptor(receipt), row['checkpoint']['sha256']
        return previous, source, code

    def test_complete_twenty_chunk_chain_and_no_native_imports(self):
        with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import):
            ref, source, code = self.chain(Path(temporary), 20)
            row = full.check_chain(ref, 2000, 'large', source, code)
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
                    full.check_chain(*args)
            first = root / '100/receipt.json'
            original = first.read_bytes()
            first.write_bytes(original.replace(b'"pass": true', b'"pass": false'))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                full.check_chain(ref, 200, 'large', source, code)
            first.write_bytes(original)
            (root / '200/resume.pt').write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                full.check_chain(ref, 200, 'large', source, code)

    def test_chain_rejects_rehashed_wrong_link_and_missing_update(self):
        with TemporaryDirectory() as temporary:
            ref, source, code = self.chain(Path(temporary))
            row = full.read_json(ref)
            for edit in ({'previous_checkpoint_sha256': 'f' * 64}, {'previous': None},
                         {'steps': row['steps'][1:]}, {'identity': {**row['identity'], 'arm': 'so400'}}):
                Path(ref['path']).write_text(json.dumps({**row, **edit}))
                changed = full.descriptor(Path(ref['path']))
                with self.assertRaises(ValueError):
                    full.check_chain(changed, 200, 'large', source, code)

    def test_frozen_scoring_sources_and_final_only(self):
        for name, (digest, fn) in score.REFERENCES.items():
            self.assertEqual(full.sha(ROOT / name), digest)
            function(ROOT / name, fn)
        run = ast.unparse(function(ROOT / 'score_siglip2_substrate_official.py', 'run'))
        self.assertIn("record['completed_step'] == full.TOTAL_UPDATES", run)
        self.assertIn('full.check_chain(ref, full.TOTAL_UPDATES', run)
        r1 = {'mean_delta': .001, 'product_lower95': .0001, 'product_upper95': .002}
        ap = {'mean_delta': -.001, 'product_lower95': -.002, 'product_upper95': 0.}
        self.assertTrue(score.quality_gate({'per_query_r1': r1, 'per_query_ap': ap}))
        for key in ('mean_delta', 'product_lower95'):
            self.assertFalse(score.quality_gate({'per_query_r1': {**r1, key: 0.}, 'per_query_ap': ap}))

    def test_scorer_rejects_intermediate_before_native_admission(self):
        with TemporaryDirectory() as temporary, patch('builtins.__import__', stdlib_import), patch.object(full, 'install_backstop'):
            root = Path(temporary)
            checkpoint = root / 'resume.pt'
            checkpoint.write_bytes(b'no native checkpoint')
            full.publish(root / 'receipt.json', {'completed_step': 100, 'checkpoint': full.descriptor(checkpoint)})
            args = SimpleNamespace(arm='large', checkpoint=checkpoint, checkpoint_sha256=full.sha(checkpoint),
                                   receipt_sha256=full.sha(root / 'receipt.json'), output=root / 'output', control=None, control_sha256=None)
            with self.assertRaisesRegex(ValueError, 'final step2000'):
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
