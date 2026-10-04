"""Bounded stdlib metadata falsifiers; run with python3 this_file.py."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name('freeze_identity_diversity_scope.py')


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SCRIPT.is_file(), 'metadata scope implementation is missing'
        spec = importlib.util.spec_from_file_location('scope_freezer', SCRIPT)
        cls.d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.d)
        cls.sources, cls.guards = cls.d.read_sources()
        cls.baseline = cls.build()

    @classmethod
    def build(cls, **changes):
        sources = dict(cls.sources, **changes)
        return cls.d.build_scope(*(sources[k] for k in
                                  ('preflight', 'roles', 'official_partition', 'mapping')))

    def test_exact_scopes_and_official_identity(self):
        value = self.baseline
        control, candidate = value['control'], value['candidate']
        self.assertEqual((len(control['rows']), len(control['class_names'])), (6355, 1008))
        self.assertEqual((len(candidate['rows']), len(candidate['class_names'])), (6355, 2016))
        self.assertEqual(len(set(candidate['original_rows'])), 6355)
        self.assertTrue(set(control['class_names']) < set(candidate['class_names']))
        self.assertEqual(len(value['consumed_outer_products']), 1008)
        self.assertEqual(len(value['remaining_outer_products']), 985)
        self.assertFalse(value['native_eligible'])
        self.assertTrue(value['metadata_only'])
        legacy = self.sources['preflight']['fit_manifest']
        indices = self.sources['roles']['panels']['train']['original_rows']
        self.assertEqual(control['original_rows'], [legacy[i]['train_row'] for i in indices])
        self.assertNotEqual(control['original_rows'], indices)
        for arm in (control, candidate):
            self.assertEqual(arm['original_rows'], sorted(arm['original_rows']))
            self.assertEqual(arm['augmentation_ids'], arm['original_rows'])
            self.assertEqual(sorted(set(arm['targets'])), list(range(len(arm['class_names']))))
            for row, target in zip(arm['rows'], arm['targets']):
                self.assertEqual(row['product'], arm['class_names'][target])
                self.assertEqual(row['augmented_rng_seed'], 179081 + row['original_train_row'])
        singles = {r['product'] for r in control['rows']
                   if control['coverage']['class_depths'][r['product']] == 1}
        self.assertEqual(len(singles), 12)
        self.assertEqual({p for p, n in candidate['coverage']['class_depths'].items() if n == 1}, singles)
        self.assertTrue(all(n >= 2 for p, n in candidate['coverage']['class_depths'].items() if p not in singles))
        excluded = value['exclusions']
        for arm in (control, candidate):
            self.assertFalse(set(arm['class_names']) & set(excluded['products']))
            self.assertFalse({r['relative_path'] for r in arm['rows']} & set(excluded['paths']))
            self.assertFalse({r['image_sha256'] for r in arm['rows']} & set(excluded['image_sha256']))
        self.assertEqual((len(value['shared_roles']['selection']['class_names']),
                          len(value['shared_roles']['validation']['class_names'])), (498, 498))
        for arm, histogram in ((control, {'8': 880, '9': 128}),
                               (candidate, {'4': 1888, '5': 128})):
            for seed in ('179061', '179069'):
                summary = arm['expected_anchor_revisits'][seed]
                self.assertEqual(summary['class_visit_histogram'], histogram)
                self.assertFalse(summary['trainer_qualified'])
                self.assertEqual(summary['seed_class_assignment'], 'pending trainer')

    def test_deterministic_scope_and_outer_order(self):
        value = self.build()
        self.assertEqual(value, self.baseline)
        preflight = dict(self.sources['preflight'])
        preflight['held_manifest'] = list(reversed(preflight['held_manifest']))
        self.assertEqual(self.build(preflight=preflight), self.baseline)
        self.assertEqual(self.d.domain_hash('outer-product', ['a']),
                         '62d319720ff9c4aae5be3300cecb466241b960486d851177dc2a6d75e255617e')

    def test_partition_path_and_mapping_mismatch(self):
        text = self.sources['official_partition'].replace(
            'img/WOMEN/Dresses/id_00000002/02_1_front.jpg',
            'img/WOMEN/Dresses/id_00000002/wrong.jpg', 1)
        with self.assertRaises(ValueError):
            self.build(official_partition=text)
        mapping = dict(self.sources['mapping'], partition_sha256='0' * 64)
        with self.assertRaises(ValueError):
            self.build(mapping=mapping)

    def test_fit_ordinal_cannot_replace_official_train_ordinal(self):
        preflight = copy.deepcopy(self.sources['preflight'])
        self.assertEqual(preflight['fit_manifest'][4]['train_row'], 9)
        preflight['fit_manifest'][4]['train_row'] = 4
        with self.assertRaisesRegex(ValueError, 'official TRAIN ordinal'):
            self.build(preflight=preflight)

    def test_duplicate_and_wrong_target_fail_closed(self):
        for kind in ('row', 'path', 'sha', 'target'):
            with self.subTest(kind=kind):
                p = copy.deepcopy(self.sources['preflight'])
                if kind == 'row':
                    p['fit_manifest'][1] = dict(p['fit_manifest'][0])
                elif kind == 'path':
                    p['held_manifest'][1]['relative_path'] = p['held_manifest'][0]['relative_path']
                elif kind == 'sha':
                    p['held_manifest'][1]['image_sha256'] = p['fit_manifest'][0]['image_sha256']
                else:
                    p['target'][0] = 1
                with self.assertRaises(ValueError):
                    self.build(preflight=p)

    def test_role_overlap_and_query_gallery_mismatch(self):
        for kind in ('row', 'class', 'query'):
            with self.subTest(kind=kind):
                roles = copy.deepcopy(self.sources['roles'])
                if kind == 'row':
                    roles['panels']['selection']['original_rows'][0] = 0
                elif kind == 'class':
                    roles['panels']['selection']['original_class_ids'][0] = 0
                else:
                    roles['panels']['selection']['query'][0] = roles['panels']['selection']['gallery'][0]
                with self.assertRaises(ValueError):
                    self.build(roles=roles)

    def test_round_robin_capacity_and_final_partial_round(self):
        members = tiny_members()
        result = self.d.allocate_rows(members, {'a'}, 6)
        self.assertEqual([r['original_train_row'] for r in result], [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(result), 6)
        self.assertEqual({r['product'] for r in result}, {'a', 'b', 'c'})
        # A second partial round must skip exhausted a, with no replacement rows.
        self.assertEqual(len(self.d.allocate_rows(members, {'a'}, 7)), 7)
        for budget in (4, 8):
            with self.assertRaises(ValueError):
                self.d.allocate_rows(members, {'a'}, budget)

    def test_singleton_and_capacity_falsifiers(self):
        for members, singles in ((tiny_members(), set()), (tiny_members(), {'b'})):
            with self.assertRaises(ValueError):
                self.d.allocate_rows(members, singles, 6)
        members = tiny_members()
        members['b'][1] = dict(members['b'][0])
        with self.assertRaises(ValueError):
            self.d.allocate_rows(members, {'a'}, 6)

    def test_pinned_input_source_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'roles.json'
            path.write_bytes(self.d.DEFAULT_PATHS['roles'].read_bytes() + b' ')
            output = Path(temp) / 'scope.json'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.d.freeze_scope(output, {'roles': path})
            self.assertFalse(output.exists())

    def test_exclusive_output_and_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'scope.json'
            command = [sys.executable, str(SCRIPT), '--output', str(output)]
            terminal = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(terminal.returncode, 0, terminal.stderr)
            original = output.read_bytes()
            value = json.loads(original)
            self.assertTrue(value['exit_rehash_pass'])
            self.assertEqual(value['control'], self.baseline['control'])
            self.assertEqual(value['candidate'], self.baseline['candidate'])
            with self.assertRaises(FileExistsError):
                self.d.freeze_scope(output)
            self.assertEqual(output.read_bytes(), original)
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(rejected.returncode, 2)
            self.assertEqual(output.read_bytes(), original)
            link = Path(temp) / 'link.json'
            link.symlink_to(output)
            with self.assertRaises(FileExistsError):
                self.d.freeze_scope(link)
            self.assertEqual(output.read_bytes(), original)

    def test_freeze_exit_tamper_invalidates_own_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'roles.json'
            path.write_bytes(self.d.DEFAULT_PATHS['roles'].read_bytes())
            output = Path(temp) / 'scope.json'
            real_fsync = self.d.os.fsync
            tampered = False

            def fsync_then_tamper(fd):
                nonlocal tampered
                real_fsync(fd)
                if output.exists() and not tampered:
                    with path.open('ab') as stream:
                        stream.write(b' ')
                    tampered = True

            # The real exclusive file is written/fsynced; only force the input
            # race at that boundary. Assertions cover rejection and cleanup.
            with mock.patch.object(self.d.os, 'fsync', side_effect=fsync_then_tamper):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    self.d.freeze_scope(output, {'roles': path})
            self.assertTrue(tampered)
            self.assertFalse(output.exists())

    def test_exit_guard_detects_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.json'
            path.write_bytes(b'original')
            guards = {str(path): hashlib.sha256(b'original').hexdigest()}
            self.d.recheck(guards)
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.d.recheck(guards)


def tiny_members():
    return {p: [{'product': p, 'original_train_row': i,
                 'relative_path': f'Img/img/{p}/{i}.jpg', 'image_sha256': f'{i + 1:064x}'}
                for i in ids] for p, ids in (('a', [0]), ('b', [1, 2, 3]), ('c', [4, 5, 6]))}


if __name__ == '__main__':
    unittest.main()
