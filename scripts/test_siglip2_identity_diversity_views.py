#!/usr/bin/env python3
"""Stdlib scope/provenance falsifiers; native/resource/quality admission unrun.

Run once after the root grants the nonoverlapping <=90s/1GiB slot:
python3 -B -S scripts/test_siglip2_identity_diversity_views.py
One named unittest may be run as a focused tiny check during implementation.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCRIPTS = Path(__file__).absolute().parent
ORIGINAL_SHA = 'e5e98f9bc85680cab013d53752e7fa140da9d5e7f7ecd4537413b29b1047f65e'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError, TypeError):
        return
    raise AssertionError('accepted ' + message)


def function(source, name):
    return ast.get_source_segment(source, next(n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name))


def expected_native_flow(original):
    """Explicit receipt/counter additions; every other original AST node retained."""
    replacements = [
        ("    mapping, witnesses, calibration = {v: [] for v in VIEWS}, {}, {}",
         "    mapping, witnesses, calibration = {v: [] for v in VIEWS}, {}, {}\n"
         "    preparation_counters = {'images_decoded': 0, 'autocast_forwards': 0, 'fp32_calibration_forwards': 0}"),
        ("                    original = opened.convert('RGB')",
         "                    original = opened.convert('RGB')\n                preparation_counters['images_decoded'] += 1"),
        ("'train_row': row['train_row'], 'target': manifest['targets'][i]",
         "'train_row': row['train_row'], 'original_fit_index': row['original_fit_index'],\n"
         "                              'global_product_id': row['global_product_id'], 'augmentation_id': row['augmentation_id'],\n"
         "                              'target': manifest['targets'][i]"),
        ("        with torch.autocast('cuda', dtype=torch.float16):\n            raw",
         "        preparation_counters['autocast_forwards'] += 1\n"
         "        with torch.autocast('cuda', dtype=torch.float16):\n            raw"),
        ("        fp32 = model(pixel_values=pixels).pooler_output.float()",
         "        preparation_counters['fp32_calibration_forwards'] += 1\n"
         "        fp32 = model(pixel_values=pixels).pooler_output.float()"),
        ("'phase': 'export', 'pass': True, 'exported': True",
         "'phase': 'export', 'pass': True, 'exported': True, 'arm': args.arm"),
        ("'classes': CLASSES, 'batch_sizes_per_view'", "'classes': len(manifest['class_names']), 'batch_sizes_per_view'"),
        ("'views': list(VIEWS), 'optimizer_updates': 0",
         "'views': list(VIEWS), 'optimizer_updates': 0,\n                           'preparation': preparation_counters"),
        ("'exit_rehash_pass': True, 'quality_read': False, 'training_qualified': False",
         "'exit_rehash_pass': True, 'quality_read': False, 'training_qualified': False, 'native_training_eligible': False"),
    ]
    result = function(original, 'export')
    for before, after in replacements:
        check(result.count(before) == 1, 'explicit native correspondence replacement: ' + before)
        result = result.replace(before, after)
    return result


class CPUState:
    def __init__(self, value=17):
        self.value, self.default_generator = value, self

    def clone(self):
        return CPUState(self.value)

    def get_rng_state(self):
        return self.clone()

    def manual_seed(self, value):
        self.value = value

    @contextmanager
    def fork_rng(self, *, devices):
        check(devices == [], 'CPU-only view RNG')
        saved = self.value
        try:
            yield
        finally:
            self.value = saved


class Falsifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load('identity_diversity_views', SCRIPTS / 'export_siglip2_identity_diversity_views.py')
        evidence = SCRIPTS.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
        cls.paths = {
            'preflight': evidence / 'pe-augmented-100-v1/preflight-v2.json',
            'roles': evidence / 'genuine-view-v1/export-source-v1/partition.json',
            'official_partition': evidence / 'identity-diversity-v1/original-partition.txt',
            'mapping': evidence / 'identity-diversity-v1/metadata-mapping.json',
            'plan': SCRIPTS.parent / 'docs/inshop_identity_diversity_plan_2026-10-04.json',
            'schedule_source': SCRIPTS / 'train_siglip2_genuine_views.py',
            'schedule_consumer': SCRIPTS / 'train_siglip2_compact_ranking.py',
        }
        cls.preflight, cls.roles, cls.mapping = [json.loads(cls.paths[k].read_text())
            for k in ('preflight', 'roles', 'mapping')]
        cls.text = cls.paths['official_partition'].read_text()
        cls.fit = json.loads((evidence / 'late-dense-v1/native256-fit-manifest-v1.json').read_text())
        cls.scope = cls.d.reconstruct_scope(cls.preflight, cls.roles, cls.text, cls.mapping)
        cls.scope.update(input_files={}, source_file={}, exit_rehash_pass=True)

    def manifest(self, scope=None, arm='candidate', preflight=None, roles=None, text=None):
        return self.d.scope_manifest(self.scope if scope is None else scope,
            self.preflight if preflight is None else preflight,
            self.roles if roles is None else roles, self.text if text is None else text,
            self.mapping, self.fit, arm)

    def test_original_source_and_complete_native_flow(self):
        original = (SCRIPTS / 'export_siglip2_genuine_views.py').read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), ORIGINAL_SHA)
        current = (SCRIPTS / 'export_siglip2_identity_diversity_views.py').read_text()
        self.assertEqual(ast.dump(ast.parse(function(current, 'export'))),
                         ast.dump(ast.parse(expected_native_flow(original))))
        for name in ('require', 'canonical', 'sha', 'object_sha', 'strict_json', 'read_json',
                     'file_json', 'closure', 'selected_manifest', 'image_rows_node',
                     'launch_descriptor', 'invocation', 'write_json', 'isolated_view'):
            self.assertEqual(ast.dump(ast.parse(function(current, name))),
                             ast.dump(ast.parse(function(original, name))), name)
        # These unchanged pins preserve the separate original source and reference.
        old = load('original_genuine_views', SCRIPTS / 'export_siglip2_genuine_views.py')
        for name in ('REF_FILES', 'ORIGINAL_REF_ROOT', 'REF_EXECUTION_SHA', 'ORIGINAL_AUTHORITY_SHA',
                     'PARTITION_SHA', 'MANIFEST_SHA', 'OLD_CACHE_SHA', 'SOURCE_CHECKPOINT_SHA',
                     'IMAGE_ROWS_AST_SHA', 'ROWS', 'WIDTH', 'BATCH', 'VIEWS'):
            self.assertEqual(getattr(self.d, name), getattr(old, name), name)
        expected = expected_native_flow(original)
        correspondence = {
            'authority': [("    selected = selected_manifest(partition, prior['fit'])\n"
                           "    selected['resolved_paths'] = [str(prior['all_images'][r]) for r in selected['original_rows']]",
                           "    selected_manifest(partition, prior['fit'])  # Unchanged historical role/FIT admission.\n"
                           "    selected = admit_scope(launch, prior, partition, args.arm, guards)")],
            'rehash': [("    selected = selected_manifest(file_json(context['launch']['partition'], {}), context['prior']['fit'])\n"
                        "    selected['resolved_paths'] = [str(context['prior']['all_images'][r]) for r in selected['original_rows']]",
                        "    partition = file_json(context['launch']['partition'], {})\n"
                        "    selected_manifest(partition, context['prior']['fit'])\n"
                        "    selected = admit_scope(context['launch'], context['prior'], partition, context['args'].arm, {})")],
            'admit_terminal': [
                ("proof['schema'] == SCHEMA and proof['phase'] == phase and proof['pass'] is True and",
                 "proof['schema'] == SCHEMA and proof['phase'] == phase and proof['pass'] is True and\n"
                 "            proof['arm'] == context['args'].arm and"),
                ('export_siglip2_genuine_views.py', 'export_siglip2_identity_diversity_views.py'),
                ("'--phase', phase, '--output'", "'--arm', context['args'].arm, '--phase', phase, '--output'")],
            'startup': [
                ('time.perf_counter() - started < 120', 'time.perf_counter() - started < 500'),
                ('complete startup exceeds120', 'complete startup exceeds500'),
                ("'phase': 'startup', 'pass': True, 'binding': binding(context)",
                 "'phase': 'startup', 'pass': True, 'arm': context['args'].arm, 'binding': binding(context)"),
                ("'native_imported': False, 'model_constructed': False, 'exported': False,",
                 "'native_imported': False, 'model_constructed': False, 'exported': False,\n"
                 "              'quality_read': False, 'training_qualified': False, 'native_training_eligible': False,")],
            'limits': [('time.perf_counter() - started < 300', 'time.perf_counter() - started < 900'),
                       ('complete genuine export exceeds300', 'complete diversity export exceeds900')],
        }
        for name, replacements in correspondence.items():
            adapted = function(original, name)
            for before, after in replacements:
                self.assertIn(before, adapted, name)
                adapted = adapted.replace(before, after)
            self.assertEqual(ast.dump(ast.parse(function(current, name))), ast.dump(ast.parse(adapted)), name)
        mutants = [
            ('model.load_state_dict(saved[\'vision\'], strict=True)', 'model.load_state_dict(saved[\'vision\'], strict=False)'),
            ("value.copy_(saved['buffers'][name])", 'pass'),
            ("model = source.construct(saved['config'], prior)", "model = source.fresh_source(prior)[0]"),
            ('torch.equal(rng, torch.random.get_rng_state())', 'True'),
            ('source.numerical_flags() == flags', 'True'),
            ('rehash(context)', 'pass'),
            ('return raw, F.normalize(raw, dim=1)', 'return raw, raw'),
            ('value.detach().cpu()', 'value.detach()'),
            ("extract.sha(path) == row['image_sha256']", 'True'),
            ('os.fsync(stream.fileno())', 'pass'),
            ("calibrate(view) == calibration[view]", 'True'),
        ]
        for before, after in mutants:
            self.assertIn(before, expected)
            self.assertNotEqual(ast.dump(ast.parse(expected.replace(before, after))),
                                ast.dump(ast.parse(function(current, 'export'))), before)

    def test_scope_correspondence_and_namespaces(self):
        for name, classes in (('control', 1008), ('candidate', 2016)):
            selected = self.manifest(arm=name)
            self.assertEqual(len(selected['rows']), 6355)
            self.assertEqual(len(selected['class_names']), classes)
            self.assertEqual(selected['augmentation_ids'], selected['original_rows'])
            self.assertEqual(selected['original_rows'], sorted(set(selected['original_rows'])))
            for ordinal, row in enumerate(selected['rows']):
                self.assertEqual(row['train_row'], row['augmentation_id'])
                self.assertEqual(row['augmented_rng_seed'], 179081 + row['train_row'])
                self.assertEqual(selected['class_names'][selected['targets'][ordinal]], row['product'])
                self.assertEqual(self.scope['global_class_names'][row['global_product_id']], row['product'])
        control = self.manifest(arm='control')
        original = self.d.selected_manifest(self.roles, self.fit)
        self.assertEqual(control['original_fit_indices'], original['original_rows'])
        self.assertEqual([r['relative_path'] for r in control['rows']],
                         [r['relative_path'] for r in original['rows']])
        self.assertTrue(any(r['original_fit_index'] is None for r in self.manifest()['rows']))
        self.assertTrue(any(r['original_fit_index'] != r['train_row'] for r in control['rows']))
        self.assertEqual(set(control['class_names']) | set(self.scope['consumed_outer_products']),
                         set(self.manifest()['class_names']))

    def test_scope_tamper_leak_and_ordinal_substitution(self):
        for mutation in ('target', 'globalid', 'fitordinal', 'augmentation', 'rowsha', 'held',
                         'partial', 'support', 'scope_swap', 'remaining', 'quality', 'extra'):
            altered = copy.deepcopy(self.scope)
            arm, row = altered['candidate'], altered['candidate']['rows'][0]
            if mutation == 'target':
                arm['targets'][0] = (arm['targets'][0] + 1) % 2016
            elif mutation == 'globalid':
                row['global_product_id'] += 1
            elif mutation == 'fitordinal':
                row = next(r for r in arm['rows'] if r['original_fit_index'] != r['original_train_row'])
                row['original_train_row'] = row['original_fit_index']
            elif mutation == 'augmentation':
                row['augmentation_id'] += 1
            elif mutation == 'rowsha':
                row['image_sha256'] = '0' * 64
            elif mutation == 'held':
                row.update(altered['shared_roles']['selection']['rows'][0])
            elif mutation == 'partial':
                arm['rows'].pop()
            elif mutation == 'support':
                arm['class_depth_counts']['1'] += 1
            elif mutation == 'scope_swap':
                altered['control'], altered['candidate'] = altered['candidate'], altered['control']
            elif mutation == 'remaining':
                altered['remaining_outer_products'].append(altered['consumed_outer_products'][0])
            elif mutation == 'quality':
                altered['native_eligible'] = True
            else:
                altered['extra'] = True
            # Updating the digest cannot make mutated inventory self-authoritative.
            arm['scope_sha256'] = self.d.domain_hash('scope', {k: v for k, v in arm.items() if k != 'scope_sha256'})
            rejects(lambda: self.manifest(altered), mutation)

    def test_historical_inventory_and_roles_mutants(self):
        for kind in ('official', 'fitordinal', 'outerduplicate', 'roles', 'query'):
            preflight, roles, text = copy.deepcopy(self.preflight), copy.deepcopy(self.roles), self.text
            if kind == 'official':
                text = text.replace('id_00000002 train', 'id_00000003 train', 1)
            elif kind == 'fitordinal':
                i = next(i for i, r in enumerate(preflight['fit_manifest']) if i != r['train_row'])
                preflight['fit_manifest'][i]['train_row'] = i
            elif kind == 'outerduplicate':
                preflight['held_manifest'][1] = preflight['held_manifest'][0]
            elif kind == 'roles':
                roles['panels']['train']['original_rows'][0] = roles['panels']['validation']['original_rows'][0]
            else:
                roles['panels']['selection']['query'][0] = roles['panels']['selection']['gallery'][0]
            rejects(lambda: self.manifest(preflight=preflight, roles=roles, text=text), kind)

    def test_file_pins_strict_json_and_closure(self):
        for name, path in self.paths.items():
            self.assertEqual(self.d.sha(path), self.d.PINS[name], name)
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            rejects(lambda: self.d.strict_json(raw), raw)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = self.d.FILES
            for name in names:
                (root / name).write_bytes((SCRIPTS / name).read_bytes())
            code = {name: self.d.sha(root / name) for name in names}
            digest = write(root / 'execution.json', code)
            self.assertEqual(self.d.closure(root, digest, names, {}), code)
            (root / next(iter(names))).write_bytes(b'tampered')
            rejects(lambda: self.d.closure(root, digest, names, {}), 'source tamper')
            for mutated in (dict(code, extra='0' * 64), {}):
                other = write(root / 'execution.json', mutated)
                rejects(lambda: self.d.closure(root, other, names, {}), 'nonexact closure')
            artifact = root / 'guard.json'
            descriptor = {'path': str(artifact), 'sha256': write(artifact, {'metadata': True})}
            guards = {}
            self.d.file_json(descriptor, guards)
            artifact.write_bytes(b'changed')
            rejects(lambda: self.d.guard_file(descriptor, guards), 'metadata input changed')
            rejects(lambda: self.d.file_json(descriptor, {}), 'scope JSON changed')

    def test_selected_path_bytes_and_exit_recheck(self):
        # Synthetic byte files, never actual image decode or native imports.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'Img').mkdir()
            path = root / 'Img' / 'synthetic.bin'
            path.write_bytes(b'selected synthetic bytes')
            manifest = {'rows': [{'relative_path': 'Img/synthetic.bin', 'image_sha256': self.d.sha(path)}]}
            guards = {}
            self.assertEqual(self.d.resolve_scope_images(manifest, root, guards), [str(path)])
            path.write_bytes(b'tampered selected bytes')
            rejects(lambda: self.d.resolve_scope_images(manifest, root, {}), 'selected image exit hash')
            for relative in ('../synthetic.bin', '/absolute.bin', 'Img/../synthetic.bin'):
                altered = copy.deepcopy(manifest)
                altered['rows'][0]['relative_path'] = relative
                rejects(lambda: self.d.resolve_scope_images(altered, root, {}), 'path escape')
            link = root / 'Img' / 'alias.bin'
            link.symlink_to(path)
            altered = copy.deepcopy(manifest)
            altered['rows'][0]['relative_path'] = 'Img/alias.bin'
            rejects(lambda: self.d.resolve_scope_images(altered, root, {}), 'resolved alias')

    def test_selected_symlink_paths_preserve_containment_hash_and_guards(self):
        # Reproduce Img -> img with synthetic bytes, without decoding images.
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'dataset'
            directory = root / 'img' / 'img'
            directory.mkdir(parents=True)
            (root / 'Img').symlink_to('img', target_is_directory=True)
            path = directory / 'synthetic.bin'
            raw = b'selected synthetic bytes'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            manifest = {'rows': [{'relative_path': 'Img/img/synthetic.bin', 'image_sha256': digest}]}
            frozen = copy.deepcopy(manifest)
            guards = {}
            self.assertEqual(self.d.resolve_scope_images(manifest, root, guards), [str(path)])
            self.assertEqual(guards, {str(path): digest})
            self.assertEqual(manifest, frozen)
            rejects(lambda: self.d.canonical(root / 'Img' / 'img' / 'synthetic.bin'),
                    'noncanonical generic FILE path')
            rejects(lambda: self.d.resolve_scope_images(manifest, root / 'Img', {}),
                    'noncanonical dataset root')

            alias = directory / 'alias.bin'
            alias.symlink_to('synthetic.bin')
            duplicates = {'rows': manifest['rows'] + [dict(manifest['rows'][0], relative_path='Img/img/alias.bin')]}
            with self.assertRaisesRegex(ValueError, 'selected scope resolved paths collide'):
                self.d.resolve_scope_images(duplicates, root, {})

            outside = root.parent / 'outside.bin'
            outside.write_bytes(raw)
            (directory / 'escape.bin').symlink_to(outside)
            (directory / 'dangling.bin').symlink_to('missing.bin')
            for name, message in (('escape.bin', 'scope image escaped canonical dataset root'),
                                  ('dangling.bin', 'canonical regular file required')):
                altered = {'rows': [dict(manifest['rows'][0], relative_path='Img/img/' + name)]}
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    self.d.resolve_scope_images(altered, root, {})
            altered = {'rows': [dict(manifest['rows'][0], image_sha256='0' * 64)]}
            with self.assertRaisesRegex(ValueError, 'scope input FILE SHA256 differs'):
                self.d.resolve_scope_images(altered, root, {})
            path.write_bytes(b'tampered selected bytes')
            with self.assertRaisesRegex(ValueError, 'scope input FILE SHA256 differs'):
                self.d.resolve_scope_images(manifest, root, {})
            for guarded_path, expected in guards.items():
                with self.assertRaisesRegex(ValueError, 'scope input FILE SHA256 differs'):
                    self.d.guard_file({'path': guarded_path, 'sha256': expected}, {})

    def test_original_provenance_and_staged_file_binding(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {k: {'path': str(p), 'sha256': self.d.PINS[k]} for k, p in self.paths.items()}
            staged = {}
            for key, descriptor in inputs.items():
                destination = root / (key + Path(descriptor['path']).suffix)
                destination.write_bytes(Path(descriptor['path']).read_bytes())
                staged[key] = {'path': str(destination), 'sha256': descriptor['sha256']}
            freezer = root / 'freeze_identity_diversity_scope.py'
            freezer.write_bytes(b'# synthetic authenticated metadata source\n')
            staged['freezer'] = {'path': str(freezer), 'sha256': self.d.sha(freezer)}
            scope = copy.deepcopy(self.scope)
            scope['input_files'] = inputs
            scope['source_file'] = {'path': '/original/provenance/freeze_identity_diversity_scope.py',
                                    'sha256': staged['freezer']['sha256']}
            scope_path = root / 'scope.json'
            scope_sha = write(scope_path, scope)
            launch = {'scope': {'path': str(scope_path), 'sha256': scope_sha},
                      'metadata': staged, 'partition': staged['roles']}
            self.assertEqual(self.d.admit_scope_metadata(launch, {'fit': self.fit}, self.roles, 'candidate', {}),
                             self.manifest())
            for field in ('freezer', 'plan', 'roles', 'scope'):
                altered = copy.deepcopy(launch)
                descriptor = altered['scope'] if field == 'scope' else altered['metadata'][field]
                descriptor['sha256'] = '0' * 64
                rejects(lambda: self.d.admit_scope_metadata(altered, {'fit': self.fit}, self.roles, 'candidate', {}), field)
            original = copy.deepcopy(scope)
            scope['source_file']['sha256'] = '0' * 64
            launch['scope']['sha256'] = write(scope_path, scope)
            rejects(lambda: self.d.admit_scope_metadata(launch, {'fit': self.fit}, self.roles, 'candidate', {}), 'freezer provenance SHA')
            scope = original
            scope['input_files']['preflight']['sha256'] = '0' * 64
            launch['scope']['sha256'] = write(scope_path, scope)
            rejects(lambda: self.d.admit_scope_metadata(launch, {'fit': self.fit}, self.roles, 'candidate', {}), 'original input SHA')

    def test_view_mapping_and_rng_namespace(self):
        manifest = self.manifest()
        manifest['resolved_paths'] = [str(Path(self.fit['dataset_root']) / r['relative_path']) for r in manifest['rows']]
        rgb = {'mode': 'RGB', 'size': [300, 400], 'sha256': 'b' * 64}
        pixel = {'shape': [3, 256, 256], 'dtype': 'torch.float32', 'sha256': 'a' * 64}
        views = {view: [dict(view=view, ordinal=i, original_row=row['train_row'], train_row=row['train_row'],
                    original_fit_index=row['original_fit_index'], augmentation_id=row['augmentation_id'],
                    global_product_id=row['global_product_id'], target=manifest['targets'][i],
                    path=manifest['resolved_paths'][i], relative_path=row['relative_path'],
                    image_sha256=row['image_sha256'], original_rgb=rgb,
                    rgb=rgb if view == 'canonical' else {**rgb, 'size': [256, 256]},
                    rng_seed=None if view == 'canonical' else 179081 + row['train_row'], pixels=pixel)
                for i, row in enumerate(manifest['rows'])] for view in self.d.VIEWS}
        self.d.validate_views(manifest, views)
        for field in ('augmentation_id', 'original_fit_index', 'global_product_id', 'rng_seed', 'train_row', 'target'):
            changed = copy.deepcopy(views)
            fact = changed['augmented'][0]
            fact[field] = -1
            rejects(lambda: self.d.validate_views(manifest, changed), field)
        state = CPUState()
        fake = SimpleNamespace(random=state, equal=lambda a, b: a.value == b.value)
        def transform(image):
            value = state.value
            state.value += 99
            return image, value
        self.assertEqual(self.d.isolated_view(fake, 'synthetic', transform, 73), ('synthetic', 179154))
        self.assertEqual(state.value, 17)
        def fail(image):
            state.value += 99
            raise OSError('synthetic transform failure')
        rejects(lambda: self.d.isolated_view(fake, 'synthetic', fail, 73), 'failed view RNG')
        self.assertEqual(state.value, 17)
        rejects(lambda: self.d.isolated_view(fake, 'synthetic', transform, True), 'bool ordinal')

    def test_launch_scope_arm_and_terminal(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = {'schema': self.d.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
                'reference': {'root': str(self.d.ORIGINAL_REF_ROOT), 'execution_sha256': self.d.REF_EXECUTION_SHA,
                    'authority': {'path': str(self.d.ORIGINAL_REF_ROOT / 'authority.json'), 'sha256': self.d.ORIGINAL_AUTHORITY_SHA}},
                'partition': {'path': str(self.paths['roles']), 'sha256': self.d.PARTITION_SHA},
                'image_rows': {'path': str(SCRIPTS / 'train_sop_siglip2_compact.py'), 'sha256': self.d.sha(SCRIPTS / 'train_sop_siglip2_compact.py')},
                'scope': {'path': str(root / 'scope.json'), 'sha256': '1' * 64},
                'metadata': {k: {'path': str(self.paths[k]), 'sha256': self.d.PINS[k]}
                             for k in self.d.PINS},
                'startup_policy': self.d.STARTUP_POLICY, 'export_policy': self.d.EXPORT_POLICY}
            launch['metadata']['freezer'] = {'path': str(root / 'freeze_identity_diversity_scope.py'), 'sha256': '2' * 64}
            self.d.check_launch(launch, 'd' * 64)
            for field in ('scope', 'metadata', 'startup_policy', 'export_policy', 'reference'):
                changed = copy.deepcopy(launch)
                if field == 'metadata':
                    changed[field]['official_partition']['sha256'] = '0' * 64
                elif field == 'reference':
                    changed[field]['execution_sha256'] = '0' * 64
                elif field == 'scope':
                    changed[field]['extra'] = True
                else:
                    changed[field]['seconds'] += 1
                rejects(lambda: self.d.check_launch(changed, 'd' * 64), field)
            terminal_checks(self.d, root)

    def test_exclusive_outputs_and_no_old_cache(self):
        extract = load('original_extract', SCRIPTS / 'extract_siglip2_vision_source.py')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'canonical.npy'
            with extract.exclusive(cache) as stream:
                stream.write(b'synthetic output bytes')
            original = cache.read_bytes()
            rejects(lambda: extract.exclusive(cache), 'existing cache overwrite')
            self.assertEqual(cache.read_bytes(), original)
            dangling = root / 'augmented.npy'
            dangling.symlink_to(root / 'absent')
            rejects(lambda: extract.exclusive(dangling), 'dangling cache symlink')
            rejects(lambda: extract.new_output(root), 'existing output directory')
        source = function((SCRIPTS / 'export_siglip2_identity_diversity_views.py').read_text(), 'export')
        self.assertEqual(source.count('np.load('), 1)
        self.assertIn('loaded = np.load(cache, allow_pickle=False)', source)
        self.assertNotIn('old_cache', source)
        self.assertNotIn('fit.npy', source)

    def test_limits_cli_and_no_native_import(self):
        low = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0))
        high = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 10_000_000_000))
        self.d.limits(self.d.time.perf_counter(), low)
        rejects(lambda: self.d.limits(self.d.time.perf_counter(), high), 'CUDA exclusive10GB')
        rejects(lambda: self.d.limits(self.d.time.perf_counter() - 901, low), 'whole900')
        base = ['--execution-sha256', 'a' * 64, '--authority', '/actual/authority.json',
                '--authority-sha256', 'b' * 64, '--phase', 'startup', '--output', '/new/output']
        self.assertEqual(self.d.parser().parse_args(base + ['--arm', 'candidate']).arm, 'candidate')
        with self.assertRaises(SystemExit):
            self.d.parser().parse_args(base)
        for optimize in ('-O', '-OO'):
            result = subprocess.run([sys.executable, '-B', '-S', optimize,
                str(SCRIPTS / 'export_siglip2_identity_diversity_views.py'), '--help'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('optimized mode is forbidden', result.stderr)
        self.assertFalse(any(name in sys.modules for name in ('torch', 'numpy', 'PIL', 'transformers',
                                                              'torchvision', 'safetensors')))


def terminal_checks(d, root):
    spec = importlib.util.spec_from_file_location('old_fit_metadata', Path(__file__).with_name('export_siglip2_substrate_fit.py'))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = SimpleNamespace(execution_sha256='d' * 64, authority=root / 'authority.json', authority_sha256='e' * 64, arm='candidate')
    prior_identity = {'python': '/immutable/python', 'python_sha256': 'f' * 64, 'python_version': 'fixture'}
    context = {'args': args, 'root': root, 'code': {}, 'launch': {'reference': {}, 'partition': {}, 'image_rows': {}, 'scope': {}, 'metadata': {}},
               'selected': {'scope_sha256': 'c' * 64}, 'guards': {}, 'reference': old,
               'prior': {'proof': {'invocation': prior_identity}, 'guards': {}, 'export_args': SimpleNamespace(arm='so400', execution_sha256='a' * 64),
                         'authority_sha256': 'a' * 64, 'own_code': {}, 'root': root,
                         'args': SimpleNamespace(execution_sha256='a' * 64, sources_sha256='a' * 64, fit_manifest_sha256='a' * 64),
                         'launch': {'source_cpu': {'so400': {}}}}}
    unit, invocation = 'genuine-startup-fixture', '1' * 32
    memory = {'path': '/sys/fs/cgroup/' + unit + '.service',
              'values': {'memory.max': str(8 * 1024**3), 'memory.current': '100', 'memory.peak': '200',
                         'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                         'memory.events': 'max 0\noom 0\noom_kill 0'}}
    proof_path, log_path = root / 'proof.json', root / 'startup.log'
    proof = {'schema': d.SCHEMA, 'phase': 'startup', 'pass': True, 'arm': args.arm, 'binding': d.binding(context),
             'exit_rehash_pass': True, 'resource_policy': d.STARTUP_POLICY, 'wall_seconds': 10,
             'process_peak_rss_kib': 10, 'native_imported': False, 'model_constructed': False, 'exported': False,
             'input_guards': {}, 'original_input_guards': {}, 'cgroup_before': memory, 'cgroup_after': memory,
             'invocation': {**prior_identity, 'invocation_id': invocation, 'optimize': 0, 'cuda_visible_devices': '',
                            'argv': [str(root / 'export_siglip2_identity_diversity_views.py'), '--execution-sha256', args.execution_sha256,
                                     '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
                                     '--arm', args.arm, '--phase', 'startup', '--output', str(root)]}}
    original_log = f'Running as unit: {unit}.service; invocation ID: {invocation}\n' + '\n'.join([
        '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
        '\tSwaps: 0', 'Memory swap peak: 0B', 'Service runtime: 11s', '\tMaximum resident set size (kbytes): 20',
        'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation})]) + '\n'
    log_path.write_text(original_log)
    descriptor = {'proof': {'path': str(proof_path), 'sha256': write(proof_path, proof)},
                  'log': {'path': str(log_path), 'sha256': d.sha(log_path)}, 'unit': unit,
                  'invocation_id': invocation, 'service_seconds': 11, 'native_peak_rss_kib': 20, 'both_locks_held': True}
    d.admit_terminal(context, descriptor, proof, 'startup')
    wrong_proof = copy.deepcopy(descriptor)
    wrong_proof['proof']['sha256'] = '0' * 64
    rejects(lambda: d.admit_terminal(context, wrong_proof, proof, 'startup'), 'terminal proof bytes')
    extra_log = copy.deepcopy(descriptor)
    extra_log['log']['extra'] = True
    rejects(lambda: d.admit_terminal(context, extra_log, proof, 'startup'), 'extra log descriptor field')
    minute_proof, minute_descriptor = copy.deepcopy(proof), copy.deepcopy(descriptor)
    minute_proof['wall_seconds'] = 60
    minute_descriptor['service_seconds'] = 61.125
    minute_descriptor['proof']['sha256'] = write(proof_path, minute_proof)
    log_path.write_text(original_log.replace('Service runtime: 11s', 'Service runtime: 1min 1.125s'))
    minute_descriptor['log']['sha256'] = d.sha(log_path)
    d.admit_terminal(context, minute_descriptor, minute_proof, 'startup')
    for duration in ('1min 61.125s', '1min 1.126s', '61.125', 'NaNs', '1min -1s'):
        log_path.write_text(original_log.replace('Service runtime: 11s', 'Service runtime: ' + duration))
        minute_descriptor['log']['sha256'] = d.sha(log_path)
        rejects(lambda: d.admit_terminal(context, minute_descriptor, minute_proof, 'startup'), 'service runtime ' + duration)
    log_path.write_text(original_log)
    write(proof_path, proof)
    for change in ('arm', 'locks', 'cap', 'argv', 'guards', 'binding', 'hidden', 'native', 'swap', 'events'):
        altered, terminal = copy.deepcopy(proof), copy.deepcopy(descriptor)
        if change == 'arm':
            altered['arm'] = 'control'
        elif change == 'locks':
            terminal['both_locks_held'] = False
        elif change == 'cap':
            terminal['service_seconds'] = 501
        elif change == 'argv':
            altered['invocation']['argv'][-1] += '-other'
        elif change == 'guards':
            altered['original_input_guards']['foreign'] = '0' * 64
        elif change == 'binding':
            altered['binding']['ordered_input_sha256'] = '0' * 64
        elif change == 'hidden':
            altered['invocation']['cuda_visible_devices'] = '0'
        elif change == 'native':
            altered['native_imported'] = True
        elif change == 'swap':
            altered['cgroup_after']['values']['memory.swap.peak'] = '1'
        else:
            altered['cgroup_after']['values']['memory.events'] = 'max 1\noom 0\noom_kill 0'
        terminal['proof']['sha256'] = write(proof_path, altered)
        rejects(lambda: d.admit_terminal(context, terminal, altered, 'startup'), 'startup ' + change)
    write(proof_path, proof)
    log_path.write_text(original_log.replace('\tExit status: 0', '\tExit status: 1'))
    changed = copy.deepcopy(descriptor)
    changed['log']['sha256'] = d.sha(log_path)
    rejects(lambda: d.admit_terminal(context, changed, proof, 'startup'), 'failed original exit')
    log_path.write_text(original_log + 'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation}) + '\n')
    changed['log']['sha256'] = d.sha(log_path)
    rejects(lambda: d.admit_terminal(context, changed, proof, 'startup'), 'duplicate footer')


if __name__ == '__main__':
    unittest.main(verbosity=2)
