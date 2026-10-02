#!/usr/bin/env python3
"""Stdlib contract falsifiers only; native CPU/CUDA qualification is separate."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

PATH = Path(__file__).with_name('train_siglip2_postln_adaptation.py')
spec = importlib.util.spec_from_file_location('postln_contract', PATH)
driver = importlib.util.module_from_spec(spec) if PATH.exists() else None
if driver is not None:
    spec.loader.exec_module(driver)


class Contract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(driver, 'bounded post-layernorm driver is missing')

    def test_json_and_uncached_same_stat_tamper(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                driver.strict_json(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_bytes(b'abcd')
            guard = {}
            digest = hashlib.sha256(b'abcd').hexdigest()
            driver.bound_file(guard, path, digest)
            stat = path.stat()
            path.write_bytes(b'efgh')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                driver.bound_file(guard, path, digest)

    def launch(self):
        unit = {'receipt': {'path': '/actual/receipt.json', 'sha256': 'a' * 64},
                'log': {'path': '/actual/job.log', 'sha256': 'b' * 64},
                'unit': 'actual-job', 'invocation_id': 'c' * 32, 'service_seconds': 90.,
                'native_peak_rss_kib': 1000, 'both_locks_held': True}
        warm = {'launch': dict(driver.WARM_LAUNCH), 'terminal': unit,
                'checkpoint': dict(driver.WARM_CHECKPOINT), 'terminal_state_sha256': driver.WARM_STATE_SHA}
        warm['terminal']['receipt'] = dict(driver.WARM_RECEIPT)
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
                  'phase': 'cpu', 'arm': 'control', 'seed': 179061,
                  'genuine_reference': dict(driver.GENUINE_REFERENCE), 'warm_start': warm,
                  'partition': {'path': '/actual/partition.json', 'sha256': driver.PARTITION_SHA},
                  'recipe': copy.deepcopy(driver.RECIPE), 'resource_policy': driver.policy('cpu'),
                  'both_locks_held': True, 'selected_cpu': None, 'selected_mechanics': None}
        return launch, SimpleNamespace(phase='cpu', arm='control', seed=179061, execution_sha256='d' * 64)

    def test_authority_rejects_recipe_source_partial_and_wrong_roles(self):
        launch, args = self.launch()
        driver.check_launch(launch, args)
        for key, value in [('schema', 'old'), ('selected_cpu', {}), ('both_locks_held', False)]:
            bad = copy.deepcopy(launch); bad[key] = value
            with self.assertRaises(ValueError):
                driver.check_launch(bad, args)
        for mutate in (
                lambda x: x['recipe'].update(native_learning_rate=1e-4),
                lambda x: x['warm_start']['checkpoint'].update(sha256='0' * 64),
                lambda x: x['genuine_reference'].update(execution_sha256='0' * 64),
                lambda x: x['warm_start']['terminal'].pop('log')):
            bad = copy.deepcopy(launch); mutate(bad)
            with self.assertRaises((ValueError, KeyError)):
                driver.check_launch(bad, args)

    def test_optimizer_members_are_disjoint_ordered_five_or_seven(self):
        class Tensor:
            dtype = 'torch.float32'
            requires_grad = True
            def __init__(self, shape):
                self.shape = shape
        for arm in driver.ARMS:
            names = driver.parameter_names(arm)
            pairs = [(n, Tensor(s)) for n, s in zip(names, driver.parameter_shapes(arm), strict=True)]
            groups = [[p for _, p in pairs]]
            driver.check_membership(pairs, groups, arm)
            bad = list(pairs); bad[-1] = (bad[-1][0], bad[0][1])
            with self.assertRaises(ValueError):
                driver.check_membership(bad, [[p for _, p in bad]], arm)
            with self.assertRaises(ValueError):
                driver.check_membership(pairs, [list(reversed(groups[0]))], arm)
        self.assertEqual(len(driver.parameter_names('control')), 5)
        self.assertEqual(len(driver.parameter_names('candidate')), 7)

    def test_schedule_is_exact_first_hundred_without_offset(self):
        full = [[(i + j) % 6355 for j in range(64)] for i in range(1000)]
        driver.check_continuation(full, full[:100])
        for selected in (full[1:101], full[:99], full[:101]):
            with self.assertRaises(ValueError):
                driver.check_continuation(full, selected)
        bad = copy.deepcopy(full); bad[999][0] = 6355
        with self.assertRaises(ValueError):
            driver.check_continuation(bad, full[:100])

    def test_sequential_model_ownership_is_enforced(self):
        context = {}
        class Model:
            pass
        model = Model()
        driver.claim_model(context, model)
        with self.assertRaises(ValueError):
            driver.require_no_model(context)
        del model
        driver.require_no_model(context)

    def test_train_local_original_fit_train_row_mapping(self):
        rows = [{'train_row': 90, 'product': 'a', 'relative_path': 'Img/img/a', 'image_sha256': 'a' * 64},
                {'train_row': 177, 'product': 'b', 'relative_path': 'Img/img/b', 'image_sha256': 'b' * 64}]
        prior = {'fit': {'dataset_root': '/dataset', 'rows': rows, 'targets': [0, 1], 'class_names': ['a', 'b']},
                 'all_images': [Path('/dataset/Img/img/a'), Path('/dataset/Img/img/b')]}
        manifest = {'original_rows': [1], 'rows': [rows[1]], 'targets': [0],
                    'resolved_paths': ['/dataset/Img/img/b']}
        context = {'prior': prior, 'selected': {'genuine': {'selected': manifest}}}
        state = {'original_rows': [1], 'target': [0], 'partition': {'panels': {'train': {'original_class_ids': [1]}}}}
        row, path, fact = driver.canonical_row(context, state, 0)
        self.assertEqual((fact['train_local'], fact['fit_ordinal'], fact['train_row']), (0, 1, 177))
        self.assertEqual(path, prior['all_images'][1])
        bad = copy.deepcopy(context); bad['selected']['genuine']['selected']['original_rows'][0] = 0
        with self.assertRaises(ValueError):
            driver.canonical_row(bad, state, 0)

    def fake_payload(self, arm, step):
        class Tensor:
            def __init__(self, shape, dtype='torch.float32', value=0):
                self.shape, self.dtype, self.value = shape, dtype, value
            def __float__(self):
                return float(self.value)
        launch, _ = self.launch()
        count = len(driver.parameter_names(arm))
        defaults = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05, 'amsgrad': False,
                    'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
        rates = ([1e-5] if arm == 'candidate' else []) + [1e-4, 1e-4]
        members = [[0, 1], [2, 3, 4, 5], [6]] if arm == 'candidate' else [[0, 1, 2, 3], [4]]
        groups = [{**defaults, 'lr': rate} for rate in rates]
        ident = {'source': {}, 'config': {}, 'numerical_flags': {}, 'optimizer_defaults': defaults,
                 'seed': 179061, 'arm': arm, 'parameter_names': driver.parameter_names(arm),
                 'method': driver.method(launch), 'positive_shape': (6355, 110), 'device': 'cpu',
                 'optimizer_groups': groups,
                 'optimizer_serial_groups': [dict(g, params=p) for g, p in zip(groups, members, strict=True)]}
        saved = {'schema': driver.SCHEMA, 'identity': ident, 'source': {}, 'config': {},
                 'numerical_flags': {}, 'optimizer_defaults': defaults, 'counter': step, 'seed': 179061,
                 'norm': {n: Tensor((1152,)) for n in driver.NORM},
                 'head': {n: Tensor(s) for n, s in driver.HEAD_LAYOUT.items()},
                 'buffers': {'embeddings.position_ids': Tensor((1, 256), 'torch.int64')},
                 'classifier': Tensor((1008, 128)), 'bank': Tensor((6355, 128)),
                 'target': Tensor((6355,), 'torch.int64'), 'positive': Tensor((6355, 110), 'torch.int64'),
                 'original_rows': Tensor((6355,), 'torch.int64'),
                 'pca': {'mean': Tensor((1152,)), 'components': Tensor((128, 1152))},
                 'schedules': {str(s): Tensor((1000, 64), 'torch.int64') for s in driver.SEEDS},
                 'masks': {str(s): Tensor((1000, 64), 'torch.bool') for s in driver.SEEDS},
                 'continuation': {str(s): Tensor((100, 64), 'torch.int64') for s in driver.SEEDS},
                 'partition': {'schema': 'siglip2-identity-mix-partition-v1'},
                 'warm_start': {'endpoint': launch['warm_start'], 'warm_source_updates': 1000,
                                'local_initial_counter': 0, 'bank_context': 'original canonical B32'},
                 'optimizer': {'param_groups': ident['optimizer_serial_groups'], 'state': {}},
                 'scaler': {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                            'growth_interval': 2000, '_growth_tracker': step},
                 'cpu_rng': Tensor((5000,), 'torch.uint8'), 'cuda_rng': []}
        if step:
            for i, shape in enumerate(driver.parameter_shapes(arm)):
                saved['optimizer']['state'][i] = {'step': Tensor((), value=step),
                    'exp_avg': Tensor(shape), 'exp_avg_sq': Tensor(shape)}
        return saved, ident

    def test_complete_payload_rejects_every_omission_and_bad_counter(self):
        for arm in driver.ARMS:
            for step in (0, 8, 17, 100):
                saved, ident = self.fake_payload(arm, step)
                driver.check_payload(saved, ident, step)
                for key in driver.PAYLOAD_KEYS:
                    bad = saved.copy(); del bad[key]
                    with self.assertRaises((ValueError, KeyError), msg=key):
                        driver.check_payload(bad, ident, step)
                for member in ('center', 'preactivation_std'):
                    bad = copy.deepcopy(saved); del bad['head'][member]
                    with self.assertRaises(ValueError):
                        driver.check_payload(bad, ident, step)
                if step:
                    bad = copy.deepcopy(saved); bad['optimizer']['state'][0]['step'].value = step - 1
                    with self.assertRaises(ValueError):
                        driver.check_payload(bad, ident, step)
                bad = copy.deepcopy(saved); bad['scaler']['scale'] = 64
                with self.assertRaises(ValueError):
                    driver.check_payload(bad, ident, step)
                bad, bad_ident = self.fake_payload(arm, step)
                bad_ident['optimizer_defaults']['weight_decay'] = .01
                for group in bad_ident['optimizer_groups'] + bad_ident['optimizer_serial_groups']:
                    group['weight_decay'] = .01
                with self.assertRaises(ValueError):
                    driver.check_payload(bad, bad_ident, step)

    def test_native_inventory_rejects_original_205_roles(self):
        launch, _ = self.launch()
        _, ident = self.fake_payload('candidate', 0)
        ident['device'] = 'cuda'
        ident['native_inventory'] = [
            {'name': 'frozen.' + str(i), 'shape': [1], 'dtype': 'torch.float32', 'role': 'frozen'} for i in range(446)]
        ident['native_inventory'] += [dict(name=n, shape=[1152], dtype='torch.float32', role='trainable') for n in driver.NORM]
        for key in ('complement_sha256', 'buffers_sha256', 'head_buffers_sha256', 'static_sha256',
                    'warm_members_sha256', 'schedule_sha256', 'full_schedule_sha256'):
            ident[key] = 'a' * 64
        driver.check_identity_record(ident, launch, 'candidate', 179061, 'cuda')
        for change in ('old_roles', 'wrong_norm_shape', 'fp16_norm', 'missing_native'):
            bad = copy.deepcopy(ident)
            if change == 'old_roles':
                for row in bad['native_inventory'][:203]:
                    row['role'] = 'trainable'
            elif change == 'wrong_norm_shape':
                bad['native_inventory'][-1]['shape'] = [1153]
            elif change == 'fp16_norm':
                bad['native_inventory'][-1]['dtype'] = 'torch.float16'
            else:
                bad['native_inventory'].pop(0)
            with self.assertRaises(ValueError):
                driver.check_identity_record(bad, launch, 'candidate', 179061, 'cuda')

    def test_complement_uses_fresh_bytes_not_versions(self):
        class Tensor:
            _version = 0
            value = 'original'
            def detach(self):
                return self
            def cpu(self):
                return self
        tensor = Tensor()
        model = SimpleNamespace(named_parameters=lambda: [('frozen', tensor)])
        context = {'source_driver': SimpleNamespace(tensor_fact=lambda t: {'sha256': t.value}),
                   'prior': {'mapping': {'frozen': {'sha256': 'original'}}}}
        driver.check_complement(context, model, excluded=())
        tensor.value = 'mutated-through-data'
        with self.assertRaises(ValueError):
            driver.check_complement(context, model, excluded=())

    def test_native_origins_require_authenticated_file_and_module_union(self):
        cases = ('cpu_subset', 'late_warm', 'unknown_file', 'unknown_module', 'changed_sha',
                 'changed_module_path', 'file_union_conflict', 'module_union_conflict', 'original_guard_conflict')
        for initial in (True, False):
            for case in cases:
                with self.subTest(initial=initial, case=case), tempfile.TemporaryDirectory() as directory:
                    paths, hashes = {}, {}
                    for name in ('cpu', 'spare', 'warm', 'unknown'):
                        path = Path(directory) / (name + '.py')
                        raw = name.encode()
                        path.write_bytes(raw)
                        paths[name] = str(path)
                        hashes[name] = hashlib.sha256(raw).hexdigest()
                    cpu = {'packages': {}, 'modules': {'torch': paths['cpu'], 'torch.spare': paths['spare']},
                           'native_files': [], 'files': {paths[n]: hashes[n] for n in ('cpu', 'spare')}}
                    warm = copy.deepcopy(cpu)
                    warm['modules']['torch.warm'] = paths['warm']
                    warm['files'][paths['warm']] = hashes['warm']
                    actual = {'packages': {}, 'modules': {'torch': paths['cpu']}, 'native_files': [],
                              'files': {paths['cpu']: hashes['cpu']}}
                    context = {'source_driver': SimpleNamespace(imported_origins=lambda extract, packages: actual),
                               'extract': None, 'selected': {'packages': {}, 'source_cpu': {'origins': cpu}},
                               'warm_record': {'origins': warm}, 'guards': dict(cpu['files']),
                               'prior': {'guards': dict(cpu['files'])}}
                    if case == 'late_warm':
                        actual['modules']['torch.warm'] = paths['warm']
                        actual['files'][paths['warm']] = hashes['warm']
                    elif case == 'unknown_file':
                        actual['native_files'] = [paths['unknown']]
                        actual['files'][paths['unknown']] = hashes['unknown']
                        context['guards'][paths['unknown']] = hashes['unknown']
                    elif case == 'unknown_module':
                        actual['modules']['torch.alias'] = paths['cpu']
                    elif case == 'changed_sha':
                        Path(paths['cpu']).write_bytes(b'changed')
                        actual['files'][paths['cpu']] = hashlib.sha256(b'changed').hexdigest()
                    elif case == 'changed_module_path':
                        actual['modules']['torch'] = paths['spare']
                        actual['files'][paths['spare']] = hashes['spare']
                    elif case == 'file_union_conflict':
                        warm['files'][paths['spare']] = '0' * 64
                    elif case == 'module_union_conflict':
                        warm['modules']['torch.spare'] = paths['warm']
                    elif case == 'original_guard_conflict':
                        context['prior']['guards'][paths['cpu']] = '0' * 64
                    before = copy.deepcopy((context['guards'], context['prior']['guards']))
                    if case == 'cpu_subset' or case == 'late_warm' and not initial:
                        driver.audit_origins(context, initial=initial)
                        self.assertEqual(context['origins'], actual)
                        for path, digest in actual['files'].items():
                            self.assertEqual(context['guards'][path], digest)
                            self.assertEqual(context['prior']['guards'][path], digest)
                    else:
                        with self.assertRaises(ValueError):
                            driver.audit_origins(context, initial=initial)
                        self.assertEqual((context['guards'], context['prior']['guards']), before)
                        self.assertNotIn('origins', context)

    def test_native_imports_are_lazy_and_forbidden_paths_absent(self):
        tree = ast.parse(PATH.read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(set(n.split('.')[0] for n in names) & driver.NATIVE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('half', 'reset_peak_memory_stats', 'training_features',
                                                 'initializer', 'pixels_and_raw', 'augmented_pixels'))
                if node.func.attr == 'head_from':
                    self.assertEqual(ast.literal_eval(node.args[0]), 'control')
        self.assertNotIn('torch', sys.modules)

    def test_help_and_optimized_mode(self):
        help_result = subprocess.run([sys.executable, '-B', str(PATH), '--help'], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for word in ('--phase', '--arm', '--seed', '--authority-sha256'):
            self.assertIn(word, help_result.stdout)
        optimized = subprocess.run([sys.executable, '-B', '-O', str(PATH), '--help'], capture_output=True, text=True)
        self.assertNotEqual(optimized.returncode, 0)
        self.assertIn('optimized mode is forbidden', optimized.stderr)


if __name__ == '__main__':
    unittest.main()
