#!/usr/bin/env python3
"""Bounded stdlib source/stand-in falsifiers; no Torch/native qualification."""
import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('train_siglip2_nearest_ranking.py')
spec = importlib.util.spec_from_file_location('nearest_test_driver', PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Tensor:
    """Small scalar/list arithmetic stand-in; rejects every negative gather."""
    dtype = 'float32'
    device = SimpleNamespace(type='cpu')

    def __init__(self, value):
        self.value = list(value) if isinstance(value, tuple) else value

    def item(self):
        return self.value

    def tolist(self):
        return self.value

    def cpu(self):
        return self

    def detach(self):
        return self

    def binary(self, other, op):
        other = other.value if isinstance(other, Tensor) else other
        def apply(a, b):
            if isinstance(a, list):
                return [apply(x, y) for x, y in zip(a, b, strict=True)] if isinstance(b, list) else [apply(x, b) for x in a]
            if isinstance(b, list):
                return [apply(a, y) for y in b]
            return op(a, b)
        return Tensor(apply(self.value, other))

    def __add__(self, v):
        return self.binary(v, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, v):
        return self.binary(v, lambda a, b: a - b)

    def __mul__(self, v):
        return self.binary(v, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, v):
        return self.binary(v, lambda a, b: a / b)

    def __ge__(self, v):
        return self.binary(v, lambda a, b: a >= b)

    def __gt__(self, v):
        return self.binary(v, lambda a, b: a > b)

    def square(self):
        return self * self

    def sum(self, axis=None):
        def total(v):
            return sum(total(x) for x in v) if isinstance(v, list) else v
        return Tensor([total(row) for row in self.value] if axis == 1 else total(self.value))

    def all(self):
        def all_items(v):
            return all(all_items(x) for x in v) if isinstance(v, list) else bool(v)
        return Tensor(all_items(self.value))

    def any(self):
        return Tensor(any(self.value))

    def __int__(self):
        return int(self.value)

    def __getitem__(self, key):
        key = key.value if isinstance(key, Tensor) else key
        if isinstance(key, list):
            if all(type(i) is bool for i in key):
                return Tensor([v for v, keep in zip(self.value, key, strict=True) if keep])
            if any(i < 0 for i in key):
                raise AssertionError('invalid positive was indexed')
            return Tensor([self.value[i] for i in key])
        if type(key) is int and key < 0:
            raise AssertionError('invalid positive was indexed')
        return Tensor(self.value[key])

    @property
    def T(self):
        return Tensor([list(row) for row in zip(*self.value, strict=True)])

    def __matmul__(self, other):
        return Tensor([[sum(a * b for a, b in zip(row, col, strict=True)) for col in zip(*other.value, strict=True)]
                       for row in self.value])


def fake_torch():
    torch = ModuleType('torch')
    nn, functional = ModuleType('torch.nn'), ModuleType('torch.nn.functional')
    torch.float32 = 'float32'
    torch.tensor = lambda v, **kwargs: Tensor(v)
    torch.no_grad = lambda: nullcontext()
    torch.autocast = lambda *args, **kwargs: nullcontext()
    torch.isfinite = lambda v: v.binary(0, lambda a, _: math.isfinite(a))
    functional.normalize = lambda v, dim: Tensor([[x / math.sqrt(sum(y * y for y in row)) for x in row] for row in v.value])
    functional.relu = lambda v: v.binary(0, lambda a, _: max(0., a))
    nn.functional = functional
    torch.nn = nn
    return {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}


class NearestRankingTests(unittest.TestCase):
    def launch(self, phase='cpu', arm='control'):
        unit = {'receipt': {'path': '/receipt', 'sha256': 'a' * 64},
                'log': {'path': '/log', 'sha256': 'b' * 64}, 'unit': 'new-unit',
                'invocation_id': 'c' * 32, 'service_seconds': 1., 'native_peak_rss_kib': 1,
                'both_locks_held': True}
        args = SimpleNamespace(execution_sha256='d' * 64, phase=phase, arm=arm, seed=driver.SEED)
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': args.execution_sha256,
            'phase': phase, 'arm': arm, 'seed': driver.SEED, 'fitter': copy.deepcopy(driver.FITTER),
            'accepted': copy.deepcopy(driver.ACCEPTED), 'recipe': copy.deepcopy(driver.RECIPE),
            'resource_policy': driver.policy(phase), 'both_locks_held': True,
            'selected_cpu': None if phase == 'cpu' else unit,
            'selected_mechanics': {a: copy.deepcopy(unit) for a in driver.ARMS} if phase == 'train' else None}
        return launch, args

    def test_exact_authority_and_phase_admission(self):
        for phase in ('cpu', 'mechanics', 'train'):
            launch, args = self.launch(phase)
            driver.check_launch(launch, args)
            for key, value in [('seed', 1), ('recipe', {}), ('fitter', {}), ('accepted', {}),
                               ('both_locks_held', False), ('resource_policy', {})]:
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    driver.check_launch({**launch, key: value}, args)
        launch, args = self.launch('cpu', 'candidate')
        with self.assertRaises(ValueError):
            driver.check_launch(launch, args)
        launch, args = self.launch('train')
        with self.assertRaises(ValueError):
            driver.check_launch({**launch, 'selected_mechanics': {'control': launch['selected_cpu']}}, args)
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                driver.strict_json(raw)

    def test_source_pins_match_committed_evidence(self):
        root = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1'
        receipt = json.loads((root / 'signed-concat-selection-score-v1/receipt.json').read_text())
        self.assertEqual(driver.FITTER, receipt['spec']['training'])
        self.assertEqual(driver.ACCEPTED, receipt['spec']['endpoints'][1])
        for name, sha in driver.FITTER['code'].items():
            self.assertEqual(hashlib.sha256(PATH.with_name(name).read_bytes()).hexdigest(), sha)
        self.assertEqual(driver.policy('cpu')['seconds'], 500)
        self.assertEqual(driver.policy('train')['seconds'], 300)
        self.assertEqual(driver.FILES, {PATH.name, Path(__file__).name, 'nearest_ranking_readout.py'})

    def test_current_bytes_reject_same_size_restored_mtime(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_bytes(b'abcd')
            sha = hashlib.sha256(b'abcd').hexdigest()
            driver.bound_file({}, path, sha)
            stamp = path.stat()
            path.write_bytes(b'abce')
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaises(ValueError):
                driver.bound_file({}, path, sha)
            with self.assertRaises(ValueError):
                driver.bound_file({str(path): 'b' * 64}, path, hashlib.sha256(b'abce').hexdigest())
            link = Path(directory) / 'link'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                driver.bound_file({}, link, hashlib.sha256(b'abce').hexdigest())

    def test_closure_is_exact_and_full_byte_authenticated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'x.py').write_text('x=1\n')
            code = {'x.py': hashlib.sha256((root / 'x.py').read_bytes()).hexdigest()}
            (root / 'execution.json').write_text(json.dumps(code))
            sha = hashlib.sha256((root / 'execution.json').read_bytes()).hexdigest()
            self.assertEqual(driver.closure(root, sha, {'x.py'}, {}), code)
            with self.assertRaises(ValueError):
                driver.closure(root, sha, {'x.py', 'extra.py'}, {})
            (root / 'x.py').write_text('x=2\n')
            with self.assertRaises(ValueError):
                driver.closure(root, sha, {'x.py'}, {})

    def test_self_exclusion_wrong_identity_original_row_ties_singleton(self):
        scores = [1., .8, .8, .8, .8]
        targets, rows = [0, 0, 0, 1, 2], [40, 30, 10, 50, 20]
        self.assertEqual(driver.select_nearest(scores, targets, rows, 0), (2, 4))
        self.assertEqual(driver.select_nearest(scores, targets, rows, 3), (-1, 0))
        with self.assertRaises(ValueError):
            driver.select_nearest([1., float('nan')], [0, 1], [0, 1], 0)
        with self.assertRaises(ValueError):
            driver.select_nearest([1., 1.], [0, 0], [0, 1], 0)
        with self.assertRaises(ValueError):
            driver.select_nearest([1., 1.], [0, 1], [0, 0], 0)

    def test_actual_loss_unequal_micro_valid_counts_matches_B64(self):
        state = {'teachers': {'V': Tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                             'P': Tensor([[0., 0.]] * 3), 'e0': Tensor(2.)},
                 'target': Tensor([0, 0, 1, 2, 2]), 'target_list': [0, 0, 1, 2, 2],
                 'row_list': [20, 10, 30, 40, 50], 'count_list': [2, 1, 2]}
        batches = [[0] * 16, [1] * 8 + [2] * 8, [2] * 16, [4] + [2] * 15]
        with patch.dict(sys.modules, fake_torch()):
            loss = [driver.loss_terms(state, Tensor([[.6, .8]] * 16), batch, 25) for batch in batches]
            self.assertEqual([info['valid'] for _, _, info in loss], [16, 8, 0, 1])
            self.assertAlmostEqual(sum(m.item() for m, _, _ in loss), .5)
            expected = (24 * (.05 + .8 - .6) + .05) / (25 * .05)
            self.assertAlmostEqual(sum(r.item() for _, r, _ in loss), expected)
            all_invalid = driver.loss_terms(state, Tensor([[.6, .8]] * 16), [2] * 16, 0)
            self.assertEqual(all_invalid[1].item(), 0.)
            self.assertEqual(all_invalid[2]['positive'], [-1] * 16)
            # A microbatch average would overweight the final single valid anchor.
            wrong = sum(r.item() * 25 / valid / 4 for (_, r, _), valid in zip(loss, [16, 8, 1, 1], strict=True))
            self.assertNotAlmostEqual(expected, wrong)

    def test_freeze_all_before_exact_four_roles_with_standin(self):
        class Parameter:
            dtype, grad = 'torch.float32', None
            def __init__(self, shape):
                self.shape, self.requires_grad = shape, True
            def requires_grad_(self, value):
                self.requires_grad = value
                return self
            def numel(self):
                return math.prod(self.shape)
        class Model:
            def __init__(self):
                self.parameters = {**{f'frozen{i}': Parameter((1,)) for i in range(444)},
                                   **{n: Parameter(s) for n, s in zip(driver.NAMES, driver.SHAPES, strict=True)}}
            def requires_grad_(self, value):
                for p in self.parameters.values():
                    p.requires_grad_(value)
            def named_parameters(self):
                return self.parameters.items()
            def state_dict(self):
                return self.parameters
            def eval(self):
                self.training = False
        model = Model()
        pairs = driver.configure_roles(model)
        self.assertEqual([n for n, _ in pairs], driver.NAMES)
        self.assertEqual(sum(p.requires_grad for p in model.parameters.values()), 4)
        self.assertFalse(model.training)
        model.parameters[driver.NAMES[0]].shape = (1,)
        with self.assertRaises(ValueError):
            driver.configure_roles(model)

    def test_state_export_source_separation_and_no_legacy_mutation(self):
        tree = ast.parse(PATH.read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        native = ast.unparse(functions['native_raw'])
        self.assertNotIn('no_grad', native)
        self.assertIn('isolated_A', native)
        self.assertIn('features.detach()', native)
        self.assertNotIn('pooled.detach()', native)
        forward = ast.unparse(functions['update'])
        self.assertIn('torch.autograd.grad(rank', forward)
        self.assertIn("mse + rank if state['arm'] == 'candidate' else mse", forward)
        self.assertNotIn('reset_peak_memory', PATH.read_text())
        for name in ('load_inference', 'inference_payload', 'check_inference'):
            text = ast.unparse(functions[name])
            for forbidden in ('canonical.npy', "['teachers']", "['schedule']", "['target']"):
                self.assertNotIn(forbidden, text)
        self.assertIn('vision', driver.PAYLOAD_KEYS)
        self.assertIn('provenance', driver.PAYLOAD_KEYS)
        self.assertNotIn('teachers', driver.INFERENCE_KEYS)
        self.assertNotIn('optimizer', driver.INFERENCE_KEYS)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        self.assertNotIn(target.value.id, ('fitter', 'old', 'source', 'legacy'))

    def test_payload_rejects_bad_identity_before_native_reads(self):
        with patch.dict(sys.modules, fake_torch()):
            with self.assertRaises(ValueError):
                driver.check_payload({}, {'schema': 'old-immutable-encoder'}, {}, 0)
            with self.assertRaises(ValueError):
                driver.check_optimizer({'optimizer': {'state': {}, 'param_groups': [{'params': [0, 1, 2, 3, 4]}]}},
                                       {'optimizer_groups': [{}]}, 0)
        self.assertTrue(driver.rejected(lambda: driver.require(False, 'tamper'), 'not rejected'))
        with self.assertRaises(ValueError):
            driver.rejected(lambda: None, 'not rejected')

    def test_fresh_optimizer_keeps_cpu_scaler_and_rejects_old_moments(self):
        scaler = {'scale': 128., 'growth_factor': 2., 'backoff_factor': .5,
                  'growth_interval': 2000, '_growth_tracker': 0}
        saved = {'optimizer': {'state': {}, 'param_groups': [{'params': [0, 1, 2, 3]}]},
                 'scaler': scaler, 'numerical_flags': {'threads': 8}}
        ident = {'optimizer_groups': [{}], 'device': 'cpu', 'initial_scaler': scaler,
                 'numerical_flags': saved['numerical_flags']}
        with patch.dict(sys.modules, fake_torch()):
            driver.check_optimizer(saved, ident, 0)
            with self.assertRaises(ValueError):
                driver.check_optimizer({**saved, 'scaler': {}}, ident, 0)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][0] = {'warm_optimizer': True}
            with self.assertRaises(ValueError):
                driver.check_optimizer(wrong, ident, 0)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['params'] = [3, 2, 1, 0]
            with self.assertRaises(ValueError):
                driver.check_optimizer(wrong, ident, 0)

    def test_updated_inference_substitution_and_no_concurrent_models(self):
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        saved = {k: {} for k in driver.INFERENCE_KEYS}
        saved.update(schema=driver.INFERENCE_SCHEMA, source={'original': 'pinned'},
                     vision={'mlp': [2., 3.]}, A={'fixed': [1.]})
        saved['vision_sha256'] = digest(saved['vision'])
        saved['fixed_sha256'] = digest({k: saved[k] for k in
            ('source', 'config', 'buffers', 'processor', 'head', 'classifier', 'A', 'means', 'numerical_flags')})
        context = {'source': saved['source'], 'old': SimpleNamespace(finite_tree=lambda value: None)}
        with patch.object(driver, 'fingerprint', side_effect=lambda context, value: digest(value)):
            driver.check_inference(context, saved, digest(saved))
            replaced = {**saved, 'vision': {'mlp': [0., 1.]}}
            with self.assertRaises(ValueError):
                driver.check_inference(context, replaced, digest(saved))
            with self.assertRaises(ValueError):
                driver.check_inference(context, replaced, digest(replaced))
            corrupted = {**saved, 'A': {'fixed': [2.]}}
            with self.assertRaises(ValueError):
                driver.check_inference(context, corrupted, digest(corrupted))
        import weakref
        class Model:
            pass
        model = Model()
        context = {'live_model': weakref.ref(model)}
        with self.assertRaises(ValueError):
            driver.require_no_model(context)
        del model
        driver.require_no_model(context)

    def test_original_image_mapping_rejects_source_and_identity_swap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'relative_path': 'image.jpg', 'train_row': 8, 'image_sha256': 'a' * 64, 'product': 'p'}
            manifest = {'original_rows': [0], 'rows': [row], 'resolved_paths': [str(root / 'image.jpg')], 'targets': [0]}
            fit = {'dataset_root': str(root), 'rows': [row], 'targets': [7], 'class_names': {7: 'p'}}
            state = {'original_rows': Tensor([0]), 'target': Tensor([0]),
                     'partition': {'panels': {'train': {'original_class_ids': [7]}}}}
            context = {'legacy': {'selected': {'genuine': {'selected': manifest}},
                                  'prior': {'fit': fit, 'all_images': [root / 'image.jpg']}}}
            self.assertEqual(driver.canonical_row(context, state, 0)[2]['image_sha256'], 'a' * 64)
            state['target'] = Tensor([1])
            with self.assertRaises(ValueError):
                driver.canonical_row(context, state, 0)
            state['target'] = Tensor([0])
            manifest['resolved_paths'] = ['/outside/image.jpg']
            with self.assertRaises(ValueError):
                driver.canonical_row(context, state, 0)

    def test_help_optimization_rejection_and_syntax(self):
        ast.parse(PATH.read_text())
        ast.parse(Path(__file__).read_text())
        help_result = subprocess.run([sys.executable, str(PATH), '--help'], capture_output=True, text=True, timeout=5)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--phase {cpu,mechanics,train}', help_result.stdout)
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, flag, str(PATH), '--help'], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('optimized mode is forbidden', result.stderr)


if __name__ == '__main__':
    unittest.main()
