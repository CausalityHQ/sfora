#!/usr/bin/env python3
"""Default: bounded stdlib checks. --torch-fp32: opt-in CPU arithmetic test only."""
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_connected_gradient_decomposition import Dual, Tensor, dual_fixture, dual_torch

HERE = Path(__file__).resolve().parent
TORCH_FP32 = '--torch-fp32' in sys.argv
if TORCH_FP32:
    sys.argv.remove('--torch-fp32')
ARCHIVED_TRAINER = (HERE.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1' /
                    'export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py')


def load(name, path=None):
    path = HERE / (name + '.py') if path is None else path
    if not path.is_file():
        raise AssertionError('missing requested source: ' + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_reference(scores, positive, eligible, ks=(1, 2, 4, 8, 16)):
    def sigmoid(x):
        x = min(50., max(-50., x))
        return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    if not positive:
        return [0.]
    ranks = [1 + sum(sigmoid((scores[j] - scores[p]) / .01)
                     for j in eligible if j != p) for p in positive]
    return [1 - min(k, sum(sigmoid(k - rank) for rank in ranks)) / min(k, len(positive))
            for k in ks]


def fake_native(stack, modules=None):
    torch, defaults = dual_torch()
    if modules is not None:
        torch = modules['torch']
    else:
        modules = defaults
    torch.stack = lambda values: Tensor([v.data[0] for v in values])
    torch.full_like = lambda tensor, value: Tensor([value] * len(tensor.data), tensor.shape)
    torch.minimum = lambda a, b: a.binary(b, lambda x, y:
                                        x if x.value < y.value else y if y.value < x.value else (x + y) * .5)
    def clamp(tensor, *, min=-math.inf, max=math.inf):
        return Tensor([x if min <= x.value <= max else Dual(min if x.value < min else max)
                       for x in tensor.data], tensor.shape)
    stack.enter_context(patch.object(Tensor, 'clamp', clamp, create=True))
    stack.enter_context(patch.object(Tensor, '__neg__', lambda t: t * -1, create=True))
    stack.enter_context(patch.object(Tensor, '__rtruediv__',
        lambda t, x: t.binary(x, lambda a, b: b / a), create=True))
    stack.enter_context(patch.object(Tensor, 'exp', lambda t: Tensor([
        Dual(math.exp(x.value), [math.exp(x.value) * d for d in x.derivative], x.reach)
        for x in t.data], t.shape), create=True))
    stack.enter_context(patch.dict(sys.modules, modules))
    return torch


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.loss = load('connected_rsk_loss')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        fake_native(self.stack)

    def values(self, scores, positive, eligible, ks=(1, 2, 4, 8, 16)):
        return self.loss.rsk_terms(Tensor(scores), positive, eligible, k_values=ks)

    def test_default_k_values_have_five_known_tied_score_losses(self):
        actual = self.loss.rsk_terms(Tensor([1., .2, .2, .2, .2]), [1, 2], [1, 2, 3, 4])
        # Four eligible tied columns give each positive rank 1 + 3/2 = 2.5.
        expected = [.6351489523872873, .6224593312018546, .18242552380635635,
                    .004070137715896073, .000001370957206869416]
        for value, want in zip(actual.data, expected, strict=True):
            self.assertAlmostEqual(value.value, want, places=14)

    def test_asymmetric_scores_match_literal_reference(self):
        scores = [.99, .317, .305, .33, -.1, .321]
        positive, eligible = [1, 2, 5], [1, 2, 3, 4, 5]
        actual = self.values(scores, positive, eligible)
        for value, want in zip(actual.data, scalar_reference(scores, positive, eligible), strict=True):
            self.assertAlmostEqual(value.value, want, places=14)

    def test_k1_gradient_promotes_positive_and_demotes_impostor(self):
        scores = [Dual(1.), Dual(.3, (1., 0., 0., 0., 0., 0.), (0,)),
                  Dual(.31, (0., 1., 0., 0., 0., 0.), (1,))]
        value = self.values(scores, [1], [1, 2], (1,)).data[0]
        self.assertLess(value.derivative[0], 0.)
        self.assertGreater(value.derivative[1], 0.)
        epsilon = 1e-6
        for index in (1, 2):
            hi, lo = [1., .3, .31], [1., .3, .31]
            hi[index] += epsilon
            lo[index] -= epsilon
            expected = (scalar_reference(hi, [1], [1, 2], (1,))[0] -
                        scalar_reference(lo, [1], [1, 2], (1,))[0]) / (2 * epsilon)
            self.assertAlmostEqual(value.derivative[index - 1], expected, places=7)

    def test_excluded_self_is_inert_and_co_positives_still_compete(self):
        a = self.values([100., .2, .2, -.5], [1, 2], [1, 2, 3], (1,))
        b = self.values([-100., .2, .2, -.5], [1, 2], [1, 2, 3], (1,))
        self.assertEqual(a.item(), b.item())
        self.assertAlmostEqual(a.item(), 1 - 2 / (1 + math.exp(.5)), places=14)

    def test_retrieved_cap_prevents_negative_loss_and_empty_is_zero(self):
        value = self.values([0.] * 11, list(range(1, 11)), list(range(1, 11)), (8,))
        self.assertEqual(value.item(), 0.)
        self.assertEqual(self.values([1., 0.], [], [1]).item(), 0.)

    def test_invalid_membership_and_nonfinite_eligible_scores_reject(self):
        for positives, eligible in (([1, 1], [1, 2]), ([0], [1, 2]), ([1], [1, 1]),
                                    ([1], [1, 3]), ([True], [1, 2])):
            with self.assertRaises(ValueError):
                self.values([1., .3, .2], positives, eligible)
        for ks in ((), (0,), (True,), (1, 1)):
            with self.assertRaises(ValueError):
                self.values([1., .3, .2], [1], [1, 2], ks)
        with self.assertRaises(ValueError):
            self.values([1., float('nan'), .2], [1], [1, 2])

    def test_official_exponent_clip_has_zero_gradient_outside_fifty(self):
        scores = [Dual(1.), Dual(.6, (1., 0., 0., 0., 0., 0.), (0,)), Dual(0.)]
        value = self.values(scores, [1], [1, 2], (1,)).data[0]
        self.assertEqual(value.derivative[0], 0.)

    def test_official_expression_keeps_positive_tail_gradient(self):
        for logit in (-60., -50., -20., 0., 20., 50., 60.):
            value = self.loss.official_sigmoid(Tensor([Dual(logit, (1.,) * 6)])).data[0]
            exponent = math.exp(min(50., max(-50., -logit)))
            expected = exponent / (1 + exponent) ** 2 if abs(logit) <= 50. else 0.
            self.assertEqual(value.value, 1 / (1 + exponent))
            self.assertAlmostEqual(value.derivative[0] / expected if expected else value.derivative[0],
                                   1. if expected else 0., places=14)

    def test_both_rsk_sigmoids_keep_tail_score_gradients(self):
        for other_score, k, rank_logit in ((.5, 1, 50.), (0., 51, 0.)):
            scores = Tensor([Dual(0., (1.,) * 6), Dual(other_score)])
            actual = self.loss.rsk_terms(scores, [0], [0, 1], k_values=(k,)).data[0]
            rank_exp = math.exp(-rank_logit)
            rank = 1 + 1 / (1 + rank_exp)
            membership_exp = math.exp(-(k - rank))
            expected = -(membership_exp / (1 + membership_exp) ** 2) * \
                       (rank_exp / (1 + rank_exp) ** 2) / .01
            self.assertLess(actual.derivative[0], 0.)
            self.assertAlmostEqual(actual.derivative[0] / expected, 1., places=14)


@unittest.skipUnless(TORCH_FP32, 'real Torch CPU FP32 check is opt-in; source-only gate')
class TorchFP32Tests(unittest.TestCase):
    def test_fp32_values_and_autograd_match_literal_official_expression(self):
        import torch
        torch.set_num_threads(1)
        loss = load('connected_rsk_loss')
        for temperature in (.01, 1.):
            x = (torch.tensor([-60., -50., -20., 0., 20., 50., 60.], dtype=torch.float32) *
                 temperature).requires_grad_(True)
            actual = loss.official_sigmoid(x, temperature)
            expected = 1.0 / (1.0 + torch.exp(torch.clamp(-x / temperature, min=-50, max=50)))
            actual_grad, = torch.autograd.grad(actual.sum(), x)
            expected_grad, = torch.autograd.grad(expected.sum(), x)
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0., atol=0.)
            self.assertGreater(actual_grad[4].item(), 0.)
            self.assertGreater(actual_grad[5].item(), 0.)
        # Exercise both production call sites, including saturated membership logits.
        scores = torch.tensor([0., -.6, -.5, -.2, 0., .2, .5, .6], dtype=torch.float32, requires_grad=True)
        positive, eligible, ks = [1, 4, 6], list(range(1, 8)), (1, 2, 4, 8, 16, 50, 60)
        def sigmoid(x, temperature):
            return 1.0 / (1.0 + torch.exp(torch.clamp(-x / temperature, min=-50, max=50)))
        ranks = [1 + sum(sigmoid(scores[j] - scores[p], .01) for j in eligible if j != p)
                 for p in positive]
        expected = torch.stack([1 - torch.minimum(sum(sigmoid(k - r, 1.) for r in ranks),
                                                  scores.new_tensor(float(k))) / min(k, len(positive))
                                for k in ks])
        actual = loss.rsk_terms(scores, positive, eligible, k_values=ks)
        actual_grad, = torch.autograd.grad(actual.sum(), scores)
        expected_grad, = torch.autograd.grad(expected.sum(), scores)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0.)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=0.)


class MutationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        fake_native(self.stack)

    def mutant_values(self, before, after, scores, positives, eligible, ks):
        source = (HERE / 'connected_rsk_loss.py').read_text()
        self.assertEqual(source.count(before), 1)
        namespace = {}
        exec(compile(source.replace(before, after), '<RS@k source mutant>', 'exec'), namespace)
        return [x.value for x in namespace['rsk_terms'](Tensor(scores), positives, eligible, k_values=ks).data]

    def assert_killed(self, actual, scores, positives, eligible, ks):
        expected = scalar_reference(scores, positives, eligible, ks)
        self.assertGreater(max(abs(a - b) for a, b in zip(actual, expected, strict=True)), 1e-6)

    def test_kill_wrong_rank_temperature(self):
        case = ([.9, .3, .31], [1], [1, 2], (1,))
        actual = self.mutant_values(', .01)', ', .1)', *case)
        self.assert_killed(actual, *case)

    def test_kill_missing_retrieved_count_clip(self):
        case = ([0.] * 11, list(range(1, 11)), list(range(1, 11)), (8,))
        actual = self.mutant_values('torch.minimum(retrieved, torch.full_like(retrieved, float(k)))',
                                    'retrieved', *case)
        self.assert_killed(actual, *case)
        self.assertLess(actual[0], 0.)

    def test_kill_wrong_min_k_positive_count_normalizer(self):
        case = ([1., .2, .2, .2, .2], [1, 2], [1, 2, 3, 4], (1, 16))
        actual = self.mutant_values(' / min(k, len(positive))', ' / len(positive)', *case)
        self.assert_killed(actual, *case)

    def test_kill_self_column_leakage(self):
        case = ([1., .3, .31], [1], [1, 2], (1,))
        actual = self.mutant_values('    comparisons =', '    eligible = [0] + eligible\n    comparisons =', *case)
        self.assert_killed(actual, *case)


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.arm = load('train_siglip2_connected_rsk')
        self.trainer = load('train_siglip2_identity_diversity', ARCHIVED_TRAINER)
        self.connected = load('train_siglip2_connected_mlp')

    def test_original_control_ast_pins_are_unchanged(self):
        pins = {
            'train_siglip2_connected_mlp.py': {
                'update': '5f3e94b8c1d5fe5fb68368265999e30a960ff19db63c38c229032f88b5148258',
                'bind_optimizer': '4475ddc06af0316b9bd75b7dd6f65748cb161802df7f6abae18f641550aa2245',
                'parameter_roles': '64449c04d0d168b2f2edbe5b6ec8ccdb5e24ada5d63860dc10474cd7c0fa7019'},
            'train_siglip2_identity_diversity.py': {
                'loss_terms': 'd9cdceea0ac03e072e6a33f86053823a786e12a3cf180e645eebe002edcffc26',
                'smooth_ap_terms': 'a90d81810d4295691ff1829215a07b9ab1793cd040ad92530d6bffaefa1eb698',
                'ranking_membership': 'c56dd67441cc81a47e8fcfb71032e4c751b7d5b70f2b6df19ca6236b9301813f',
                'loss_denominators': 'ce86de123238eee428db8c62ff8ed4c09c2cde5ad6fbe52a85139623802a78e5'}}
        for filename, expected in pins.items():
            raw = (ARCHIVED_TRAINER if filename == ARCHIVED_TRAINER.name else HERE / filename).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), self.arm.PINS[filename])
            functions = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
            for name, digest in expected.items():
                self.assertEqual(hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest(), digest)
        current = (HERE / ARCHIVED_TRAINER.name).read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(),
                         '094d97259b5f417b334186305351f02d6b76983cf88dbf4021a22d0f70a235e9')
        with self.assertRaisesRegex(ValueError, 'frozen source bytes'):
            self.arm.make_loss_terms(load('train_siglip2_identity_diversity'), 'rsk')

    def test_control_is_original_candidate_changes_only_ranking_binding(self):
        control = self.arm.make_loss_terms(self.trainer, 'control')
        candidate = self.arm.make_loss_terms(self.trainer, 'rsk')
        self.assertIs(control, self.trainer.loss_terms)
        self.assertIs(candidate.__code__, control.__code__)
        self.assertIsNot(candidate.__globals__, control.__globals__)
        self.assertEqual(candidate.__globals__.keys(), control.__globals__.keys())
        changed = [k for k in control.__globals__ if candidate.__globals__[k] is not control.__globals__[k]]
        self.assertEqual(changed, ['smooth_ap_terms'])
        self.assertIs(control.__globals__['smooth_ap_terms'], self.trainer.smooth_ap_terms)

    def test_source_and_live_function_tampering_reject(self):
        original_code = self.trainer.loss_terms.__code__
        try:
            self.trainer.loss_terms.__code__ = (lambda *a: 0).__code__
            with self.assertRaises(ValueError):
                self.arm.make_loss_terms(self.trainer, 'rsk')
        finally:
            self.trainer.loss_terms.__code__ = original_code
        with patch.object(self.trainer, 'smooth_ap_terms', lambda *a: 0):
            with self.assertRaises(ValueError):
                self.arm.make_loss_terms(self.trainer, 'rsk')
        with self.assertRaises(ValueError):
            self.arm.make_loss_terms(self.trainer, 'candidate')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_siglip2_identity_diversity.py'
            path.write_bytes((HERE / path.name).read_bytes() + b'\n')
            with patch.object(self.trainer, '__file__', str(path)):
                with self.assertRaises(ValueError):
                    self.arm.make_loss_terms(self.trainer, 'rsk')

    def test_loss_pair_preserves_regression_exclusion_singletons_and_full_batch_scale(self):
        torch, modules, fixture, state, _, raw = dual_fixture(SimpleNamespace(require=require), count=8)
        state['arm'] = 'control'
        # One duplicated original row (self exclusion by row, not column) and one singleton.
        labels, rows = [0, 0, 0, 1, 1, 2, 2, 3], [0, 0, 2, 3, 4, 5, 6, 7]
        state['target'] = Tensor(labels)
        state['ranking_bank'] = fixture.ranking_bank(labels, rows)
        adapted = self.arm.make_loss_terms(self.trainer, 'rsk')
        # Execute the actual production loss bytecode with the existing dual arithmetic fixture.
        fixture.loss_terms.__globals__['smooth_ap_terms'] = adapted.__globals__['smooth_ap_terms']
        with ExitStack() as stack:
            fake_native(stack, modules)
            anchors = [0, 1, 2, 7]
            full_valid = 7
            mse, rank, facts = fixture.loss_terms({}, state, raw(anchors), anchors, full_valid)
            self.assertEqual(facts['positive'], [[2], [2], [0, 1], []])
            self.assertEqual(facts['eligible_counts'], [6, 6, 7, 7])
            self.assertEqual(facts['valid'], 3)
            twice = fixture.loss_terms({}, state, raw(anchors), anchors, 14)
            self.assertEqual(twice[0].item(), mse.item())
            self.assertAlmostEqual(twice[1].item() * 2, rank.item(), places=14)
            fixture.loss_terms.__globals__['smooth_ap_terms'] = self.trainer.smooth_ap_terms
            old_mse, old_rank, old_facts = fixture.loss_terms({}, state, raw(anchors), anchors, full_valid)
            self.assertEqual(old_mse.item(), mse.item())
            self.assertNotEqual(old_rank.item(), rank.item())
            for key in ('positive', 'eligible_counts', 'eligible_sha256', 'valid'):
                self.assertEqual(facts[key], old_facts[key])
            for loss in (self.trainer.smooth_ap_terms, adapted.__globals__['smooth_ap_terms']):
                fixture.loss_terms.__globals__['smooth_ap_terms'] = loss
                singleton = fixture.loss_terms({}, state, raw([7]), [7], 0)
                self.assertEqual(singleton[1].item(), 0.)
                self.assertEqual(singleton[2]['valid'], 0)

    def test_update_context_is_private_and_keeps_control_parameter_roles(self):
        context = {'trainer': self.trainer, 'guards': {}}
        for experiment_arm in ('control', 'rsk'):
            adapted = self.arm.arm_context(context, experiment_arm)
            self.assertIs(context['trainer'], self.trainer)
            self.assertIs(adapted['guards'], context['guards'])
            changed = [k for k in vars(self.trainer) if getattr(adapted['trainer'], k) is not getattr(self.trainer, k)]
            self.assertEqual(changed, [] if experiment_arm == 'control' else ['loss_terms'])
        self.assertEqual(self.connected.parameter_roles('control')[:2], (['A', 'C'], [[128, 160], [128, 1152]]))

    def test_encoder_training_or_wrong_optimizer_roles_reject_before_native_update(self):
        identity = {'arm': 'control', 'parameter_names': ['A', 'C'],
                    'parameter_shapes': [[128, 160], [128, 1152]]}
        for state, ident in (({'arm': 'candidate'}, identity),
                             ({'arm': 'control'}, {**identity, 'arm': 'candidate'}),
                             ({'arm': 'control'}, {**identity, 'parameter_names': ['A']})):
            with self.assertRaisesRegex(ValueError, 'frozen-encoder CONTROL'):
                self.arm.update(self.connected, {'trainer': self.trainer}, state, ident, 1, arm='rsk')

    def test_update_labels_experiment_and_charges_adapter_time(self):
        context = {'trainer': self.trainer, 'guards': {}}
        state = {'arm': 'control'}
        identity = {'arm': 'control', 'parameter_names': ['A', 'C'],
                    'parameter_shapes': [[128, 160], [128, 1152]]}
        seen = []
        def native_step(private, actual_state, actual_identity, step):
            seen.append((private, actual_state, actual_identity, step))
            return {'arm': 'control', 'step': step, 'core_seconds': .5}
        # Only the native step is stood in; math and source admission have independent checks.
        with patch.object(self.arm, 'authenticate'), patch.object(self.connected, 'update', native_step), \
                patch.object(self.arm.time, 'perf_counter', side_effect=(1., 3.)):
            record = self.arm.update(self.connected, context, state, identity, 1, arm='rsk')
        self.assertEqual(record['schema'], 'siglip2-connected-rsk-update-v1')
        self.assertEqual((record['arm'], record['ranking']), ('rsk', 'RS@k'))
        self.assertEqual(record['control_update']['step'], 1)
        self.assertEqual(record['core_seconds'], 2.)
        self.assertIs(seen[0][1], state)
        self.assertIs(seen[0][2], identity)
        self.assertIs(context['trainer'], self.trainer)
        self.assertIsNot(seen[0][0]['trainer'].loss_terms, self.trainer.loss_terms)

    def test_candidate_failure_names_the_actual_ranking_term(self):
        identity = {'arm': 'control', 'parameter_names': ['A', 'C'],
                    'parameter_shapes': [[128, 160], [128, 1152]]}
        for message in ('actual SmoothAP member gradient absent/nonfinite',
                        'both-view actual SmoothAP gradients must be nonzero', 'unrelated guard'):
            for arm in ('control', 'rsk'):
                expected = message.replace('SmoothAP', 'RS@k') if arm == 'rsk' else message
                with patch.object(self.arm, 'authenticate'), \
                        patch.object(self.connected, 'update', side_effect=ValueError(message)):
                    with self.assertRaises(ValueError) as caught:
                        self.arm.update(self.connected, {'trainer': self.trainer}, {'arm': 'control'},
                                        identity, 1, arm=arm)
                self.assertEqual(str(caught.exception), expected)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.probe = load('preflight_siglip2_connected_rsk')

    def test_first_positive_ranks_exclude_self_and_include_nonboundary_anchors(self):
        for scores, expected in (([9., .9, .8, .7, .6], (1, 1, 1)),
                                 ([9., .7, .8, .6, .5], (1, 2, 2)),
                                 ([9., .6, .8, .7, .5], (1, 2, 3)),
                                 ([9., .5, .8, .7, .6], (1, 2, 4))):
            self.assertEqual(self.probe.first_positive_pair(scores, [1], [1, 2, 3, 4]), expected)
        self.assertIsNone(self.probe.first_positive_pair([9., .9, .8, .7], [], [1, 2, 3]))

    def test_gradient_fraction_uses_absolute_mass_only_on_eligible_columns(self):
        self.assertEqual(self.probe.gradient_fraction([100., -3., 2., 5.], 1, 2, [1, 2, 3]), .5)
        self.assertEqual(self.probe.gradient_fraction([0., 0.], 0, 1, [0, 1]), 0.)

    @staticmethod
    def rows(count, smooth=.2, rsk=.4):
        return [{name: {'fraction': fraction, 'margin_delta_A_first_order': -2.,
                       'margin_delta_C_first_order': 3., 'margin_delta_AC_first_order': 1.,
                       'score_gradient_pair_l1': 2. * fraction, 'score_gradient_l1': 2.,
                       'parameter_cosine_AC': .5}
                 for name, fraction in (('smooth_ap', smooth), ('rsk', rsk), ('hinge_calibration', 1.))}
                for _ in range(count)]

    def test_mechanism_present_requires_twenty_boundary_anchors_on_both_seeds(self):
        for count, expected in ((19, 'INCONCLUSIVE_HOLD'), (20, 'MECHANISM_PRESENT')):
            for counts in ((count, 20), (20, count)):
                results = [self.probe.summarize(self.rows(n), [2] * n) for n in counts]
                self.assertEqual(self.probe.preflight_decision(results), expected)

    def test_skip_requires_one_qualified_seed_below_twofold(self):
        passed = self.probe.summarize(self.rows(20), [2] * 20)
        for count, expected in ((19, 'INCONCLUSIVE_HOLD'), (20, 'SKIP_NATIVE')):
            result = self.probe.summarize(self.rows(count, rsk=.399), [2] * count)
            self.assertEqual(result['decision'], expected)
            self.assertEqual(self.probe.preflight_decision([result, passed]), expected)
        # A qualified failure remains a failure even if the other seed is inconclusive.
        for inconclusive in (self.probe.summarize([], []),
                             self.probe.summarize(self.rows(20, smooth=0.), [3] * 20)):
            for results in ([result, inconclusive], [inconclusive, result]):
                self.assertEqual(self.probe.preflight_decision(results), 'SKIP_NATIVE')

    def test_zero_baseline_and_empty_cohort_hold_instead_of_skip(self):
        passed = self.probe.summarize(self.rows(20), [2] * 20)
        for count in (0, 19, 20):
            result = self.probe.summarize(self.rows(count, smooth=0.), [3] * count)
            self.assertIsNone(result['fraction_ratio'])
            self.assertEqual(result['decision'], 'INCONCLUSIVE_HOLD')
            self.assertEqual(self.probe.preflight_decision([passed, result]), 'INCONCLUSIVE_HOLD')

    def test_seed_summary_always_reports_histogram_fractions_ratio_and_ac_split(self):
        rows = self.rows(1, smooth=.1, rsk=.2) + self.rows(1, smooth=.3, rsk=.6)
        rows[1]['rsk'].update(margin_delta_A_first_order=4., margin_delta_C_first_order=-1.,
                              margin_delta_AC_first_order=3.)
        result = self.probe.summarize(rows, [1, 1, 2, 3, 4, 8, None])
        self.assertEqual(result['anchors'], 2)
        self.assertEqual(result['first_positive_rank_histogram'], {'1': 2, '2': 1, '3': 1, '>3': 2})
        self.assertEqual(result['no_positive_anchors'], 1)
        self.assertEqual(result['mean_fraction'], {'smooth_ap': .2, 'rsk': .4, 'hinge_calibration': 1.})
        self.assertEqual(result['fraction_ratio'], 2.)
        self.assertEqual(result['mean_margin_delta_first_order'], {
            'smooth_ap': {'A': -2., 'C': 3., 'AC': 1.}, 'rsk': {'A': 1., 'C': 1., 'AC': 2.},
            'hinge_calibration': {'A': -2., 'C': 3., 'AC': 1.}})
        empty = self.probe.summarize([], [1, 1, None])
        self.assertEqual(empty['anchors'], 0)
        self.assertEqual(empty['first_positive_rank_histogram'], {'1': 2, '2': 0, '3': 0, '>3': 0})
        self.assertEqual(empty['mean_fraction'], {'smooth_ap': 0., 'rsk': 0., 'hinge_calibration': 0.})
        self.assertIsNone(empty['fraction_ratio'])
        self.assertEqual(empty['mean_margin_delta_first_order'], {
            name: {'A': None, 'C': None, 'AC': None} for name in ('smooth_ap', 'rsk', 'hinge_calibration')})

    def test_mass_weighting_and_cosine_are_diagnostics_only(self):
        rows = self.rows(1, .1, .2) + self.rows(1, .3, .6)
        for arm in rows[1].values():
            arm['score_gradient_l1'] = 8.
            arm['score_gradient_pair_l1'] = 8. * arm['fraction']
            arm['parameter_cosine_AC'] = -.5
        result = self.probe.summarize(rows, [2, 3])
        self.assertEqual(result['diagnostics']['mass_weighted_fraction'],
                         {'smooth_ap': .26, 'rsk': .52, 'hinge_calibration': 1.})
        self.assertEqual(result['diagnostics']['mass_weighted_fraction_ratio'], 2.)
        self.assertEqual(result['diagnostics']['mean_parameter_cosine_AC'],
                         {'smooth_ap': 0., 'rsk': 0., 'hinge_calibration': 0.})
        self.assertEqual(result['decision'], 'INCONCLUSIVE_HOLD')
        for row in rows:
            row['rsk']['parameter_cosine_AC'] = None
        result = self.probe.summarize(rows, [2, 3])
        self.assertIsNone(result['diagnostics']['mean_parameter_cosine_AC']['rsk'])
        self.assertEqual(result['diagnostics']['parameter_cosine_count']['rsk'], 0)
        empty = self.probe.summarize([], [])
        self.assertIsNone(empty['diagnostics']['mass_weighted_fraction_ratio'])

    def test_killed_hinge_calibration_uses_the_same_top_pair(self):
        with ExitStack() as stack:
            fake_native(stack)
            for positive, expected, slope in ((.3, .06, -1.), (.5, 0., 0.)):
                scores = Tensor([Dual(positive, (1.,) * 6), Dual(.31)])
                result = self.probe.top_pair_hinge(scores, 0, 1).data[0]
                self.assertAlmostEqual(result.value, expected, places=14)
                self.assertEqual(result.derivative[0], slope)

    def test_source_facts_bind_loss_and_probe_and_detect_later_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(self.probe, 'HERE', Path(directory)):
            paths = [Path(directory) / name for name in ('connected_rsk_loss.py', 'preflight_siglip2_connected_rsk.py')]
            for path in paths:
                path.write_text(path.name)
            facts = self.probe.source_facts()
            self.assertEqual({fact['path'] for fact in facts}, {str(p) for p in paths})
            for fact in facts:
                self.probe.check_file(fact)
            paths[0].write_text('changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.probe.check_file(next(f for f in facts if f['path'] == str(paths[0])))

    def test_fixed_inputs_are_fit_cache_and_both_cpu_v5_control_initializers(self):
        self.assertEqual(self.probe.CACHE['sha256'], '55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8')
        self.assertEqual(set(self.probe.INITIALIZERS), {179061, 179069})
        for seed, fact in self.probe.INITIALIZERS.items():
            self.assertEqual(fact['path'], f'/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-{seed}.pt')

    def test_file_absence_and_hash_mismatch_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            fact = {'path': str(path), 'sha256': hashlib.sha256(b'abc').hexdigest()}
            with self.assertRaises(FileNotFoundError):
                self.probe.check_file(fact)
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                self.probe.check_file(fact)
            path.write_bytes(b'abc')
            self.probe.check_file(fact)

    def test_small_numpy_cache_parser_rejects_object_or_wrong_length(self):
        header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (1, 2)}).encode() + b'\n'
        raw = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<2f', 1., 2.)
        shape, payload = self.probe.npy_payload(raw)
        self.assertEqual(shape, (1, 2))
        self.assertEqual(struct.unpack('<2f', payload), (1., 2.))
        for bad in (raw[:-1], raw.replace(b'<f4', b'|O8')):
            with self.assertRaises(ValueError):
                self.probe.npy_payload(bad)

    def test_parameter_cosine_and_step_match_explicit_ac_gradients(self):
        loss_raw = Tensor([1., 2., 3., 4., 5., 6.], (3, 2))
        margin_raw = Tensor([.1, .3, -.2, .4, .6, -.5], (3, 2))
        basis = Tensor([1., 2., 3., 4., 5., 6.], (3, 2))
        indices = [0, 1, 2]
        bases = (basis, Tensor([3., -1., 2., 4., 0., -2.], (3, 2)))
        gradients = [[sum(raw.data[i * 2 + o].value * b.data[i * 2 + j].value for i in range(3))
                      for b in bases for o in range(2) for j in range(2)] for raw in (loss_raw, margin_raw)]
        dots = [sum(a * b for a, b in zip(gradients[0][start:start + 4], gradients[1][start:start + 4]))
                for start in (0, 4)]
        expected_cosine = -sum(dots) / math.sqrt(math.prod(sum(x * x for x in g) for g in gradients))
        with patch.object(Tensor, 'double', lambda value: value, create=True):
            actual = self.probe.parameter_diagnostics(loss_raw, margin_raw, bases, indices)
            self.assertAlmostEqual(actual['margin_delta_A_first_order'], -1e-4 * dots[0], places=14)
            self.assertAlmostEqual(actual['margin_delta_C_first_order'], -1e-4 * dots[1], places=14)
            self.assertAlmostEqual(actual['margin_delta_AC_first_order'], -1e-4 * sum(dots), places=14)
            self.assertAlmostEqual(actual['parameter_cosine_AC'], expected_cosine, places=14)
            scaled = self.probe.parameter_diagnostics(loss_raw * 7., margin_raw, bases, indices)
            self.assertAlmostEqual(scaled['parameter_cosine_AC'], expected_cosine, places=14)
            zero = self.probe.parameter_diagnostics(loss_raw * 0., margin_raw, bases, indices)
            self.assertIsNone(zero['parameter_cosine_AC'])

    def test_score_row_jacobian_includes_query_and_gallery_paths(self):
        raw = [[1., 2.], [3., 1.], [-1., 1.]]
        norms = [math.sqrt(sum(x * x for x in row)) for row in raw]
        unit = Tensor([x / norm for row, norm in zip(raw, norms) for x in row], (3, 2))
        weights = [0., -.3, .2]
        def assign(tensor, index, row):
            self.assertEqual(row.shape, (tensor.shape[1],))
            tensor.data[index * 2:(index + 1) * 2] = row.data
        with patch.object(Tensor, '__setitem__', assign, create=True):
            actual = self.probe.score_gradient_to_raw(unit, Tensor(norms), 0, Tensor(weights))
        def loss(values):
            units = [[x / math.sqrt(sum(z * z for z in row)) for x in row] for row in values]
            return sum(g * sum(x * y for x, y in zip(units[0], row)) for g, row in zip(weights, units))
        for i in range(3):
            for j in range(2):
                hi, lo = [r[:] for r in raw], [r[:] for r in raw]
                hi[i][j] += 1e-6
                lo[i][j] -= 1e-6
                self.assertAlmostEqual(actual.data[i * 2 + j].value, (loss(hi) - loss(lo)) / 2e-6, places=9)


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('source checks require assertions')
    result = unittest.main(defaultTest='TorchFP32Tests' if TORCH_FP32 else
                           ['RankingTests', 'MutationTests', 'ArmTests', 'PreflightTests'],
                           verbosity=2, exit=False).result
    if not TORCH_FP32:
        assert not {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'} & {
            name.split('.')[0] for name in sys.modules}, 'native imports escaped source-only fixtures'
    raise SystemExit(not result.wasSuccessful())
