#!/usr/bin/env python3
"""Bounded stdlib falsifiers only; torch, CUDA, images, Cutile and every original native gate are UNRUN.

Run narrow (root contract): ulimit -v 1048576; timeout 15 python -B scripts/test_connected_asymmetric_cache.py
"""
import ast
import contextlib
import copy
import gc
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
DRIVER = HERE/'diagnose_connected_asymmetric_cache.py'


def load_driver():
    spec = importlib.util.spec_from_file_location('_asymmetric_test', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v3():
    spec = importlib.util.spec_from_file_location('_asymmetric_v3_test', HERE/'diagnose_connected_gallery_freshness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree(path):
    return ast.parse(Path(path).read_bytes())


def functions(node):
    return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


def signature(node):
    return ast.unparse(node.args)


def frozen_v5():
    """The v5 observer pinned by execution.json lives only in git history (a20c1f35)."""
    try:
        return {n: subprocess.run(['git', 'show', 'a20c1f35:scripts/'+n], cwd=ROOT, check=True, capture_output=True,
                                  timeout=10).stdout for n in ('observe_connected_control_batch_execution.py',
                                                              'test_connected_control_batch_execution.py')}
    except (OSError, subprocess.SubprocessError):
        return None


def fact(path, sha):
    return {'path': path, 'sha256': sha}


def good_launch(d, output='/tmp/asymmetric-out'):
    return {'schema': d.SCHEMA, 'execution_sha256': 'e'*64, 'stage': 'first', 'seeds': [179061], 'output': output,
        'historical': {'source': fact('/r/v3/diagnose_connected_gallery_freshness.py', d.V3['source']),
                       'execution': fact('/r/v3/execution.json', d.V3['execution']),
                       'authority': fact('/r/v3/authority.json', d.V3['authority'])},
        'observer': {'source': fact('/r/v5/observe_connected_control_batch_execution.py', d.V5['source']),
                     'test': fact('/r/v5/test_connected_control_batch_execution.py', d.V5['test']),
                     'execution': fact('/r/v5/execution.json', 'a'*64)},
        'evaluator': copy.deepcopy(d.EVALUATOR),
        'evaluation_authority': {'path': d.EVALUATOR['root']+'/'+d.EVALUATION_AUTHORITY['name'],
                                 'sha256': d.EVALUATION_AUTHORITY['sha256']},
        'native': {'authority': fact('/r/n/native-authority.json', d.NATIVE_AUTHORITY_SHA),
                   'library': fact('/r/n/candidate.so', d.BINARY_SHA), 'archived_control_binary_ack': True},
        'serving': {'request_driver': fact('/r/s/qualify_connected_serving_requests.py', d.REQUEST_DRIVER_SHA),
                    'observer': fact('/r/s/observe_connected_serving.py', d.SERVING_OBSERVER_SHA),
                    'control_native': fact('/r/s/connected_control_native_authority.py', d.CONTROL_NATIVE_SHA),
                    'native_wrapper': fact('/r/w/sfora/cutile_int8.py', d.WRAPPER_SHA),
                    'packed': fact('/r/w/sfora/packed_int8.py', d.PACKED_SHA)},
        'controls': {d.KEY: {'bundle': fact('/r/c/bundle.json', d.CONTROL_BUNDLE_SHA),
                             'export': fact('/r/c/receipt.json', d.EXPORT_SHA)}},
        'tail_oracle': {**copy.deepcopy(d.TAIL), 'images': copy.deepcopy(d.TAIL_IMAGES)},
        'resource_policy': dict(d.LIMITS), 'locks': [{'path': '/r/l1', 'fd': 8}, {'path': '/r/l2', 'fd': 9}],
        'both_locks_held': True, 'candidate_status': 'KILL', 'qualification_eligible': False,
        'state_reuse_eligible': False}


def synthetic_panel(d):
    tail = d.TAIL['selection_rows']
    other = [i for i in range(d.PANEL_ROWS) if i not in tail]
    query, gallery = other[:1728]+tail, other[1728:]
    rows = list(range(d.PANEL_ROWS))
    for ordinal, original in zip(tail, d.TAIL['fit_rows'], strict=True):
        rows[ordinal] = original
    assert len(query) == 1734 and len(gallery) == 1715
    fit = {'targets': [0]*13283, 'rows': [{'train_row': i} for i in range(13283)]}
    return {'original_rows': rows, 'query': query, 'gallery': gallery}, fit


def output_bytes(d):
    return {n: bytes(d.PANEL_ROWS*d.WIDTHS[n]) for n in d.OUTPUT_NAMES}


def differences(d, mutate):
    """Real v3 localization over synthetic equal byte outputs with the given (output, row) bytes flipped."""
    v3 = load_v3()
    panel, fit = synthetic_panel(d)
    expected = output_bytes(d)
    actual = {n: bytearray(v) for n, v in expected.items()}
    for name, row in mutate:
        actual[name][row*d.WIDTHS[name]] ^= 1
    return v3.control_byte_differences({n: bytes(v) for n, v in actual.items()}, expected, panel, fit)


class PinsAndApis(unittest.TestCase):
    def test_pins_equal_current_repository_sources(self):
        d = load_driver()
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.assertEqual(digest(HERE/'evaluate_siglip2_connected_mlp.py'), d.EVALUATOR['code'][d.EVALUATOR_NAME])
        self.assertEqual(digest(HERE/'connected_control_native_authority.py'), d.CONTROL_NATIVE_SHA)
        self.assertEqual(digest(HERE/'qualify_connected_serving_requests.py'), d.REQUEST_DRIVER_SHA)
        self.assertEqual(digest(HERE/'observe_connected_serving.py'), d.SERVING_OBSERVER_SHA)
        self.assertEqual(digest(ROOT/'src/sfora/cutile_int8.py'), d.WRAPPER_SHA)
        self.assertEqual(digest(ROOT/'src/sfora/packed_int8.py'), d.PACKED_SHA)
        preflight = json.loads((EVIDENCE/'connected-asymmetry-plan-20261009/root-input-preflight.json').read_text())
        self.assertEqual(d.TAIL_IMAGES, [{'path': r['path'], 'sha256': r['image_sha256']} for r in preflight['batch']['rows']])
        self.assertEqual(digest(HERE/'diagnose_connected_gallery_freshness.py'), d.V3['source'])
        freeze = EVIDENCE/'connected-gallery-freshness-v3-freeze'
        self.assertEqual(digest(freeze/'execution.json'), d.V3['execution'])
        self.assertEqual(digest(freeze/'authority.json'), d.V3['authority'])
        v5 = EVIDENCE/'connected-control-batch-execution-v5-freeze/execution.json'
        self.assertEqual(json.loads(v5.read_text()), {d.V5_NAMES[k]: d.V5[k] for k in d.V5})
        pinned = frozen_v5()
        if pinned is not None:
            self.assertEqual({k: hashlib.sha256(v).hexdigest() for k, v in pinned.items()},
                             {d.V5_NAMES['source']: d.V5['source'], d.V5_NAMES['test']: d.V5['test']})

    def test_oracle_source_is_the_admitted_function(self):
        d = load_driver()
        for name in ('evaluate_siglip2_connected_mlp.py', 'evaluate_siglip2_connected_probe.py'):
            d.check_oracle_source((HERE/name).read_bytes())
        for raw in ((HERE/'evaluate_siglip2_identity_diversity.py').read_bytes(),
                    (HERE/'evaluate_siglip2_connected_mlp.py').read_bytes().replace(
                        b'F.linear(features-mu,C)', b'F.linear(features,C)', 1)):
            with self.assertRaises(ValueError):
                d.check_oracle_source(raw)

    def test_evaluator_chain_source_contract(self):
        d = load_driver()
        native = tree(HERE/'connected_control_native_authority.py')
        loader = functions(native)['load_evaluator_source']
        names = [n.value for n in ast.walk(loader) if isinstance(n, ast.Constant) and n.value == d.EVALUATOR_NAME]
        self.assertEqual(len(names), 1, 'unchanged helper must already name the MLP evaluator')
        combined = {n.name: {m.name: signature(m) for m in n.body if isinstance(m, ast.FunctionDef)}
                    for n in native.body if isinstance(n, ast.ClassDef)}['CombinedAuthority']
        self.assertEqual(combined['__init__'], 'self, context, fact, observer, request')
        self.assertEqual(combined['install'], 'self, evaluator_source, evaluation_context')
        self.assertEqual(signature(functions(native)['validate_runtime_compiler']), 'record, observer')
        evaluator = functions(tree(HERE/'evaluate_siglip2_connected_mlp.py'))
        for name, expected in (('authority', 'args'), ('native_start', 'context'),
                               ('fullfeature_oracle', 'context, state, features'), ('merge_guards', 'target, values'),
                               ('guard_helpers', 'context'), ('resources', 'context, before'),
                               ('exit_rehash', 'context, original_guard'), ('check_code', 'value, names, pins=None'),
                               ('closure', 'root, expected, names, guards'), ('label', 'endpoint')):
            self.assertEqual(signature(evaluator[name]), expected, name)
        wrapper = functions(tree(HERE/'evaluate_siglip2_connected_mlp.py'))['_capture_source_builtins']
        inner = [n for n in ast.walk(wrapper) if isinstance(n, ast.FunctionDef) and n.name == 'source_live_guard']
        self.assertEqual([signature(n) for n in inner], ['module, digest, guards, names=None, class_name=None'],
                         'v3/v5/driver call source_live_guard(module, sha, guards, class_name=...) through this wrapper')
        target = ast.dump(ast.parse("t['nearest'].native_source_api(t)", mode='eval').body)
        hits = [n for n in ast.walk(evaluator['exit_rehash']) if ast.dump(n) == target]
        self.assertEqual(len(hits), 1, 'install substitution target must occur exactly once')

    def test_helper_signatures_in_pinned_sources(self):
        v3 = functions(tree(HERE/'diagnose_connected_gallery_freshness.py'))
        for name, expected in (('prepare', 'args'), ('terminal_admission', 'context, sources'),
                ('fresh_values', 'context, sources, endpoint'), ('scorer', 'sources, context'),
                ('native_wire_quality', 'wire, labels, query, gallery'), ('replay', 'expected, actual'),
                ('stale_values', 'context, sources, endpoint, features, budget, *, mutant=None'),
                ('endpoint_payload', 'context, sources, endpoint'),
                ('control_byte_differences', 'actual, expected, panel, fit'), ('byte_differences', 'actual, expected'),
                ('require_control_tap', 'differences'), ('compose', 'stale, fresh, query, gallery, cell'),
                ('output_bytes', 'values'), ('exact_wire', 'actual, expected'),
                ('final_state', 'budget, source, initializer, before, before_rng, flags, receipt'),
                ('write_json', 'path, value'), ('decode_wire', 'wire, count'), ('origin_audit', 'context, sources')):
            self.assertEqual(signature(v3[name]), expected, name)
        pinned = frozen_v5()
        if pinned is None:
            self.skipTest('frozen v5 observer blob unavailable')
        v5 = functions(ast.parse(pinned['observe_connected_control_batch_execution.py']))
        for name, expected in (('authenticated', 'fact, guards, *, keep=False'),
                ('capture_inference_outputs', 'connected, endpoint, images, expected_pixels=None'),
                ('readout', 'connected, owner, state, features'), ('exact', 'actual, expected, message'),
                ('tensor_bytes', 'value'), ('capture_workspace_owner', 'torch, context'),
                ('check_capture_ast', 'original_raw, observer_raw=None'), ('exact_four', 'context, sources, origins')):
            self.assertEqual(signature(v5[name]), expected, name)

    def test_serving_and_native_apis(self):
        requests = tree(HERE/'qualify_connected_serving_requests.py')
        classes = {n.name: {m.name: signature(m) for m in n.body if isinstance(m, ast.FunctionDef)}
                   for n in requests.body if isinstance(n, ast.ClassDef)}
        self.assertEqual(classes['Source']['__init__'], 'self, module, fact, *, packed_source=None')
        self.assertEqual(classes['Source']['load'], 'cls, fact')
        self.assertEqual(classes['Locks']['__init__'], 'self, rows')
        fns = functions(requests)
        self.assertEqual(signature(fns['native_ties']), 'packed, gallery_type, native, observer')
        self.assertEqual(signature(fns['raise_failures']), 'failures')
        observer = functions(tree(HERE/'observe_connected_serving.py'))
        self.assertEqual(signature(observer['file_bytes']), 'fact, *, keep=False')
        self.assertEqual(signature(observer['native_snapshot']), 'result')
        wrapper = {n.name: {m.name: signature(m) for m in n.body if isinstance(m, ast.FunctionDef)}
                   for n in tree(ROOT/'src/sfora/cutile_int8.py').body if isinstance(n, ast.ClassDef)}
        gallery = wrapper['CutilePackedInt8Gallery']
        self.assertEqual(gallery['open_packed'], 'cls, library_path: Path, embeddings: PackedInt8Embeddings')
        self.assertEqual(gallery['search_packed'], 'self, embeddings: PackedInt8Embeddings, *, k: int=_TOP_K')
        packed = {n.name: {m.name: signature(m) for m in n.body if isinstance(m, ast.FunctionDef)}
                  for n in tree(ROOT/'src/sfora/packed_int8.py').body if isinstance(n, ast.ClassDef)}
        self.assertEqual(packed['PackedInt8Embeddings']['from_bytes'], 'cls, wire: bytes, *, count: int, dimensions: int')

    def test_kernel_score_formula_matches_reference_arithmetic(self):
        """(int32 dot -> f32) * query_inv * gallery_inv, lowest-ordinal ties: the reference_topk contract."""
        kernel = (ROOT/'rust/sfora-cutile-int8-score/src/topk.rs').read_text()
        self.assertIn('* query_scale.broadcast', kernel)
        self.assertIn('* gallery_scale.broadcast', kernel)
        self.assertIn('reduce_min(eligible_ordinals', kernel)
        source = DRIVER.read_text()
        self.assertIn(').float()*inv[rows, None]*inv[None, gallery]', source)
        self.assertIn('stable=True', source)


class DriverStructure(unittest.TestCase):
    def setUp(self):
        self.source = DRIVER.read_text()
        self.tree = ast.parse(self.source)
        self.functions = functions(self.tree)

    def calls(self, name):
        return [ast.unparse(n.func) for n in ast.walk(self.functions[name]) if isinstance(n, ast.Call)]

    def test_no_native_import_at_module_level_and_hygiene(self):
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, 'module', None) or '']
                self.assertFalse({n.split('.')[0] for n in names} &
                                 {'torch', 'numpy', 'PIL', 'sfora', 'transformers', 'safetensors', 'torchvision'})
        code = ("import importlib.util,sys,json;s=importlib.util.spec_from_file_location('d',%r);m=importlib.util.module_from_spec(s);"
                "s.loader.exec_module(m);print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in "
                "{'torch','numpy','PIL','sfora','transformers','safetensors','torchvision'})))") % str(DRIVER)
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), [], result.stderr)

    def test_sfora_and_images_only_where_allowed(self):
        owners = {}
        for name, node in self.functions.items():
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module == 'sfora':
                    owners.setdefault('sfora', []).append(name)
                if isinstance(child, ast.ImportFrom) and child.module == 'PIL':
                    owners.setdefault('PIL', []).append(name)
                if isinstance(child, ast.Call) and ast.unparse(child.func) == 'Image.open':
                    owners.setdefault('Image.open', []).append(name)
        self.assertEqual(owners, {'sfora': ['load_native'], 'PIL': ['tail_replay'], 'Image.open': ['tail_replay']})
        self.assertEqual(sum(1 for c in self.calls('tail_replay') if c == 'v5.authenticated'), 1)

    def test_forbidden_gates_are_never_used(self):
        for forbidden in ('bundle_reads_only', 'grouped_md_origin', 'images_outputs', 'export_pass', 'fullfeature_raw_features'):
            self.assertNotIn(forbidden, self.source.replace('fullfeature_oracle', ''), forbidden)
        names = {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn('exact_four', {c.split('.')[-1] for c in self.calls('native_stage')+self.calls('load_native')})
        self.assertIn('exact_four', names)

    def test_no_decision_ss_sf_or_effects_cells(self):
        keys = {k.value for n in ast.walk(self.tree) if isinstance(n, ast.Dict) for k in n.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        self.assertFalse(keys & {'decision', 'SS', 'SF', 'effects', 'GO', 'CONTINUE'})
        self.assertEqual({'FF', 'FS'}, {k for k in keys if k in ('FF', 'FS')})
        self.assertIn("'FS'", self.source)
        compose_cells = [ast.unparse(n.args[-1]) for n in ast.walk(self.tree)
                         if isinstance(n, ast.Call) and ast.unparse(n.func).endswith('.compose')]
        self.assertEqual(set(compose_cells), {"'FS'"})

    def sequence(self, function, names):
        """First call site of each name must appear in the given order."""
        first = {}
        for node in ast.walk(self.functions[function]):
            if isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                if text in names:
                    first[text] = min(first.get(text, 10**9), node.lineno)
        self.assertEqual(set(first), set(names), 'missing call in '+function)
        lines = [first[n] for n in names]
        self.assertEqual(lines, sorted(lines), dict(first))

    def test_body_order_replays_tail_candidate_before_fs_native_before_quality(self):
        self.sequence('body', ['archive_replays', 'control_phase', 'tail_replay', 'candidate_phase', 'S.audit',
                               'S.v5.exact_four', 'fs_phase', 'native_stage', 'quality_stage'])
        self.assertLess(self.source.index('def load_native'), self.source.index('def native_stage'))
        stage = self.calls('native_stage')
        self.assertLess(stage.index('reference_topk'), stage.index('load_native'))
        self.assertIn('compare_native', stage)
        # No metric function is reachable before native parity: scorer/fixed only in archive_replays and quality_stage.
        users = [n for n, f in self.functions.items() if any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'S.fixed'
                                                             for c in ast.walk(f))]
        self.assertEqual(sorted(users), ['archive_replays', 'quality_stage'])

    def test_admit_and_native_start_order(self):
        self.sequence('admit', ['prepare', 'requests.Source.load', 'native.load_evaluator_source', 'd.prepare',
                                'evaluator.authority', 'native.CombinedAuthority', 'combined.install'])
        start = [ast.unparse(n.func) for n in ast.walk(self.functions['start_native']) if isinstance(n, ast.Call)]
        self.assertLess(start.index('evaluator.native_start'), start.index('v5.capture_workspace_owner'))
        text = ast.get_source_segment(self.source, self.functions['admit'])
        self.assertLess(text.index('combined.install'), text.index("S.prospective['output'].mkdir()"))

    def test_run_cleans_up_in_finally_and_returns_only_the_published_receipt(self):
        run = self.functions['run']
        trys = [n for n in run.body if isinstance(n, ast.Try)]
        self.assertEqual(len(trys), 1)
        self.assertEqual([ast.unparse(s) for s in trys[0].finalbody], ['cleanup(S, error)'])
        self.assertEqual([ast.unparse(s) for s in run.body[run.body.index(trys[0])+1:]], ['return S.receipt'])
        handler = trys[0].handlers[0]
        self.assertEqual(ast.unparse(handler.type), 'BaseException')
        self.assertEqual([ast.unparse(s) for s in handler.body],
                         ['error = failure', 'report(failure)', 'clear_frames(failure)'])
        self.assertNotIn('__traceback__ = None', ast.get_source_segment(self.source, run))
        text = ast.get_source_segment(self.source, run)
        self.assertNotIn('write_json', text)      # nothing is published from run(); cleanup() owns publication

    def test_cleanup_action_order_is_fixed(self):
        text = ast.get_source_segment(self.source, self.functions['cleanup'])
        marks = ["actions.append(checked(lambda: snap(S, 'before_exit')))", 'actions.append(checked(release))',
                 'actions.append(checked(maps))', 'actions.append(exit_evaluator)', "rehash(S.context['guards'])",
                 "rehash(S.prospective['guards'])", 'actions.append(S.budget.check)',
                 'actions.append(lambda: final_guard(S))', 'actions.append(checked(S.sources.close))',
                 'actions.append(checked(final_resources))', 'actions.append(lambda: terminal_checks(S))',
                 'failures = attempt(actions)', 'attempt([lambda: stage_receipt(S)])',
                 'attempt([lambda: registry_dispose(S)])', 'attempt([lambda: publish_receipt(S)])',
                 "receipt.json.partial').unlink(missing_ok=True)", 'raise_primary(error, failures)']
        positions = [text.index(m) for m in marks]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('api.evaluator_exit(S.context_e, S.exit_guard)', text)
        self.assertIn('S.evaluator.exit_rehash(S.context_e, S.exit_guard)', text)
        self.assertIn('independent_exit(S)', text)
        self.assertNotIn('load_native', text, 'Cutile is never mapped to rescue an error')
        self.assertIn('error is None and not failures and S.receipt is not None', text)

class LaunchShape(unittest.TestCase):
    def setUp(self):
        self.d = load_driver()
        self.launch = good_launch(self.d)

    def check(self, launch=None, output='/tmp/asymmetric-out'):
        self.d.check_launch(self.launch if launch is None else launch, 'e'*64, Path(output))

    def test_accepts_the_exact_authority(self):
        self.check()

    def test_rejects_each_fixed_value_and_pin_change(self):
        mutations = {
            'extra key': lambda l: l.update(extra=1), 'missing key': lambda l: l.pop('locks'),
            'schema': lambda l: l.update(schema='x'), 'execution': lambda l: l.update(execution_sha256='f'*64),
            'stage': lambda l: l.update(stage='full'), 'seeds': lambda l: l.update(seeds=[179061, 179069]),
            'seed': lambda l: l.update(seeds=[179069]), 'policy seconds': lambda l: l['resource_policy'].update(seconds=701),
            'policy host': lambda l: l['resource_policy'].update(host_bytes=9*1024**3),
            'policy swap': lambda l: l['resource_policy'].update(swap_bytes=1),
            'policy cuda': lambda l: l['resource_policy'].update(cuda_allocated_bytes_exclusive=10**10+1),
            'locks flag': lambda l: l.update(both_locks_held=False),
            'candidate status': lambda l: l.update(candidate_status='GO'),
            'qualification': lambda l: l.update(qualification_eligible=True),
            'state reuse': lambda l: l.update(state_reuse_eligible=True),
            'output': lambda l: l.update(output='/tmp/other'),
            'one lock': lambda l: l.update(locks=l['locks'][:1]),
            'lock fd': lambda l: l['locks'][0].update(fd=2),
            'v3 source': lambda l: l['historical']['source'].update(sha256='0'*64),
            'v3 execution role': lambda l: l['historical']['execution'].update(path='/r/other/execution.json'),
            'v5 source': lambda l: l['observer']['source'].update(sha256='0'*64),
            'v5 test name': lambda l: l['observer']['test'].update(path='/r/v5/other.py'),
            'evaluator code': lambda l: l['evaluator']['code'].update({self.d.EVALUATOR_NAME: '0'*64}),
            'evaluator exec': lambda l: l['evaluator'].update(execution_sha256='0'*64),
            'evaluation authority': lambda l: l['evaluation_authority'].update(sha256='0'*64),
            'native authority': lambda l: l['native']['authority'].update(sha256='0'*64),
            'native library': lambda l: l['native']['library'].update(sha256='0'*64),
            'native ack': lambda l: l['native'].update(archived_control_binary_ack=False),
            'control native': lambda l: l['serving']['control_native'].update(sha256='0'*64),
            'serving role name': lambda l: l['serving']['observer'].update(path='/r/s/other.py'),
            'serving split wheel': lambda l: l['serving']['packed'].update(path='/r/z/sfora/packed_int8.py'),
            'request driver sha (arbitrary launch hash)': lambda l: l['serving']['request_driver'].update(sha256='1'*64),
            'serving observer sha (arbitrary launch hash)': lambda l: l['serving']['observer'].update(sha256='2'*64),
            'wrapper sha (arbitrary launch hash)': lambda l: l['serving']['native_wrapper'].update(sha256='3'*64),
            'packed sha (arbitrary launch hash)': lambda l: l['serving']['packed'].update(sha256='4'*64),
            'forbidden image with a valid sha': lambda l: l['tail_oracle']['images'].__setitem__(
                0, fact('/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Dresses/id_00000041/03_1_front.jpg', 'f'*64)),
            'tail images reordered': lambda l: l['tail_oracle']['images'].reverse(),
            'tail image path substituted': lambda l: l['tail_oracle']['images'][2].update(path='/r/other.jpg'),
            'tail image sha substituted': lambda l: l['tail_oracle']['images'][3].update(sha256='0'*64),
            'control export': lambda l: l['controls'][self.d.KEY]['export'].update(sha256='0'*64),
            'control bundle': lambda l: l['controls'][self.d.KEY]['bundle'].update(sha256='0'*64),
            'extra control': lambda l: l['controls'].update({'control-179069': {}}),
            'tail rows': lambda l: l['tail_oracle']['selection_rows'].reverse(),
            'tail batch': lambda l: l['tail_oracle'].update(batch_index=53),
            'tail fit rows': lambda l: l['tail_oracle']['fit_rows'].__setitem__(0, 13216),
            'tail image count': lambda l: l['tail_oracle']['images'].pop(),
            'tail image duplicate': lambda l: l['tail_oracle']['images'].__setitem__(1, l['tail_oracle']['images'][0]),
            'relative file': lambda l: l['historical']['source'].update(path='relative.py'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                launch = copy.deepcopy(self.launch)
                mutate(launch)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.check(launch)


class TapPredicate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_driver()
        cls.tail = cls.d.TAIL['selection_rows']
        cls.clean = differences(cls.d, [])                      # the real v3 localization is ~0.4s per call: cache it
        cls.all_tail = differences(cls.d, [(n, r) for n in cls.d.OUTPUT_NAMES for r in cls.tail])
        cls.one_tail = differences(cls.d, [('raw', cls.tail[2])])

    def test_accepts_no_difference_and_tail_only_differences(self):
        self.assertEqual(self.d.require_tail_oracle_tap(self.clean), dict.fromkeys(self.d.OUTPUT_NAMES, []))
        self.assertEqual(self.d.require_tail_oracle_tap(self.all_tail), dict.fromkeys(self.d.OUTPUT_NAMES, self.tail))
        one = self.d.require_tail_oracle_tap(self.one_tail)
        self.assertEqual(one['raw'], [self.tail[2]])

    def test_rejects_any_gallery_or_non_tail_row_in_any_output(self):
        panel, _ = synthetic_panel(self.d)
        # last gallery row is the B19 gallery tail; query[1727] is the last row of the full B32 before the B6 tail
        for row in (panel['gallery'][-1], panel['gallery'][0], panel['query'][0], panel['query'][1727]):
            bad = differences(self.d, [(n, row) for n in self.d.OUTPUT_NAMES])
            for name in self.d.OUTPUT_NAMES:
                partial = copy.deepcopy(self.clean)
                partial[name] = bad[name]      # exactly one output carries the offending row
                with self.subTest(row=row, output=name), self.assertRaises(ValueError):
                    self.d.require_tail_oracle_tap(partial)

    def test_rejects_a_tail_difference_hidden_beside_a_gallery_difference(self):
        panel, _ = synthetic_panel(self.d)
        with self.assertRaises(ValueError):
            self.d.require_tail_oracle_tap(differences(self.d, [('wire', self.tail[0]), ('wire', panel['gallery'][5])]))

    def test_rejects_truncation_unavailable_localization_and_unattributed_bytes(self):
        base = self.one_tail
        for name, edit in (
                ('truncated actual', lambda x: x['raw']['row_localization'].update(actual_bytes=x['raw']['row_localization']['actual_bytes']-512)),
                ('truncated expected', lambda x: x['unit']['row_localization'].update(expected_bytes=0)),
                ('unavailable', lambda x: x['codes']['row_localization'].update(status='UNAVAILABLE')),
                ('unattributed bytes', lambda x: x['inverse_norms'].update(different_bytes=7)),
                ('wrong width', lambda x: x['wire']['row_localization'].update(row_bytes=128)),
                ('missing output', lambda x: x.pop('wire')),
                ('wrong batch size', lambda x: x['raw']['row_localization']['rows'][0].update(live_encoder_batch_size=32)),
                ('wrong batch index', lambda x: x['raw']['row_localization']['rows'][0].update(live_encoder_batch_index=53)),
                ('not tail flag', lambda x: x['raw']['row_localization']['rows'][0].update(live_encoder_tail=False)),
                ('wrong role', lambda x: x['raw']['row_localization']['rows'][0].update(role='gallery')),
                ('wrong fit row', lambda x: x['raw']['row_localization']['rows'][0].update(original_fit_index=13216)),
                ('duplicate row', lambda x: x['raw']['row_localization']['rows'].append(dict(x['raw']['row_localization']['rows'][0])))):
            with self.subTest(name):
                broken = copy.deepcopy(base)
                edit(broken)
                with self.assertRaises((ValueError, KeyError)):
                    self.d.require_tail_oracle_tap(broken)


class Falsifiers(unittest.TestCase):
    def setUp(self):
        self.d = load_driver()
        self.panel, _ = synthetic_panel(self.d)

    def wires(self):
        d = self.d
        f = b''.join(struct.pack('<128be', *([i % 7]*128), 1.0) for i in range(d.PANEL_ROWS))
        s = bytearray(f)
        return f, bytes(s)

    def test_deny_group_images_allows_only_the_exact_six(self):
        d = self.d
        d.deny_group_images(list(d.TAIL['fit_rows']))
        for rows in ([13216] + d.TAIL['fit_rows'][1:], d.TAIL['fit_rows'] + [13248], d.TAIL['fit_rows'][::-1], [],
                     d.TAIL['fit_rows'][:5], [13248] * 6):
            with self.assertRaises(ValueError):
                d.deny_group_images(rows)

    def test_fs_composition_detects_flipped_query_or_gallery_byte(self):
        d, panel = self.d, self.panel
        f, s = self.wires()
        s = bytearray(s)
        g = panel['gallery'][3]
        s[g*130] ^= 1
        fs = bytearray(f)
        for i in panel['gallery']:
            fs[i*130:(i+1)*130] = s[i*130:(i+1)*130]
        d.check_fs_composition(bytes(fs), f, bytes(s), panel['query'], panel['gallery'])
        for index, name in ((panel['query'][4], 'query'), (panel['gallery'][7], 'gallery')):
            broken = bytearray(fs)
            broken[index*130+5] ^= 1
            with self.subTest(name), self.assertRaises(ValueError):
                d.check_fs_composition(bytes(broken), f, bytes(s), panel['query'], panel['gallery'])
        with self.assertRaises(ValueError):
            d.check_fs_composition(bytes(fs)[:-130], f, bytes(s), panel['query'], panel['gallery'])
        with self.assertRaises(ValueError):
            d.check_fs_composition(bytes(fs), f, bytes(s), panel['query'][:-1], panel['gallery'])

    def test_gallery_difference_counts_are_descriptive_and_zero_rows_mean_no_asymmetry(self):
        f, _ = self.wires()
        zero = self.d.gallery_difference(f, f, self.panel['gallery'])
        self.assertEqual((zero['rows'], zero['bytes'], zero['of_rows'], zero['descriptive_only']), (0, 0, 1715, True))
        s = bytearray(f)
        s[self.panel['gallery'][-1]*130+129] ^= 1
        s[self.panel['gallery'][3]*130] ^= 3
        s[self.panel['gallery'][3]*130+1] ^= 1
        counts = self.d.gallery_difference(bytes(s), f, self.panel['gallery'])
        self.assertEqual((counts['rows'], counts['bytes']), (2, 3))
        s = bytearray(f)
        s[self.panel['query'][0]*130] ^= 1   # a differing QUERY row is not asymmetry evidence
        self.assertEqual(self.d.gallery_difference(bytes(s), f, self.panel['gallery'])['rows'], 0)

    def test_native_parity_is_exact_ids_ties_and_score_bits(self):
        d, rows = self.d, 3
        ids = struct.pack('<30q', *[(r*10+i) % 97 for r in range(rows) for i in range(10)])
        scores = struct.pack('<30f', *[1.0 - 0.01*i for r in range(rows) for i in range(10)])
        d.compare_native(ids, scores, ids, scores, rows)
        swapped = bytearray(ids)
        swapped[0:8], swapped[8:16] = ids[8:16], ids[0:8]   # swapped tied ordinals
        one_ulp = bytearray(scores)
        one_ulp[0] ^= 1
        for name, (a, b, rows_) in {'swap': (bytes(swapped), scores, rows), 'ulp': (ids, bytes(one_ulp), rows),
                                    'short': (ids[:-8], scores, rows), 'rows33': (ids, scores, 33),
                                    'rows0': (b'', b'', 0)}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                d.compare_native(a, b, ids, scores, rows_)
        with self.assertRaises(ValueError):
            d.compare_native(ids, scores, ids, bytes(one_ulp), rows)
        big = 33
        with self.assertRaises(ValueError):
            d.compare_native(bytes(big*80), bytes(big*40), bytes(big*80), bytes(big*40), big)

    def test_transitions_are_descriptive_never_a_gate(self):
        fs = {'per_query_r1': [1, 1, 0, 1], 'per_query_ap': [1., .5, .2, 1.], 'recall_at_1': .75, 'map_at_r': .675}
        ff = {'per_query_r1': [1, 0, 1, 1], 'per_query_ap': [1., .5, .5, 1.], 'recall_at_1': .75, 'map_at_r': .75}
        result = self.d.transitions(fs, ff)
        self.assertEqual((result['r1_up'], result['r1_down'], result['r1_net'], result['gate']), (1, 1, 0, False))
        with self.assertRaises(ValueError):
            self.d.transitions(fs, {**ff, 'per_query_r1': [1]})

    def test_persisted_wire_is_exclusive_and_read_back(self):
        d = self.d
        with tempfile.TemporaryDirectory() as temp:
            roles = {'schema': 'r', 'query': {'ordinals': [1, 2]}, 'gallery': {'ordinals': [0]}, 'causal_freshness_claim': False}
            persisted = d.persist_fs(temp, b'abc', roles)
            self.assertEqual(Path(persisted['wire']['path']).read_bytes(), b'abc')
            self.assertEqual(persisted['wire']['sha256'], hashlib.sha256(b'abc').hexdigest())
            self.assertEqual(json.loads(Path(persisted['roles']['path']).read_text()), roles)
            with self.assertRaises(FileExistsError):
                d.persist_fs(temp, b'abc', roles)
            real = Path.read_bytes
            with tempfile.TemporaryDirectory() as other, patch.object(Path, 'open', autospec=True) as opened:
                opened.side_effect = lambda self_, mode='r', *a, **k: __import__('io').BytesIO(b'tampered') if 'r' in mode else open(self_, mode)
                with self.assertRaises((ValueError, OSError)):
                    d.write_exclusive(Path(other)/'x.bin', b'payload')

    def test_authenticated_rejects_hash_symlink_relative_and_changed_files(self):
        d = self.d
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()/'a.json'
            path.write_bytes(b'{"a": 1}')
            good = {'path': str(path), 'sha256': hashlib.sha256(b'{"a": 1}').hexdigest()}
            guards = {}
            self.assertEqual(d.authenticated(good, guards, keep=True), b'{"a": 1}')
            self.assertEqual(guards, {str(path): good['sha256']})
            with self.assertRaises(ValueError):
                d.authenticated({**good, 'sha256': '0'*64}, {})
            with self.assertRaises(ValueError):
                d.authenticated(good, {str(path): '1'*64})
            link = Path(temp).resolve()/'link.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                d.authenticated({'path': str(link), 'sha256': good['sha256']}, {})
            with self.assertRaises(ValueError):
                d.authenticated({'path': 'a.json', 'sha256': good['sha256']}, {})
            with self.assertRaises(ValueError):
                d.read_json({'path': str(path), 'sha256': good['sha256']}, {}) and d.strict_json(b'{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                d.strict_json(b'{"a": NaN}')


class Binding(unittest.TestCase):
    def setUp(self):
        self.d = load_driver()
        d = self.d
        panel, _ = synthetic_panel(d)
        rows = []
        self.fit_rows = {}
        for ordinal, original in zip(d.TAIL['selection_rows'], d.TAIL['fit_rows'], strict=True):
            record = {'train_row': 100000+original, 'relative_path': f'Img/{original}.jpg', 'image_sha256': f'{original:064x}'}
            self.fit_rows[original] = record
            rows.append({'panel_ordinal': ordinal, 'original_row': original, 'role': 'query', 'path': f'/data/{original}.jpg',
                         **record})
        batch = {'role': 'query', 'rows': rows, 'rgb_sha256': 'r'*64, 'pixels_sha256': 'p'*64, 'outputs_sha256': 'o'*64}
        self.context = {'partition': {'panels': {'selection': panel, 'train': {'original_rows': [1, 2, 3]},
                                                 'validation': {'original_rows': [20000, 20001]}}},
                        'exports': {d.KEY: {'images': [None]*54+[batch]}}, 'fit': {'rows': self.fit_rows}}
        self.launch = {'tail_oracle': {'images': [{'path': r['path'], 'sha256': r['image_sha256']} for r in rows]}}

    def test_accepts_the_accepted_tail_and_rejects_every_deviation(self):
        d = self.d
        self.assertIs(d.bind_tail(self.context, self.launch), self.context['exports'][d.KEY]['images'][54])
        cases = {
            'swapped images': lambda c, l: l['tail_oracle']['images'].reverse(),
            'extra image sha': lambda c, l: l['tail_oracle']['images'][0].update(sha256='9'*64),
            'gallery role': lambda c, l: c['exports'][d.KEY]['images'][54].update(role='gallery'),
            'five rows': lambda c, l: c['exports'][d.KEY]['images'][54]['rows'].pop(),
            'row order': lambda c, l: c['exports'][d.KEY]['images'][54]['rows'].reverse(),
            'sealed group row': lambda c, l: c['exports'][d.KEY]['images'][54]['rows'][0].update(original_row=13216),
            'train overlap': lambda c, l: c['partition']['panels']['train'].update(original_rows=[13241]),
            'validation overlap': lambda c, l: c['partition']['panels']['validation'].update(original_rows=[13257]),
            'fit metadata': lambda c, l: self.fit_rows[13239].update(relative_path='Img/other.jpg'),
            'query slice': lambda c, l: c['partition']['panels']['selection']['query'].__setitem__(1730, 3440),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                context, launch = copy.deepcopy(self.context), copy.deepcopy(self.launch)
                if name == 'fit metadata':
                    context['fit']['rows'][13239] = {**context['fit']['rows'][13239], 'relative_path': 'Img/other.jpg'}
                else:
                    mutate(context, launch)
                with self.assertRaises((ValueError, KeyError, IndexError)):
                    self.d.bind_tail(context, launch)

    def test_bind_endpoints_requires_the_accepted_control_roles(self):
        d = self.d
        control = {'seed': 179061, 'arm': 'control', 'bundle': fact('/c/bundle.json', d.CONTROL_BUNDLE_SHA)}
        candidate = {'seed': 179061, 'arm': 'candidate', 'bundle': fact('/k/bundle.json', 'c'*64)}
        other = {'seed': 179069, 'arm': 'control', 'bundle': fact('/o/bundle.json', 'd'*64)}
        receipt = fact('/c/receipt.json', d.EXPORT_SHA)
        context = {'score': {'launch': {'endpoints': [control, candidate, other], 'exports': {d.KEY: {'receipt': receipt}}}},
                   'launch': {'sources': {'connected': {'sha256': d.CONNECTED_SHA}}}}
        launch = {'controls': {d.KEY: {'bundle': control['bundle'], 'export': receipt}}}
        self.assertEqual(d.bind_endpoints(context, launch), (control, candidate))
        for edit in (lambda c, l: l['controls'][d.KEY].update(export=fact('/c/receipt.json', '0'*64)),
                     lambda c, l: l['controls'][d.KEY].update(bundle=other['bundle']),
                     lambda c, l: c['launch']['sources']['connected'].update(sha256='0'*64)):
            c, l = copy.deepcopy(context), copy.deepcopy(launch)
            edit(c, l)
            with self.assertRaises(ValueError):
                d.bind_endpoints(c, l)


class PrepareCoversEveryFile(unittest.TestCase):
    def test_prepare_authenticates_exactly_the_non_image_files_and_never_opens_a_tail_image(self):
        d = load_driver()
        allowlist = {f['path'] for f in d.TAIL_IMAGES}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()/'new'
            launch = good_launch(d, str(output))
            seen = []
            v5_closure = {d.V5_NAMES[k]: d.V5[k] for k in d.V5}
            holder = [v5_closure]
            def fake(fact_, guards, *, keep=False):
                if fact_['path'] in allowlist:
                    raise AssertionError('tail image touched by prepare: '+fact_['path'])
                seen.append(fact_['path'])
                guards[fact_['path']] = fact_['sha256']
                if fact_['path'] == str(HERE/'execution.json'):
                    return json.dumps(dict.fromkeys(d.FILES, 'a'*64)).encode()
                if fact_['path'] == '/r/auth.json':
                    return json.dumps(launch).encode()
                if fact_['path'] == launch['observer']['execution']['path']:
                    return json.dumps(holder[0]).encode()
                return Path(fact_['path'])
            args = SimpleNamespace(execution_sha256='e'*64, authority=Path('/r/auth.json'), authority_sha256='b'*64, output=output)
            with patch.object(d, 'authenticated', fake):
                result = d.prepare(args)
            self.assertEqual(result['output'], output)
            expected = {f['path'] for k in ('historical', 'observer') for f in launch[k].values()}
            expected |= {launch['evaluation_authority']['path'], launch['native']['authority']['path'],
                         launch['native']['library']['path']}
            expected |= {f['path'] for f in launch['serving'].values()}
            expected |= {f['path'] for f in launch['controls'][d.KEY].values()}
            self.assertEqual(len(expected), 16)
            self.assertLessEqual(expected, set(seen))
            self.assertFalse(allowlist & set(seen))
            for bad in ({**v5_closure, 'extra': 'x'}, {**v5_closure, d.V5_NAMES['test']: '0'*64}):
                holder[0] = bad
                with patch.object(d, 'authenticated', fake), self.assertRaises(ValueError):
                    d.prepare(args)
            output.mkdir()
            holder[0] = v5_closure
            with patch.object(d, 'authenticated', fake), self.assertRaises(ValueError):
                d.prepare(args)

    def test_a_forbidden_image_with_a_valid_sha_is_rejected_with_zero_opens(self):
        d = load_driver()
        with tempfile.TemporaryDirectory() as temp:
            forbidden = Path(temp).resolve()/'sealed_val.jpg'
            forbidden.write_bytes(b'sealed validation image bytes')
            launch = good_launch(d)
            launch['tail_oracle']['images'][0] = {'path': str(forbidden), 'sha256': hashlib.sha256(forbidden.read_bytes()).hexdigest()}
            real_open = Path.open
            opened = []
            def tripwire(self_, *a, **k):
                opened.append(str(self_))
                return real_open(self_, *a, **k)
            with patch.object(Path, 'open', tripwire), patch('builtins.open', side_effect=AssertionError('open() called')):
                with self.assertRaises(ValueError):
                    d.check_launch(launch, 'e'*64, Path('/tmp/asymmetric-out'))
            self.assertNotIn(str(forbidden), opened)


def genuine(path, names, constants=(), **extra):
    """Genuine pinned functions extracted by AST (no import of the script, no torch)."""
    source_tree = ast.parse(Path(path).read_bytes())
    namespace = {'__name__': '_genuine_'+Path(path).stem, 'sys': sys, 'gc': gc, 'weakref': weakref, 'hashlib': hashlib,
                 'importlib': importlib, 'os': os, 're': re, 'time': time, 'Path': Path, **extra}
    for node in source_tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in constants):
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    body = [n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), names
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def silent():
    return contextlib.redirect_stderr(io.StringIO())


class FakeTorch:
    def __init__(self, allocated=0, cpu=1, cuda=(1,), peak=1):
        self.allocated, self.cpu, self.cuda_rng = allocated, cpu, list(cuda)
        self.cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: None,
                                    memory_allocated=lambda: self.allocated, max_memory_allocated=lambda: peak,
                                    get_rng_state_all=lambda: list(self.cuda_rng))
        self.random = SimpleNamespace(get_rng_state=lambda: self.cpu)

    def equal(self, a, b):
        return a == b


class World:
    """A fake S carrying everything cleanup()/final_guard()/terminal_checks()/stage_receipt() touch."""
    def __init__(self, test, *, fail=(), api=True, state=True, receipt=False, cap=False, allocated=0, flags_ok=True,
                 rng_ok=True, cgroup_ok=True, locks_fail=False):
        self.d, self.log, self.fs, self.fail = load_driver(), [], [], set(fail)
        self.temp = tempfile.TemporaryDirectory()
        test.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()/'out'
        self.output.mkdir()
        test.enterContext(patch.dict(os.environ, {'INVOCATION_ID': 'i'*32}))
        log, d = self.log, self.d

        def step(name):
            def call(*a, **k):
                log.append(name)
                if name in self.fail:
                    raise RuntimeError('secondary '+name)
            return call

        def budget():
            log.append('budget.check')
            self.fs.append(((self.output/'receipt.json.partial').exists(), (self.output/'receipt.json').exists()))
            if cap:
                raise ValueError('cap')
            if 'budget.check' in self.fail:
                raise RuntimeError('secondary budget.check')

        def locks():
            if locks_fail:
                raise ValueError('lock lost')
        connected = SimpleNamespace(release_inference=step('release'), mapping_absent=lambda p: log.append(('map', Path(p).name)))
        self.initializer = SimpleNamespace(admit_cgroup=step('terminal.admit_cgroup'))
        modules = {'connected': connected, 'initializer': self.initializer}
        self.sources = SimpleNamespace(modules=modules, guard=lambda: log.append('sources.guard'))

        def close():
            log.append('sources.close')
            modules.clear()
            if 'sources.close' in self.fail:
                raise RuntimeError('secondary sources.close')
        self.sources.close = close
        self.registry = dict(sys.modules)
        self.owned_module = types.ModuleType('_asym_world_owned')
        self.listed_module = types.ModuleType('_asym_world_listed')
        for module in (self.owned_module, self.listed_module):
            sys.modules[module.__name__] = module
            test.addCleanup(sys.modules.pop, module.__name__, None)
        source = SimpleNamespace(numerical_flags=lambda: 'F' if flags_ok else 'X',
                                 cgroup_memory=lambda: (log.append('terminal.cgroup_read'),
                                                        {'path': '/c/u.service' if cgroup_ok else '/c/other.service'})[1])

        def evaluator_exit(*a):
            log.append('evaluator_exit')
            if 'evaluator_exit' in self.fail:
                raise RuntimeError('secondary evaluator_exit')
        self.checked_sources = []
        owner = lambda name: SimpleNamespace(check=lambda: (self.checked_sources.append(name), log.append('guard.'+name)))
        self.S = SimpleNamespace(
            budget=SimpleNamespace(check=budget), registry=self.registry, sources=self.sources,
            state=object() if state else None, serving_checks=[], v5=SimpleNamespace(memory_snapshot=step('snap')),
            checks=[], diagnostic_source=owner('v3'), v5_source=owner('v5'), self_source=owner('self'),
            request_source=owner('requests'), observer_source=owner('observer'), native_source=owner('native'),
            evaluator_source=owner('evaluator'), wrapper_source=None, packed_source=None, dispose_check=lambda: None,
            context={'launch': {'original_cache': {'path': '/runs/cache/fit.npy'}},
                     'score': {'launch': {'endpoints': [{'bundle': {'path': '/runs/e1/bundle.json'}}],
                                          'exports': {}}, 'invocation': {'python_sha256': 'p'*64}}, 'guards': {}},
            control={'bundle': {'path': '/runs/control/bundle.json'}}, candidate={'bundle': {'path': '/runs/cand/bundle.json'}},
            context_e={'guards': {}} , exit_guard='exit-guard', before='before', locks=SimpleNamespace(check=locks),
            api=(SimpleNamespace(evaluator_exit=evaluator_exit, evidence=lambda: (log.append('evidence'), {'combined': True})[1],
                                 authenticate=lambda: log.append('api.authenticate')) if api else None),
            evaluator=SimpleNamespace(exit_rehash=step('plain_exit_rehash'), guard_helpers=lambda c: None,
                                      resources=step('final_guard.resources')),
            prospective={'output': self.output, 'guards': {}, 'launch': {'id': 1}}, workspace_dispose=step('workspace_dispose'),
            diagnostic=SimpleNamespace(final_state=step('final_state'), origin_audit=lambda c, s: (lambda: None)),
            source=source, v3_before={'path': '/c/u.service'}, rng=(1, [1]), flags='F',
            receipt={'schema': 'receipt'} if receipt else None, torch=FakeTorch(allocated, 1 if rng_ok else 2),
            initializer=self.initializer, owned=[SimpleNamespace(module=self.owned_module)], modules=[self.listed_module])


def parent_accepts(output, terminal, limit=700):
    """Test-side MODEL of root's UNIT acceptance: a receipt counts only for a normal, in-cap, lock-holding terminal."""
    receipt = Path(output)/'receipt.json'
    if (Path(output)/'receipt.json.partial').exists() or not receipt.exists():
        return False, 'no published receipt'
    record = json.loads(receipt.read_text())
    if terminal['exit_status'] != 0:
        return False, 'nonzero terminal'
    if not terminal['both_locks_held'] or terminal['service_seconds'] >= limit or record['wall_seconds'] >= limit:
        return False, 'cap or locks'
    if record['terminal_binding']['invocation_id'] != terminal['invocation_id']:
        return False, 'invocation differs'
    return True, 'accepted'


class ExceptionOrdering(unittest.TestCase):
    """The genuine combined exit and the terminal checks run on every error and never mask the primary error."""

    ORDER = ['snap', 'release', 'evaluator_exit', 'evidence', 'budget.check', 'guard.self', 'final_guard.resources',
             'sources.close', 'workspace_dispose', 'final_state', 'terminal.cgroup_read']

    def world(self, **kw):
        return World(self, **kw)

    def test_success_runs_every_action_in_order_then_stages_disposes_and_links_last(self):
        w = self.world(receipt=True)
        w.d.cleanup(w.S, None)
        positions = [w.log.index(x) for x in self.ORDER]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual([x for x in w.log if isinstance(x, tuple)],
                         [('map', 'fit.npy'), ('map', 'endpoint.pt'), ('map', 'vision.pt')])
        self.assertTrue(all(state == (False, False) for state in w.fs[:-2]), w.fs)
        self.assertEqual(w.fs[-2:], [(True, False), (False, True)], 'staged, then linked; partial never visible after')
        recorded = json.loads((w.output/'receipt.json').read_text())
        self.assertTrue(recorded['exit_rehash_pass'] and recorded['cleanup_pass'])
        self.assertEqual(recorded['registry']['conflicts'], [])
        self.assertEqual(recorded['combined_native'], {'combined': True})
        self.assertFalse((w.output/'receipt.json.partial').exists())
        self.assertTrue(set(w.checked_sources) >= {'self', 'requests', 'observer', 'native', 'evaluator', 'v3', 'v5'})
        self.assertNotIn(w.owned_module.__name__, sys.modules)
        self.assertNotIn(w.listed_module.__name__, sys.modules)

    def test_nothing_is_staged_or_published_after_any_single_failure(self):
        cases = [dict(fail=(x,), receipt=True) for x in ('release', 'evaluator_exit', 'budget.check', 'workspace_dispose',
                 'final_state', 'sources.close', 'final_guard.resources', 'terminal.admit_cgroup', 'snap')]
        cases += [dict(cap=True, receipt=True), dict(locks_fail=True, receipt=True), dict(allocated=4096, receipt=True),
                  dict(flags_ok=False, receipt=True), dict(rng_ok=False, receipt=True), dict(cgroup_ok=False, receipt=True)]
        for case in cases:
            with self.subTest(**case):
                w = self.world(**case)
                with self.assertRaises((RuntimeError, ValueError)):
                    w.d.cleanup(w.S, None)
                self.assertFalse((w.output/'receipt.json').exists())
                self.assertFalse((w.output/'receipt.json.partial').exists())

    def test_body_error_still_runs_combined_exit_workspace_final_state_and_terminal_checks_and_stays_primary(self):
        for fail in ((), ('release',), ('evaluator_exit',), ('workspace_dispose',), ('final_state',), ('budget.check',),
                     ('release', 'evaluator_exit', 'workspace_dispose', 'final_state', 'sources.close')):
            with self.subTest(fail=fail):
                w = self.world(fail=fail, receipt=True)
                error = ValueError('primary body failure')
                with self.assertRaises(ValueError) as caught:
                    w.d.cleanup(w.S, error)
                self.assertIs(caught.exception, error)
                self.assertEqual(str(caught.exception), 'primary body failure')
                notes = '\n'.join(error.__notes__) if fail else ''
                for name in fail:
                    self.assertIn('secondary '+name, notes)
                for step in ('release', 'evaluator_exit', 'budget.check', 'sources.close', 'workspace_dispose', 'final_state',
                             'terminal.cgroup_read', 'final_guard.resources'):
                    self.assertIn(step, w.log, step+' skipped after another failure')
                self.assertLess(w.log.index('workspace_dispose'), w.log.index('final_state'))
                self.assertLess(w.log.index('budget.check'), w.log.index('workspace_dispose'))
                self.assertFalse((w.output/'receipt.json').exists())
                self.assertFalse((w.output/'receipt.json.partial').exists())
                self.assertNotIn(w.owned_module.__name__, sys.modules)

    def test_secondary_failure_with_primary_cause_cannot_mask_primary(self):
        primary = ValueError('primary')
        w = self.world()
        def exiting(*a):
            w.log.append('evaluator_exit')
            raise RuntimeError('chained secondary') from primary
        w.S.api.evaluator_exit = exiting
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, primary)
        self.assertIs(caught.exception, primary)
        self.assertIn('chained secondary', ''.join(primary.__notes__))
        self.assertIn('final_state', w.log)

    def test_disposal_failure_does_not_skip_final_state_and_final_failure_is_noted(self):
        w = self.world(fail=('workspace_dispose', 'final_state'))
        error = ValueError('primary')
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, error)
        self.assertIs(caught.exception, error)
        self.assertLess(w.log.index('workspace_dispose'), w.log.index('final_state'))
        notes = '\n'.join(error.__notes__)
        self.assertIn('secondary final_state', notes)
        self.assertIn('workspace cleanup: RuntimeError', notes)

    def test_without_primary_error_the_first_cleanup_failure_is_raised_and_others_noted(self):
        w = self.world(fail=('release', 'workspace_dispose'))
        with self.assertRaises(RuntimeError) as caught:
            w.d.cleanup(w.S, None)
        self.assertEqual(str(caught.exception), 'secondary release')
        self.assertIn('secondary workspace_dispose', '\n'.join(caught.exception.__notes__))
        self.assertIn('final_state', w.log)

    def test_plain_exit_rehash_when_combined_authority_was_never_installed(self):
        w = self.world(api=False, state=False)
        w.S.locks = None
        with self.assertRaises(ValueError):
            w.d.cleanup(w.S, ValueError('early'))
        self.assertIn('plain_exit_rehash', w.log)
        self.assertNotIn('evaluator_exit', w.log)
        self.assertNotIn('release', w.log)
        self.assertNotIn('final_guard.resources', w.log, 'final guard needs the installed api and both locks')

    def test_cleanup_before_any_admission_is_safe(self):
        log = []
        S = SimpleNamespace(registry=dict(sys.modules), sources=None, state=None, serving_checks=[], v5=None, checks=[],
                            diagnostic_source=None, v5_source=None, dispose_check=None, context=None, control=None,
                            candidate=None, context_e=None, exit_guard=None, api=None, evaluator=None, prospective=None,
                            budget=SimpleNamespace(check=lambda: log.append('budget.check')), workspace_dispose=None,
                            diagnostic=None, source=None, v3_before=None, rng=None, flags=None, receipt=None, owned=[],
                            modules=[], locks=None, torch=None, initializer=None, self_source=None, request_source=None,
                            observer_source=None, native_source=None, evaluator_source=None, wrapper_source=None,
                            packed_source=None, before=None)
        with self.assertRaises(KeyError) as caught:
            load_driver().cleanup(S, KeyError('first failure'))
        self.assertEqual(log, ['budget.check'])
        self.assertEqual(caught.exception.args, ('first failure',))

    def test_whole_cap_is_checked_in_the_exit_and_again_at_publication(self):
        w = self.world(cap=True, receipt=True)
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, None)
        self.assertEqual(str(caught.exception), 'cap')
        self.assertTrue(w.log.count('budget.check') >= 2, 'exit cap check and final_guard cap check')
        self.assertIn('terminal.cgroup_read', w.log, 'expired cap must not skip the mandatory terminal checks')
        self.assertIn('final_state', w.log)
        self.assertEqual(w.d.LIMITS['seconds'], 700)

    def run_with(self, body, **world):
        """Functional run(): fake admission/native/body, REAL cleanup, so publication ordering is exercised."""
        d = load_driver()
        w = World(self, receipt=False, **world)
        def admit(S, args):
            for key, value in vars(w.S).items():
                setattr(S, key, value)
            S.receipt = None
        with patch.object(d, 'admit', admit), patch.object(d, 'start_native', lambda S: None), patch.object(d, 'body', body), \
                patch.object(d, 'NATIVE_PACKAGES', set()), patch.object(sys, 'dont_write_bytecode', True), silent():
            try:
                return d.run(SimpleNamespace(authority=Path('/a'), authority_sha256='a'*64)), w
            except BaseException as error:
                return error, w

    def test_run_publishes_nothing_after_a_body_error_and_exits_first(self):
        error = ValueError('body failed')
        def body(S):
            raise error
        result, w = self.run_with(body)
        self.assertIs(result, error)
        self.assertFalse((w.output/'receipt.json').exists())
        self.assertIn('evaluator_exit', w.log)
        self.assertIn('final_state', w.log)

    def test_run_links_the_receipt_only_after_every_cleanup_step(self):
        result, w = self.run_with(lambda S: {'status': 'MEASURED', 'cells': {'FF': 1, 'FS': 2}})
        self.assertIsInstance(result, dict, result)
        recorded = json.loads((w.output/'receipt.json').read_text())
        self.assertEqual(recorded['status'], 'MEASURED')
        self.assertFalse(recorded['validation_read'] or recorded['causal_freshness_claim'] or recorded['official_read'])
        self.assertNotIn('decision', recorded)
        self.assertEqual(w.fs[-2:], [(True, False), (False, True)])
        self.assertTrue(parent_accepts(w.output, {'exit_status': 0, 'service_seconds': 600.0, 'both_locks_held': True,
                                                  'invocation_id': 'i'*32})[0])

    def test_run_withholds_everything_when_exit_cap_or_terminal_checks_fail(self):
        ok = lambda S: {'status': 'MEASURED', 'cells': {'FF': 1, 'FS': 2}}
        for world, kind in (({'fail': ('final_state',)}, RuntimeError), ({'cap': True}, ValueError),
                            ({'allocated': 1}, ValueError), ({'fail': ('evaluator_exit',)}, RuntimeError),
                            ({'locks_fail': True}, ValueError)):
            with self.subTest(**world):
                result, w = self.run_with(ok, **world)
                self.assertIsInstance(result, kind)
                self.assertFalse((w.output/'receipt.json').exists())
                self.assertFalse((w.output/'receipt.json.partial').exists())

    def test_failure_after_the_link_leaves_a_receipt_that_the_parent_must_reject(self):
        w = self.world(receipt=True)
        real_unlink = os.unlink
        def failing(path, *a, **k):
            if str(path).endswith('receipt.json.partial'):
                real_unlink(path)
                raise OSError('post-link failure')
            return real_unlink(path, *a, **k)
        with patch.object(os, 'unlink', failing), self.assertRaises(OSError):
            w.d.cleanup(w.S, None)
        self.assertTrue((w.output/'receipt.json').exists(), 'published before the failure')
        terminal = {'exit_status': 1, 'service_seconds': 650.0, 'both_locks_held': True, 'invocation_id': 'i'*32}
        self.assertEqual(parent_accepts(w.output, terminal), (False, 'nonzero terminal'))

    def test_parent_model_rejects_nonzero_cap_exceeded_lockless_and_mismatched_terminals(self):
        w = self.world(receipt=True)
        w.d.cleanup(w.S, None)
        good = {'exit_status': 0, 'service_seconds': 650.0, 'both_locks_held': True, 'invocation_id': 'i'*32}
        self.assertEqual(parent_accepts(w.output, good), (True, 'accepted'))
        for edit, why in (({'exit_status': 1}, 'nonzero terminal'), ({'service_seconds': 700.0}, 'cap or locks'),
                          ({'service_seconds': 701.5}, 'cap or locks'), ({'both_locks_held': False}, 'cap or locks'),
                          ({'invocation_id': 'x'*32}, 'invocation differs')):
            self.assertEqual(parent_accepts(w.output, {**good, **edit}), (False, why), edit)
        (w.output/'receipt.json').rename(w.output/'receipt.json.partial')
        self.assertEqual(parent_accepts(w.output, good), (False, 'no published receipt'))


class TerminalChecks(unittest.TestCase):
    """Mandatory allocator/RNG/flags/cgroup checks are independent of the historical final_state and of the cap."""

    def world(self, **kw):
        return World(self, **kw)

    def test_final_state_alone_is_not_sufficient_on_the_error_path(self):
        """The REAL v3.final_state: no allocator check without a receipt, no checks at all after an expired cap."""
        d = load_driver()
        v3 = load_v3()
        def run(allocated, receipt, expired):
            fake = FakeTorch(allocated)
            reads = []
            source = SimpleNamespace(numerical_flags=lambda: 'F', cgroup_memory=lambda: (reads.append(1), {'path': '/c/u.service'})[1])
            initializer = SimpleNamespace(admit_cgroup=lambda a, u: reads.append('admit'))
            budget = SimpleNamespace(check=lambda: d.require(not expired, 'cap'))
            with patch.dict(sys.modules, {'torch': fake}):
                try:
                    v3.final_state(budget, source, initializer, {'path': '/c/u.service'}, (1, [1]), 'F', receipt)
                    return 'ok', len(reads)
                except ValueError as error:
                    return str(error), len(reads)
        self.assertEqual(run(4096, None, False), ('ok', 2), 'residual allocation invisible without a receipt')
        self.assertEqual(run(4096, {}, False)[0], 'final CUDA tensor cleanup differs')
        self.assertEqual(run(0, None, True), ('cap', 0), 'expired cap skips cgroup/flags/RNG entirely')

    def test_each_terminal_check_is_attempted_and_failures_are_collected(self):
        w = self.world(allocated=4096, flags_ok=False, rng_ok=False, cgroup_ok=False)
        with self.assertRaises(ValueError) as caught:
            w.d.terminal_checks(w.S)
        text = str(caught.exception)+'\n'+'\n'.join(caught.exception.__notes__)
        for expected in ('residual CUDA allocation', 'numerical flags changed', 'RNG changed', 'enclosing cgroup changed'):
            self.assertIn(expected, text)
        clean = self.world()
        clean.d.terminal_checks(clean.S)
        self.assertEqual(clean.log, ['terminal.cgroup_read', 'terminal.admit_cgroup'])

    def test_error_path_with_residual_allocation_and_expired_cap_keeps_both_and_publishes_nothing(self):
        w = self.world(allocated=4096, cap=True, receipt=True)
        error = RuntimeError('body failed')
        with self.assertRaises(RuntimeError) as caught:
            w.d.cleanup(w.S, error)
        self.assertIs(caught.exception, error)
        notes = '\n'.join(error.__notes__)
        self.assertIn('ValueError(\'cap\')', notes, 'the original budget failure is retained')
        self.assertIn('residual CUDA allocation', notes, 'allocator check ran despite the expired cap')
        self.assertIn('terminal.cgroup_read', w.log)
        self.assertFalse((w.output/'receipt.json').exists())


class IndependentExit(unittest.TestCase):
    """An unmapped S makes the unchanged combined exit reject early; the uncached checks still complete."""

    def normalized(self, node):
        class Mirror(ast.NodeTransformer):
            def visit_Attribute(self, n):
                self.generic_visit(n)
                if isinstance(n.value, ast.Name) and n.value.id == 'e':
                    return ast.copy_location(ast.Name(id=n.attr, ctx=n.ctx), n)
                return n

            def visit_Name(self, n):
                return ast.copy_location(ast.Name(id={'c': 'context'}.get(n.id, n.id), ctx=n.ctx), n)
        return ast.dump(Mirror().visit(copy.deepcopy(node)))

    def test_independent_exit_mirrors_the_genuine_exit_rehash_after_its_native_audit(self):
        mine = functions(tree(DRIVER))['independent_exit']
        genuine_exit = functions(tree(HERE/'evaluate_siglip2_connected_mlp.py'))['exit_rehash']
        loops_mine = [n for n in ast.walk(mine) if isinstance(n, ast.For)]
        loops_genuine = [n for n in ast.walk(genuine_exit) if isinstance(n, ast.For)]
        self.assertEqual(len(loops_mine), len(loops_genuine), 2)
        for a, b in zip(loops_mine, loops_genuine, strict=True):
            self.assertEqual(self.normalized(a), self.normalized(b))
        calls = lambda fn: {self.normalized(n.value) for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)}
        statements, genuine_statements = calls(mine), calls(genuine_exit)
        self.assertEqual(len(statements), 6)
        self.assertLessEqual(statements, genuine_statements, 'every non-loop step is a step of the genuine exit')
        self.assertFalse(any('audit_origins' in ast.unparse(n) or 'exit_rehash' in ast.unparse(n) or 'original_guard' in ast.unparse(n)
                             for n in ast.walk(mine) if isinstance(n, ast.Call)), 'no native audit inside the independent part')

    def evaluator(self, log, fail=None):
        def step(name, then=None):
            def call(*a, **k):
                log.append(name)
                if name == fail:
                    raise RuntimeError('independent '+name)
                return then
            return call
        launch = {'training': {'root': '/t', 'execution_sha256': 't'*64, 'code': {'t.py': '1'*64}},
                  'evaluator_reference': {'root': '/er', 'execution_sha256': 'e'*64}, 'nearest_evaluator': {'root': '/ne', 'execution_sha256': 'n'*64},
                  'genuine_evaluator': {'root': '/ge', 'execution_sha256': 'g'*64}, 'reference': {'root': '/rf', 'execution_sha256': 'r'*64, 'code': {'r.py': '2'*64}},
                  'endpoints': [{'bundle': {'path': '/b/one/bundle.json', 'sha256': 'b'*64}}, {'bundle': {'path': '/b/two/bundle.json', 'sha256': 'c'*64}}]}
        pins = {'EVALUATOR_PINS': {'x.py': '3'*64}, 'NEAREST_EVALUATOR': {'code': {'n.py': '4'*64}}, 'GENUINE_PINS': {'g.py': '5'*64},
                'FILES': {'f.py'}, 'TRAIN_FILES': {'t.py'}}
        by_root = {'/r': {'f.py': '0'*64}, '/t': {'t.py': '1'*64}, '/er': pins['EVALUATOR_PINS'], '/ne': pins['NEAREST_EVALUATOR']['code'],
                   '/ge': pins['GENUINE_PINS'], '/rf': {'r.py': '2'*64}}
        def closure(root, execution, names, guards):
            log.append(('closure', root))
            if fail == 'closure':
                raise ValueError('closure rejected')
            return by_root[root]
        trainer = SimpleNamespace(batch_bound_files=step('batch_bound_files'), admit_bundle=lambda d_, sha: (log.append(('bundle', Path(d_).name)), (None, {str(d_): sha}))[1])
        t = {'trainer': SimpleNamespace(require_no_training=step('require_no_training'), helper_guard=step('helper_guard')),
             'guards': {}, 'legacy': {'guards': {}}}
        context = {'training_context': t, 'trainer': trainer, 'guards': {}, 'root': '/r', 'args': SimpleNamespace(execution_sha256='x'*64),
                   'code': {'f.py': '0'*64}, 'launch': launch}
        evaluator = SimpleNamespace(guard_helpers=step('guard_helpers'), merge_guards=lambda target, values: target.update(values),
                                    closure=closure, ORIGINAL_EXPORT_OWNER=None, **pins)
        return evaluator, context

    def cleanup_with_rejecting_combined_exit(self, fail=None):
        w = World(self, receipt=True)
        evaluator, context = self.evaluator(w.log, fail)
        w.S.evaluator, w.S.context_e = evaluator, context
        def reject(*a):
            w.log.append('evaluator_exit')
            raise ValueError('exact supplemental native inventory required')
        w.S.api.evaluator_exit = reject
        return w

    def test_unmapped_s_rejection_is_preserved_and_the_uncached_checks_still_complete(self):
        w = self.cleanup_with_rejecting_combined_exit()
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, None)
        self.assertEqual(str(caught.exception), 'exact supplemental native inventory required', 'original rejection kept')
        self.assertIn('independent uncached context/closure/bundle checks completed', '\n'.join(caught.exception.__notes__))
        roots = [x[1] for x in w.log if isinstance(x, tuple) and x[0] == 'closure']
        self.assertEqual(roots, ['/r', '/t', '/er', '/ne', '/ge', '/rf'])
        self.assertEqual([x[1] for x in w.log if isinstance(x, tuple) and x[0] == 'bundle'], ['one', 'two'])
        for step in ('batch_bound_files', 'require_no_training', 'helper_guard'):
            self.assertIn(step, w.log)
        self.assertNotIn('evidence', w.log, 'no combined evidence after a rejected exit')
        self.assertFalse((w.output/'receipt.json').exists())
        self.assertNotIn('sfora', sys.modules.keys() - w.registry.keys(), 'Cutile is never mapped to rescue an error')

    def test_independent_failures_are_notes_on_the_original_rejection(self):
        for fail in ('closure', 'batch_bound_files'):
            with self.subTest(fail=fail):
                w = self.cleanup_with_rejecting_combined_exit(fail)
                with self.assertRaises(ValueError) as caught:
                    w.d.cleanup(w.S, None)
                self.assertEqual(str(caught.exception), 'exact supplemental native inventory required')
                self.assertIn('independent uncached exit checks also failed', '\n'.join(caught.exception.__notes__))


class RegistryOwnership(unittest.TestCase):
    """Complete owned-registry identity tracking with the GENUINE extracted trainer loaders."""

    def setUp(self):
        self.d = load_driver()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name).resolve()
        self.trainer = HERE/'train_siglip2_connected_mlp.py'

    def state(self, control=True):
        return SimpleNamespace(registry=dict(sys.modules), prospective={'guards': {}}, context=None, context_e=None,
                               control={'bundle': {'path': str(self.bundle/'bundle.json')}} if control else None,
                               candidate=None, modules=[], owned=[])

    def snapshot_equals(self, S, extra=()):
        current = dict(sys.modules)
        added = {k for k in current if k not in S.registry}
        return added <= set(extra) and all(current.get(k) is v for k, v in S.registry.items())

    def test_success_path_genuine_load_authenticated_modules_are_disposed_exactly(self):
        ns = genuine(self.trainer, ('require', 'file_fact', 'bound_file', 'load_authenticated'))
        S = self.state(control=False)
        guards = {}
        for i in range(3):
            path = self.bundle/f'helper_{i}.py'
            path.write_text(f'VALUE = {i}\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            ns['load_authenticated'](f'_asym_registry_ok_{i}', path, digest, guards)
            self.addCleanup(sys.modules.pop, f'_asym_registry_ok_{i}', None)
        S.prospective['guards'].update(guards)
        owned, foreign, conflicts = self.d.registry_plan(S)
        self.assertEqual(sorted(owned), [f'_asym_registry_ok_{i}' for i in range(3)])
        self.assertEqual(conflicts, [])
        self.d.registry_dispose(S)
        self.assertTrue(self.snapshot_equals(S), 'registry equals the pre-admission snapshot')

    def test_partial_serving_loader_failure_leaves_owned_entries_that_are_disposed(self):
        names = ('qualify_siglip2_substrate_cpu.py', 'extract_siglip2_vision_source.py', 'train_siglip2_cached_readout.py',
                 'train_siglip2_substrate_adaptation.py', 'prototype_residual_readout.py', 'quadratic_readout.py',
                 'joint_relational_compaction.py')
        code = {}
        for name in names:
            (self.bundle/name).write_text('VALUE = 1\n')
            code[name] = hashlib.sha256((self.bundle/name).read_bytes()).hexdigest()
        ns = genuine(self.trainer, ('require', 'file_fact', 'bound_file', 'load_authenticated', 'load_inference'),
                     constants=('SERVING_FILES', 'INFERENCE_KEYS'),
                     admit_bundle=lambda directory, digest: ({'code': code}, {}), mapping_absent=lambda path: None)
        fake_torch = types.ModuleType('torch')
        fake_torch.load = lambda *a, **k: {}                         # genuine identity requirement then fails
        with patch.dict(sys.modules, {'torch': fake_torch}):
            S = self.state()
            with self.assertRaises(ValueError):
                ns['load_inference'](self.bundle, 'a'*64, 'cuda')
            leaked = sorted(set(sys.modules) - set(S.registry))
            self.addCleanup(lambda: [sys.modules.pop(k, None) for k in leaked])
            self.assertEqual(len(leaked), 7)
            self.assertTrue(all(k.startswith('_connected_serving_') for k in leaked), 'the genuine loader leaks them')
            owned, foreign, conflicts = self.d.registry_plan(S)
            self.assertEqual(sorted(owned), leaked, 'owned by bundle-directory origin although no guard dict names them')
            self.d.registry_dispose(S)
            self.assertTrue(self.snapshot_equals(S))

    def test_foreign_native_and_stdlib_entries_survive_and_conflicts_are_fatal(self):
        S = self.state(control=False)
        outside = Path(self.temp.name).resolve()/'..'/'elsewhere.py'
        for name, origin in (('_asym_foreign_new', str(outside)), ('PIL', str(self.bundle/'pil.py')),
                             ('sfora.cutile_int8', str(self.bundle/'c.py'))):
            module = types.ModuleType(name)
            module.__file__ = origin
            sys.modules[name] = module
            self.addCleanup(sys.modules.pop, name, None)
        S.prospective['guards'].update({str(self.bundle/'pil.py'): 'a'*64, str(self.bundle/'c.py'): 'b'*64})
        owned, foreign, conflicts = self.d.registry_plan(S)
        self.assertEqual(owned, {})
        self.assertEqual(sorted(foreign), ['PIL', '_asym_foreign_new', 'sfora.cutile_int8'])
        self.d.registry_dispose(S)
        self.assertTrue(all(k in sys.modules for k in ('PIL', '_asym_foreign_new', 'sfora.cutile_int8')))
        replaced = types.ModuleType('_asym_preexisting')
        sys.modules['_asym_preexisting'] = replaced
        self.addCleanup(sys.modules.pop, '_asym_preexisting', None)
        S2 = self.state(control=False)
        sys.modules['_asym_preexisting'] = types.ModuleType('_asym_preexisting')
        with self.assertRaisesRegex(ValueError, 'preexisting module replaced'):
            self.d.registry_dispose(S2)
        sys.modules.pop('_asym_preexisting')
        with self.assertRaisesRegex(ValueError, 'preexisting module removed'):
            self.d.registry_dispose(S2)

    def real_module(self, name, path):
        module = types.ModuleType(name)
        module.__file__ = path
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=path)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def foreign_clone(self, original):
        foreign = types.ModuleType(original.__name__)
        foreign.__file__ = original.__file__
        foreign.__spec__ = importlib.machinery.ModuleSpec(original.__name__, None, origin=original.__file__)
        return foreign

    def test_a_foreign_replacement_with_an_identical_origin_is_retained_while_legitimate_modules_dispose(self):
        """Real sys.modules: a tracked name replaced by a foreign module carrying the SAME __file__/spec.origin."""
        S = self.state(control=False)
        guarded = {name: str(self.bundle/f'{name}.py') for name in ('a', 'b', 'c', 'd')}
        S.prospective['guards'].update({path: 'a'*64 for path in guarded.values()})
        A, B, C, D = (self.real_module(f'_asym_cas_{n}', p) for n, p in guarded.items())
        S.modules, S.owned = [A], [SimpleNamespace(module=B)]
        replacement = self.foreign_clone(B)
        sys.modules[B.__name__] = replacement
        preexisting = S.registry
        owned, foreign, conflicts = self.d.registry_plan(S)
        self.assertEqual(sorted(owned), ['_asym_cas_a', '_asym_cas_c', '_asym_cas_d'], 'a foreign alias is never owned')
        self.assertTrue(any(c.startswith('_asym_cas_b:') and 'foreign' in c for c in conflicts), conflicts)
        with self.assertRaisesRegex(ValueError, 'owned source registry not restored.*_asym_cas_b'):
            self.d.registry_dispose(S)
        self.assertIs(sys.modules[B.__name__], replacement, 'the foreign module is left untouched')
        for module in (A, C, D):
            self.assertNotIn(module.__name__, sys.modules, 'legitimate owned modules still dispose despite the rejection')
        self.assertTrue(all(sys.modules.get(k) is v for k, v in preexisting.items()))

    def test_deletion_is_compare_and_delete_even_when_the_entry_is_swapped_after_the_plan(self):
        S = self.state(control=False)
        path = str(self.bundle/'swap.py')
        S.prospective['guards'][path] = 'a'*64
        owned_module = self.real_module('_asym_cas_swap', path)
        other = self.real_module('_asym_cas_other', str(self.bundle/'other.py'))
        S.prospective['guards'][str(self.bundle/'other.py')] = 'b'*64
        real_plan = self.d.registry_plan
        plan = real_plan(S)
        swapped = self.foreign_clone(owned_module)
        calls = []
        def stale_plan(state, disposed=False):
            calls.append(disposed)
            if len(calls) == 1:
                sys.modules[owned_module.__name__] = swapped      # replaced between the plan and the deletion
                return plan
            return real_plan(state, disposed)
        with patch.object(self.d, 'registry_plan', stale_plan), self.assertRaisesRegex(ValueError, 'replaced before deletion'):
            self.d.registry_dispose(S)
        self.assertIs(sys.modules['_asym_cas_swap'], swapped)
        self.assertNotIn('_asym_cas_other', sys.modules, 'the legitimate owned module was still disposed')

    def test_a_removed_tracked_module_is_a_conflict_before_disposal_and_expected_after(self):
        S = self.state(control=False)
        module = self.real_module('_asym_cas_gone', str(self.bundle/'gone.py'))
        S.modules = [module]
        del sys.modules['_asym_cas_gone']
        self.assertTrue(any('removed before its final guards' in c for c in self.d.registry_plan(S)[2]))
        self.assertEqual(self.d.registry_plan(S, disposed=True)[2], [])

    def test_authenticated_guarded_origin_is_owned_but_an_unguarded_one_is_only_reported(self):
        S = self.state(control=False)
        guarded, unguarded = self.bundle/'guarded.py', self.bundle/'unguarded.py'
        for path, name in ((guarded, '_asym_guarded'), (unguarded, '_asym_unguarded')):
            module = types.ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            self.addCleanup(sys.modules.pop, name, None)
        S.prospective['guards'][str(guarded)] = 'a'*64
        owned, foreign, _ = self.d.registry_plan(S)
        self.assertEqual((sorted(owned), foreign), (['_asym_guarded'], ['_asym_unguarded']))


class PrimaryPreservingPhases(unittest.TestCase):
    """protected() keeps the primary error and detaches retained frames before the GENUINE release_inference runs."""

    class Model:
        def __init__(self):
            self.params = []

        def parameters(self):
            return iter(self.params)

        def buffers(self):
            return iter(())

    def endpoint(self, test):
        name = f'_asym_serving_{id(self)}_{time.time_ns()}'
        module = types.ModuleType(name)
        sys.modules[name] = module
        test.addCleanup(sys.modules.pop, name, None)
        cache = SimpleNamespace(cache_clear=lambda: None, cache_info=lambda: SimpleNamespace(currsize=0))
        endpoint = {k: self.Model() for k in ('model', 'processor_object', 'head_object', 'A', 'C', 'mu_train')}
        endpoint.update(processor_cache=cache, modules={'m.py': module}, guards={})
        return endpoint, module

    def setUp(self):
        self.d = load_driver()
        self.release = genuine(HERE/'train_siglip2_connected_mlp.py', ('require', 'release_inference'),
                               _processor_cache=lambda processor, guards: self.cache)['release_inference']

    def capture(self, endpoint):
        model = endpoint['model']       # a failing frame keeps its OWN alias, like encoder_facts() does
        raise ValueError('PRIMARY capture failure')

    def test_the_old_unprotected_shape_masks_the_primary_and_fails_the_genuine_lifetime_check(self):
        endpoint, module = self.endpoint(self)
        self.cache = endpoint['processor_cache']
        def old_phase():
            try:
                self.capture(endpoint)
            finally:
                self.release(endpoint)
        with self.assertRaises(ValueError) as caught:
            old_phase()
        self.assertIn('lifetime survived release', str(caught.exception))
        self.assertEqual(str(caught.exception.__context__), 'PRIMARY capture failure', 'the primary was demoted to __context__')

    def test_protected_keeps_the_primary_detaches_frames_and_the_genuine_release_succeeds(self):
        endpoint, module = self.endpoint(self)
        self.cache = endpoint['processor_cache']
        done = []
        def phase():
            self.capture(endpoint)
        with silent(), self.assertRaises(ValueError) as caught:
            self.d.protected(phase, [lambda: self.release(endpoint), lambda: done.append('mapping')])
        self.assertEqual(str(caught.exception), 'PRIMARY capture failure')
        self.assertFalse(getattr(caught.exception, '__notes__', []), 'release succeeded: no secondary failure')
        self.assertEqual(done, ['mapping'])
        self.assertNotIn(module.__name__, sys.modules, 'genuine release unregistered its owned module')

    def test_primary_and_secondary_failures_every_closer_runs_and_identity_is_preserved(self):
        endpoint, module = self.endpoint(self)
        self.cache = endpoint['processor_cache']
        ran = []
        def failing_close():
            ran.append('images')
            raise RuntimeError('secondary image close failed')
        with silent(), self.assertRaises(ValueError) as caught:
            self.d.protected(lambda: self.capture(endpoint),
                             [failing_close, lambda: self.release(endpoint), lambda: ran.append('mapping')])
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(ran, ['images', 'mapping'])
        self.assertIn('secondary image close failed', '\n'.join(caught.exception.__notes__))
        self.assertNotIn(module.__name__, sys.modules)

    def test_chained_primary_frames_are_detached_not_only_the_newest_traceback(self):
        endpoint, module = self.endpoint(self)
        self.cache = endpoint['processor_cache']
        def phase():
            try:
                self.capture(endpoint)
            except ValueError:
                raise RuntimeError('secondary raised while handling the primary')
        with silent(), self.assertRaises(RuntimeError) as caught:
            self.d.protected(phase, [lambda: self.release(endpoint)])
        self.assertIsInstance(caught.exception.__context__, ValueError)
        self.assertFalse(getattr(caught.exception, '__notes__', []))
        # the newest-only approach (traceback = None) provably leaves the chained primary frame alive
        endpoint2, module2 = self.endpoint(self)
        self.cache = endpoint2['processor_cache']
        try:
            try:
                self.capture(endpoint2)
            except ValueError:
                raise RuntimeError('secondary')
        except RuntimeError as newest:
            newest.__traceback__ = None
            with self.assertRaises(ValueError) as old:
                self.release(endpoint2)
            self.assertIn('lifetime survived release', str(old.exception))
            self.d.clear_frames(newest)

    def test_clear_frames_walks_causes_contexts_and_exception_groups_and_keeps_identity(self):
        leaked = []
        def frame_with(model):
            local = model
            raise ValueError('inner')
        model = self.Model()
        ref = weakref.ref(model)
        try:
            frame_with(model)
        except ValueError as inner:
            group = ExceptionGroup('group', [inner])
        del model
        gc.collect()
        self.assertIsNotNone(ref(), 'the group member traceback pins the model')
        self.d.clear_frames(group)
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(str(group.exceptions[0]), 'inner')
        self.assertIsNotNone(group.exceptions[0].__traceback__, 'traceback objects are kept, only frame locals are cleared')

    def test_report_is_bounded_and_printed_once(self):
        buffer = io.StringIO()
        error = ValueError('x'*200_000)
        with contextlib.redirect_stderr(buffer):
            self.d.report(error)
            self.d.report(error)
        text = buffer.getvalue()
        self.assertLessEqual(len(text), 64*1024+64)
        self.assertEqual(text.count('[report truncated]'), 1)
        self.assertEqual(str(error), 'x'*200_000, 'message untouched')

    def test_tail_replay_runs_its_phase_under_protected_and_releases_once(self):
        text = DRIVER.read_text()
        fn = ast.get_source_segment(text, functions(ast.parse(text))['tail_replay'])
        self.assertIn('return protected(phase, [close_images, release_state, lambda: connected.mapping_absent', fn)
        self.assertIn('state, S.state = S.state, None', fn)
        self.assertNotIn('finally:', fn)


def real_modules(test):
    """Real requests/v3/v5 loaded through the driver's own authenticated loader (stdlib-only modules)."""
    d = load_driver()
    pinned = frozen_v5()
    if pinned is None:
        test.skipTest('frozen v5 observer blob unavailable')
    temp = tempfile.TemporaryDirectory()
    test.addCleanup(temp.cleanup)
    v5_path = Path(temp.name).resolve()/'observe_connected_control_batch_execution.py'
    v5_path.write_bytes(pinned['observe_connected_control_batch_execution.py'])
    facts = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in
             (('requests', HERE/'qualify_connected_serving_requests.py'),
              ('v3', HERE/'diagnose_connected_gallery_freshness.py'), ('v5', v5_path))}
    names, loaded = {}, {}
    for key in ('requests', 'v3', 'v5'):
        names[key] = f'_asymmetric_live_{key}_{os.getpid()}_{id(test)}'
        loaded[key] = d.load_module(names[key], facts[key], {})
        test.addCleanup(sys.modules.pop, names[key], None)
    sources = {k: loaded['requests'].Source(loaded[k], facts[k]) for k in ('v3', 'v5')}
    return d, loaded, facts, sources, names


class LiveSourceOwnership(unittest.TestCase):
    """v3 and v5 are guarded by NO Sources/evaluator/api guard; the driver owns them with the real requests.Source."""

    def test_no_existing_guard_covers_the_v3_or_v5_modules(self):
        v3_source = (HERE/'diagnose_connected_gallery_freshness.py').read_text()
        sources = {n.name: n for n in ast.parse(v3_source).body if isinstance(n, ast.ClassDef)}['Sources']
        admit = [m for m in sources.body if isinstance(m, ast.FunctionDef) and m.name == 'admit'][0]
        guarded = [ast.unparse(n.args[0]) for n in ast.walk(admit) if isinstance(n, ast.Call) and
                   ast.unparse(n.func).endswith('source_live_guard')]
        self.assertEqual(guarded, ['module'], 'only SOURCE_NAMES modules loaded through self.load(name)')
        v3 = load_v3()
        self.assertNotIn('diagnose_connected_gallery_freshness.py', set(v3.SOURCE_FILES.values()))
        self.assertEqual(set(v3.SOURCE_NAMES), set(v3.SOURCE_FILES))
        pinned = frozen_v5()
        if pinned is None:
            self.skipTest('frozen v5 observer blob unavailable')
        text = pinned['observe_connected_control_batch_execution.py'].decode()
        self.assertIn("checks.append(guard(diagnostic, historical['source']['sha256'], context['guards'], class_name='Sources'))", text)
        self.assertIn("checks.append(guard(sys.modules[__name__], prospective['code'][Path(__file__).name], context['guards']))", text)

    def test_every_tamper_class_on_the_real_modules_is_detected(self):
        d, mods, facts, sources, names = real_modules(self)
        v3, v5 = mods['v3'], mods['v5']
        for source in sources.values():
            source.check()
        kw, defaults = v3.stale_values.__kwdefaults__, v3.Budget.__init__.__defaults__
        label, check, guard, tensor = v3.label, v3.Budget.check, v3.Sources.guard, v5.tensor_bytes
        score_sha, limits = v3.SCORE_SHA, dict(v3.LIMITS)
        code = {'label': v3.label.__code__, 'tensor': v5.tensor_bytes.__code__}
        cases = {
            'v3 function __code__': ('v3', lambda: setattr(v3.label, '__code__', v3.exact_wire.__code__),
                                     lambda: setattr(v3.label, '__code__', code['label'])),
            'v3 function kwdefaults': ('v3', lambda: setattr(v3.stale_values, '__kwdefaults__', {'mutant': 'omitted_C'}),
                                       lambda: setattr(v3.stale_values, '__kwdefaults__', kw)),
            'v3 method defaults': ('v3', lambda: setattr(v3.Budget.__init__, '__defaults__', (1.0, defaults[1])),
                                   lambda: setattr(v3.Budget.__init__, '__defaults__', defaults)),
            'v3 class attribute': ('v3', lambda: setattr(v3.Budget, 'check', lambda self: None),
                                   lambda: setattr(v3.Budget, 'check', check)),
            'v3 Sources.guard class attribute': ('v3', lambda: setattr(v3.Sources, 'guard', lambda self: None),
                                                 lambda: setattr(v3.Sources, 'guard', guard)),
            'v3 literal mutated in place': ('v3', lambda: v3.LIMITS.__setitem__('seconds', 9999),
                                            lambda: v3.LIMITS.__setitem__('seconds', limits['seconds'])),
            'v3 global rebound': ('v3', lambda: setattr(v3, 'SCORE_SHA', 'x'), lambda: setattr(v3, 'SCORE_SHA', score_sha)),
            'v3 global added': ('v3', lambda: setattr(v3, 'injected', 1), lambda: delattr(v3, 'injected')),
            'v3 global function rebound': ('v3', lambda: setattr(v3, 'label', lambda e: 'x'), lambda: setattr(v3, 'label', label)),
            'v5 function __code__': ('v5', lambda: setattr(v5.tensor_bytes, '__code__', v5.rehash.__code__),
                                     lambda: setattr(v5.tensor_bytes, '__code__', code['tensor'])),
            'v5 global function rebound': ('v5', lambda: setattr(v5, 'tensor_bytes', lambda v: b''),
                                           lambda: setattr(v5, 'tensor_bytes', tensor)),
            'v5 literal mutated': ('v5', lambda: v5.LIMITS.__setitem__('seconds', 9999),
                                   lambda: v5.LIMITS.__setitem__('seconds', 700)),
        }
        for name, (which, mutate, restore) in cases.items():
            with self.subTest(name):
                mutate()
                try:
                    with self.assertRaisesRegex(ValueError, 'authenticated live source changed'):
                        sources[which].check()
                finally:
                    restore()
                sources[which].check()

    def test_registry_replacement_and_disk_change_are_detected(self):
        d, mods, facts, sources, names = real_modules(self)
        original = sys.modules[names['v3']]
        sys.modules[names['v3']] = types.ModuleType(names['v3'])
        try:
            with self.assertRaises(ValueError):
                sources['v3'].check()
        finally:
            sys.modules[names['v3']] = original
        sources['v3'].check()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()/'diagnose_connected_gallery_freshness.py'
            path.write_bytes((HERE/'diagnose_connected_gallery_freshness.py').read_bytes())
            fact_ = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            module = d.load_module('_asymmetric_disk_test', fact_, {})
            self.addCleanup(sys.modules.pop, '_asymmetric_disk_test', None)
            source = mods['requests'].Source(module, fact_)
            source.check()
            with path.open('ab') as stream:
                stream.write(b'\n# changed after load\n')
            with self.assertRaises(ValueError):
                source.check()

    def test_the_driver_module_itself_is_compatible_with_a_true_main_source(self):
        """The real Source on a true __main__: the driver text with its last block replaced by a probe, run as a script."""
        text = DRIVER.read_text()
        head = text[:text.index("if __name__ == '__main__':")]
        requests_path = HERE/'qualify_connected_serving_requests.py'
        # No module-level assignment may happen while a Source exists (it would change the guarded inventory),
        # exactly like run(): the probe is a function and the block only calls it.
        probe = f"""def _probe():
    fact = lambda p: {{'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}}
    requests = load_module('_main_probe_requests', fact({str(requests_path)!r}), {{}})
    source = requests.Source(sys.modules[__name__], fact(Path(__file__).absolute()))
    source.check()
    print('MAIN_SOURCE_OK', sys.modules[__name__].__spec__)


if __name__ == '__main__':
    _probe()
"""
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp).resolve()/'diagnose_connected_asymmetric_cache.py'
            script.write_text(head+probe)
            result = subprocess.run([sys.executable, '-B', '-I', str(script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), 'MAIN_SOURCE_OK None', result.stderr)

    def test_pin_detects_runtime_closure_code_defaults_cells_globals_and_attributes(self):
        d = load_driver()
        def build():
            cell = ['state']
            def audit(x, y=1, *, z=2):
                return (x, y, z, cell[0])
            audit.tag = object()
            return audit, cell
        audit, cell = build()
        check = d.pin(audit)
        check()
        self.assertEqual(audit(1), (1, 1, 2, 'state'))
        cell[0] = 'dynamic state may change: only identities are pinned'
        check()
        other, _ = build()
        mutations = {'code': lambda a: setattr(a, '__code__', other.__code__.replace(co_consts=(None, 7, 7))),
                     'defaults': lambda a: setattr(a, '__defaults__', (5,)),
                     'kwdefaults': lambda a: setattr(a, '__kwdefaults__', {'z': 9}),
                     'attribute replaced': lambda a: setattr(a, 'tag', object()),
                     'attribute added': lambda a: setattr(a, 'extra', 1)}
        for name, mutate in mutations.items():
            audit, cell = build()
            check = d.pin(audit)
            mutate(audit)
            with self.subTest(name), self.assertRaises(ValueError):
                check()
        replaced = (lambda a: (lambda: a))('one')
        check = d.pin(replaced)
        replaced.__closure__[0].cell_contents = 'two'
        with self.assertRaises(ValueError):
            check()
        with self.assertRaises(ValueError):
            d.pin(len)

    def test_pin_uses_identity_so_a_rebound_equal_default_is_detected(self):
        d = load_driver()
        def build():
            def audit(x, y=(1, 2), *, z={'k': 'v'}):
                return x
            return audit
        audit = build()
        check = d.pin(audit)
        check()
        rebinds = {
            'equal tuple rebound': lambda a: setattr(a, '__defaults__', (*a.__defaults__,)),
            'equal kwdefaults dict rebound': lambda a: setattr(a, '__kwdefaults__', dict(a.__kwdefaults__)),
            'kwdefaults value rebound to an equal object': lambda a: a.__kwdefaults__.__setitem__('z', {'k': 'v'}),
            'kwdefaults key added in place': lambda a: a.__kwdefaults__.__setitem__('extra', 1),
            'kwdefaults key removed in place': lambda a: a.__kwdefaults__.pop('z'),
        }
        self.assertEqual(audit.__defaults__, (*audit.__defaults__,), 'the rebound tuple is EQUAL, only not identical')
        for name, rebind in rebinds.items():
            audit = build()
            check = d.pin(audit)
            rebind(audit)
            with self.subTest(name), self.assertRaises(ValueError):
                check()

    def test_pin_never_invokes_equality_so_opaque_defaults_cannot_raise_or_pass_ambiguously(self):
        d = load_driver()
        class Ambiguous:
            calls = 0
            def __eq__(self, other):
                Ambiguous.calls += 1
                raise ValueError('The truth value of an array is ambiguous')
            __hash__ = object.__hash__
        default, bound = Ambiguous(), Ambiguous()
        def build():
            def audit(x, y=default, *, z=bound):
                return x
            return audit
        audit = build()
        check = d.pin(audit)
        check()
        check()
        self.assertEqual(Ambiguous.calls, 0, 'identity comparison only')
        audit.__defaults__ = (Ambiguous(),)
        with self.assertRaises(ValueError) as caught:
            check()
        self.assertEqual(str(caught.exception), 'pinned runtime callable changed', 'detected, not an ambiguity error')
        audit = build()
        check = d.pin(audit)
        audit.__kwdefaults__['z'] = Ambiguous()
        with self.assertRaises(ValueError) as caught:
            check()
        self.assertEqual(str(caught.exception), 'pinned runtime callable changed')
        self.assertEqual(Ambiguous.calls, 0)

    def test_genuine_sources_close_removes_only_its_own_entries_and_every_driver_guard_survives_it(self):
        """Real v3.Sources.close with real pinned sources; the driver-owned modules use other registry names."""
        d = load_driver()
        v3 = load_v3()
        facts = lambda p: {'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
        requests = d.load_module(f'_asym_close_requests_{id(self)}', facts(HERE/'qualify_connected_serving_requests.py'), {})
        self.addCleanup(sys.modules.pop, requests.__name__, None)
        native = requests.Source.load(facts(HERE/'connected_control_native_authority.py'))
        evaluator = native.module.load_evaluator_source(facts(HERE/'evaluate_siglip2_connected_mlp.py'), requests)
        diagnostic_name = f'_asym_close_v3_{id(self)}'
        diagnostic = d.load_module(diagnostic_name, facts(HERE/'diagnose_connected_gallery_freshness.py'), {})
        self.addCleanup(sys.modules.pop, diagnostic_name, None)
        v3_source = requests.Source(diagnostic, facts(HERE/'diagnose_connected_gallery_freshness.py'))
        for module in (native.module, evaluator.module):
            self.addCleanup(sys.modules.pop, module.__name__, None)
        files = {'readout': 'prototype_residual_readout.py', 'quadratic': 'quadratic_readout.py',
                 'connected': 'train_siglip2_connected_mlp.py', 'evaluator': 'evaluate_siglip2_connected_mlp.py'}
        sources = v3.Sources({'launch': {'sources': {n: facts(HERE/f) for n, f in files.items()}}, 'guards': {}})
        before = set(sys.modules)
        for name in files:
            sources.load(name)
        loaded = {n for n in sys.modules if n not in before and n.startswith('_gallery_freshness_')}
        self.addCleanup(lambda: [sys.modules.pop(n, None) for n in loaded])
        self.assertEqual(loaded, {'_gallery_freshness_'+n for n in files})
        for guard in (native, evaluator, v3_source):
            guard.check()
        sources.close()
        self.assertFalse(loaded & set(sys.modules), 'the genuine close removed exactly its own registered entries')
        for guard in (native, evaluator, v3_source):
            guard.check()
        self.assertTrue(all(m.__name__ in sys.modules for m in (native.module, evaluator.module, diagnostic)))

    def test_driver_and_evaluator_registry_namespaces_are_disjoint_from_the_v3_sources_namespace(self):
        text = (HERE/'diagnose_connected_gallery_freshness.py').read_text()
        self.assertEqual(re.findall(r"module_name='([^']+)'\+name", text), ['_gallery_freshness_'])
        for path in ('evaluate_siglip2_connected_mlp.py', 'train_siglip2_connected_mlp.py'):
            for name in re.findall(r"load_authenticated\(\s*'([^']+)'", (HERE/path).read_text()):
                self.assertFalse(name.startswith('_gallery_freshness_'), name)
        for name in re.findall(r"load_module\('([^']+)'", DRIVER.read_text()):
            self.assertFalse(name.startswith('_gallery_freshness_'), name)

    def test_final_guard_itself_runs_sources_guard_every_owner_pin_locks_resources_and_cap(self):
        w = World(self)
        w.S.checks.append(lambda: w.log.append('pin'))
        w.d.final_guard(w.S)
        for marker in ('sources.guard', 'guard.self', 'guard.requests', 'guard.observer', 'guard.native', 'guard.evaluator',
                       'guard.v3', 'guard.v5', 'pin', 'api.authenticate', 'final_guard.resources', 'budget.check'):
            self.assertIn(marker, w.log, marker)
        broken = World(self, locks_fail=True)
        with self.assertRaises(ValueError):
            broken.d.final_guard(broken.S)

    def test_every_live_guard_runs_before_any_sources_or_registry_removal(self):
        w = World(self, receipt=True)
        w.d.cleanup(w.S, None)
        first_removal = w.log.index('sources.close')
        for marker in ('guard.self', 'guard.requests', 'guard.observer', 'guard.native', 'guard.evaluator', 'guard.v3',
                       'guard.v5', 'final_guard.resources', 'sources.guard'):
            self.assertLess(w.log.index(marker), first_removal, marker+' must precede Sources.close')
        text = DRIVER.read_text()
        cleanup = ast.get_source_segment(text, functions(ast.parse(text))['cleanup'])
        self.assertLess(cleanup.index('final_guard(S)'), cleanup.index('S.sources.close'))
        self.assertLess(cleanup.index('S.sources.close'), cleanup.index('registry_dispose(S)'))
        self.assertLess(cleanup.index('registry_dispose(S)'), cleanup.index('publish_receipt(S)'))

    def test_driver_wiring_owns_v3_and_v5_with_real_sources_before_first_use(self):
        text = DRIVER.read_text()
        fns = functions(ast.parse(text))
        admit = ast.get_source_segment(text, fns['admit'])
        self.assertLess(admit.index("requests.Source(d, history['source'])"), admit.index('d.prepare('))
        self.assertLess(admit.index("requests.Source(v5, L['observer']['source'])"), admit.index('v5.authenticated('))
        guard_tuple = [n for n in ast.walk(fns['start_native']) if isinstance(n, ast.For) and
                       isinstance(n.iter, ast.Tuple) and any(ast.unparse(e) == 'S.self_source' for e in n.iter.elts)]
        self.assertEqual(len(guard_tuple), 1)
        self.assertLessEqual({'S.diagnostic_source', 'S.v5_source', 'S.wrapper_source', 'S.packed_source'},
                             {ast.unparse(e) for e in guard_tuple[0].iter.elts})
        start = ast.get_source_segment(text, fns['start_native'])
        self.assertIn('S.light = lambda: (S.budget.check(), S.sources.guard(), live(S))', start)
        self.assertLess(start.index('v5.capture_workspace_owner'), start.index('S.dispose_check = pin(S.workspace_dispose)'))
        self.assertLess(start.index('S.fixed = d.scorer'), start.index('S.checks.append(pin(S.fixed))'))
        run = ast.get_source_segment(text, fns['run'])
        self.assertLess(run.index('S.audit = S.diagnostic.origin_audit'), run.index('S.checks.append(pin(S.audit))'))
        self.assertLess(run.index('S.checks.append(pin(S.audit))'), run.index('S.audit()'))
        cleanup = ast.get_source_segment(text, fns['cleanup'])
        self.assertLess(cleanup.index('S.dispose_check()'), cleanup.index('S.workspace_dispose()'))
        self.assertNotIn('d.write_json', text)
        self.assertNotIn('S.diagnostic.write_json', text)

    def cleanup_world(self, tamper):
        """Production-shaped: the REAL requests.Source on the REAL v3 module inside the real cleanup()."""
        d, mods, facts, sources, names = real_modules(self)
        w = World(self, receipt=False)
        w.S.diagnostic, w.S.diagnostic_source, w.S.v5_source = mods['v3'], sources['v3'], sources['v5']
        w.log_dispose = []
        w.S.workspace_dispose = lambda: w.log.append('workspace_dispose')
        tamper(mods)
        return w

    def test_cleanup_never_executes_a_tampered_real_v3_function_and_keeps_the_primary_error(self):
        def evil(*a, **k):
            raise RuntimeError('TAMPERED_FUNCTION_EXECUTED')
        w = self.cleanup_world(lambda m: setattr(m['v3'].final_state, '__code__', evil.__code__))
        error = ValueError('primary body failure')
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, error)
        self.assertIs(caught.exception, error)
        notes = '\n'.join(error.__notes__)
        self.assertIn('authenticated live source changed', notes)
        self.assertNotIn('TAMPERED_FUNCTION_EXECUTED', notes)
        self.assertNotIn('workspace_dispose', w.log, 'checked(final_resources) refused before disposal')
        self.assertNotIn('release', w.log)
        self.assertIn('evaluator_exit', w.log, 'the genuine combined exit is self-authenticating and still runs')
        self.assertIn('terminal.cgroup_read', w.log, 'independent terminal checks still run')

    def test_cleanup_stops_before_disposal_when_the_pinned_closure_changed(self):
        w = World(self, receipt=False)
        calls = []
        def dispose():
            calls.append('disposed')
        w.S.workspace_dispose = dispose
        w.S.dispose_check = w.d.pin(dispose)
        dispose.__defaults__ = (1,)
        error = ValueError('primary')
        with self.assertRaises(ValueError) as caught:
            w.d.cleanup(w.S, error)
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [], 'a changed disposal closure must not be invoked')
        self.assertIn('pinned runtime callable changed', '\n'.join(error.__notes__))
        self.assertIn('final_state', w.log, 'original final state still runs after the refused disposal')

    def test_pinned_runtime_callables_are_checked_at_every_phase_boundary_and_before_publication(self):
        w = World(self, receipt=True)
        flips = []
        w.S.checks.append(lambda: flips.append('pin'))
        w.d.cleanup(w.S, None)
        self.assertGreaterEqual(len(flips), 3, 'checked() wrappers plus final_guard')
        w2 = World(self, receipt=True)
        def broken():
            raise ValueError('pinned runtime callable changed')
        w2.S.checks.append(broken)
        with self.assertRaises(ValueError):
            w2.d.cleanup(w2.S, None)
        self.assertFalse((w2.output/'receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
