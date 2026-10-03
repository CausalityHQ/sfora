#!/usr/bin/env python3
"""Bounded stdlib falsifiers; no native imports or quality data."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_siglip2_nearest_ranking import NativeAdmissionFixture, driver

PATH = Path(__file__).with_name('evaluate_siglip2_nearest_ranking.py')
if PATH.exists():
    spec = importlib.util.spec_from_file_location('_nearest_evaluation_test', PATH)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
else:
    evaluator = None


def quality(r1, ap):
    return {'recall_at_1': sum(r1)/len(r1), 'map_at_r': sum(ap)/len(ap),
            'per_query_r1': r1, 'per_query_ap': ap}


def evaluator_exit_boundary(tree):
    """Reverse only the owned audit dispatch; preserve the entire base AST."""
    dump = lambda n: ast.dump(n, include_attributes=False)
    before = ast.parse("trainer.native_source_api(t).audit_origins(t['legacy'])", mode='eval').body
    after = ast.parse("t['old'].audit_origins(t['legacy'])", mode='eval').body
    exit_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
    class Inverse(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if dump(node) == dump(before):
                self.count += 1
                return ast.copy_location(copy.deepcopy(after), node)
            return self.generic_visit(node)
    inverse = Inverse()
    inverse.visit(exit_node)
    evaluator.require(inverse.count == 1 and hashlib.sha256(dump(tree).encode()).hexdigest() ==
        'b2780835cc8f588a69794e07d57d1abf4a965bd060b09412424a48555ef71e8b',
        'only ordinary evaluator exit audit dispatch may change')


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluator, 'procedure-owned evaluator is required')

    def test_exit_real_owned_api_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f = NativeAdmissionFixture(root)
            api = f.admit()
            def descriptor(name, names):
                destination = root / name; destination.mkdir()
                for n in names:
                    (destination / n).write_bytes(PATH.with_name(n).read_bytes())
                code = {n: hashlib.sha256((destination / n).read_bytes()).hexdigest() for n in names}
                manifest = destination / 'execution.json'; manifest.write_text(json.dumps(code))
                return {'root': str(destination), 'code': code,
                        'execution_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
            own = descriptor('evaluator', evaluator.FILES)
            training = descriptor('trainer', evaluator.TRAIN_FILES)
            reference = descriptor('reference', evaluator.REFERENCE['code'])
            f.context.update(root=Path(training['root']), code=training['code'],
                             args=SimpleNamespace(execution_sha256=training['execution_sha256']))
            context = {'trainer': driver, 'training_context': f.context, 'guards': {},
                'root': Path(own['root']), 'args': SimpleNamespace(execution_sha256=own['execution_sha256']),
                'code': own['code'], 'launch': {'training': training}}
            def exit_call():
                f.context['fit_context']['phase_seconds'].clear()
                return evaluator.exit_rehash(context)
            # No model was constructed: only its disconnected helper snapshot is stubbed.
            with patch.object(driver, 'helper_guard'), patch.object(evaluator, 'REFERENCE', reference), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'unknown or changed'):
                    f.old.audit_origins(f.legacy)
                self.assertIs(exit_call(), f.legacy['origins'])
                self.assertEqual(set(f.legacy['origins']['files']), set(f.files))
                self.assertEqual(set(f.legacy['origins']['native_files']), set(f.files))
                with patch.object(driver, 'native_source_api', None), self.assertRaises(TypeError):
                    exit_call()
                with self.assertRaisesRegex(ValueError, 'owned legacy'):
                    api.audit_origins(dict(f.legacy))
                private = next(c.cell_contents for c in api.audit_origins.__closure__ if
                    isinstance(c.cell_contents, driver.FunctionType) and c.cell_contents.__name__ == 'audit_origins')
                with patch.dict(private.__globals__, {'_nearest_supplement': {'files': {}, 'modules': {}}}), \
                        self.assertRaisesRegex(ValueError, 'global binding changed'):
                    exit_call()
                with patch.dict(f.context['source'], {'native_authority': {}}), \
                        self.assertRaisesRegex(ValueError, 'owned API/supplement changed'):
                    exit_call()
                f.set_origins({**f.files, f.bulk[0]['path']: f.bulk[0]['sha256']})
                with self.assertRaisesRegex(ValueError, 'unknown or changed'):
                    exit_call()
                f.set_origins(f.files)
                prior = f.legacy['selected']['genuine']['prior']
                prior['images'] = ['fitter exit must reject this']
                with self.assertRaisesRegex(ValueError, 'FIT image resolution'):
                    exit_call()
                prior['images'] = []
                manifest = Path(own['root']) / 'execution.json'
                raw = manifest.read_bytes(); manifest.write_bytes(raw + b' ')
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    exit_call()
                manifest.write_bytes(raw)
                f.unchanged_originals(self)

    def test_exit_dispatch_inverse_rejects_retained_guard_mutations(self):
        tree = ast.parse(PATH.read_bytes())
        evaluator_exit_boundary(copy.deepcopy(tree))
        for statement in ("trainer.exit_rehash(t)", "trainer.helper_guard(t)",
                          "trainer.native_source_api(t).audit_origins(t['legacy'])"):
            mutant = copy.deepcopy(tree)
            node = next(n for n in mutant.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
            target = ast.dump(ast.parse(statement).body[0], include_attributes=False)
            node.body = [n for n in node.body if ast.dump(n, include_attributes=False) != target]
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                evaluator_exit_boundary(mutant)
        mutant = copy.deepcopy(tree)
        node = next(n for n in mutant.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
        node.body = [n for n in node.body if not isinstance(n, ast.For)]
        with self.assertRaises(ValueError):
            evaluator_exit_boundary(mutant)

    def test_cost_fresh_matched_core_and_whole_service(self):
        records = {'control': {'service_seconds': 100, 'total_training_core_seconds': 60},
                   'candidate': {'service_seconds': 150, 'total_training_core_seconds': 90}}
        self.assertTrue(evaluator.paired_cost(records)['pass'])
        records['candidate']['total_training_core_seconds'] = 90.001
        self.assertFalse(evaluator.paired_cost(records)['pass'])
        for bad in (0, float('nan'), True, 101):
            records['control']['total_training_core_seconds'] = bad
            with self.assertRaises(ValueError):
                evaluator.paired_cost(records)

    def test_diagnostic_negative_never_vetoes_selection(self):
        d = evaluator.diagnostic_result([.1, -.1], [.09, -.2])
        self.assertFalse(d['mechanism_demonstrated'])
        self.assertTrue(d['diagnostic_only'])
        self.assertFalse(d['utility_veto'])
        self.assertEqual(evaluator.diagnostic_result([], [])['triples'], 0)
        with self.assertRaises(ValueError):
            evaluator.diagnostic_result([float('nan')], [0])

    def test_batch_roles_and_exact_tails(self):
        self.assertEqual(evaluator.batch_sizes(1734), [32]*54+[6])
        self.assertEqual(evaluator.batch_sizes(1715), [32]*53+[19])
        self.assertEqual(evaluator.batch_sizes(1749)[-1], 21)
        self.assertEqual(evaluator.batch_sizes(1730)[-1], 2)

    def test_same_size_restored_mtime_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'state'; p.write_bytes(b'first')
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            evaluator.bound_file({}, p, sha)
            old = p.stat(); p.write_bytes(b'other'); os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns))
            with self.assertRaises(ValueError):
                evaluator.bound_file({}, p, sha)

    def test_quality_gate_source_floors_and_immediate_failure(self):
        source = quality([1,0,1,0], [.5]*4)
        concat = quality([1,0,1,0], [.6]*4)
        arms = {'control': quality([1,0,1,0], [.6]*4),
                'candidate': quality([1,1,1,0], [.7]*4)}
        costs = evaluator.paired_cost({a:{'service_seconds':100,'total_training_core_seconds':50}
                                      for a in evaluator.ARMS})
        pair = evaluator.metric_deltas(arms,source,concat,4)['candidate_minus_control']
        intervals = {m: {'mean_delta':sum(pair[m])/4,'product_lower95':.01,
                          'product_upper95':.3,'query_lower95':0,'query_upper95':.4}
                     for m in evaluator.METRICS}
        self.assertEqual(evaluator.decide(arms,source,concat,4,intervals,costs)['decision'],'GO')
        arms['candidate'] = quality([1,0,1,0],[.8]*4)
        self.assertEqual(evaluator.decide(arms,source,concat,4,{},costs)['decision'],'KILL')
        with self.assertRaises(ValueError):
            evaluator.decide(arms,source,concat,4,intervals,costs)

    def test_readiness_precedes_quality_call(self):
        calls=[]
        def score():
            calls.append('quality')
        for k in evaluator.READINESS:
            ready = dict.fromkeys(evaluator.READINESS, True); ready[k] = False
            with self.assertRaises(ValueError):
                evaluator.quality_after_readiness(ready,score)
        self.assertEqual(calls,[])
        evaluator.quality_after_readiness(dict.fromkeys(evaluator.READINESS,True),score)
        self.assertEqual(calls,['quality'])

    def test_exact_schema_duplicate_nonfinite_and_closure_tamper(self):
        for raw in ('{"phase":1,"phase":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                evaluator.strict_json(raw)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'worker.py'; p.write_text('pass\n')
            code = {'worker.py': hashlib.sha256(p.read_bytes()).hexdigest()}
            manifest = root/'execution.json'; manifest.write_text(json.dumps(code))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(evaluator.closure(root,digest,{'worker.py'},{}),code)
            p.write_text('fail\n')
            with self.assertRaises(ValueError):
                evaluator.closure(root,digest,{'worker.py'},{})
            with self.assertRaises(ValueError):
                evaluator.closure(root,digest,{'worker.py','extra.py'},{})

    def test_resource_failed_or_partial_endpoint_rejected(self):
        for phase in ('cpu','export','score'):
            base = {'resource_policy':evaluator.policy(phase),'wall_seconds':1,
                    'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':1 if phase == 'export' else 0,
                    'cuda_initialized':phase == 'export'}
            evaluator.check_resource_facts(base,phase)
            for key,value in (('wall_seconds',evaluator.policy(phase)['seconds']),
                              ('wall_seconds',float('nan')),('process_peak_rss_kib',8*1024**2+1),
                              ('peak_cuda_allocated_bytes',10_000_000_000),('cuda_initialized',phase != 'export')):
                with self.subTest(phase=phase,key=key), self.assertRaises(ValueError):
                    evaluator.check_resource_facts({**base,key:value},phase)

    def test_authority_roles_foreign_cpu_and_future_source_inventory(self):
        file = {'path':'/tmp/file','sha256':'a'*64}
        unit = {'receipt':file,'log':file,'unit':'actual-unit','invocation_id':'b'*32,
                'service_seconds':1,'native_peak_rss_kib':1,'both_locks_held':True}
        endpoints = [{'arm':a,'launch':file,'terminal':unit,'checkpoint':{'path':'/tmp/resume.pt','sha256':'c'*64},
                      'terminal_state_sha256':'d'*64,'inference':{'path':'/tmp/inference.pt','sha256':'e'*64},
                      'inference_state_sha256':'f'*64} for a in evaluator.ARMS]
        args = SimpleNamespace(execution_sha256='1'*64,phase='cpu',arm=None)
        launch = {'schema':evaluator.AUTHORITY_SCHEMA,'execution_sha256':args.execution_sha256,
                  'training':{'root':'/tmp/training','execution_sha256':'2'*64,
                              'code':dict.fromkeys(evaluator.TRAIN_FILES,'3'*64)},
                  'reference':evaluator.REFERENCE,'phase':'cpu','arm':None,'panel':'selection',
                  'endpoints':endpoints,'selected_cpu':None,'exports':{},'selection_go':None,
                  'resource_policies':{p:evaluator.policy(p) for p in ('cpu','export','score')},
                  'cost_policy':evaluator.COST_POLICY,'both_locks_held':True}
        evaluator.check_launch(launch,args)
        for key,value in (('selected_cpu',unit),('panel','validation'),('arm','candidate'),('exports',{'candidate':unit}),
                          ('reference',{**evaluator.REFERENCE,'execution_sha256':'0'*64})):
            with self.subTest(key=key),self.assertRaises(ValueError):
                evaluator.check_launch({**launch,key:value},args)
        foreign = copy.deepcopy(launch); foreign['training']['code']['old-encoder.py'] = '4'*64
        with self.assertRaises(ValueError):
            evaluator.check_launch(foreign,args)
        missing = copy.deepcopy(endpoints[0]); missing.pop('inference')
        with self.assertRaises(ValueError):
            evaluator.check_endpoint(missing)

    def test_exact_fraction_threshold_product_zero_and_perquery_reduction(self):
        n = 500; left = [i%2 for i in range(n)]; right = left.copy(); right[0] = 1
        source=concat=quality(left,[.4]*n)
        arms={'control':quality(left,[.5]*n),'candidate':quality(right,[.502]*n)}
        pair=evaluator.metric_deltas(arms,source,concat,n)['candidate_minus_control']
        costs=evaluator.paired_cost({a:{'service_seconds':10,'total_training_core_seconds':5} for a in evaluator.ARMS})
        intervals={m:{'mean_delta':sum(pair[m])/n,'product_lower95':.0001,'product_upper95':.01,
                      'query_lower95':0,'query_upper95':.01} for m in evaluator.METRICS}
        self.assertEqual(evaluator.decide(arms,source,concat,n,intervals,costs)['decision'],'GO')
        intervals['per_query_r1']['product_lower95']=0
        self.assertEqual(evaluator.decide(arms,source,concat,n,intervals,costs)['decision'],'KILL')
        intervals['per_query_r1']['mean_delta']=.1
        with self.assertRaises(ValueError):
            evaluator.decide(arms,source,concat,n,intervals,costs)
        bad=copy.deepcopy(arms); bad['candidate']['per_query_ap'][0]=float('nan')
        with self.assertRaises(ValueError):
            evaluator.metric_deltas(bad,source,concat,n)

    def test_native_source_and_qualification_predicate_order(self):
        tree=ast.parse(PATH.read_text()); functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        score=ast.get_source_segment(PATH.read_text(),functions['score_exports'])
        self.assertLess(score.index('archived_replay('),score.index('quality_after_readiness('))
        self.assertLess(score.index('values=read_wires('),score.index('quality_after_readiness('))
        export=ast.get_source_segment(PATH.read_text(),functions['native_export'])
        self.assertLess(export.index('authenticate_payloads('),export.index('export_pass('))
        self.assertEqual(export.count('trainer.load_inference('),2)
        self.assertNotIn('cache_rows',export)
        self.assertNotIn('reset_peak',PATH.read_text())
        self.assertNotIn('globals()[',PATH.read_text())
        self.assertNotIn('if False',PATH.read_text())
        self.assertTrue(all(not isinstance(n,ast.Import) or all(a.name.split('.')[0] not in evaluator.NATIVE for a in n.names)
                            for n in tree.body))


if __name__ == '__main__':
    unittest.main()
