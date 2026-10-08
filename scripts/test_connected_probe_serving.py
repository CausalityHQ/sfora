#!/usr/bin/env python3
"""Bounded stdlib falsifier for the installed-probe parity gate; native work UNRUN.

Narrow<=30s, final serial affected gate<=120s, RLIMIT_AS 1GiB. No Torch/NumPy/PIL/
sfora/third-party import, SSH, image, checkpoint, GPU, cloud or native job. Real
sources are read as bytes/AST (and the real probe evaluator is loaded through the
derived loader); stand-ins replace only live native objects. The accepted-looking
path is never self-regenerated: every pin below is an independent literal.
"""
import ast
import base64
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import observe_connected_serving as observer  # noqa: E402
import qualify_connected_probe_serving as gate  # noqa: E402
import qualify_connected_serving_requests as requests  # noqa: E402

DRIVER = HERE/'qualify_connected_probe_serving.py'
SRC = ROOT/'src'/'sfora'
# Independent literal base (ec20d929) pins of every reused original source.
REUSED = {
    'qualify_connected_serving_requests.py':'6a4d310d4eb883b7bf3d2ea74a1e96223cee096d0bdb9bc6c2a651eecf9ff163',
    'observe_connected_serving.py':'b255c6e835ad3d66b1143f2ca2192e500958fe8ffd6f62ec135a0326a6ad48de',
    'connected_control_native_authority.py':'fc8795be3cca792aa328f087b1908c8fc12ff75362802042a35e874c02732442',
    'qualify_connected_control_serving.py':'2961c2b0129b762403273c1d550b44c0ba7e4e9ec231a3e39848a9ee57d68815',
    'evaluate_siglip2_connected_probe.py':'6b60df9162a5cdfc253c61535dac09681604ba1868a6b2135a75fd65b8ffdd1d',
    'evaluate_siglip2_connected_mlp.py':'b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4'}
SHA = lambda raw: hashlib.sha256(raw).hexdigest()  # noqa: E731


def file_sha(path):
    return SHA(Path(path).read_bytes())


def fact(path):
    return {'path':str(path),'sha256':file_sha(path)}


def tree_of(path):
    return ast.parse(Path(path).read_bytes())


def function(tree, name, owner=None):
    scope = tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == owner).body
    return next(n for n in scope if isinstance(n,ast.FunctionDef) and n.name == name)


def signature(node):
    return ast.unparse(node.args)


def b64(sha):
    return base64.urlsafe_b64encode(bytes.fromhex(sha)).rstrip(b'=').decode()


class World:
    """Synthetic installed wheel + copied bundle built from REAL reviewed bytes."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.site = self.root/'site'
        self.pkg = self.site/'sfora'
        self.pkg.mkdir(parents=True)
        self.bundle = self.root/'bundle'
        self.bundle.mkdir()
        for name in ('connected_probe_inference.py','packed_int8.py','cutile_int8.py','joint_relational_compaction.py'):
            (self.pkg/name).write_bytes((SRC/name).read_bytes())
        # Synthetic ninth historical file: the real joint copy is an out-of-repo artifact.
        for name,_ in gate.HISTORICAL:
            source = HERE/name
            (self.bundle/name).write_bytes(source.read_bytes() if source.exists() else b'# synthetic historical joint copy\n')
        self.historical = tuple((n,file_sha(self.bundle/n)) for n,_ in gate.HISTORICAL)
        joint = dict(gate.HISTORICAL)['joint_relational_compaction.py']
        ledger = (SRC/'_connected_probe_inference_authority.py').read_text().replace(
            joint,dict(self.historical)['joint_relational_compaction.py'])
        (self.pkg/'_connected_probe_inference_authority.py').write_text(ledger)
        self.ledger_sha = file_sha(self.pkg/'_connected_probe_inference_authority.py')
        bridge = (SRC/'connected_compact_serving.py').read_text().replace(gate.LEDGER_SHA,self.ledger_sha)
        (self.pkg/'connected_compact_serving.py').write_text(bridge)
        self.bridge_sha = file_sha(self.pkg/'connected_compact_serving.py')
        for name,payload in (('vision.pt',b'v'),('endpoint.pt',b'e'),('processor.json',b'{}')):
            (self.bundle/name).write_bytes(payload)
        manifest = {'schema':gate.BUNDLE_SCHEMA,'code':dict(self.historical),
            'files':{n:file_sha(self.bundle/n) for n in ('vision.pt','endpoint.pt','processor.json')}}
        (self.bundle/'bundle.json').write_text(json.dumps(manifest))
        dist = self.site/'sfora-9.9.9.dist-info'
        dist.mkdir()
        rows = [(f'sfora/{p.name}','sha256='+b64(file_sha(p)),str(p.stat().st_size)) for p in sorted(self.pkg.iterdir())]
        with (dist/'RECORD').open('w',newline='') as stream:
            csv.writer(stream).writerows(rows)
        self.record = dist/'RECORD'
        self.patches = [patch.object(gate,'HISTORICAL',self.historical),patch.object(gate,'LEDGER_SHA',self.ledger_sha),
            patch.object(gate,'BRIDGE_SHA',self.bridge_sha)]

    def __enter__(self):
        for p in self.patches: p.start()
        return self

    def __exit__(self, *args):
        for p in self.patches: p.stop()

    def sources(self):
        s = {'probe_driver':fact(DRIVER),'probe_test':fact(Path(__file__).resolve()),
            'request_driver':fact(HERE/'qualify_connected_serving_requests.py'),
            'request_test':fact(HERE/'test_connected_serving_requests.py'),
            'observer':fact(HERE/'observe_connected_serving.py'),'observer_test':fact(HERE/'test_observe_connected_serving.py'),
            'control_native':fact(HERE/'connected_control_native_authority.py')}
        for role,name in gate.INSTALLED_NAMES.items():
            s[role] = fact(self.pkg/name)
        return s

    def bundle_descriptor(self):
        return {'directory':str(self.bundle),'manifest':fact(self.bundle/'bundle.json'),
            'code':{n:fact(self.bundle/n) for n,_ in self.historical},
            'files':{n:fact(self.bundle/n) for n in ('vision.pt','endpoint.pt','processor.json')}}

    def wheel(self, direct_url=None):
        return {'site_root':str(self.site),'distribution':'sfora','version':'9.9.9','record':fact(self.record),
            'direct_url':direct_url}

    def authority(self):
        def f(name):
            path = self.root/name
            path.write_bytes(name.encode())
            return fact(path)
        unit = {'receipt':f('receipt.json'),'log':f('unit.log'),'unit':'sfora-probe-export-unit','invocation_id':'a'*32,
            'service_seconds':12.5,'native_peak_rss_kib':1024,'both_locks_held':True}
        images = [f(f'image{i}.png') for i in range(32)]
        tail = [f(f'tail{i}.png') for i in range(5)]
        return {'schema':gate.SCHEMA,'sources':self.sources(),
            'evaluator':{'root':str(self.root/'evaluator'),'execution_sha256':'b'*64,
                'code':{n:'c'*64 for n in gate.EVALUATOR_FILES}},
            'evaluation_authority':f('launch.json'),'endpoint':{'arm':'candidate','seed':179061,'stage':'first','panel':'selection'},
            'train_export':unit,'bundle':self.bundle_descriptor(),
            'gallery':{'file':f('gallery.bin'),'count':10},'images':{'fixed32':images,'tail':{'role':'gallery','files':tail}},
            'native':{'authority':f('native.json'),'library':{'path':str(self.root/'candidate.so'),
                'sha256':gate.ARCHIVED_BINARY_SHA},'archived_control_binary_ack':True},
            'wheel':self.wheel(),'locks':[],'resource_policy':dict(gate.POLICY)}


@contextmanager
def world():
    with tempfile.TemporaryDirectory() as raw, World(raw) as value:
        yield value


class ReviewedPins(unittest.TestCase):
    def test_probe_runtime_pins_equal_real_reviewed_bytes(self):
        self.assertEqual(file_sha(SRC/'connected_probe_inference.py'),gate.RUNTIME_SHA)
        self.assertEqual(file_sha(SRC/'_connected_probe_inference_authority.py'),gate.LEDGER_SHA)
        self.assertEqual(file_sha(SRC/'connected_compact_serving.py'),gate.BRIDGE_SHA)
        self.assertEqual(file_sha(SRC/'packed_int8.py'),gate.PACKED_SHA)
        record = gate.literal_record((SRC/'_connected_probe_inference_authority.py').read_bytes())
        self.assertEqual(record['SCHEMA'],gate.LEDGER_SCHEMA)
        self.assertEqual(record['HISTORICAL_CODE'],gate.HISTORICAL)
        self.assertEqual((record['RUNTIME_SHA256'],record['PACKED_SHA256']),(gate.RUNTIME_SHA,gate.PACKED_SHA))
        literal = function(tree_of(SRC/'connected_compact_serving.py'),'_installed_probe_authority')
        self.assertEqual(literal.body[0].value.elts[1].value,gate.LEDGER_SHA)
        self.assertEqual(sorted(dict(gate.HISTORICAL)),[n for n,_ in gate.HISTORICAL])

    def test_probe_trainer_and_historical_vector_equal_real_files(self):
        pins = dict(gate.HISTORICAL)
        self.assertEqual(file_sha(HERE/gate.TRAINER),pins[gate.TRAINER])
        self.assertEqual(file_sha(HERE/'test_siglip2_connected_probe.py'),pins['test_siglip2_connected_probe.py'])
        for name,digest in gate.HISTORICAL:
            if (HERE/name).exists():
                self.assertEqual(file_sha(HERE/name),digest,name)

    def test_archived_binary_pins_equal_the_original_native_module(self):
        values = {}
        for node in tree_of(HERE/'connected_control_native_authority.py').body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ARCHIVE_SHA','BINARY_SHA'):
                values[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(values,{'ARCHIVE_SHA':gate.ARCHIVED_BUILD_SHA,'BINARY_SHA':gate.ARCHIVED_BINARY_SHA})
        self.assertTrue(gate.ARCHIVED_BINARY_SHA.startswith('3d1ec796') and gate.ARCHIVED_BUILD_SHA.startswith('c2d6'))

    def test_every_reused_original_is_byte_pinned_at_base(self):
        for name,digest in REUSED.items():
            self.assertEqual(file_sha(HERE/name),digest,name)

    def test_driver_literals_are_exact_policy_and_no_future_pin(self):
        self.assertEqual(gate.POLICY,{'body_seconds':300,'host_bytes':8*1024**3,'swap_bytes':0,
            'cuda_allocated_bytes_exclusive':10_000_000_000,'whole_process_seconds':1500,'exit_reserve_seconds':300})
        digests = set(re.findall(r'[0-9a-f]{64}',DRIVER.read_text()))
        allowed = {gate.RUNTIME_SHA,gate.LEDGER_SHA,gate.BRIDGE_SHA,gate.PACKED_SHA,gate.ARCHIVED_BINARY_SHA,
            gate.ARCHIVED_BUILD_SHA,*(h for _,h in gate.HISTORICAL)}
        self.assertEqual(digests,allowed)


class ReusedSeams(unittest.TestCase):
    def test_original_helper_signatures_are_unchanged(self):
        req = tree_of(HERE/'qualify_connected_serving_requests.py')
        for name,args in (('read_file','fact'),('strict_json','raw'),('raise_failures','failures'),
                ('owner','factory, guard, charges'),('check_resources','facts, policy, *, reserve'),
                ('native_ties','packed, gallery_type, native, observer'),('decode_images','observer, image_api, facts, paths')):
            self.assertEqual(signature(function(req,name)),args,name)
        self.assertEqual(signature(function(req,'load','Source')),'cls, fact')
        self.assertEqual(signature(function(req,'check','Source')),'self')
        self.assertEqual(signature(function(req,'__init__','Locks')),'self, rows')
        obs = tree_of(HERE/'observe_connected_serving.py')
        for name,args in (('file_bytes','fact, *, keep=False'),('canonical','value'),('native_snapshot','result'),
                ('tensor_snapshot','value, dtype, shape, width'),('pairs','items')):
            self.assertEqual(signature(function(obs,name)),args,name)
        native = tree_of(HERE/'connected_control_native_authority.py')
        self.assertEqual(signature(function(native,'load_evaluator_source')),'fact, request')
        self.assertEqual(signature(function(native,'validate_runtime_compiler')),'record, observer')
        self.assertEqual(signature(function(native,'__init__','CombinedAuthority')),'self, context, fact, observer, request')
        self.assertEqual(signature(function(native,'install','CombinedAuthority')),'self, evaluator_source, evaluation_context')
        for name in ('provenance_facts','check','collect'):
            function(native,name,'CombinedAuthority')

    def test_probe_evaluator_surface_is_the_one_the_driver_calls(self):
        ev = tree_of(HERE/'evaluate_siglip2_connected_probe.py')
        for name,args in (('authority','args'),('accept_unit','context, unit, phase, arm=None, seed=None, stage=None, panel=None'),
                ('exit_rehash','context, original_guard'),('native_start','context'),('endpoint_scope','context, endpoint'),
                ('authenticate_payloads','context, endpoint'),('resources','context, before'),('guard_helpers','context'),
                ('merge_guards','target, values'),('check_code','value, names, pins=None'),
                ('closure','root, expected, names, guards'),('load_authenticated','name, path, digest, guards'),
                ('label','endpoint'),('batch_sizes','count'),('policy','phase')):
            self.assertEqual(signature(function(ev,name)),args,name)
        names = {t.id for n in ev.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
        self.assertTrue({'FILES','ARMS','SEEDS'} <= names)

    def test_native_composition_keys_are_read_by_the_probe_flows(self):
        """CombinedAuthority needs only keys the probe trainer/evaluator themselves use on the same context."""
        native = function(tree_of(HERE/'connected_control_native_authority.py'),'CombinedAuthority') if False else next(
            n for n in tree_of(HERE/'connected_control_native_authority.py').body if isinstance(n,ast.ClassDef) and n.name == 'CombinedAuthority')
        needed = {n.slice.value for n in ast.walk(native) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and
            n.value.id in {'context','legacy'} and isinstance(n.slice,ast.Constant)}
        produced = {'control_native_owned','native_source_owned'}  # created by CombinedAuthority / nearest.native_source_api
        used = set()
        for name in ('evaluate_siglip2_connected_probe.py','train_siglip2_connected_probe.py'):
            used |= {n.slice.value for n in ast.walk(tree_of(HERE/name)) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and
                isinstance(n.slice.value,str)}
        self.assertEqual(needed-produced-used,set())
        # Both connected trainers build the SAME historical context (nearest/fitter/old/legacy/fit_context/source).
        for name in ('train_siglip2_connected_mlp.py','train_siglip2_connected_probe.py'):
            self.assertIn('context = trainer.authority(historical)',(HERE/name).read_text())
        self.assertIn("'nearest': nearest, 'fitter': fitter, 'fit_context': original, 'legacy': original['legacy']",
            (HERE/'train_siglip2_identity_diversity.py').read_text())

    def test_probe_exit_rehash_has_the_sole_native_api_target_and_exact_inverse(self):
        """The same substitution CombinedAuthority.install derives for the MLP evaluator applies once to the probe."""
        node = function(tree_of(HERE/'evaluate_siglip2_connected_probe.py'),'exit_rehash')
        dump = lambda value: ast.dump(value,include_attributes=False)  # noqa: E731
        source = ast.parse("t['nearest'].native_source_api(t)",mode='eval').body
        target = ast.parse('_control_native_api(t)',mode='eval').body
        hits = [n for n in ast.walk(node) if dump(n) == dump(source)]
        self.assertEqual(len(hits),1)
        original = copy.deepcopy(node)

        class Swap(ast.NodeTransformer):
            def __init__(self, before, after):
                self.before, self.after, self.count = before, after, 0

            def visit(self, value):
                if dump(value) == dump(self.before):
                    self.count += 1
                    return copy.deepcopy(self.after)
                return super().visit(value)
        forward, back = Swap(source,target), None
        changed = forward.visit(node)
        back = Swap(target,source)
        restored = back.visit(copy.deepcopy(changed))
        self.assertEqual((forward.count,back.count),(1,1))
        self.assertEqual(dump(restored),dump(original))
        self.assertNotEqual(dump(changed),dump(original))


class DerivedLoader(unittest.TestCase):
    RAW = (HERE/'connected_control_native_authority.py').read_bytes()

    def derive(self, raw=None):
        return gate.derive_evaluator_loader(raw or self.RAW,{'marker':1},'/x/derived.py')

    def test_exact_one_constant_substitution_and_inverse(self):
        loader = self.derive()
        self.assertEqual(loader.__name__,'load_evaluator_source')
        self.assertIn(gate.PROBE_EVALUATOR_NAME,loader.__code__.co_consts)
        self.assertNotIn(gate.MLP_EVALUATOR_NAME,loader.__code__.co_consts)
        text = self.RAW.decode()
        self.assertEqual(text.count(gate.MLP_EVALUATOR_NAME),1)
        original = function(ast.parse(text),'load_evaluator_source')
        derived = ast.parse(text.replace(gate.MLP_EVALUATOR_NAME,gate.PROBE_EVALUATOR_NAME))
        a = ast.dump(function(derived,'load_evaluator_source'),include_attributes=False)
        self.assertEqual(ast.dump(original,include_attributes=False).replace(gate.MLP_EVALUATOR_NAME,gate.PROBE_EVALUATOR_NAME),a)

    def test_missing_extra_or_absent_function_is_rejected(self):
        text = self.RAW.decode()
        with self.assertRaises(ValueError): self.derive(text.replace(gate.MLP_EVALUATOR_NAME,'other.py').encode())
        extra = text.replace("        return SimpleNamespace(module=module,fact=source.fact,check=check)",
            "        unused = 'evaluate_siglip2_connected_mlp.py'\n        return SimpleNamespace(module=module,fact=source.fact,check=check)")
        self.assertNotEqual(extra,text)
        with self.assertRaises(ValueError): self.derive(extra.encode())
        with self.assertRaises(ValueError): self.derive(b'x = 1\n')

    def test_original_module_and_function_identity_are_untouched(self):
        before = dict(vars(requests))
        self.derive()
        self.assertEqual(vars(requests).keys(),before.keys())
        self.assertTrue(all(vars(requests)[k] is v for k,v in before.items()))

    def test_real_probe_evaluator_loads_and_checks_through_derived_loader(self):
        """Genuine Source/builtin identity composition on the REAL probe evaluator source (stdlib only)."""
        native = requests.Source.load(fact(HERE/'connected_control_native_authority.py'))
        try:
            loader = gate.derive_evaluator_loader(requests.read_file(native.fact),vars(native.module),native.fact['path'])
            evaluator_fact = fact(HERE/gate.PROBE_EVALUATOR_NAME)
            loaded = loader(evaluator_fact,requests)
            try:
                self.assertEqual(loaded.module.FILES,gate.EVALUATOR_FILES)
                self.assertEqual(loaded.module.ARMS,('control','candidate'))
                loaded.check()
                loaded.module._SOURCE_BUILTINS = ()
                with self.assertRaises(ValueError): loaded.check()
            finally:
                self.assertIs(sys.modules.pop(loaded.module.__name__),loaded.module)
            # The original MLP-named helper must still reject the probe evaluator.
            original = requests.Source.load(fact(HERE/'connected_control_native_authority.py'))
            try:
                with self.assertRaises(ValueError): original.module.load_evaluator_source(evaluator_fact,requests)
            finally:
                del sys.modules[original.module.__name__]
        finally:
            del sys.modules[native.module.__name__]
        self.assertFalse([n for n in sys.modules if n.startswith('_connected_requests_')])


class ShapeAdmission(unittest.TestCase):
    def setUp(self):
        self.cm = world()
        self.w = self.cm.__enter__()
        self.base = self.w.authority()

    def tearDown(self):
        self.cm.__exit__(None,None,None)

    def mutate(self, edit):
        value = copy.deepcopy(self.base)
        edit(value)
        return value

    def test_exact_authority_is_admitted(self):
        gate.check_shape(self.base)

    def test_each_structural_mutant_is_rejected(self):
        edits = {
            'schema':lambda a: a.__setitem__('schema','connected-control-serving-requests-authority-v2'),
            'extra key':lambda a: a.__setitem__('observation',{}),
            'missing key':lambda a: a.pop('gallery'),
            'sources missing role':lambda a: a['sources'].pop('ledger'),
            'sources extra role':lambda a: a['sources'].__setitem__('foreign',a['sources']['ledger']),
            'source relative path':lambda a: a['sources']['bridge'].__setitem__('path','connected_compact_serving.py'),
            'source bad sha':lambda a: a['sources']['bridge'].__setitem__('sha256','A'*64),
            'evaluator extra file':lambda a: a['evaluator']['code'].__setitem__('x.py','d'*64),
            'evaluator missing file':lambda a: a['evaluator']['code'].pop('test_connected_probe_evaluation.py'),
            'evaluator MLP name':lambda a: a['evaluator']['code'].__setitem__('evaluate_siglip2_connected_mlp.py','d'*64),
            'endpoint stage':lambda a: a['endpoint'].__setitem__('stage','all'),
            'endpoint seed string':lambda a: a['endpoint'].__setitem__('seed','179061'),
            'unit lock false':lambda a: a['train_export'].__setitem__('both_locks_held',False),
            'unit invocation':lambda a: a['train_export'].__setitem__('invocation_id','xyz'),
            'unit zero service':lambda a: a['train_export'].__setitem__('service_seconds',0),
            'bundle manifest elsewhere':lambda a: a['bundle']['manifest'].__setitem__('path','/tmp/bundle.json'),
            'bundle code MLP trainer':lambda a: a['bundle']['code'].__setitem__('train_siglip2_connected_mlp.py',
                a['bundle']['code'].pop(gate.TRAINER)),
            'bundle code unpinned sha':lambda a: a['bundle']['code'][gate.TRAINER].__setitem__('sha256','e'*64),
            'bundle code missing':lambda a: a['bundle']['code'].pop('quadratic_readout.py'),
            'bundle file outside':lambda a: a['bundle']['files']['vision.pt'].__setitem__('path','/tmp/vision.pt'),
            'gallery count':lambda a: a['gallery'].__setitem__('count',9),
            'gallery count bool':lambda a: a['gallery'].__setitem__('count',True) or a['gallery'].__setitem__('count',9.5),
            'images 31':lambda a: a['images']['fixed32'].pop(),
            'images duplicate':lambda a: a['images']['fixed32'].__setitem__(1,a['images']['fixed32'][0]),
            'tail role':lambda a: a['images']['tail'].__setitem__('role','train'),
            'tail empty':lambda a: a['images']['tail'].__setitem__('files',[]),
            'tail 33':lambda a: a['images']['tail'].__setitem__('files',a['images']['fixed32'] + [a['images']['fixed32'][0]]),
            'native ack':lambda a: a['native'].__setitem__('archived_control_binary_ack',False),
            'native ack truthy':lambda a: a['native'].__setitem__('archived_control_binary_ack',1),
            'native other binary':lambda a: a['native']['library'].__setitem__('sha256','f'*64),
            'native extra key':lambda a: a['native'].__setitem__('dirty_rust',True),
            'wheel version':lambda a: a['wheel'].__setitem__('version',''),
            'wheel direct url bad':lambda a: a['wheel'].__setitem__('direct_url',{'path':'x'}),
            'locks type':lambda a: a.__setitem__('locks','none'),
        }
        for name,edit in edits.items():
            with self.subTest(name), self.assertRaises(ValueError):
                gate.check_shape(self.mutate(edit))

    def test_policy_is_the_exact_typed_ceiling_not_a_variable_reserve(self):
        for key,value in (('exit_reserve_seconds',299),('exit_reserve_seconds',301),('exit_reserve_seconds',300.0),
                ('whole_process_seconds',1499),('whole_process_seconds',1501),('whole_process_seconds',True),
                ('body_seconds',299),('body_seconds',301),('host_bytes',8*1024**3+1),('swap_bytes',1),
                ('cuda_allocated_bytes_exclusive',10_000_000_001)):
            with self.subTest((key,value)), self.assertRaises(ValueError):
                gate.check_shape(self.mutate(lambda a: a['resource_policy'].__setitem__(key,value)))
        with self.assertRaises(ValueError): gate.check_shape(self.mutate(lambda a: a['resource_policy'].pop('swap_bytes')))
        with self.assertRaises(ValueError): gate.check_shape(self.mutate(lambda a: a['resource_policy'].__setitem__('extra',1)))


class InstalledAndWheel(unittest.TestCase):
    def setUp(self):
        self.cm = world()
        self.w = self.cm.__enter__()

    def tearDown(self):
        self.cm.__exit__(None,None,None)

    def check(self, w=None):
        w = w or self.w
        return gate.check_installed(w.sources(),w.bundle_descriptor(),observer)

    def test_reviewed_installed_closure_is_admitted(self):
        manifest = self.check()
        self.assertEqual(manifest['schema'],gate.BUNDLE_SCHEMA)
        gate.check_wheel(self.w.wheel(),self.w.sources(),observer,ROOT)

    def test_stale_or_tampered_installed_bytes_are_rejected(self):
        for name in ('connected_probe_inference.py','packed_int8.py','_connected_probe_inference_authority.py','connected_compact_serving.py'):
            path = self.w.pkg/name
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n# x\n')
                with self.subTest(name), self.assertRaises(ValueError): self.check()
            finally: path.write_bytes(original)
        self.check()

    def test_stale_pin_in_the_driver_is_rejected(self):
        for name in ('RUNTIME_SHA','PACKED_SHA','LEDGER_SHA','BRIDGE_SHA'):
            with patch.object(gate,name,'1'*64), self.subTest(name), self.assertRaises(ValueError): self.check()

    def test_ledger_and_bridge_binding_mutants(self):
        pkg = self.w.pkg
        ledger = (pkg/'_connected_probe_inference_authority.py').read_text()
        mutants = {
            'schema':ledger.replace(gate.LEDGER_SCHEMA,'sfora-connected-inference-extraction-v1'),
            'historical entry':ledger.replace(dict(gate.HISTORICAL)[gate.TRAINER],'9'*64),
            'runtime pin':ledger.replace(gate.RUNTIME_SHA,'8'*64),
            'extra statement':ledger+'EXTRA = 1\n',
            'not literal':ledger+'X = __import__("os")\n'}
        original_bridge = (pkg/'connected_compact_serving.py').read_text()
        for name,text in mutants.items():
            (pkg/'_connected_probe_inference_authority.py').write_text(text)
            new = file_sha(pkg/'_connected_probe_inference_authority.py')
            # The bridge literal and the driver pins follow the mutated ledger: only the ledger predicate can reject.
            (pkg/'connected_compact_serving.py').write_text(original_bridge.replace(self.w.ledger_sha,new))
            bridge_new = file_sha(pkg/'connected_compact_serving.py')
            sources = {**self.w.sources()}
            with patch.object(gate,'LEDGER_SHA',new), patch.object(gate,'BRIDGE_SHA',bridge_new), self.subTest(name), \
                    self.assertRaises((ValueError,KeyError,SyntaxError)):
                gate.check_installed(sources,self.w.bundle_descriptor(),observer)
            with patch.object(gate,'LEDGER_SHA',new), patch.object(gate,'BRIDGE_SHA',bridge_new):
                pass
        (pkg/'_connected_probe_inference_authority.py').write_text(ledger)
        (pkg/'connected_compact_serving.py').write_text(original_bridge)
        bridge = (pkg/'connected_compact_serving.py').read_text()
        (pkg/'connected_compact_serving.py').write_text(bridge.replace(self.w.ledger_sha,'7'*64))
        new = file_sha(pkg/'connected_compact_serving.py')
        with patch.object(gate,'BRIDGE_SHA',new), self.assertRaises(ValueError):
            gate.check_installed({**self.w.sources(),'bridge':{'path':str(pkg/'connected_compact_serving.py'),'sha256':new}},
                self.w.bundle_descriptor(),observer)

    def test_sibling_layout_and_mixed_bundle_closures_are_rejected(self):
        other = self.w.root/'elsewhere'
        other.mkdir()
        (other/'packed_int8.py').write_bytes((self.w.pkg/'packed_int8.py').read_bytes())
        sources = self.w.sources()
        sources['packed'] = fact(other/'packed_int8.py')
        with self.assertRaises(ValueError): gate.check_installed(sources,self.w.bundle_descriptor(),observer)
        manifest = json.loads((self.w.bundle/'bundle.json').read_text())
        for name,edit in (('mlp schema',lambda m: m.__setitem__('schema','siglip2-connected-mlp-bundle-v1')),
                ('mlp code',lambda m: m['code'].__setitem__('train_siglip2_connected_mlp.py',m['code'].pop(gate.TRAINER))),
                ('code digest',lambda m: m['code'].__setitem__(gate.TRAINER,'6'*64)),
                ('files digest',lambda m: m['files'].__setitem__('vision.pt','5'*64)),
                ('extra file',lambda m: m['files'].__setitem__('foreign.pt','4'*64))):
            value = copy.deepcopy(manifest)
            edit(value)
            (self.w.bundle/'bundle.json').write_text(json.dumps(value))
            descriptor = self.w.bundle_descriptor()
            with self.subTest(name), self.assertRaises(ValueError): gate.check_installed(self.w.sources(),descriptor,observer)
        (self.w.bundle/'bundle.json').write_text(json.dumps(manifest))
        self.check()

    def test_linked_bundle_files_are_rejected(self):
        os.link(self.w.bundle/'processor.json',self.w.root/'alias.json')
        with self.assertRaises(ValueError): self.check()

    def test_wheel_mutants(self):
        sources, wheel = self.w.sources(), self.w.wheel()
        direct = self.w.site/'sfora-9.9.9.dist-info'/'direct_url.json'
        with self.assertRaises(ValueError):  # undeclared direct_url.json
            direct.write_text('{}')
            try: gate.check_wheel(wheel,sources,observer,ROOT)
            finally: direct.unlink()
        for name,payload in (('editable',{'url':'file:///x','dir_info':{'editable':True}}),):
            direct.write_text(json.dumps(payload))
            try:
                with self.subTest(name), self.assertRaises(ValueError):
                    gate.check_wheel(self.w.wheel(fact(direct)),sources,observer,ROOT)
            finally: direct.unlink()
        direct.write_text(json.dumps({'url':'file:///x.whl','archive_info':{'hash':'sha256=0'}}))
        try: gate.check_wheel(self.w.wheel(fact(direct)),sources,observer,ROOT)
        finally: direct.unlink()
        with self.assertRaises(ValueError):  # checkout contains the wheel root
            gate.check_wheel(wheel,sources,observer,self.w.root)
        with self.assertRaises(ValueError):  # wheel root contains the checkout
            gate.check_wheel(wheel,sources,observer,self.w.pkg)
        (self.w.site/'__editable__.sfora-9.9.9.pth').write_text('x')
        with self.assertRaises(ValueError): gate.check_wheel(wheel,sources,observer,ROOT)
        (self.w.site/'__editable__.sfora-9.9.9.pth').unlink()
        (self.w.site/'x.pth').write_text(str(ROOT)+'\n')
        with self.assertRaises(ValueError): gate.check_wheel(wheel,sources,observer,ROOT)
        (self.w.site/'x.pth').unlink()
        (self.w.site/'.git').mkdir()
        with self.assertRaises(ValueError): gate.check_wheel(wheel,sources,observer,ROOT)
        (self.w.site/'.git').rmdir()
        record = self.w.record.read_text()
        self.w.record.write_text(record.replace('sha256=','sha256=A',1))
        with self.assertRaises(ValueError): gate.check_wheel(self.w.wheel(),sources,observer,ROOT)
        self.w.record.write_text(record)
        # RECORD omits one role entirely
        self.w.record.write_text('\n'.join(l for l in record.splitlines() if 'packed_int8' not in l)+'\n')
        with self.assertRaises(ValueError): gate.check_wheel(self.w.wheel(),sources,observer,ROOT)
        self.w.record.write_text(record)
        gate.check_wheel(self.w.wheel(),sources,observer,ROOT)

    def test_loaded_sfora_must_be_the_installed_wheel(self):
        good = SimpleNamespace(__file__=str(self.w.pkg/'__init__.py'),__path__=[str(self.w.pkg)])
        (self.w.pkg/'__init__.py').write_text('')
        gate.check_loaded(self.w.site,good)
        for bad in (SimpleNamespace(__file__=str(SRC/'__init__.py'),__path__=[str(SRC)]),
                SimpleNamespace(__file__=str(self.w.pkg/'__init__.py'),__path__=[str(self.w.pkg),str(SRC)])):
            with self.assertRaises(ValueError): gate.check_loaded(self.w.site,bad)


class Witnesses(unittest.TestCase):
    @staticmethod
    def outputs(count, fill=1):
        codes = bytes([fill])*(count*128)
        norms = struct.pack('<e',1.5)*count
        wire = b''.join(codes[i*128:(i+1)*128]+norms[i*2:(i+1)*2] for i in range(count))
        return {'raw':{'dtype':'torch.float32','shape':[count,128],'hex':(b'\0'*(count*512)).hex()},
            'unit':{'dtype':'torch.float32','shape':[count,128],'hex':(b'\1'*(count*512)).hex()},
            'codes':{'dtype':'torch.int8','shape':[count,128],'hex':codes.hex()},
            'inverse_norms':{'dtype':'torch.float16','shape':[count],'hex':norms.hex()},'wire_hex':wire.hex()}

    @staticmethod
    def native(count, ids=None, scores=None):
        ids = ids or list(range(10))*count
        scores = scores or [1.0]*(10*count)
        return [{'format':'q','shape':[count,10],'hex':struct.pack(f'<{len(ids)}q',*ids).hex()},
            {'format':'f','shape':[count,10],'hex':struct.pack(f'<{len(scores)}f',*scores).hex()}]

    def test_outputs_are_typed_complete_and_wire_consistent(self):
        gate.check_outputs(self.outputs(3),3)
        for name,edit in (('dtype',lambda o: o['raw'].__setitem__('dtype','torch.float64')),
                ('shape',lambda o: o['codes'].__setitem__('shape',[2,128])),
                ('hex short',lambda o: o['unit'].__setitem__('hex','00')),
                ('wire bit',lambda o: o.__setitem__('wire_hex','ff'+o['wire_hex'][2:])),
                ('wire short',lambda o: o.__setitem__('wire_hex',o['wire_hex'][:-2])),
                ('norm bit',lambda o: o['inverse_norms'].__setitem__('hex','0000'+o['inverse_norms']['hex'][4:])),
                ('extra',lambda o: o.__setitem__('foreign',{})),
                ('missing',lambda o: o.pop('unit'))):
            value = copy.deepcopy(self.outputs(3))
            edit(value)
            with self.subTest(name), self.assertRaises((ValueError,KeyError)): gate.check_outputs(value,3)

    def test_native_witness_requires_typed_finite_in_gallery_ids(self):
        gate.check_native(self.native(2),2,gallery=10)
        for name,value,gallery in (('negative id',self.native(1,ids=[-1]+list(range(1,10))),None),
                ('id outside gallery',self.native(1,ids=list(range(1,11))),10),
                ('nan score',self.native(1,scores=[float('nan')]+[1.0]*9),None),
                ('batch',self.native(1),2)):
            with self.subTest(name), self.assertRaises(ValueError): gate.check_native(value,gallery if name == 'batch' else 1,gallery if name != 'batch' else None)

    def test_select_batches_binds_membership_and_the_exact_export_tail(self):
        rows = [{'path':f'/img/{i}.png','image_sha256':f'{i:064x}','panel_ordinal':i} for i in range(80)]
        mapping = {'query':list(range(0,80,2)),'gallery':list(range(1,80,2))}
        facts = {'fixed32':[{'path':r['path'],'sha256':r['image_sha256']} for r in rows[:32]],
            'tail':{'role':'gallery','files':[{'path':r['path'],'sha256':r['image_sha256']} for r in (rows[i] for i in mapping['gallery'][-8:])]}}
        sizes = {'query':[32,8],'gallery':[32,8]}
        batches,ordinals = gate.select_batches(facts,rows,mapping,sizes,lambda n: [min(32,n-s) for s in range(0,n,32)])
        self.assertEqual([k for k,_ in batches],list(gate.KINDS))
        self.assertEqual([len(v) for _,v in batches],[1,2,32,8])
        self.assertEqual(ordinals,mapping['gallery'][-8:])
        size = lambda n: [min(32,n-s) for s in range(0,n,32)]  # noqa: E731
        for name,edit,export in (
                ('tail reordered',lambda f: f['tail']['files'].reverse(),sizes),
                ('tail wrong role rows',lambda f: f['tail'].__setitem__('role','query'),sizes),
                ('tail size',lambda f: f['tail']['files'].pop(),sizes),
                ('export tail size differs',lambda f: None,{'query':[32,8],'gallery':[32,7]}),
                ('fixed row bytes',lambda f: f['fixed32'][3].__setitem__('sha256','0'*64),sizes),
                ('fixed foreign file',lambda f: f['fixed32'][4].__setitem__('path','/img/foreign.png'),sizes)):
            value = copy.deepcopy(facts)
            edit(value)
            with self.subTest(name), self.assertRaises(ValueError): gate.select_batches(value,rows,mapping,export,size)

    def test_gallery_is_the_exact_ordered_export_row_subset(self):
        rows = list(range(6))
        wire = b''.join(bytes([i])*130 for i in rows)
        mapping = {'gallery':[1,3,5]}
        data = bytes([1])*130+bytes([3])*130+bytes([5])*130
        gallery = {'file':{'path':'/g','sha256':SHA(data)},'count':3}
        self.assertEqual(gate.bind_gallery(wire,rows,mapping,gallery),data)
        for name,args in (('order',({'gallery':[3,1,5]},gallery)),('count',(mapping,{**gallery,'count':4})),
                ('sha',(mapping,{**gallery,'file':{'path':'/g','sha256':SHA(b'x')}}))):
            with self.subTest(name), self.assertRaises(ValueError): gate.bind_gallery(wire,rows,*args)
        with self.assertRaises(ValueError): gate.bind_gallery(wire[:-1],rows,mapping,gallery)


class FakeImage:
    def __init__(self, tag):
        self.tag, self.size, self.closed = tag, (1,1), False

    def tobytes(self):
        return self.tag.encode()

    def close(self):
        self.closed = True


class Obj:
    """Weakref-able stand-in for a model/readout object."""


class Fake:
    """Stdlib stand-in world for the reference loader, installed index and native gallery."""
    PATHS = {'B1':['/i/0'],'B2':['/i/0','/i/1'],'B32':[f'/i/{i}' for i in range(32)],'TAIL':[f'/t/{i}' for i in range(5)]}

    def __init__(self):
        self.table, self.native_table, self.images = {}, {}, []
        for fill,(kind,paths) in enumerate(self.PATHS.items(),start=1):
            out = Witnesses.outputs(len(paths),fill)
            self.table[tuple(paths)] = out
            ids = [(fill*3+j) % 10 for j in range(10)]
            self.native_table[out['wire_hex']] = (ids,[1.0-j/100 for j in range(10)])
        self.live = 0
        self.cache = SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=0))
        self.tamper_current = self.leak_registry = self.leak_refs = False
        self.accept_after_close = self.reference_leak = False
        self.output_edit = self.native_edit = self.decode_edit = None
        self.kept, self.calls, self.indexes = None, 0, []

    def decode(self, paths):
        images = [FakeImage(str(p)) for p in paths]
        if self.decode_edit: images = self.decode_edit(images)
        self.images += images
        return images

    def produce(self, index, images):
        self.calls += 1
        out = copy.deepcopy(self.table[tuple(i.tag for i in images)])
        if self.output_edit: out = self.output_edit(out,self.calls) or out
        return out

    def native_result(self, out, count):
        ids,scores = self.native_table[self.table_key(out)]
        if self.native_edit: ids,scores = self.native_edit(ids,scores,self.calls)
        return (memoryview(struct.pack(f'<{10*count}q',*(ids*count))).cast('q',shape=[count,10]),
            memoryview(struct.pack(f'<{10*count}f',*(scores*count))).cast('f',shape=[count,10]))

    def table_key(self, out):
        count = len(out['wire_hex'])//260
        for key,value in self.table.items():
            if len(key) == count and value['codes'] == out['codes'] or value['wire_hex'] == out['wire_hex']:
                return value['wire_hex']
        return next(iter(self.native_table))

    def make(self):
        fake = self

        class Index:
            def __init__(self):
                self.name = f'_sfora_connected_compact_fake{len(fake.indexes)}'
                self._module = SimpleNamespace()
                self._module = type(sys)(self.name)
                exec('def inference_outputs(endpoint, images):\n    return endpoint["produce"](images)\n',vars(self._module))
                function_ = self._module.inference_outputs
                self._apis = {'inference_outputs':(function_,function_.__code__)}
                self._endpoint = {k:Obj() for k in ('model','processor_object','head_object','A','C','mu_train')}
                self._endpoint['processor_cache'] = fake.cache
                self._endpoint['produce'] = lambda images: fake.produce(self,images)
                self._closed = False
                sys.modules[self.name] = self._module
                fake.live += 1
                fake.indexes.append(self)

            def search_images(self, images):
                if self._closed and fake.accept_after_close:
                    return fake.native_result(fake.produce(self,images),len(images))
                if self._closed: raise RuntimeError('connected compact index is closed')
                try:
                    out = self._apis['inference_outputs'][0](self._endpoint,images)
                    return fake.native_result(out,len(images))
                except BaseException:
                    self.close()
                    raise

            def _check_current(self):
                if fake.tamper_current: raise ValueError('installed state changed')

            def close(self):
                if self._closed: return
                self._closed = True
                fake.live -= 1
                if not fake.leak_registry: sys.modules.pop(self.name,None)
                if fake.leak_refs: fake.kept = dict(self._endpoint)
                self._endpoint = None
        return Index

    def factory(self):
        index_type, fake = self.make(), self

        def factory(*, method='from_probe_bundle', **overrides):
            if method != 'from_probe_bundle': raise ValueError('unsupported connected bundle schema/closure')
            if any(factory.base[k] != v for k,v in overrides.items()): raise ValueError('pinned input differs')
            if fake.live: raise AssertionError('second encoder owner before release')
            return index_type()
        factory.base = {'bundle_dir':Path('/b'),'expected_bundle_sha256':'a'*64,'gallery_path':Path('/g'),
            'expected_gallery_sha256':'b'*64,'gallery_count':10,'native_library_path':Path('/n'),
            'expected_native_library_sha256':'c'*64}
        return factory

    def reference(self, items):
        registry = {}
        fake = self

        def loader(name, path, pin, guards):
            portable = Obj()
            portable.load_inference = lambda directory,sha,device: {'model':Obj()}
            portable.inference_outputs = lambda state,images: fake.produce(None,images)
            portable.release_inference = lambda state: None
            registry[name] = portable
            return portable
        rows = gate.reference_outputs(name='_connected_probe_gate_reference_x',directory=Path('/b'),bundle_sha='a'*64,
            loader=loader,pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=items,decode=self.decode,
            rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda state: None,after=lambda: None,registry=registry)
        if self.reference_leak: sys.modules['_connected_probe_gate_reference_leak'] = Obj()
        return rows

    def native(self, ref):
        cls = self
        out = {}
        for kind,row in ref.items():
            wire = row['outputs']['wire_hex']
            ids,scores = self.native_table[wire]
            count = len(wire)//260
            out[kind] = observer.native_snapshot((memoryview(struct.pack(f'<{10*count}q',*(ids*count))).cast('q',shape=[count,10]),
                memoryview(struct.pack(f'<{10*count}f',*(scores*count))).cast('f',shape=[count,10])))
        return out


class FakeDenial:
    def __init__(self):
        self.active, self.checks = False, 0

    def check(self):
        self.checks += 1
        return {'checks':self.checks,'historical_modules_absent':True}


def body_for(fake, **override):
    batches = [(k,[Path(p) for p in v]) for k,v in Fake.PATHS.items()]
    reference = fake.reference
    tail_wire = bytes.fromhex(fake.table[tuple(Fake.PATHS['TAIL'])]['wire_hex'])
    factory = fake.factory()
    capture = lambda index,images: gate.captured_search(index,images,copy.deepcopy,observer.native_snapshot)  # noqa: E731
    mutant = [('flip',lambda: None,lambda: None,'current')]
    values = dict(batches=batches,ordinals=[7,9,11,13,15],persisted_tail=lambda w: w == tail_wire,reference=reference,
        native=fake.native,factory=factory,decode=fake.decode,rgb=gate.rgb_digest,capture=capture,
        mutants=lambda index: gate.run_mutants([(n,a,r,m) for n in sorted(gate.MUTANT_NAMES) for a,r,m in
            [(lambda: setattr(fake,'tamper_current',True),lambda: setattr(fake,'tamper_current',False),'current')]],
            lambda mode: index._check_current()),
        lifecycle_check=lambda index: gate.lifecycle(index,fake.decode,[Path('/i/0')],Path('/b'),lambda: 'no mappings\n'),
        denial=FakeDenial(),guard=lambda **kw: None,started=gate.time.perf_counter(),
        mutant_factory=lambda: gate.factory_mutants(factory))
    values.update(override)
    return values


class CaptureSeam(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        self.index = self.fake.make()()

    def tearDown(self):
        self.index.close()

    def search(self, images=None, snapshot=copy.deepcopy):
        images = images or self.fake.decode(Fake.PATHS['B2'])
        return gate.captured_search(self.index,images,snapshot,observer.native_snapshot)

    def test_exactly_one_return_witness_equals_the_public_outputs(self):
        output,native = self.search()
        self.assertEqual(output,self.fake.table[tuple(Fake.PATHS['B2'])])
        gate.check_native(native,2,10)
        self.assertIsNone(sys.getprofile())

    def test_zero_or_repeated_inference_returns_are_rejected(self):
        original = self.index.search_images
        self.index.search_images = lambda images: None
        with self.assertRaises(ValueError): self.search()
        def twice(images):
            original(images)
            return original(images)
        self.index.search_images = twice
        with self.assertRaises(ValueError): self.search()
        self.assertIsNone(sys.getprofile())

    def test_existing_nonempty_callback_is_rejected_before_any_alteration_and_identity_survives(self):
        foreign = lambda frame,event,arg: None  # noqa: E731
        calls = []
        real = sys.setprofile
        def spy(callback):
            calls.append(callback)
            real(callback)
        sys.setprofile(foreign)
        try:
            with patch.object(sys,'setprofile',spy), self.assertRaisesRegex(ValueError,'unprofiled capture required'):
                self.search()
            self.assertEqual(calls,[])  # never altered, not even transiently
            self.assertIs(sys.getprofile(),foreign)
        finally: real(None)
        self.assertIsNone(sys.getprofile())

    def test_profile_identity_is_restored_after_search_errors_and_cancellation(self):
        calls = []
        real = sys.setprofile
        def spy(callback):
            calls.append(callback)
            real(callback)
        for error in (KeyError('search failed'),KeyboardInterrupt()):
            def boom(images, error=error): raise error
            self.index.search_images = boom
            calls.clear()
            with patch.object(sys,'setprofile',spy), self.assertRaises(type(error)): self.search()
            self.assertEqual(len(calls),2)
            self.assertIsInstance(calls[0],gate.Capture)
            self.assertIsNone(calls[1])
            self.assertIsNone(sys.getprofile())

    def test_snapshot_failure_keeps_only_bounded_scalar_facts_and_pins_nothing_native(self):
        marker, holder = weakref.WeakSet(), Obj()
        marker.add(holder)
        self.fake.output_edit = lambda out,calls: {**out,'native_reference':holder}
        holder = None
        def bad(value):
            tensor = value['native_reference']  # the traceback frame of this call would pin it
            raise TypeError('snapshot failed '+'x'*1000)
        kept = None
        try: self.search(snapshot=bad)
        except ValueError as error: kept = error  # deliberately retained, with its traceback
        self.assertIsNotNone(kept)
        self.assertIsNone(kept.__cause__)
        self.assertIsNone(kept.__context__)
        self.assertLessEqual(len(str(kept)),400)
        self.assertIn('TypeError',str(kept))
        gc.collect()
        self.assertEqual(len(marker),0)  # the retained exception holds no output/model reference
        self.assertIsNone(sys.getprofile())
        self.fake.output_edit = None

    def test_snapshot_failure_still_rejects_the_call(self):
        with self.assertRaises(ValueError): self.search(snapshot=lambda value: 1/0)
        self.assertIsNone(sys.getprofile())

    def test_output_snapshot_is_typed_before_copy(self):
        class Tensor:
            def __init__(self, count, dtype, size):
                self.shape, self.dtype, self.size = (count,128),dtype,size
        with self.assertRaises((ValueError,AttributeError,KeyError)):
            gate.snapshot_outputs({'raw':Tensor(1,'torch.float32',512),'unit':None,'codes':None,'inverse_norms':None,'wire':b'x'},
                lambda *a: {})
        with self.assertRaises(ValueError):
            gate.snapshot_outputs({'raw':Tensor(2,'torch.float32',1),'unit':1,'codes':1,'inverse_norms':1,'wire':b'x'*10},lambda *a: {})
        with self.assertRaises(ValueError): gate.snapshot_outputs({'raw':1},lambda *a: {})


class FakeData:
    def __init__(self, tensor, bump=False, frozen=False):
        self.t, self.bump, self.frozen = tensor, bump, frozen

    def __getitem__(self, index):
        return self.t.store[self.t.flat(index)]

    def __setitem__(self, index, value):
        if not self.frozen: self.t.store[self.t.flat(index)] = value
        if self.bump: self.t._version += 1

    def add_(self, value):
        self.t.store[0] += value

    def copy_(self, other):
        self.t.store[:] = other.store


class FakeTensor:
    def __init__(self, shape, store=None, bump=False, frozen=False):
        self.shape, self._version, self.bump, self.frozen = tuple(shape), 0, bump, frozen
        size = 1
        for d in self.shape: size *= d
        self.store = store if store is not None else [float(i) for i in range(size)]

    def dim(self): return len(self.shape)

    def flat(self, index):
        flat = 0
        for i,d in zip(index,self.shape,strict=True): flat = flat*d + i
        return flat

    def detach(self): return self

    def clone(self): return FakeTensor(self.shape,list(self.store))

    @property
    def data(self): return FakeData(self,self.bump,self.frozen)


FAKE_TORCH = SimpleNamespace(equal=lambda a,b: a.shape == b.shape and a.store == b.store)


class LeafMutant(unittest.TestCase):
    def test_every_dimensionality_changes_the_real_storage_without_a_version_bump(self):
        for shape in ((),(4,),(2,3),(2,3,4)):
            value = FakeTensor(shape)
            before = list(value.store)
            name,apply,restore,mode = gate.leaf_mutant(FAKE_TORCH,'leaf',value)
            self.assertEqual((name,mode),('leaf','outputs'))
            apply()
            self.assertNotEqual(value.store,before,shape)
            self.assertEqual(value._version,0)
            self.assertEqual(value.store[0],before[0]+.25)
            self.assertEqual(value.store[1:],before[1:])
            restore()
            self.assertEqual(value.store,before)

    def test_no_op_or_version_bumping_mutation_is_an_apply_failure_and_is_restored(self):
        for kwargs in ({'frozen':True},{'bump':True}):
            value = FakeTensor((2,3),**kwargs)
            before = list(value.store)
            mutant = gate.leaf_mutant(FAKE_TORCH,'leaf',value)
            with self.subTest(kwargs), self.assertRaisesRegex(ValueError,'real storage and bypass versions'):
                gate.run_mutants([mutant],lambda mode: (_ for _ in ()).throw(ValueError('rejected')))
            self.assertEqual(value.store,before)

    def test_restore_mismatch_is_reported(self):
        value = FakeTensor((3,))
        name,apply,restore,mode = gate.leaf_mutant(FAKE_TORCH,'leaf',value)
        apply()
        with patch.object(FakeData,'copy_',lambda self,other: None), self.assertRaisesRegex(ValueError,'restoration differs'):
            restore()

    def test_native_mutants_use_the_zero_tuple_leaf_helper_not_reshape(self):
        tree = tree_of(DRIVER)
        body = ast.unparse(function(tree,'native_live_mutants'))
        self.assertIn('leaf_mutant(torch, name, value)',body)
        self.assertNotIn('reshape',ast.unparse(function(tree,'leaf_mutant')))
        self.assertIn('(0,) * value.dim()',ast.unparse(function(tree,'leaf_mutant')))


class FailureFrames(unittest.TestCase):
    def test_clear_frames_releases_objects_pinned_by_a_retained_traceback(self):
        def deep():
            model = Obj()
            ref = weakref.ref(model)
            refs.append(ref)
            raise RuntimeError('boom')
        refs = []
        kept = None
        try: deep()
        except RuntimeError as error: kept = error
        gc.collect()
        self.assertIsNotNone(refs[0]())  # the retained traceback pins the finished frame
        gate.clear_frames(kept)
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertIsNotNone(kept.__traceback__)  # locations stay available to callers

    def test_active_frames_are_never_cleared_and_chains_are_followed(self):
        outer = None
        try:
            try: raise KeyError('inner')
            except KeyError: raise RuntimeError('outer')
        except RuntimeError as error: outer = error
        local = 'alive'
        gate.clear_frames(outer)  # this frame is still executing
        self.assertEqual(local,'alive')

    def test_reference_failure_releases_the_loader_and_pins_no_model_while_the_error_is_retained(self):
        fake, released, registry, refs = Fake(), [], {}, []
        def forward(model):
            local = model
            raise RuntimeError('forward failed')
        def loader(name, path, pin, guards):
            portable = Obj()
            def load(directory, sha, device):
                model = Obj()
                refs.append(weakref.ref(model))
                return {'model':model}
            portable.load_inference = load
            portable.inference_outputs = lambda state,images: forward(state['model'])
            portable.release_inference = lambda state: (released.append(1),state.clear())
            registry[name] = portable
            return portable
        kept = None
        try:
            gate.reference_outputs(name='_connected_probe_gate_reference_k',directory=Path('/b'),bundle_sha='a'*64,loader=loader,
                pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=[('B1',[Path('/i/0')])],decode=fake.decode,
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,registry=registry)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)
        gc.collect()
        self.assertEqual((released,registry),([1],{}))
        self.assertIsNone(refs[0]())
        self.assertTrue(all(i.closed for i in fake.images))

    def test_after_hook_and_release_failures_are_reported_alongside_the_primary(self):
        fake, registry = Fake(), {}
        def loader(name, path, pin, guards):
            portable = Obj()
            portable.load_inference = lambda d,s,device: {'model':Obj()}
            portable.inference_outputs = lambda state,images: (_ for _ in ()).throw(RuntimeError('forward failed'))
            portable.release_inference = lambda state: (_ for _ in ()).throw(OSError('release failed'))
            registry[name] = portable
            return portable
        kept = None
        try:
            gate.reference_outputs(name='_connected_probe_gate_reference_m',directory=Path('/b'),bundle_sha='a'*64,loader=loader,
                pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=[('B1',[Path('/i/0')])],decode=fake.decode,
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,registry=registry)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)
        self.assertIsInstance(kept.__cause__,BaseExceptionGroup)
        self.assertEqual(sorted(type(e).__name__ for e in kept.__cause__.exceptions) or [],['OSError'])


class ParityBody(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        self.before = set(sys.modules)

    def tearDown(self):
        for name in [n for n in set(sys.modules)-self.before if n.startswith(gate.OWNED_PREFIXES)]:
            del sys.modules[name]

    def run_body(self, **override):
        return gate.parity_body(**body_for(self.fake,**override))

    def test_green_run_orders_owners_and_compares_every_call(self):
        result = self.run_body()
        self.assertEqual([r['owner'] for r in result['installed']].count('A'),8)
        self.assertEqual(len(result['installed']),14)
        self.assertEqual(result['owner_order'],['reference','installed_A','installed_B'])
        self.assertEqual(result['factory_mutants'],dict.fromkeys(gate.FACTORY_NAMES,'rejected'))
        self.assertEqual(result['live_mutants'],dict.fromkeys(gate.MUTANT_NAMES,'rejected'))
        self.assertEqual(result['lifecycle']['post_close_rejected'],True)
        self.assertEqual(result['denial']['checks'],3)
        self.assertEqual(self.fake.live,0)
        self.assertTrue(all(i.closed for i in self.fake.images))

    def rejects(self, clean=True, **override):
        kept = None
        try: self.run_body(**override)
        except (ValueError,AssertionError) as error: kept = error  # retained exception, as a caller would
        self.assertIsNotNone(kept,'run unexpectedly accepted')
        if clean:
            self.assertEqual(gate.owned_names(),[])  # owner failure cleanup despite the retained exception
            self.assertEqual(self.fake.live,0)
            self.assertTrue(all(i.closed for i in self.fake.images))
        return kept

    def test_installed_wire_raw_unit_codes_or_norm_bit_differs(self):
        for field in ('wire_hex',):
            self.fake = Fake()
            self.fake.output_edit = lambda out,calls: ({**out,'wire_hex':'ff'+out['wire_hex'][2:]} if calls > 10 else None)
            self.rejects(clean=False)
        for name in ('raw','unit','codes','inverse_norms'):
            self.fake = Fake()
            self.fake.output_edit = lambda out,calls,name=name: (
                {**out,name:{**out[name],'hex':'0f'+out[name]['hex'][2:]}} if calls > 10 else None)
            with self.subTest(name): self.rejects(clean=False)

    def test_native_top10_ordinal_or_score_bit_differs(self):
        self.fake.native_edit = lambda ids,scores,calls: (ids[1:]+ids[:1],scores) if calls > 10 else (ids,scores)
        self.rejects()
        self.fake = Fake()
        self.fake.native_edit = lambda ids,scores,calls: (ids,[scores[0]+1e-6]+scores[1:]) if calls > 10 else (ids,scores)
        self.rejects()

    def test_repeat_reload_and_image_membership_differences(self):
        self.fake.output_edit = lambda out,calls: ({**out,'wire_hex':out['wire_hex'][:-2]+'00'} if calls == 12 else None)
        self.rejects()
        self.fake = Fake()
        self.fake.decode_edit = lambda images: images[::-1] if self.fake.calls > 9 and len(images) > 1 else images
        self.rejects()

    def test_reference_registry_survival_blocks_the_installed_owner(self):
        self.fake.reference_leak = True
        self.rejects(clean=False)

    def test_only_one_encoder_owner_may_be_live(self):
        factory = self.fake.factory()
        first = factory()
        try:
            with self.assertRaises(AssertionError): factory()
        finally: first.close()

    def test_installed_owner_registry_or_reference_leaks_are_rejected(self):
        self.fake.leak_registry = True
        self.rejects(clean=False)
        for name in gate.owned_names(): del sys.modules[name]
        self.fake = Fake()
        self.fake.leak_refs = True
        self.rejects(clean=False)

    def test_post_close_acceptance_and_double_close_errors_are_rejected(self):
        self.fake.accept_after_close = True
        self.rejects(clean=False)
        self.fake = Fake()
        factory = self.fake.factory()
        index = factory()
        original = index.close
        count = [0]
        def close():
            count[0] += 1
            if count[0] > 1: raise RuntimeError('second close failed')
            original()
        index.close = close
        with self.assertRaises(RuntimeError):
            gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '')
        original()

    def lifecycle(self, fake=None, maps=lambda: ''):
        fake = fake or self.fake
        index = fake.factory()()
        try: return gate.lifecycle(index,fake.decode,[Path('/i/0')],Path('/b'),maps)
        finally: index.close()

    def test_each_lifecycle_predicate_is_isolated(self):
        self.assertEqual(self.lifecycle()['registry_clean'],True)
        self.fake.leak_registry = True
        with self.assertRaisesRegex(ValueError,'owned registry entries'): self.lifecycle()
        for name in gate.owned_names(): del sys.modules[name]
        self.fake = Fake()
        self.fake.leak_refs = True
        with self.assertRaisesRegex(ValueError,'retained model'): self.lifecycle()
        self.fake = Fake()
        with self.assertRaisesRegex(ValueError,'bundle mappings'): self.lifecycle(maps=lambda: '/b/endpoint.pt\n')
        self.fake = Fake()
        self.fake.accept_after_close = True
        with self.assertRaisesRegex(ValueError,'post-close search accepted'): self.lifecycle()
        self.fake = Fake()
        self.fake.cache = SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=2))
        with self.assertRaisesRegex(ValueError,'processor cache'): self.lifecycle()

    def test_registry_pollution_between_owners_blocks_the_reload_owner(self):
        def polluted(index):
            result = gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '')
            sys.modules['_connected_serving_leaked_after_close'] = Obj()
            return result
        with self.assertRaisesRegex(ValueError,'second encoder owner preceded complete release'):
            self.run_body(lifecycle_check=polluted)

    def test_processor_cache_entries_surviving_close_are_rejected(self):
        self.fake.cache = SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=1))
        self.rejects(clean=False)
        self.fake = Fake()
        index = self.fake.factory()()
        index._endpoint.pop('processor_cache')
        with self.assertRaises(ValueError): gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '')
        index.close()

    def test_bundle_mapping_survival_is_rejected(self):
        self.rejects(lifecycle_check=lambda index: gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),
            lambda: '7f00 r--s /b/endpoint.pt\n'))

    def test_accepted_live_or_factory_mutant_is_rejected(self):
        accepted = lambda index: gate.run_mutants([('accepted',lambda: None,lambda: None,'current')],lambda mode: None)  # noqa: E731
        self.rejects(mutants=accepted)
        self.fake = Fake()
        factory = self.fake.factory()
        index_type = self.fake.make()
        def lax(*, method='from_probe_bundle', **overrides): return index_type()
        lax.base = factory.base
        with self.assertRaises(ValueError): gate.factory_mutants(lax)

    def test_factory_mutant_that_leaks_the_registry_is_rejected(self):
        fake = self.fake
        def leaky(*, method='from_probe_bundle', **overrides):
            sys.modules['_sfora_connected_compact_leak'] = Obj()
            raise ValueError('rejected but leaked')
        leaky.base = fake.factory().base
        with self.assertRaises(ValueError): gate.factory_mutants(leaky)
        del sys.modules['_sfora_connected_compact_leak']

    def test_restore_failure_is_reported_with_the_rejection_preserved(self):
        order = []
        def restore_bad(): raise RuntimeError('restore failed')
        reject = lambda mode: (_ for _ in ()).throw(ValueError('rejected'))  # noqa: E731
        with self.assertRaisesRegex(RuntimeError,'restore failed'):
            gate.run_mutants([('a',lambda: order.append('a'),restore_bad,'x')],reject)
        gate.run_mutants([('c',lambda: order.append('c'),lambda: order.append('restored-c'),'x')],reject)
        self.assertEqual(order,['a','c','restored-c'])

    def test_partial_apply_failure_still_restores_and_preserves_the_original_error(self):
        state = {'value':1}
        def apply():
            state['value'] = 99  # mutates, then fails
            raise RuntimeError('apply failed after mutating')
        def restore(): state['value'] = 1
        with self.assertRaisesRegex(RuntimeError,'apply failed after mutating'):
            gate.run_mutants([('partial',apply,restore,'x')],lambda mode: None)
        self.assertEqual(state['value'],1)
        def restore_bad():
            state['value'] = 1
            raise OSError('restore failed too')
        kept = None
        try: gate.run_mutants([('partial',apply,restore_bad,'x')],lambda mode: None)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)  # the original apply error stays primary
        self.assertTrue(any('restore failed too' in n for n in kept.__notes__))
        self.assertIsInstance(kept.__cause__,BaseExceptionGroup)
        self.assertEqual(state['value'],1)

    def test_apply_that_never_mutated_is_restored_harmlessly_and_cancellation_wins(self):
        calls = []
        def apply(): raise RuntimeError('apply failed')
        with self.assertRaises(RuntimeError):
            gate.run_mutants([('a',apply,lambda: calls.append('restore'),'x')],lambda mode: None)
        self.assertEqual(calls,['restore'])
        def interrupt(): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            gate.run_mutants([('a',interrupt,lambda: (_ for _ in ()).throw(RuntimeError('r')),'x')],lambda mode: None)

    def test_unrejected_mutant_is_still_restored_before_the_failure(self):
        state = {'value':1}
        with self.assertRaisesRegex(ValueError,'live mutant accepted'):
            gate.run_mutants([('m',lambda: state.update(value=2),lambda: state.update(value=1),'x')],lambda mode: None)
        self.assertEqual(state['value'],1)

    def test_persisted_tail_body_cap_and_denial_failures_are_rejected(self):
        self.rejects(persisted_tail=lambda wire: False)
        self.fake = Fake()
        self.rejects(started=gate.time.perf_counter()-301)
        class Denied(FakeDenial):
            def check(self): raise ValueError('historical source module is live')
        self.fake = Fake()
        self.rejects(denial=Denied())
        self.fake = Fake()
        calls = []
        def guard(**kw):
            calls.append(1)
            if len(calls) == 4: raise ValueError('source changed')
        self.rejects(guard=guard,clean=False)

    def test_denial_is_deactivated_even_on_failure(self):
        denial = FakeDenial()
        self.fake = Fake()
        self.rejects(denial=denial,persisted_tail=lambda w: False)
        self.assertFalse(denial.active)

    def test_reference_helper_releases_and_pops_registry_on_failure(self):
        registry, released = {}, []
        def loader(name, path, pin, guards):
            portable = Obj()
            portable.load_inference = lambda d,s,device: {'model':Obj()}
            portable.inference_outputs = lambda state,images: (_ for _ in ()).throw(RuntimeError('forward failed'))
            portable.release_inference = lambda state: released.append(1)
            registry[name] = portable
            return portable
        with self.assertRaises(RuntimeError):
            gate.reference_outputs(name='_connected_probe_gate_reference_y',directory=Path('/b'),bundle_sha='a'*64,loader=loader,
                pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=[('B1',[Path('/i/0')])],decode=self.fake.decode,
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,registry=registry)
        self.assertEqual((released,registry),([1],{}))
        self.assertTrue(all(i.closed for i in self.fake.images))

    def test_reference_registry_change_is_rejected(self):
        registry = {}
        def loader(name, path, pin, guards):
            portable = Obj()
            portable.load_inference = lambda d,s,device: {'model':Obj()}
            portable.inference_outputs = lambda state,images: self.fake.produce(None,images)
            portable.release_inference = lambda state: None
            registry[name] = portable
            registry[name] = Obj()
            return portable
        with self.assertRaises(ValueError):
            gate.reference_outputs(name='_connected_probe_gate_reference_z',directory=Path('/b'),bundle_sha='a'*64,loader=loader,
                pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=[('B1',[Path('/i/0')])],decode=self.fake.decode,
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,registry=registry)


class ReceiptValidation(unittest.TestCase):
    @classmethod
    def build(cls, active=None):
        if active is None:
            with world() as w:
                authority = w.authority()
        else: authority = active.authority()
        fake = Fake()
        parity = gate.parity_body(**body_for(fake))
        afact = {'path':'/a/authority.json','sha256':'a'*64}
        ties = {'ascending_ordinal_score_bits_exact':True,'gallery':'separate discarded duplicate e1; resident public gallery unchanged',
            'native':[Witnesses.native(1),Witnesses.native(32)]}
        record = {'schema':gate.RECEIPT,'status':'ENGINEERING_PARITY_DIAGNOSTIC','engineering_only':True,'authority':afact,
            'sources':authority['sources'],**{k:authority[k] for k in ('evaluator','evaluation_authority','endpoint','train_export',
                'bundle','gallery','images','native','wheel')},'wheel_evidence':{},'native_runtime':authority['native']['authority'],
            'native_scope':gate.NATIVE_SCOPE,'output':'/o/new','ties':ties,**parity,'installed_public_parity_pass':True,
            'full_uncached_exit_pass':True,**dict.fromkeys(('quality_read','quality_eligible','speed_eligible','release_eligible',
                'qualification_eligible','deployment_eligible','state_reuse_eligible','optimization_eligible','product_go',
                'native_launch_authorized_by_source_pass'),False),'resource_policy':dict(gate.POLICY),
            'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
            'invocation':{'argv':gate.cli(afact,'/o/new',authority['sources']['probe_driver']['path']),'optimize':0,
                'cuda_visible_devices':'0','cublas_workspace_config':':4096:8'}}
        return record,authority,afact

    def setUp(self):
        self.record,self.authority,self.afact = self.build()

    def check(self, record=None, authority=None):
        gate.validate_receipt(record or self.record,authority or self.authority,self.afact)

    def test_complete_engineering_receipt_passes(self):
        self.check()

    def test_each_field_mutant_is_rejected(self):
        flags = ('quality_read','quality_eligible','speed_eligible','release_eligible','qualification_eligible',
            'deployment_eligible','state_reuse_eligible','optimization_eligible','product_go','native_launch_authorized_by_source_pass')
        edits = {f'flag {k}':(lambda r,k=k: r.__setitem__(k,True)) for k in flags}
        edits.update({
            'status':lambda r: r.__setitem__('status','DISCARDED_DIAGNOSTIC'),
            'schema':lambda r: r.__setitem__('schema','connected-control-serving-diagnostic-v1'),
            'parity false':lambda r: r.__setitem__('installed_public_parity_pass',False),
            'exit false':lambda r: r.__setitem__('full_uncached_exit_pass',False),
            'native scope':lambda r: r.__setitem__('native_scope','current-rebuilt-binary'),
            'persisted false':lambda r: r.__setitem__('persisted_tail_exact',False),
            'owner order':lambda r: r.__setitem__('owner_order',['installed_A','reference','installed_B']),
            'policy reserve':lambda r: r['resource_policy'].__setitem__('exit_reserve_seconds',299),
            'policy float':lambda r: r['resource_policy'].__setitem__('body_seconds',300.0),
            'sources':lambda r: r['sources'].__setitem__('bridge',{'path':'/x/b.py','sha256':'1'*64}),
            'evaluator':lambda r: r['evaluator'].__setitem__('execution_sha256','9'*64),
            'native lib':lambda r: r['native']['library'].__setitem__('sha256','2'*64),
            'wheel':lambda r: r['wheel'].__setitem__('version','0'),
            'argv':lambda r: r['invocation']['argv'].append('--extra'),
            'optimize':lambda r: r['invocation'].__setitem__('optimize',1),
            'cuda env':lambda r: r['invocation'].__setitem__('cuda_visible_devices',''),
            'cublas':lambda r: r['invocation'].__setitem__('cublas_workspace_config',None),
            'batch order':lambda r: r['batches'].reverse(),
            'batch count':lambda r: r['batches'][2].__setitem__('count',31),
            'ordinals on B1':lambda r: r['batches'][0].__setitem__('ordinals',[1]),
            'tail ordinals missing':lambda r: r['batches'][3].__setitem__('ordinals',None),
            'batch wire':lambda r: r['batches'][1]['outputs'].__setitem__('wire_hex','00'+r['batches'][1]['outputs']['wire_hex'][2:]),
            'native id':lambda r: r['batches'][0]['native'][0].__setitem__('hex',struct.pack('<10q',*range(1,11)).hex()),
            'installed row dropped':lambda r: r['installed'].pop(),
            'installed row order':lambda r: r['installed'].reverse(),
            'installed hash':lambda r: r['installed'][3].__setitem__('outputs_sha256','0'*64),
            'installed native hash':lambda r: r['installed'][5].__setitem__('native_sha256','0'*64),
            'installed extra key':lambda r: r['installed'][0].__setitem__('seconds',1),
            'factory mutant':lambda r: r['factory_mutants'].pop('gallery_count'),
            'factory mutant status':lambda r: r['factory_mutants'].__setitem__('native_sha256','accepted'),
            'live mutant dropped':lambda r: r['live_mutants'].pop('cpu_rng'),
            'live mutant extra':lambda r: r['live_mutants'].__setitem__('foreign','rejected'),
            'lifecycle':lambda r: r['lifecycle'].__setitem__('post_close_rejected',False),
            'denial checks':lambda r: r['denial'].__setitem__('checks',2),
            'denial false':lambda r: r['denial'].__setitem__('historical_modules_absent',False),
            'owners one':lambda r: r['owners'].pop(),
            'owner negative':lambda r: r['owners'][0].__setitem__('admission_seconds',-1.0),
            'owner nan':lambda r: r['owners'][1].__setitem__('release_seconds',float('nan')),
            'body cap':lambda r: r.__setitem__('body_seconds',300.5),
            'charge over body':lambda r: r['owners'][0].__setitem__('admission_seconds',r['body_seconds']+1),
            'ties false':lambda r: r['ties'].__setitem__('ascending_ordinal_score_bits_exact',False),
            'ties bits':lambda r: r['ties']['native'][1][1].__setitem__('hex',struct.pack('<320f',*([2.]*320)).hex()),
            'terminal false':lambda r: r.__setitem__('normal_terminal_required',False)})
        for name,edit in edits.items():
            value = copy.deepcopy(self.record)
            edit(value)
            with self.subTest(name), self.assertRaises((ValueError,KeyError,TypeError)):
                self.check(value)

    def test_authority_binding_mutants(self):
        for name,edit in (('gallery',lambda a: a['gallery'].__setitem__('count',11)),
                ('tail length',lambda a: a['images']['tail']['files'].pop()),
                ('endpoint',lambda a: a['endpoint'].__setitem__('seed',179069))):
            value = copy.deepcopy(self.authority)
            edit(value)
            with self.subTest(name), self.assertRaises((ValueError,KeyError)):
                self.check(authority=value)


class AcceptUnitEarly(unittest.TestCase):
    """Parent normal-terminal analogue: strict receipt/authority admission before any evaluator or native object."""
    def setUp(self):
        self.cm = world()
        self.w = self.cm.__enter__()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        record,authority,_ = ReceiptValidation.build(self.w)
        self.authority, self.record = authority, record
        path = self.root/'authority.json'
        path.write_text(json.dumps(authority))
        self.afact = fact(path)
        record['authority'] = self.afact
        record['output'] = str(self.root/'out')
        record['invocation']['argv'] = gate.cli(self.afact,record['output'],authority['sources']['probe_driver']['path'])
        self.before = set(sys.modules)

    def tearDown(self):
        self.tmp.cleanup()
        self.cm.__exit__(None,None,None)

    def unit(self, record=None, receipt_path=None):
        out = self.root/'out'
        out.mkdir(exist_ok=True)
        (out/'receipt.json').write_text(json.dumps(record or self.record))
        unit = copy.deepcopy(self.authority['train_export'])
        unit['receipt'] = fact(out/'receipt.json')
        if receipt_path:
            Path(receipt_path).write_text(json.dumps(record or self.record))
            unit['receipt'] = fact(receipt_path)
        return unit

    def test_tampered_or_misplaced_receipt_is_rejected_before_anything_is_loaded(self):
        bad = copy.deepcopy(self.record)
        bad['release_eligible'] = True
        with self.assertRaisesRegex(ValueError,'engineering-only'): gate.accept_unit({},self.unit(bad),self.afact)
        with self.assertRaisesRegex(ValueError,'roles differ'):
            gate.accept_unit({},self.unit(receipt_path=str(self.root/'elsewhere.json')),self.afact)
        bad = copy.deepcopy(self.record)
        bad['native_runtime'] = {'path':'/x/native.json','sha256':'1'*64}
        with self.assertRaisesRegex(ValueError,'roles differ'): gate.accept_unit({},self.unit(bad),self.afact)
        with self.assertRaises(ValueError):
            gate.accept_unit({},self.unit(),{'path':self.afact['path'],'sha256':'0'*64})
        self.assertEqual(set(sys.modules)-self.before,set())

    def test_valid_receipt_reaches_the_native_reader_and_cleans_every_owned_source_on_failure(self):
        with self.assertRaises((KeyError,ValueError,FileNotFoundError,OSError)):
            gate.accept_unit({'training_context':{}},self.unit(),self.afact)
        self.assertEqual([n for n in set(sys.modules)-self.before if n.startswith('_connected_requests_')],[])


class DriverStructure(unittest.TestCase):
    """Guard-removal detection: every reused/owned boundary call must remain in the real source."""
    REQUIRED = {
        'run':{'check_shape','requests.Locks','requests.Source','requests.Source.load','check_installed','check_wheel','check_loaded',
            'derive_evaluator_loader','native_module.validate_runtime_compiler','native_module.CombinedAuthority','evaluator.closure',
            'evaluator.check_code','evaluator.authority','admit_probe','evaluator.merge_guards','runtime_authority.install',
            'evaluator.native_start','locks.check','source.check','observer.file_bytes','api.authenticate','evaluator.guard_helpers',
            'evaluator.resources','requests.check_resources','torch.equal','requests.native_ties','api.audit_origins',
            'evaluator.endpoint_scope','evaluator.authenticate_payloads','select_batches','bind_gallery','parity_body','api.evaluator_exit',
            'evaluator.exit_rehash','api.evidence','validate_receipt',"context['helper'].publish",'requests.raise_failures'},
        'parity_body':{'guard','owned_names','requests.owner','denial.check','reference','native','persisted_tail','mutant_factory',
            'mutants','lifecycle_check','installed_pass'},
        'reference_outputs':{'reads_only','loader','portable.load_inference','loaded','portable.release_inference','after','registry.pop',
            'close_all','requests.raise_failures','check_outputs'},
        'installed_pass':{'decode','rgb','capture','close_all','check_outputs','digest'},
        'captured_search':{'sys.getprofile','sys.setprofile','index.search_images','native_snapshot'},
        'lifecycle':{'index.close','index.search_images','gc.collect','owned_names','mappings_absent','ref'},
        'factory_mutants':{'make','index.close','owned_names'},
        'run_mutants':{'apply','restore','detect','requests.raise_failures'},
        'accept_unit':{'check_shape','validate_receipt',"context['terminal_reader']","context['helper'].zero_events",
            'requests.check_resources','requests.Source.load','native_source.module.CombinedAuthority','derive_evaluator_loader',
            'evaluator.closure','evaluator.check_code','admit_probe',"legacy['invocations'].add",'observer.file_bytes'},
        'admit_probe':{'evaluator.accept_unit'},
        'check_installed':{'observer.file_bytes','literal_record','json.loads','ast.parse'},
        'check_wheel':{'observer.canonical','observer.file_bytes','csv.reader'}}

    @staticmethod
    def calls(node):
        return {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}

    def functions(self, tree):
        return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}

    def test_required_boundary_calls_are_present(self):
        functions = self.functions(tree_of(DRIVER))
        for name,required in self.REQUIRED.items():
            self.assertEqual(required-self.calls(functions[name]),set(),name)

    def test_checker_detects_each_guard_removal(self):
        functions = self.functions(tree_of(DRIVER))

        class Remove(ast.NodeTransformer):
            def __init__(self, name): self.name = name

            def visit_Call(self, node):
                self.generic_visit(node)
                return ast.Constant(None) if ast.unparse(node.func) == self.name else node
        for name,required in self.REQUIRED.items():
            for call in sorted(required):
                mutated = Remove(call).visit(copy.deepcopy(functions[name]))
                self.assertIn(call,required-self.calls(mutated),(name,call))

    def test_native_install_guard_keys_are_all_merged_from_the_pinned_sources(self):
        """install() looks up guards for its own, requests and observer FILEs; run() merges every pinned source FILE."""
        native = next(n for n in tree_of(HERE/'connected_control_native_authority.py').body
            if isinstance(n,ast.ClassDef) and n.name == 'CombinedAuthority')
        install = function(ast.Module(body=native.body,type_ignores=[]),'install')
        lookups = {ast.unparse(n.slice) for n in ast.walk(install) if isinstance(n,ast.Subscript) and
            ast.unparse(n.value) == "evaluation_context['guards']"}
        self.assertEqual(lookups,{'__file__','request.__file__','self.observer.__file__'})
        run = function(tree_of(DRIVER),'run')
        frozen = next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'frozen')
        self.assertIn('*sources.values()',ast.unparse(frozen.value))
        self.assertTrue(set(gate.SOURCES) >= {'request_driver','observer','control_native'})
        merge = [n for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func) == 'evaluator.merge_guards']
        self.assertGreaterEqual(len(merge),2)

    def test_locks_are_checked_at_admission_every_guard_and_at_publication(self):
        run = self.functions(tree_of(DRIVER))['run']
        count = sum(isinstance(n,ast.Call) and ast.unparse(n.func) == 'locks.check' for n in ast.walk(run))
        self.assertGreaterEqual(count,3)

    def test_no_function_or_global_is_rebound_and_no_historical_execution(self):
        text = DRIVER.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute):
                        self.assertNotIn(ast.unparse(target.value),{'requests','observer','evaluator','native_module','gate','sys.modules'},ast.unparse(node))
            if isinstance(node,ast.Call) and ast.unparse(node.func) in {'setattr','exec','eval'}:
                self.assertIn(ast.unparse(node.func),{'setattr','exec'})
        execs = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func) == 'exec']
        self.assertEqual(len(execs),1)  # only the derived-loader exec
        self.assertNotIn('torch.cuda.synchronize',text)
        for name in ('importlib','runpy','subprocess','ctypes','socket'):
            self.assertNotIn('import '+name,text)

    def test_terminal_reader_blocks_equal_the_original_control_accept_unit(self):
        """The parent normal-terminal analogue keeps the original statements (names substituted only)."""
        control = self.functions(tree_of(HERE/'qualify_connected_control_serving.py'))['accept_unit']
        mine = self.functions(tree_of(DRIVER))['accept_unit']
        simple = lambda node: {ast.unparse(n) for n in ast.walk(node) if isinstance(n,(ast.Assign,ast.Expr,ast.Return,ast.For))}  # noqa: E731
        ours = simple(mine)
        theirs = {s.replace("authority['native_runtime']","authority['native']['authority']") for s in simple(control)}
        anchors = ('terminal_record = ','final = context[','for value in (resources','for p, h in record[','legacy[','historical_files = ',
            'projected = ','legacy = context','prior = legacy','combined = record','return record','require(all(record[')
        picked = [s for s in theirs if s.startswith(anchors) or 'combined.keys()' in s or "combined['historical_projection']" in s]
        self.assertGreaterEqual(len(picked),9)
        for statement in picked:
            if 'authority.keys() == KEYS' in statement or 'validate_receipt' in statement: continue
            if statement.startswith('legacy = context') or 'prior = legacy' in statement:
                self.assertIn(statement,ours)
        missing = [s for s in picked if s not in ours and any(k in s for k in
            ('terminal_record','final = context','input_guards','combined','projected','historical_files','invocations','return record'))]
        # Only the receipt-specific policy/roles lines may differ from the original.
        self.assertEqual([m for m in missing if 'resources = record' in m],[])
        for needle in ('terminal_record =','final = context[','legacy[\'invocations\'].add','return record','combined.keys()'):
            self.assertTrue(any(needle in s for s in ours),needle)
            self.assertTrue(any(needle in s for s in theirs),needle)

    def test_run_finally_and_native_install_statements_equal_the_original_control_driver(self):
        control = self.functions(tree_of(HERE/'qualify_connected_control_serving.py'))['run']
        mine = self.functions(tree_of(DRIVER))['run']
        simple = lambda node: {ast.unparse(n) for n in ast.walk(node) if isinstance(n,(ast.Assign,ast.Expr))}  # noqa: E731
        ours, theirs = simple(mine), simple(control)
        for statement in ("context['training_context']['fit_context']['unit_started'] = STARTED",
                'api = runtime_authority.install(evaluator_source, context)','before = evaluator.native_start(context)',
                "api.audit_origins(context['training_context']['legacy'])",'api.evaluator_exit(context, exit_guard)',
                'evaluator.exit_rehash(context, exit_guard)',"record['combined_native'] = api.evidence()",
                "record['full_uncached_exit_pass'] = True","record['resources'] = final_resources",
                "record['input_guards'] = dict(context['guards'])","record['whole_process_seconds'] = time.perf_counter() - STARTED",
                "context['helper'].publish(output / 'receipt.json', record)",'output.mkdir()',
                "evaluator.merge_guards(context['guards'], {f['path']: f['sha256'] for f in runtime_authority.provenance_facts()})"):
            self.assertIn(statement,theirs,statement)
            self.assertIn(statement,ours,statement)


class DenialHook(unittest.TestCase):
    def test_historical_sources_cannot_be_executed_or_imported_while_active_and_check_flags_live_modules(self):
        with tempfile.TemporaryDirectory() as raw:
            denial = gate.Denial(raw)
            path = str(Path(raw)/gate.TRAINER)
            try:
                compile('x = 1',path,'exec')  # inactive: allowed
                denial.active = True
                with self.assertRaises(ValueError): compile('x = 1',path,'exec')
                with self.assertRaises(ValueError): exec(compile('x = 1',str(Path(raw)/'other.py'),'exec'),{}) or sys.audit('exec',SimpleNamespace(co_filename=path))
                with self.assertRaises(ValueError): sys.audit('import','m',path,[],[],[])
                compile('x = 1',str(Path(raw)/'unrelated.py'),'exec')
                self.assertEqual(denial.check()['historical_modules_absent'],True)
                sys.modules['_leaked'] = SimpleNamespace(__file__=path)
                with self.assertRaises(ValueError): denial.check()
                del sys.modules['_leaked']
                sys.modules['train_siglip2_connected_probe'] = SimpleNamespace(__file__=None)
                with self.assertRaises(ValueError): denial.check()
            finally:
                denial.active = False
                sys.modules.pop('_leaked',None)
                sys.modules.pop('train_siglip2_connected_probe',None)
            compile('x = 1',path,'exec')

    def test_mappings_absent_requires_no_bundle_lines(self):
        self.assertTrue(gate.mappings_absent(Path('/b'),'7f r-xp /usr/lib/x.so\n'))
        self.assertFalse(gate.mappings_absent(Path('/b'),'7f r--s /b/endpoint.pt\n'))


class RunAdmission(unittest.TestCase):
    """Real run() through locks, sources, installed closure, wheel, native pins, derived loader and evaluator.authority."""
    def setUp(self):
        self.cm = world()
        self.w = self.cm.__enter__()
        self.files, self.fds = [], []

    def tearDown(self):
        for f in self.files: f.close()
        self.cm.__exit__(None,None,None)

    def locks(self):
        import fcntl
        rows = []
        for i in range(2):
            path = self.w.root/f'lock{len(self.files)}'
            path.write_text('')
            stream = path.open('rb')
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)  # never block a fixture on its own descriptor
            self.files.append(stream)
            rows.append({'path':str(path),'fd':stream.fileno()})
        return rows

    def authority(self, **edit):
        a = self.w.authority()
        evaluator = self.w.root/'evaluator'
        evaluator.mkdir(exist_ok=True)
        code = {}
        for name in sorted(gate.EVALUATOR_FILES):
            (evaluator/name).write_bytes((HERE/name).read_bytes())
            code[name] = file_sha(evaluator/name)
        (evaluator/'execution.json').write_text(json.dumps(code))
        exe = self.w.root/'tileiras'
        exe.write_text('#!/bin/sh\n')
        exe.chmod(0o755)
        library = a['native']['library']
        (self.w.root/'native.json').write_text(json.dumps({'schema':'connected-control-native-authority-v2','library':library,
            'build_receipt':{'path':'/x','sha256':gate.ARCHIVED_BUILD_SHA},'build_evidence':{},'source_manifest':{'path':'/y','sha256':'0'*64},
            'supplemental':[],'runtime_compiler':fact(exe)}))
        a['native']['authority'] = fact(self.w.root/'native.json')
        a['evaluator'] = {'root':str(evaluator),'execution_sha256':file_sha(evaluator/'execution.json'),'code':code}
        (self.w.root/'launch.json').write_text('{"junk": true}')
        a['evaluation_authority'] = fact(self.w.root/'launch.json')
        a['locks'] = self.locks()
        a.update(edit)
        self.exe = exe
        return a

    def invoke(self, a, argv=None, output=None):
        path = self.w.root/'authority.json'
        path.write_text(json.dumps(a))
        output = output or self.w.root/'out'
        authority_fact = fact(path)
        args = SimpleNamespace(authority=path,authority_sha256=authority_fact['sha256'],output=output)
        argv = argv or gate.cli(authority_fact,str(output),a['sources']['probe_driver']['path'])
        before = set(sys.modules)
        with patch.object(sys,'argv',argv), patch.object(sys,'dont_write_bytecode',True), \
                patch.dict(os.environ,{'CUTILE_TILEIRAS_PATH':str(self.exe)} if hasattr(self,'exe') else {}):
            try: gate.run(args)
            finally:
                leftover = sorted(n for n in set(sys.modules)-before)
                self.assertEqual([n for n in leftover if n.startswith(('_connected_requests_','_connected_eval_'))],[])
                self.assertFalse([n for n in sys.modules if n.split('.')[0] in {'torch','numpy','PIL','sfora'}])
                self.assertIsNone(sys.getprofile())
        self.fail('run unexpectedly completed')

    def test_malformed_authority_cli_and_output_are_rejected_before_any_load(self):
        a = self.authority()
        bad = copy.deepcopy(a)
        bad['resource_policy']['exit_reserve_seconds'] = 250
        with self.assertRaisesRegex(ValueError,'policy'): self.invoke(bad)
        with self.assertRaisesRegex(ValueError,'CLI'): self.invoke(a,argv=['x'])
        with self.assertRaisesRegex(ValueError,'role differs'):
            bad = copy.deepcopy(a)
            bad['sources']['observer'] = fact(HERE/'test_observe_connected_serving.py')
            self.invoke(bad)
        (self.w.root/'out').mkdir()
        with self.assertRaisesRegex(ValueError,'output'): self.invoke(a)

    def test_lifetime_locks_are_required(self):
        a = self.authority(locks=[])
        with self.assertRaisesRegex(ValueError,'lock'): self.invoke(a)

    def test_full_admission_reaches_the_real_evaluator_and_fails_closed_with_clean_registry(self):
        a = self.authority()
        with self.assertRaises(ValueError) as caught: self.invoke(a)
        self.assertNotIn('role differs',str(caught.exception))
        self.assertFalse((self.w.root/'out').exists())

    def test_other_native_binary_or_missing_ack_stops_without_reconstruction(self):
        a = self.authority()
        for name,edit in (('binary',lambda v: v['native']['library'].__setitem__('sha256','1'*64)),
                ('ack',lambda v: v['native'].__setitem__('archived_control_binary_ack',False))):
            value = copy.deepcopy(a)
            edit(value)
            with self.subTest(name), self.assertRaises(ValueError): self.invoke(value)

    def test_stale_installed_pin_and_editable_wheel_stop_before_native_authority(self):
        a = self.authority()
        with patch.object(gate,'RUNTIME_SHA','1'*64), self.assertRaisesRegex(ValueError,'reviewed installed probe pins'):
            self.invoke(a)
        a = self.authority()
        direct = self.w.site/'sfora-9.9.9.dist-info'/'direct_url.json'
        direct.write_text(json.dumps({'dir_info':{'editable':True}}))
        a['wheel'] = self.w.wheel(fact(direct))
        with self.assertRaisesRegex(ValueError,'editable'): self.invoke(a)

    def test_stale_authority_sha_is_rejected_before_any_load(self):
        a = self.authority()
        path = self.w.root/'authority.json'
        path.write_text(json.dumps(a))
        args = SimpleNamespace(authority=path,authority_sha256='0'*64,output=self.w.root/'out')
        with patch.object(sys,'dont_write_bytecode',True), self.assertRaisesRegex(ValueError,'SHA256 differs'):
            gate.run(args)


if __name__ == '__main__':
    limit = 1024**3
    soft,hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft > limit:
        resource.setrlimit(resource.RLIMIT_AS,(limit,hard if hard != resource.RLIM_INFINITY and hard < limit else limit))
    signal.alarm(120)
    unittest.main(verbosity=1)
