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
        digests = set(re.findall(r'[0-9a-f]{64}',installed_identity_inverse(DRIVER.read_bytes()).decode()))
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

    def reference(self, items, deadline):
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
            rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda state: None,after=lambda: None,deadline=deadline,
            registry=registry)
        if self.reference_leak: sys.modules['_connected_probe_gate_reference_leak'] = Obj()
        return rows

    def native(self, ref, deadline):
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
        mutants=lambda index,deadline: gate.run_mutants([(n,lambda: setattr(fake,'tamper_current',True),
            lambda: setattr(fake,'tamper_current',False),'guard' if n.endswith('_rng') else 'current')
            for n in sorted(gate.MUTANT_NAMES)],lambda mode: index._check_current() if mode == 'current' else
            gate.require(not fake.tamper_current,'whole-unit RNG changed'),deadline),
        lifecycle_check=lambda index,deadline: gate.lifecycle(index,fake.decode,[Path('/i/0')],Path('/b'),
            lambda: 'no mappings\n',deadline),
        denial=FakeDenial(),guard=lambda **kw: None,started=gate.time.perf_counter(),
        mutant_factory=lambda deadline: gate.factory_mutants(factory,deadline))
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
        refs, alive = [], []
        def produce(out, calls):
            marker = Obj()  # created during output production, like a live native tensor
            refs.append(weakref.ref(marker))
            return {**out,'native_reference':marker}
        self.fake.output_edit = produce
        def bad(value):
            tensor = value['native_reference']  # the traceback frame of this call would pin it
            alive.append(tensor is refs[-1]())
            raise TypeError('snapshot failed '+'x'*1000)
        kept = None
        try: self.search(snapshot=bad)
        except ValueError as error: kept = error  # deliberately retained, with its traceback
        self.assertIsNotNone(kept)
        self.assertEqual((len(refs),alive),(1,[True]))  # the marker was live at the failing snapshot
        gc.collect()
        self.assertIsNone(refs[0]())  # ...and is released while the outward exception is retained
        self.assertIsNotNone(kept.__traceback__)
        self.assertIsNone(kept.__cause__)
        self.assertIsNone(kept.__context__)
        self.assertLessEqual(len(str(kept)),400)
        self.assertIn('TypeError',str(kept))
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
                gate.run_mutants([mutant],lambda mode: (_ for _ in ()).throw(ValueError('rejected')),lambda: None)
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
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,deadline=lambda: None,
                registry=registry)
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
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,deadline=lambda: None,
                registry=registry)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)
        self.assertIsInstance(kept.__cause__,BaseExceptionGroup)
        self.assertEqual(sorted(type(e).__name__ for e in kept.__cause__.exceptions) or [],['OSError'])


    def test_release_failure_still_runs_after_and_registry_removal_and_pins_nothing(self):
        fake, registry, refs, calls = Fake(), {}, [], []
        def loader(name, path, pin, guards):
            portable = Obj()
            def load(directory, sha, device):
                model = Obj()
                refs.append(weakref.ref(model))
                return {'model':model}
            def release(state):
                held = state['model']  # the failing release frame would pin the model
                raise OSError('release failed')
            portable.load_inference = load
            portable.inference_outputs = lambda state,images: (_ for _ in ()).throw(RuntimeError('forward failed'))
            portable.release_inference = release
            registry[name] = portable
            return portable
        kept = None
        try:
            gate.reference_outputs(name='_connected_probe_gate_reference_r',directory=Path('/b'),bundle_sha='a'*64,loader=loader,
                pin='d'*64,guards={},reads_only=lambda: io.StringIO(),batches=[('B1',[Path('/i/0')])],decode=fake.decode,
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: calls.append('after'),
                deadline=lambda: None,registry=registry)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)
        self.assertEqual((calls,registry),(['after'],{}))  # attempted independently of the failed release
        self.assertEqual([type(e) for e in kept.__cause__.exceptions],[OSError])
        gc.collect()
        self.assertIsNone(refs[0]())  # nothing pinned while the aggregated error is retained
        self.assertTrue(all(i.closed for i in fake.images))


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
        self.assertEqual(result['factory_mutants'],{**dict.fromkeys(gate.FACTORY_NAMES,
            {'type':'ValueError','message':'pinned input differs'}),
            'mlp_factory_on_probe_bundle':{'type':'ValueError','message':'unsupported connected bundle schema/closure'}})
        self.assertEqual(result['live_mutants'],{n:{'type':'ValueError','message':'whole-unit RNG changed' if n.endswith('_rng')
            else 'installed state changed'} for n in gate.MUTANT_NAMES})
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
            gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '',lambda: None)
        original()

    def lifecycle(self, fake=None, maps=lambda: ''):
        fake = fake or self.fake
        index = fake.factory()()
        try: return gate.lifecycle(index,fake.decode,[Path('/i/0')],Path('/b'),maps,lambda: None)
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
        def polluted(index, deadline):
            result = gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '',deadline)
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
        with self.assertRaises(ValueError): gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),lambda: '',lambda: None)
        index.close()

    def test_bundle_mapping_survival_is_rejected(self):
        self.rejects(lifecycle_check=lambda index,deadline: gate.lifecycle(index,self.fake.decode,[Path('/i/0')],Path('/b'),
            lambda: '7f00 r--s /b/endpoint.pt\n',deadline))

    def test_accepted_live_or_factory_mutant_is_rejected(self):
        accepted = lambda index,deadline: gate.run_mutants([('accepted',lambda: None,lambda: None,'current')],lambda mode: None,deadline)  # noqa: E731
        self.rejects(mutants=accepted)
        self.fake = Fake()
        factory = self.fake.factory()
        index_type = self.fake.make()
        def lax(*, method='from_probe_bundle', **overrides): return index_type()
        lax.base = factory.base
        with self.assertRaises(ValueError): gate.factory_mutants(lax,lambda: None)

    def test_factory_mutant_that_leaks_the_registry_is_rejected(self):
        fake = self.fake
        def leaky(*, method='from_probe_bundle', **overrides):
            sys.modules['_sfora_connected_compact_leak'] = Obj()
            raise ValueError('rejected but leaked')
        leaky.base = fake.factory().base
        with self.assertRaises(ValueError): gate.factory_mutants(leaky,lambda: None)
        del sys.modules['_sfora_connected_compact_leak']

    def test_restore_failure_is_reported_with_the_rejection_preserved(self):
        order = []
        def restore_bad(): raise RuntimeError('restore failed')
        reject = lambda mode: (_ for _ in ()).throw(ValueError('rejected'))  # noqa: E731
        with self.assertRaisesRegex(RuntimeError,'restore failed'):
            gate.run_mutants([('a',lambda: order.append('a'),restore_bad,'x')],reject,lambda: None)
        gate.run_mutants([('c',lambda: order.append('c'),lambda: order.append('restored-c'),'x')],reject,lambda: None)
        self.assertEqual(order,['a','c','restored-c'])

    def test_partial_apply_failure_still_restores_and_preserves_the_original_error(self):
        state = {'value':1}
        def apply():
            state['value'] = 99  # mutates, then fails
            raise RuntimeError('apply failed after mutating')
        def restore(): state['value'] = 1
        with self.assertRaisesRegex(RuntimeError,'apply failed after mutating'):
            gate.run_mutants([('partial',apply,restore,'x')],lambda mode: None,lambda: None)
        self.assertEqual(state['value'],1)
        def restore_bad():
            state['value'] = 1
            raise OSError('restore failed too')
        kept = None
        try: gate.run_mutants([('partial',apply,restore_bad,'x')],lambda mode: None,lambda: None)
        except RuntimeError as error: kept = error
        self.assertIsNotNone(kept)  # the original apply error stays primary
        self.assertTrue(any('restore failed too' in n for n in kept.__notes__))
        self.assertIsInstance(kept.__cause__,BaseExceptionGroup)
        self.assertEqual(state['value'],1)

    def test_apply_that_never_mutated_is_restored_harmlessly_and_cancellation_wins(self):
        calls = []
        def apply(): raise RuntimeError('apply failed')
        with self.assertRaises(RuntimeError):
            gate.run_mutants([('a',apply,lambda: calls.append('restore'),'x')],lambda mode: None,lambda: None)
        self.assertEqual(calls,['restore'])
        def interrupt(): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            gate.run_mutants([('a',interrupt,lambda: (_ for _ in ()).throw(RuntimeError('r')),'x')],lambda mode: None,lambda: None)

    def test_unrejected_mutant_is_still_restored_before_the_failure(self):
        state = {'value':1}
        with self.assertRaisesRegex(ValueError,'live mutant accepted'):
            gate.run_mutants([('m',lambda: state.update(value=2),lambda: state.update(value=1),'x')],lambda mode: None,lambda: None)
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
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,deadline=lambda: None,
                registry=registry)
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
                rgb=gate.rgb_digest,snapshot=copy.deepcopy,loaded=lambda s: None,after=lambda: None,deadline=lambda: None,
                registry=registry)


    def test_fake_clock_crossing_during_an_expensive_call_prevents_the_next_call(self):
        # produce calls: reference 1..8, installed A 9..16, A_restored 17..18, reload B 19..22
        for at in (1,2,8,9,12,16,17,19):
            self.fake, clock = Fake(), [0.0]
            produce = self.fake.produce
            def timed(index, images, produce=produce, fake=self.fake, clock=clock, at=at):
                out = produce(index,images)
                if fake.calls == at: clock[0] = 301.0  # this call crossed body300
                return out
            self.fake.produce = timed
            with self.subTest(at), patch.object(gate.time,'perf_counter',lambda clock=clock: clock[0]):
                kept = self.rejects(started=0.0)  # cleanup (release, registry, owner close, images) still ran
                self.assertIn('body300',str(kept))
                self.assertEqual(self.fake.calls,at)  # the NEXT expensive call never ran

    def test_mutant_deadline_is_between_mutants_and_never_a_rejection(self):
        expired, order = [False], []
        deadline = lambda: gate.require(not expired[0],'diagnostic body300 cap exceeded')  # noqa: E731
        def crossing(mode):
            expired[0] = True  # the clock crosses inside the expensive detection
            raise ValueError('genuine rejection')
        mutants = [(n,lambda n=n: order.append('apply-'+n),lambda n=n: order.append('restore-'+n),'outputs') for n in 'ab']
        with self.assertRaisesRegex(ValueError,'body300'): gate.run_mutants(mutants,crossing,deadline)
        self.assertEqual(order,['apply-a','restore-a'])  # restored even though expired; b never applied
        order.clear()
        with self.assertRaisesRegex(ValueError,'body300'): gate.run_mutants(mutants,lambda mode: None,deadline)
        self.assertEqual(order,[])
        expired[0] = False
        def accepting(mode): expired[0] = True
        with self.assertRaisesRegex(ValueError,'live mutant accepted'): gate.run_mutants(mutants[:1],accepting,deadline)

    def test_only_an_exact_valueerror_is_a_live_rejection_with_bounded_evidence(self):
        class Sub(ValueError): pass
        for error in (TypeError('bad call'),MemoryError(),RuntimeError('CUDA error: out of memory'),Sub('subclass')):
            restored = []
            with self.subTest(type(error).__name__), self.assertRaises(type(error)):
                gate.run_mutants([('m',lambda: None,lambda: restored.append(1),'outputs')],
                    lambda mode, error=error: (_ for _ in ()).throw(error),lambda: None)
            self.assertEqual(restored,[1])
        result = gate.run_mutants([('m',lambda: None,lambda: None,'outputs')],
            lambda mode: (_ for _ in ()).throw(ValueError('x'*1000)),lambda: None)
        self.assertEqual(result,{'m':{'type':'ValueError','message':'x'*256}})

    def test_factory_negatives_accept_only_an_exact_valueerror_and_the_clock_stops_the_next(self):
        factory = self.fake.factory()
        result = gate.factory_mutants(factory,lambda: None)
        self.assertEqual(result['mlp_factory_on_probe_bundle'],{'type':'ValueError','message':'unsupported connected bundle schema/closure'})
        self.assertEqual(result['gallery_count'],{'type':'ValueError','message':'pinned input differs'})
        for error in (TypeError('unexpected keyword'),MemoryError()):
            def crashing(*, method='from_probe_bundle', error=error, **overrides): raise error
            crashing.base = factory.base
            with self.subTest(type(error).__name__), self.assertRaises(type(error)): gate.factory_mutants(crashing,lambda: None)
        calls, expired = [], [False]
        def slow(*, method='from_probe_bundle', **overrides):
            calls.append(method)
            expired[0] = True
            raise ValueError('rejected')
        slow.base = factory.base
        with self.assertRaisesRegex(ValueError,'body300'):
            gate.factory_mutants(slow,lambda: gate.require(not expired[0],'diagnostic body300 cap exceeded'))
        self.assertEqual(calls,['from_probe_bundle'])

    def test_native_search_crossing_prevents_the_next_search_and_still_closes(self):
        with tempfile.TemporaryDirectory() as raw:
            library = Path(raw)/'lib.so'
            library.write_bytes(b'x')
            calls, closed, expired = [], [], [False]

            class Gallery:
                @staticmethod
                def open_packed(path, packed): return Gallery()

                def search_packed(self, packed, k):
                    calls.append(k)
                    expired[0] = True
                    return (memoryview(struct.pack('<10q',*range(10))).cast('q',shape=[1,10]),
                        memoryview(struct.pack('<10f',*([1.]*10))).cast('f',shape=[1,10]))

                def close(self): closed.append(1)
            packed = SimpleNamespace(from_bytes=lambda wire,count,dimensions: wire)
            ref = {k:{'outputs':Witnesses.outputs(1,i)} for i,k in enumerate(('B1','B2'),start=1)}
            with self.assertRaisesRegex(ValueError,'body300'):
                gate.reference_native(packed,Gallery,fact(library),b'',10,ref,observer,
                    lambda: gate.require(not expired[0],'diagnostic body300 cap exceeded'))
        self.assertEqual((calls,closed),([10],[1]))

    def test_live_mutant_failure_releases_gate_references_before_owner_teardown(self):
        at_close = []
        index_type = self.fake.make()
        def factory():
            index = index_type()
            factory.ref = weakref.ref(index._endpoint['model'])
            close = index.close
            def checked():
                close()  # drops the endpoint, like the genuine release
                gc.collect()
                at_close.append(factory.ref() is None)  # the genuine release requires this lifetime
            index.close = checked
            return index
        def build(index, torch):
            model, backup = index._endpoint['model'], Obj()  # closure-held parameter and tensor backup
            def apply():
                held = (model,backup)  # partial apply mutates, then fails
                raise RuntimeError('apply failed after mutating')
            def restore():
                held = (model,backup)
                raise OSError('restore failed')
            return [('partial',apply,restore,'outputs')]
        kept = None
        with patch.object(gate,'native_live_mutants',build):
            try:
                with requests.owner(factory,lambda: None,{}) as index:
                    gate.live_mutants(index,None,[],lambda: None,lambda: None)
            except RuntimeError as error: kept = error  # retained, as a caller would
        self.assertIsNotNone(kept)
        self.assertEqual(str(kept),'apply failed after mutating')
        self.assertEqual([type(e) for e in kept.__cause__.exceptions],[OSError])
        self.assertEqual(at_close,[True])  # released BEFORE the owner's teardown, not merely afterwards
        gc.collect()
        self.assertIsNone(factory.ref())
        self.assertIsNotNone(kept.__traceback__)

    def test_real_denial_audit_hook_through_the_body(self):
        with tempfile.TemporaryDirectory() as raw:
            historical, bare = str(Path(raw)/gate.TRAINER), gate.TRAINER.removesuffix('.py')
            denial = gate.Denial(raw)
            result = self.run_body(denial=denial)  # green with the genuine audit hook
            self.assertEqual(result['denial'],{'checks':3,'historical_modules_absent':True})
            self.assertFalse(denial.active)
            compile('x = 1',historical,'exec')  # inactive after the body: allowed
            self.fake = Fake()
            produce = self.fake.produce
            def compiling(index, images):
                if index is not None: compile('x = 1',historical,'exec')  # installed inference only
                return produce(index,images)
            self.fake.produce = compiling
            kept = self.rejects(denial=denial)
            self.assertIn('execute a historical source',str(kept))
            self.assertFalse(denial.active)
            self.fake = Fake()
            factory = self.fake.factory()
            def leaking(**overrides):
                sys.modules[bare] = SimpleNamespace(__file__=None)  # bare historical module name
                return factory(**overrides)
            leaking.base = factory.base
            try: kept = self.rejects(denial=denial,factory=leaking)
            finally: sys.modules.pop(bare,None)
            self.assertIn('historical source module is live: '+bare,str(kept))
            self.assertFalse(denial.active)
            compile('x = 1',historical,'exec')


class LiveMutantIdentity(unittest.TestCase):
    """Stdlib stand-ins run the real native_live_mutants: every restore returns the exact original object."""
    def world(self, tensor=FakeTensor):
        class Tensor(tensor):
            is_leaf, requires_grad = True, False

            def numel(self): return len(self.store)

            def reshape(self, shape): return FakeTensor(shape,self.store)

            def requires_grad_(self, flag):
                self.requires_grad = flag
                return self

        class State(int):
            def clone(self): return self
        params = {'head.probe':Tensor((2,)),'encoder.w':Tensor((2,),[5.,6.])}
        head_weight, readout_c, mu = Tensor((2,)), Tensor((3,)), Tensor((1,))
        config = SimpleNamespace(_attn_implementation='sdpa')
        registration = {'position_ids'}
        model = SimpleNamespace(named_parameters=lambda: iter(params.items()),config=config,
            embeddings=SimpleNamespace(_non_persistent_buffers_set=registration))
        processor = SimpleNamespace(image_mean=(0.5,0.5,0.5))
        module = type(sys)('_gate_identity_runtime')
        exec("PROBE = ('head.probe',)\nARMS = ('control', 'candidate')\n"
            "def owned_copy(value, device='cpu'): return value\ndef inference_outputs(endpoint, images): return None\n",vars(module))
        index = SimpleNamespace(_module=module,_apis={'inference_outputs':(module.inference_outputs,module.inference_outputs.__code__)},
            _endpoint={'model':model,'head_object':SimpleNamespace(parameters=lambda: iter([head_weight])),'C':readout_c,
                'mu_train':mu,'processor_object':processor})
        rng, flags = {'cpu':State(7),'cuda':[State(9)]}, {'precision':'highest'}
        def rand(*shape, device='cpu'):
            if device == 'cpu': rng['cpu'] = State(rng['cpu']+1)
            else: rng['cuda'] = [State(rng['cuda'][0]+1)]
        torch = SimpleNamespace(equal=FAKE_TORCH.equal,rand=rand,
            get_float32_matmul_precision=lambda: flags['precision'],set_float32_matmul_precision=lambda v: flags.update(precision=v),
            random=SimpleNamespace(get_rng_state=lambda: rng['cpu'],set_rng_state=lambda v: rng.update(cpu=v)),
            cuda=SimpleNamespace(get_rng_state_all=lambda: list(rng['cuda']),set_rng_state_all=lambda v: rng.update(cuda=list(v))))
        tensors = [*params.values(),head_weight,readout_c,mu]
        def snapshot():
            return (processor.image_mean,config._attn_implementation,frozenset(registration),module.inference_outputs.__code__,
                module.owned_copy.__defaults__,module.ARMS,frozenset(vars(module)),flags['precision'],rng['cpu'],
                tuple(rng['cuda']),[list(t.store) for t in tensors],head_weight.requires_grad)
        return index,torch,snapshot

    def test_every_live_mutant_changes_state_and_restores_the_original_objects(self):
        index,torch,snapshot = self.world()
        module, processor = index._module, index._endpoint['processor_object']
        before, mean = snapshot(), processor.image_mean
        mutants = gate.native_live_mutants(index,torch)
        self.assertEqual({m[0] for m in mutants},gate.MUTANT_NAMES)
        for name,apply,restore,mode in mutants:
            with self.subTest(name):
                apply()
                self.assertNotEqual(snapshot(),before)
                restore()
                self.assertEqual(snapshot(),before)
                self.assertIs(processor.image_mean,mean)  # the original object itself, not an equal copy
                self.assertIs(module.inference_outputs.__code__,before[3])
                self.assertIs(module.owned_copy.__defaults__,before[4])
                self.assertIs(module.ARMS,before[5])

    def test_failed_backup_clone_leaves_no_earlier_backup_pinned_by_the_retained_error(self):
        clones = []
        class Failing(FakeTensor):
            def clone(self):
                if clones: raise MemoryError('CUDA out of memory')  # second leaf backup fails
                copy_ = FakeTensor(self.shape,list(self.store))
                clones.append(weakref.ref(copy_))
                return copy_
        index,torch,_ = self.world(Failing)
        kept = None
        try: gate.live_mutants(index,torch,[],lambda: None,lambda: None)
        except MemoryError as error: kept = error  # retained, as the owner would while closing
        self.assertIsNotNone(kept)
        self.assertEqual(len(clones),1)
        gc.collect()
        self.assertIsNone(clones[0]())


class ReceiptValidation(unittest.TestCase):
    @classmethod
    def build(cls, active=None):
        if active is None:
            with world() as w: return cls.build(w)
        authority = active.authority()
        fake = Fake()
        parity = gate.parity_body(**body_for(fake))
        afact = {'path':'/a/authority.json','sha256':'a'*64}
        ties = {'ascending_ordinal_score_bits_exact':True,'gallery':'separate discarded duplicate e1; resident public gallery unchanged',
            'native':[Witnesses.native(1),Witnesses.native(32)]}
        record = {'schema':gate.RECEIPT,'status':'ENGINEERING_PARITY_DIAGNOSTIC','engineering_only':True,'authority':afact,
            'sources':authority['sources'],**{k:authority[k] for k in ('evaluator','evaluation_authority','endpoint','train_export',
                'bundle','gallery','images','native','wheel')},
            'wheel_evidence':gate.check_wheel(authority['wheel'],authority['sources'],observer,ROOT),
            'native_runtime':authority['native']['authority'],
            'native_scope':gate.NATIVE_SCOPE,'output':'/o/new','ties':ties,**parity,'installed_public_parity_pass':True,
            'full_uncached_exit_pass':True,**dict.fromkeys(('quality_read','quality_eligible','speed_eligible','release_eligible',
                'qualification_eligible','deployment_eligible','state_reuse_eligible','optimization_eligible','product_go',
                'native_launch_authorized_by_source_pass'),False),'resource_policy':dict(gate.POLICY),
            'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
            'combined_native':{},'resources':{},'input_guards':{},'whole_process_seconds':1.0,
            'invocation':{'argv':gate.cli(afact,'/o/new',authority['sources']['probe_driver']['path']),'python':sys.executable,
                'python_sha256':'e'*64,'python_version':sys.version,'pid':1,'invocation_id':'d'*32,'optimize':0,
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
            'terminal false':lambda r: r.__setitem__('normal_terminal_required',False),
            'receipt extra key':lambda r: r.__setitem__('foreign',1),
            'receipt missing key':lambda r: r.pop('whole_process_seconds'),
            'invocation extra key':lambda r: r['invocation'].__setitem__('foreign',1),
            'invocation missing id':lambda r: r['invocation'].pop('invocation_id'),
            'mutant bare string':lambda r: r['factory_mutants'].__setitem__('gallery_count','rejected'),
            'mutant other type':lambda r: r['live_mutants']['frozen_leaf'].__setitem__('type','TypeError'),
            'mutant long message':lambda r: r['live_mutants']['readout_C'].__setitem__('message','x'*257),
            'mutant empty message':lambda r: r['factory_mutants']['native_sha256'].__setitem__('message',''),
            'mutant extra evidence':lambda r: r['live_mutants']['readout_C'].__setitem__('traceback','frames'),
            'rng other message':lambda r: r['live_mutants']['cuda_rng'].__setitem__('message','installed state changed'),
            'wheel evidence site':lambda r: r['wheel_evidence'].__setitem__('site_root','/elsewhere'),
            'wheel evidence direct url':lambda r: r['wheel_evidence'].__setitem__('direct_url',{'path':'/x','sha256':'1'*64}),
            'wheel evidence rows':lambda r: r['wheel_evidence']['rows'].pop('bridge'),
            'wheel evidence extra':lambda r: r['wheel_evidence'].__setitem__('editable',False)})
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
        unit['invocation_id'] = (record or self.record)['invocation']['invocation_id']
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

    def rebind(self, direct_url):
        """Admission-time wheel evidence for a declared direct_url FILE (or canonical absence)."""
        self.authority['wheel'] = self.record['wheel'] = self.w.wheel(direct_url)
        self.record['wheel_evidence'] = gate.check_wheel(self.authority['wheel'],self.authority['sources'],observer,ROOT)
        path = self.root/'authority.json'
        path.write_text(json.dumps(self.authority))
        self.afact = self.record['authority'] = fact(path)
        self.record['invocation']['argv'] = gate.cli(self.afact,self.record['output'],self.authority['sources']['probe_driver']['path'])

    def reaches_native(self):
        with self.assertRaises(KeyError) as caught: gate.accept_unit({},self.unit(),self.afact)
        self.assertEqual(caught.exception.args,('training_context',))  # past the fresh wheel check
        self.assertEqual([n for n in set(sys.modules)-self.before if n.startswith('_connected_requests_')],[])

    def test_receipt_invocation_must_be_the_diagnostic_unit(self):
        unit = self.unit()
        unit['invocation_id'] = 'f'*32
        with self.assertRaisesRegex(ValueError,'invocation roles differ'): gate.accept_unit({},unit,self.afact)
        self.assertEqual(set(sys.modules)-self.before,set())

    def test_wheel_changes_after_admission_are_rejected_fresh_at_parent_acceptance(self):
        self.reaches_native()
        dist = self.w.site/'sfora-9.9.9.dist-info'
        for name,change,undo in (
                ('direct_url created after declared absence',lambda: (dist/'direct_url.json').write_text('{}'),
                    lambda: (dist/'direct_url.json').unlink()),
                ('editable .pth',lambda: (self.w.site/'sfora.pth').write_text(str(ROOT)+'\n'),
                    lambda: (self.w.site/'sfora.pth').unlink()),
                ('editable finder',lambda: (self.w.site/'__editable___sfora_finder.py').write_text(''),
                    lambda: (self.w.site/'__editable___sfora_finder.py').unlink())):
            change()
            try:
                with self.subTest(name), self.assertRaises(ValueError) as caught: gate.accept_unit({},self.unit(),self.afact)
                self.assertNotIn('training_context',str(caught.exception))
            finally: undo()
        self.reaches_native()
        direct = dist/'direct_url.json'
        direct.write_text(json.dumps({'url':'file:///x.whl','archive_info':{}}))
        self.rebind(fact(direct))
        self.reaches_native()
        for name,change in (('declared direct_url edited',lambda: direct.write_text(json.dumps({'dir_info':{'editable':True}}))),
                ('declared direct_url removed',lambda: direct.unlink())):
            change()
            with self.subTest(name), self.assertRaises((ValueError,OSError)): gate.accept_unit({},self.unit(),self.afact)
        self.assertEqual([n for n in set(sys.modules)-self.before if n.startswith('_connected_requests_')],[])

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
            'mutants','lifecycle_check','installed_pass','deadline'},
        'reference_outputs':{'reads_only','loader','portable.load_inference','loaded','portable.release_inference','after','registry.pop',
            'close_all','requests.raise_failures','check_outputs','deadline','clear_frames'},
        'reference_native':{'deadline','gallery.search_packed','gallery.close','requests.raise_failures'},
        'installed_pass':{'decode','rgb','capture','close_all','check_outputs','digest','deadline'},
        'captured_search':{'sys.getprofile','sys.setprofile','index.search_images','native_snapshot'},
        'lifecycle':{'index.close','index.search_images','gc.collect','owned_names','mappings_absent','ref','deadline'},
        'factory_mutants':{'make','index.close','owned_names','deadline','rejection'},
        'run_mutants':{'apply','restore','detect','requests.raise_failures','deadline','rejection'},
        'live_mutants':{'native_live_mutants','run_mutants','clear_frames','index._check_current'},
        'accept_unit':{'check_shape','validate_receipt','check_wheel',"context['terminal_reader']","context['helper'].zero_events",
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

    def test_wheel_is_rechecked_at_exit_and_declared_direct_url_is_a_guarded_input(self):
        functions = self.functions(tree_of(DRIVER))
        run = functions['run']
        self.assertEqual(sum(isinstance(n,ast.Call) and ast.unparse(n.func) == 'check_wheel' for n in ast.walk(run)),2)
        for name in ('run','accept_unit'):
            self.assertIn("authority['wheel']['direct_url']",ast.unparse(functions[name]))

    def test_no_function_or_global_is_rebound_and_no_historical_execution(self):
        text = installed_identity_inverse(DRIVER.read_bytes()).decode()
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


class WheelBoundaryCallsites(unittest.TestCase):
    """The checkout argument of every production check_wheel call is the frozen source directory.

    Frozen source folder and separately installed wheel are siblings under a shared, non-Git parent (/runs
    stand-in). Arguments are extracted from the real call sites and executed, then given to the real check_wheel.
    """
    BASE_DRIVER_SHA = 'bf8d63d4ba0a9e852f0da33fe6cf14b844e89c3d79a8c3824a9ec3ed77b71c94'  # 36dec573 bytes

    @staticmethod
    def callsites():
        tree = tree_of(DRIVER)
        found = []
        for name in ('run','accept_unit'):
            for node in ast.walk(function(tree,name)):
                if isinstance(node,ast.Call) and ast.unparse(node.func) == 'check_wheel':
                    found.append((name,node.args[3]))
        return found

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name).resolve()/'runs'
        self.source = self.runs/'source-v1'
        self.source.mkdir(parents=True)
        self.driver = self.source/'qualify_connected_probe_serving.py'
        self.driver.write_bytes(DRIVER.read_bytes())
        self.w = World(self.runs/'wheel-v1')

    def checkout(self, node, driver=None):
        driver = driver or self.driver
        namespace = {'Path':Path,'here':Path(driver).absolute(),'sources':{'probe_driver':{'path':str(driver)}}}
        return eval(compile(ast.Expression(node),'<callsite>','eval'),namespace)

    def check(self, node, driver=None):
        return gate.check_wheel(self.w.wheel(),self.w.sources(),observer,self.checkout(node,driver))

    def test_there_are_exactly_three_callsites_and_each_names_the_frozen_source_directory(self):
        sites = self.callsites()
        self.assertEqual([n for n,_ in sites],['run','run','accept_unit'])
        for name,node in sites:
            with self.subTest(name): self.assertEqual(self.checkout(node),self.source)

    def test_sibling_wheel_outside_the_source_directory_is_admitted_at_every_callsite(self):
        self.assertFalse((self.runs/'.git').exists())
        for name,node in self.callsites():
            with self.subTest(name): self.assertEqual(self.check(node)['site_root'],str(self.w.site))

    def test_genuine_git_ancestor_nested_equal_and_editable_layouts_still_fail_at_every_callsite(self):
        site = self.w.site
        for name,node in self.callsites():
            with self.subTest(name):
                for label,driver in (('source equals wheel root',site/'d.py'),('source inside wheel root',site/'sfora'/'d.py'),
                        ('source contains wheel root',self.runs/'d.py'),('source contains wheel root deeper',self.w.root/'d.py')):
                    with self.subTest(label), self.assertRaises(ValueError): self.check(node,driver)
                (self.runs/'.git').mkdir()
                try:
                    with self.assertRaises(ValueError): self.check(node)
                finally: (self.runs/'.git').rmdir()
                (self.w.site/'x.pth').write_text(str(self.source)+'\n')
                try:
                    with self.assertRaises(ValueError): self.check(node)
                finally: (self.w.site/'x.pth').unlink()
                (self.w.site/'__editable__.sfora-9.9.9.pth').write_text('x')
                try:
                    with self.assertRaises(ValueError): self.check(node)
                finally: (self.w.site/'__editable__.sfora-9.9.9.pth').unlink()
                self.check(node)

    def test_the_three_argument_edits_are_the_whole_production_change(self):
        data = diagnostic_inverse(DRIVER.read_bytes())
        tree = ast.parse(data)
        found = []
        for name in ('run','accept_unit'):
            for node in ast.walk(function(tree,name)):
                if isinstance(node,ast.Call) and ast.unparse(node.func) == 'check_wheel':
                    found.append((name,node.args[3]))
        lines = data.splitlines(keepends=True)
        starts = [sum(map(len,lines[:i])) for i in range(len(lines)+1)]
        ends = []
        for _,node in found:
            self.assertIsInstance(node,ast.Attribute)
            self.assertEqual(node.attr,'parent')
            self.assertNotIsInstance(node.value,ast.Attribute)
            ends.append(starts[node.end_lineno-1]+node.end_col_offset)
        self.assertEqual(len(ends),3)
        for end in sorted(ends,reverse=True): data = data[:end]+b'.parent'+data[end:]
        self.assertEqual(hashlib.sha256(data).hexdigest(),self.BASE_DRIVER_SHA)


# BEGIN loader observation tests
OBSERVATION_BLOCK_SHA = '9706fee5c071322f577a8a482f83ff6f374166888c45bcff3d63221cb23c40fb'
OBSERVATION_ASSIGN = b"            state = portable.load_inference(Path(directory),bundle_sha,'cuda')\n"
OBSERVATION_SCOPE = b"""            observation = _LoaderObservation(portable,name,directory,pin,guards,registry)
            try:
                observation.start()
                state = portable.load_inference(Path(directory),bundle_sha,'cuda')
            finally:
                observation.close()
                observation = None
"""
OBSERVATION_CATCH = b"        _observation_emit({'event':'reference_catch_before_clear_frames','errors':_observation_errors(error)})\n"


def diagnostic_inverse(raw):
    raw = installed_identity_inverse(raw)
    begin,end = b'# BEGIN bounded loader observation\n',b'# END bounded loader observation\n\n\n'
    if raw.count(begin) != 1 or raw.count(end) != 1: raise ValueError('diagnostic block cardinality')
    start,stop = raw.index(begin),raw.index(end)+len(end)
    block = raw[start:stop]
    if SHA(block) != OBSERVATION_BLOCK_SHA: raise ValueError('diagnostic definitions differ')
    nodes = ast.parse(block).body
    names = [n.name if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else ast.unparse(n) for n in nodes]
    if names != ['import types as _observation_types','import functools as _observation_functools',
            '_observation_json','_observation_emit','_observation_text','_observation_type','_observation_message',
            '_observation_errors','_observation_maps','_observation_dict','_observation_member','_observation_storage','_observation_owners',
            '_LoaderObservation']:
        raise ValueError('diagnostic node inventory differs')
    raw = raw[:start]+raw[stop:]
    if raw.count(OBSERVATION_SCOPE) != 1 or raw.count(OBSERVATION_CATCH) != 1:
        raise ValueError('diagnostic seam cardinality')
    raw = raw.replace(OBSERVATION_SCOPE,OBSERVATION_ASSIGN).replace(OBSERVATION_CATCH,b'')
    if SHA(raw) != '149808c99e9216afdf7aacdc7e4610ea810598ea2dee8d1b963243ac394253dd':
        raise ValueError('complete baseline bytes differ')
    if SHA(ast.dump(ast.parse(raw),include_attributes=False).encode()) != 'e9e7ebf63c06be4063a539c6ca4285bcd32359c656135150e1527b7a0479b44b':
        raise ValueError('complete baseline AST differs')
    return raw


class LoaderObservation(unittest.TestCase):
    def test_all_original_test_bytes_and_ast_are_preserved_except_authorized_inverse(self):
        raw = installed_identity_test_inverse(Path(__file__).read_bytes())
        start = raw.index(b'# BEGIN loader observation tests\n')
        end_marker = b'# END loader observation tests\n\n\n'
        end = raw.index(end_marker)+len(end_marker)
        raw = raw[:start]+raw[end:]
        added = b"""        data = diagnostic_inverse(DRIVER.read_bytes())
        tree = ast.parse(data)
        found = []
        for name in ('run','accept_unit'):
            for node in ast.walk(function(tree,name)):
                if isinstance(node,ast.Call) and ast.unparse(node.func) == 'check_wheel':
                    found.append((name,node.args[3]))
"""
        self.assertEqual(raw.count(added),1)
        raw = raw.replace(added,b'        data = DRIVER.read_bytes()\n')
        self.assertEqual(raw.count(b'        for _,node in found:\n'),1)
        raw = raw.replace(b'        for _,node in found:\n',b'        for _,node in self.callsites():\n')
        self.assertEqual(SHA(raw),'37e209bf12e823571a828136d85bb8e9dd5da28cbc2309f4414bd7d17e35db84')
        self.assertEqual(SHA(ast.dump(ast.parse(raw),include_attributes=False).encode()),
            'e31c35d7d86b60d5495f938d6822981e0bb3a38a387957a18cdb60aa6f122c1e')

    def test_source_type_authentication_origin_registry_and_line_mutations(self):
        import types
        for mode in ('comment','pin','guard','origin','registry','module_type','filename','line','code','globals'):
            with self.subTest(mode), tempfile.TemporaryDirectory() as tmp:
                w = ObservationWorld(Path(tmp),'success')
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',io.StringIO()) as output:
                    method = w.portable.load_inference
                    if mode == 'comment':
                        with Path(w.portable.__file__).open('a') as stream: stream.write('\n# source mutation\n')
                    elif mode == 'pin': w.pin = '0'*64
                    elif mode == 'guard': w.guards[str(w.path)] = 'bad'
                    elif mode == 'origin': w.portable.__spec__.origin += '.wrong'
                    elif mode == 'registry': sys.modules[w.name] = types.ModuleType(w.name)
                    elif mode == 'module_type': w.portable = SimpleNamespace(**vars(w.portable))
                    elif mode == 'filename': w.portable.__file__ += '.wrong'
                    elif mode == 'line':
                        w.portable.load_inference = types.FunctionType(method.__code__.replace(
                            co_firstlineno=method.__code__.co_firstlineno+1),method.__globals__)
                    elif mode == 'code':
                        w.portable.load_inference = lambda *args: None
                    elif mode == 'globals': w.portable.load_inference = types.FunctionType(method.__code__,{})
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    self.assertFalse(watch.eligible,mode)
                    self.assertTrue(observation_records(output))
                    watch.close()

    def test_live_inode_registry_and_guard_mutations_are_incomplete_without_changing_loader(self):
        import types
        for mode in ('inode','registry','guard'):
            with self.subTest(mode), tempfile.TemporaryDirectory() as tmp:
                w = ObservationWorld(Path(tmp),'success')
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',io.StringIO()) as output:
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    self.assertTrue(watch.eligible)
                    w.portable.construct_encoder = w.construct
                    if mode == 'inode':
                        w.path.rename(w.directory/'old.pt'); w.path.write_bytes(b'n'*4096)
                    elif mode == 'registry': sys.modules[w.name] = types.ModuleType(w.name)
                    else: w.guards[str(w.path)] = '0'*64
                    try:
                        watch.start()
                        state = w.portable.load_inference(w.directory,'bundle','cuda')
                    finally: watch.close()
                self.assertIs(type(state['A'].storage),bytes)
                record = next(r for r in observation_records(output) if r['event'] == 'before_original_mapping_guard')
                self.assertFalse(record['diagnostic_complete'])
                self.assertNotIn('owners',record)
                w.portable.mapping_absent(w.path)

    def test_existing_trace_is_ineligible_and_profile_identity_is_restored(self):
        def previous(frame,event,arg): return previous
        def profile(frame,event,arg): pass
        saved_trace,saved_profile = sys.gettrace(),sys.getprofile()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                w = ObservationWorld(Path(tmp),'success')
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',io.StringIO()) as output:
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    w.portable.construct_encoder = w.construct
                    sys.settrace(previous); sys.setprofile(profile)
                    try:
                        watch.start()
                        self.assertIs(sys.gettrace(),previous)
                        w.portable.load_inference(w.directory,'bundle','cuda')
                    finally: watch.close()
                    self.assertIs(sys.gettrace(),previous)
                    self.assertIs(sys.getprofile(),profile)
                self.assertFalse(any(r['event'] == 'before_original_mapping_guard' for r in observation_records(output)))
                sys.settrace(None)
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',io.StringIO()):
                    # Recreate the code contract before a second, independent hook test.
                    w = ObservationWorld(Path(tmp),'success')
                    sys.modules[w.name] = w.portable
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    watch.start()
                    sys.setprofile(None)
                    watch.close()
                    self.assertIsNone(sys.gettrace())
                    self.assertIs(sys.getprofile(),profile)
        finally:
            sys.settrace(saved_trace); sys.setprofile(saved_profile)

    def test_diagnostic_install_inspection_sink_failures_leave_original_cleanup(self):
        for mode in ('install','maps','sink'):
            with self.subTest(mode), tempfile.TemporaryDirectory() as tmp:
                w = ObservationWorld(Path(tmp),'construct_error')
                saved = sys.settrace
                def settrace(callback):
                    if callback is not None: raise RuntimeError('diagnostic install failure')
                    return saved(callback)
                events,output = [],io.StringIO()
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',output):
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    self.assertTrue(watch.eligible)
                    w.portable.construct_encoder = w.construct
                    # The real reference_outputs seam is used, including the original clear_frames and cleanup.
                    def loader(*args): return w.portable
                    def after():
                        events.append('after')
                        w.portable.mapping_absent(w.path)
                    target = {'install':'sys.settrace','maps':'qualify_connected_probe_serving._observation_maps',
                              'sink':'builtins.print'}[mode]
                    replacement = settrace if mode == 'install' else None
                    with patch('qualify_connected_probe_serving._LoaderObservation',return_value=watch), \
                            patch(target,side_effect=replacement or RuntimeError('diagnostic failure')):
                        with self.assertRaisesRegex(ValueError,'checkpoint mapping survived') as caught:
                            gate.reference_outputs(name=w.name,directory=w.directory,bundle_sha='bundle',loader=loader,
                                pin=w.pin,guards=w.guards,reads_only=lambda: io.StringIO(),batches=[],decode=None,rgb=None,
                                snapshot=None,loaded=lambda state: events.append('loaded'),after=after,deadline=lambda: None)
                    self.assertIs(type(caught.exception.__context__),RuntimeError)
                    self.assertEqual(events,['after'])
                    self.assertNotIn(w.name,sys.modules)
                    self.assertIsNone(sys.gettrace())
                    w.portable.mapping_absent(w.path)
                    if mode != 'sink':
                        record = next(r for r in observation_records(output) if r['event'] == 'reference_catch_before_clear_frames')
                        self.assertIn('injected constructor failure',json.dumps(record))

    def test_custom_error_and_owner_properties_are_not_executed(self):
        import types
        calls = []
        class Error(Exception):
            def __str__(self): calls.append('str'); return 'wrong'
            @property
            def __notes__(self): calls.append('notes'); return []
        class Owner:
            @property
            def __dict__(self): calls.append('dict'); return {}
        self.assertFalse(gate._observation_errors(Error())['diagnostic_complete'])
        self.assertIsNone(gate._observation_dict(Owner()))
        class Descriptor:
            def __get__(self, *args): calls.append('tensor property'); return None
        class Tensor:
            __module__ = 'torch'
            untyped_storage = device = shape = dtype = Descriptor()
        class Storage:
            __module__ = 'torch.storage'
            data_ptr = nbytes = device = Descriptor()
        torch = types.ModuleType('torch'); torch.Tensor = Tensor; torch.UntypedStorage = Storage
        with self.assertRaisesRegex(ValueError,'unsupported native metadata'):
            gate._observation_storage(Tensor(),torch)
        self.assertEqual(calls,[])

    def test_owner_quota_retains_scalar_partial_paths_without_retaining_objects(self):
        import types
        class Candidate: pass
        candidate = Candidate()
        reference = weakref.ref(candidate)
        torch = types.ModuleType('torch'); torch.__file__ = '/fake-authenticated-torch'
        # Scalar reader stand-in tests traversal only, and establishes no Torch/native attribution.
        def storage(value, module):
            if type(value) is Candidate:
                return {'object_id':id(value),'address':12,'end':16}
        def observe(value):
            guards = {torch.__file__:'f'*64}
            return gate._observation_owners(sys._getframe(),None,[{'start':10,'end':20}])
        with patch.dict(sys.modules,{'torch':torch}), patch.object(gate,'_observation_storage',storage):
            record = observe({'overlap':candidate,'excess':list(range(600))})
        self.assertFalse(record['diagnostic_complete'])
        self.assertEqual(record['candidates'][0]['path'],'loader_local.value.overlap')
        self.assertIn('quota',record['reason'])
        self.assertLessEqual(len(gate._observation_json(record)),65536)
        candidate = None
        self.assertIsNone(reference())
        self.assertNotIn('gc.get_objects',DRIVER.read_text())
        self.assertNotIn('gc.get_referrers',DRIVER.read_text())

    def test_full_byte_and_ast_inverse_rejects_extra_missing_and_multiple_nodes(self):
        raw = DRIVER.read_bytes()
        self.assertEqual(SHA(diagnostic_inverse(raw)),
            '149808c99e9216afdf7aacdc7e4610ea810598ea2dee8d1b963243ac394253dd')
        for changed in (raw.replace(OBSERVATION_SCOPE,OBSERVATION_ASSIGN),
                        raw.replace(OBSERVATION_SCOPE,OBSERVATION_SCOPE*2),
                        raw.replace(OBSERVATION_CATCH,b''),
                        raw.replace(OBSERVATION_CATCH,OBSERVATION_CATCH*2),
                        raw+b'\nunknown = 1\n',
                        raw.replace(b'def _observation_maps(',b'def extra_node('),
                        raw.replace(b'gc.collect()',b'gc.collect(0)'),
                        raw.replace(b"'body_seconds':300",b"'body_seconds':301")):
            with self.assertRaises(ValueError): diagnostic_inverse(changed)

    def test_original_exception_graph_includes_suppressed_context_and_cycles(self):
        self.assertTrue(hasattr(gate, '_observation_errors'), 'bounded scalar exception observation is missing')
        context = RuntimeError('original copy failure')
        primary = ValueError('mapping guard')
        primary.__context__ = context
        primary.__cause__ = ExceptionGroup('independent exit', [OSError('exact four')])
        context.__context__ = primary
        primary.add_note('original note')
        record = gate._observation_errors(primary)
        self.assertTrue(record['diagnostic_complete'])
        self.assertEqual(len(record['nodes']), 4)
        self.assertEqual(record['nodes'][0]['notes'], ['original note'])
        self.assertTrue(record['nodes'][0]['suppress_context'])
        self.assertEqual({edge['kind'] for edge in record['edges']}, {'cause', 'context', 'member'})
        self.assertIn('original copy failure', json.dumps(record))

    def test_exception_quotas_and_bad_inspection_are_incomplete(self):
        for error in (ExceptionGroup('too many', [ValueError(str(i)) for i in range(65)]),
                      ValueError('x'*65537)):
            self.assertFalse(gate._observation_errors(error)['diagnostic_complete'])
        class BadError(Exception):
            def __str__(self): raise RuntimeError('inspection failed')
        self.assertFalse(gate._observation_errors(BadError())['diagnostic_complete'])
        self.assertEqual(gate._observation_errors(None),
            {'diagnostic_complete':True,'root':None,'nodes':[],'edges':[]})

    def test_extracted_original_loader_with_real_mapping(self):
        import traceback
        for mode in ('success','external_owner','construct_error','copy_error','head_error'):
            with self.subTest(mode), tempfile.TemporaryDirectory() as tmp:
                w = ObservationWorld(Path(tmp),mode)
                output = io.StringIO()
                old_trace,old_profile = sys.gettrace(),sys.getprofile()
                error = state = None
                with patch.dict(sys.modules,w.registered), patch('sys.stdout',output):
                    watch = gate._LoaderObservation(w.portable,w.name,w.directory,w.pin,w.guards,sys.modules)
                    self.assertTrue(watch.eligible,output.getvalue())
                    # Stand-ins model native dependencies only; attribution to these objects is forbidden.
                    w.portable.construct_encoder = w.construct
                    try:
                        watch.start()
                        state = w.portable.load_inference(w.directory,'bundle','cuda')
                    except BaseException as caught:
                        error = caught
                    finally: watch.close()
                self.assertIs(sys.gettrace(),old_trace)
                self.assertIs(sys.getprofile(),old_profile)
                records = observation_records(output)
                at_guard = next(r for r in records if r['event'] == 'before_original_mapping_guard')
                self.assertEqual(at_guard['function'],'load_inference')
                self.assertEqual(at_guard['line'],1352)
                self.assertTrue(at_guard['binding_complete'])
                self.assertFalse(at_guard['diagnostic_complete'])
                self.assertFalse(at_guard['locals_present']['disk'])
                self.assertEqual(bool(at_guard['mappings']),mode in ('external_owner','construct_error','copy_error'))
                self.assertEqual(at_guard['errors']['root'] is None,mode in ('success','external_owner'))
                self.assertEqual(at_guard['owners']['owner'],'unresolved')
                if mode == 'success':
                    self.assertIsNone(error)
                    self.assertIs(type(state['A'].storage),bytes)
                    self.assertIs(type(state['processor']['metadata_tensor'].storage),bytes)
                    self.assertEqual(w.consumed,8)
                elif mode == 'head_error':
                    self.assertIs(type(error),RuntimeError)
                    w.portable.mapping_absent(w.path)
                else:
                    self.assertIs(type(error),ValueError)
                    self.assertEqual(str(error),'checkpoint mapping survived independent copy/release')
                    if mode == 'external_owner': self.assertIsNone(error.__context__)
                    else: self.assertIs(type(error.__context__),RuntimeError)
                if error is not None:
                    before = gate._observation_errors(error)
                    gate.clear_frames(error)
                    native_exit = ValueError('observed native difference must be exact four')
                    try: requests.raise_failures([error,native_exit])
                    except BaseException as final:
                        self.assertIs(final,error)
                        self.assertIn('exact four',''.join(traceback.format_exception(final)))
                        if mode in ('construct_error','copy_error'):
                            self.assertIn('injected',json.dumps(before))
                            self.assertNotIn('injected',''.join(traceback.format_exception(final)))
                        gate.clear_frames(final)
                w.external = error = state = None
                gc.collect()
                w.portable.mapping_absent(w.path)


def observation_records(output):
    return [json.loads(line.removeprefix('LOADER_OBSERVATION '))
        for line in output.getvalue().splitlines() if line.startswith('LOADER_OBSERVATION ')]


class ObservationTensor:
    def __init__(self, storage):
        self.storage = storage
        self.device = SimpleNamespace(type='cpu')
        self.requires_grad = False
    def to(self, device='cpu', copy=False):
        import mmap
        if ObservationWorld.current.mode == 'copy_error' and type(self.storage) is mmap.mmap:
            raise RuntimeError('injected copy failure')
        return ObservationTensor(bytes(self.storage)) if copy else self
    def __deepcopy__(self, memo):
        result = ObservationTensor(bytes(self.storage)); memo[id(self)] = result; return result
    def is_contiguous(self): return True
    def data_ptr(self): return id(self.storage)
    def numel(self): return len(self.storage)
    def element_size(self): return 1
    def detach(self): return self


class ObservationModel:
    def state_dict(self): return {'model_vision':True}
    def requires_grad_(self, value): return self
    def eval(self): return self
    def train(self): return self
    def to(self, device): return self


class ObservationWorld:
    """Real mmap + unmodified extracted source; fake native dependencies prove no native owner."""
    current = None

    def __init__(self, directory, mode):
        import types
        self.directory,self.mode = directory,mode
        self.external = None; self.consumed = 0
        self.path = directory/'endpoint.pt'
        self.path.write_bytes(b'x'*4096)
        (directory/'vision.pt').write_bytes(b'v'*4096)
        source = directory/gate.TRAINER
        raw = (HERE/gate.TRAINER).read_bytes()
        source.write_bytes(raw)
        self.pin = dict(gate.HISTORICAL)[gate.TRAINER]
        self.guards = {str(source):self.pin,str(self.path):'e'*64,str(directory/'vision.pt'):'f'*64}
        self.name = '_connected_probe_gate_observation_test'
        self.portable = types.ModuleType(self.name)
        self.portable.__file__ = str(source)
        self.portable.__spec__ = SimpleNamespace(origin=str(source))
        space = vars(self.portable)
        space.update(Path=Path,os=os,copy=copy,gc=gc,time=__import__('time'))
        tree = ast.parse(raw)
        compiled = compile(raw,str(source),'exec')
        for name in ('require','mapping_absent','owned_copy','inference_readout_tree','load_inference','construct_encoder'):
            code = next(c for c in compiled.co_consts if type(c) is types.CodeType and c.co_name == name)
            defaults = tuple(ast.literal_eval(v) for v in function(tree,name).args.defaults)
            space[name] = types.FunctionType(code,space,name,defaults)
        for name in ('INFERENCE_KEYS','INFERENCE_SCHEMA','CONTROL_SHA256','SERVING_FILES'):
            node = next(n for n in tree.body if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id == name for t in n.targets))
            space[name] = ast.literal_eval(node.value)
        space['ARMS'] = ('control',)
        pages = {}
        original_path = HERE/'train_siglip2_substrate_adaptation.py'
        original_tree = tree_of(original_path)
        for name in ('copy','consume'):
            exec(compile(ast.Module(body=[function(original_tree,name,'CheckpointPages')],type_ignores=[]),
                str(original_path),'exec'),pages)
        pages['require'] = space['require']
        world = self
        class Pages:
            copy = pages['copy']
            consume = pages['consume']
            def __init__(self, stream): self.fd = stream.fileno()
            def release(self, address, count): world.consumed += 1
        modules = {'train_siglip2_substrate_adaptation.py':SimpleNamespace(fingerprint=self.fingerprint,CheckpointPages=Pages),
            'qualify_siglip2_substrate_cpu.py':SimpleNamespace(numerical_flags=lambda: {}),
            'extract_siglip2_vision_source.py':SimpleNamespace(),
            'train_siglip2_cached_readout.py':SimpleNamespace(head_from=self.head)}
        manifest = {'endpoint_state_sha256':'endpoint','vision_sha256':'vision','encoder_identity':{'runtime':'structure'},
            'base_vision_sha256':'vision','environment':{'packages':{},'vision_constructor':'fake'},
            'files':{'vision.pt':'fake'},'code':{n:'fake' for n in space['SERVING_FILES'] | {'joint_relational_compaction.py'}}}
        space.update(admit_bundle=lambda *args: (manifest,self.guards),
            load_authenticated=lambda name,path,*args: modules.get(path.name,SimpleNamespace()),
            encoder_facts=lambda *args,**kwargs: {'vision_sha256':'vision'})
        self.registered = {self.name:self.portable,'torch':SimpleNamespace(Tensor=ObservationTensor,load=self.load,
            nn=SimpleNamespace(Parameter=lambda value,requires_grad: value))}
        ObservationWorld.current = self

    def load(self, path, **kwargs):
        import mmap
        assert kwargs == dict(map_location='cpu',weights_only=True,mmap=True)
        with Path(path).open('rb') as stream:
            storage = mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_COPY)
        def tensor(): return ObservationTensor(storage)
        if self.mode == 'external_owner': self.external = tensor()
        disk = {k:None for k in self.portable.INFERENCE_KEYS}
        disk.update(schema=self.portable.INFERENCE_SCHEMA,arm='control',fixed_sha256='fixed',config={},
            buffers={'position':tensor()},processor={'config':{},'metadata_tensor':tensor()},head={'weight':tensor()},
            A=tensor(),C=tensor(),means=[tensor(),(tensor(),)],mu_train=tensor(),common_statistics={'nested':[tensor(),(tensor(),)]},
            mu_train_provenance={'role':'FIT'},numerical_flags={},base_vision={'sha256':'vision'},
            encoder={'overlay':tensor()},encoder_identity={'runtime':'structure'},vision_sha256='vision',
            scope={'arm':'control','payload':{'scope_sha256':self.portable.CONTROL_SHA256,'class_names':['x']*1008}})
        return disk

    def fingerprint(self, value):
        if isinstance(value,dict) and 'model_vision' in value: return 'vision'
        if isinstance(value,dict) and 'schema' in value: return 'endpoint' if 'fixed_sha256' in value else 'fixed'
        return 'readout'

    def construct(self, source, original, context, config, buffers, processor_config, base, overlay):
        if self.mode == 'construct_error': raise RuntimeError('injected constructor failure with endpoint overlay')
        return ObservationModel(),SimpleNamespace(),SimpleNamespace(currsize=0),{},'structure'

    def head(self, arm, tensors):
        if self.mode == 'head_error': raise RuntimeError('injected head failure after independent copies')
        return ObservationModel()

# END loader observation tests


# BEGIN installed identity tests
IDENTITY_BLOCK_SHA = '7870ea3c5a51f0e26052a772b57115fae24bb6d0b503dda6266ab4b94d8e4988'


def installed_identity_inverse(raw):
    begin,end = b'# BEGIN installed identity boundary\n',b'# END installed identity boundary\n\n\n'
    if raw.count(begin) != 1 or raw.count(end) != 1: raise ValueError('identity block cardinality')
    start,stop = raw.index(begin),raw.index(end)+len(end)
    block = raw[start:stop]
    if SHA(block) != IDENTITY_BLOCK_SHA: raise ValueError('identity definitions differ')
    if [n.name for n in ast.parse(block).body] != ['_InstalledIdentityBoundary']:
        raise ValueError('identity node inventory differs')
    raw = raw[:start]+raw[stop:]
    changes = (
        (b"        identity_boundary = _InstalledIdentityBoundary(context,endpoint,authority['wheel'])\n",b''),
        (b'        identity_boundary.bind_native(runtime_authority,api,native_source)\n',b''),
        (b'            identity_boundary.check()\n',b''),
        (b'            reads_only = identity_boundary.scope\n',
            b"            def reads_only(): return context['evaluator_reference'].bundle_reads_only(context,endpoint)\n"))
    for added, original in changes:
        if raw.count(added) != 1: raise ValueError('identity dispatch cardinality')
        raw = raw.replace(added,original)
    if SHA(raw) != '81828ad4c0ed7bf2cebec57c46e0db7215f244ea4ae5f07aa699adaf971d1de1':
        raise ValueError('complete original production bytes differ')
    if SHA(ast.dump(ast.parse(raw),include_attributes=False).encode()) != '61ed2b38290f53b9dd1893dc7fed661c75d4a56af8d9185e95dfe71c75affcf8':
        raise ValueError('complete original production AST differs')
    return raw


def installed_identity_test_inverse(raw):
    begin,end = b'# BEGIN installed identity tests\n',b'# END installed identity tests\n\n\n'
    if raw.count(begin) != 1 or raw.count(end) != 1: raise ValueError('identity test block cardinality')
    start,stop = raw.index(begin),raw.index(end)+len(end)
    names = [n.name for n in ast.parse(raw[start:stop]).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    if names != ['installed_identity_inverse','installed_identity_test_inverse','InstalledIdentityBoundary','GroupedNativeComposition']:
        raise ValueError('identity test node inventory differs')
    raw = raw[:start]+raw[stop:]
    changes = ((b'def diagnostic_inverse(raw):\n    raw = installed_identity_inverse(raw)\n',b'def diagnostic_inverse(raw):\n'),
        (b'        raw = installed_identity_test_inverse(Path(__file__).read_bytes())\n',b'        raw = Path(__file__).read_bytes()\n'),
        (b"        digests = set(re.findall(r'[0-9a-f]{64}',installed_identity_inverse(DRIVER.read_bytes()).decode()))\n",
            b"        digests = set(re.findall(r'[0-9a-f]{64}',DRIVER.read_text()))\n"),
        (b'        text = installed_identity_inverse(DRIVER.read_bytes()).decode()\n',b'        text = DRIVER.read_text()\n'))
    for added, original in changes:
        if raw.count(added) != 1: raise ValueError('historical identity inverse normalization differs')
        raw = raw.replace(added,original)
    if SHA(raw) != '1adbc43bd9bf1a2b131bd9c8a04d39099c71d789f45898c8ab6bf717bd17f9e9':
        raise ValueError('complete original test bytes differ')
    if SHA(ast.dump(ast.parse(raw),include_attributes=False).encode()) != '03eb997161dae093cd2a0956c8bdc798bf5df8eeaf2999f3f7d8edf420e16f97':
        raise ValueError('complete original test AST differs')
    return raw


class InstalledIdentityBoundary(unittest.TestCase):
    @contextmanager
    def identity_world(self):
        import importlib.metadata as metadata
        from importlib.machinery import PathFinder
        from types import FunctionType
        import sysconfig
        from concurrent.futures import ThreadPoolExecutor
        stdlib = Path(sysconfig.get_path('stdlib'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root/'reference'/'evaluate_siglip2_identity_diversity.py'
            source_path.parent.mkdir(); source_path.write_bytes((HERE/source_path.name).read_bytes())
            source = requests.Source.load(fact(source_path))
            original = source.module
            site, installed = root/'original-site', root/'installed-site'
            site.mkdir(); installed.mkdir()
            guards = {source.fact['path']:source.fact['sha256']}
            def row(path, raw):
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
                return [path.relative_to(path.parents[1] if path.parent.name.endswith('.dist-info') else site).as_posix(),
                    'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode(),str(len(raw))]
            for distribution, names in original.RUNTIME_SOURCES.items():
                rows = []
                dist_name = next((n.split('/')[0] for n in names if n.endswith('/METADATA')),distribution+'-9.dist-info')
                for name in sorted(names | {dist_name+'/METADATA'}):
                    raw = ('Name: '+distribution+'\nVersion: 9\n').encode() if name.endswith('/METADATA') else b'# original pinned source\n'
                    path = site/name
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
                    rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode(),str(len(raw))])
                record = site/dist_name/'RECORD'
                stream = io.StringIO(); csv.writer(stream).writerows(rows); record.write_text(stream.getvalue())
                guards[str(record)] = file_sha(record)
            dist = installed/'sfora-0.3.0rc4.dist-info'
            metadata_path = dist/'METADATA'
            rows = [row(metadata_path,b'Name: sfora\nVersion: 0.3.0rc4\n'),
                ['sfora/__init__.py','sha256='+base64.urlsafe_b64encode(hashlib.sha256(b'').digest()).rstrip(b'=').decode(),'0'],
                [dist.name+'/RECORD','','']]
            (installed/'sfora').mkdir(); (installed/'sfora'/'__init__.py').write_bytes(b'')
            record = dist/'RECORD'
            stream = io.StringIO(); csv.writer(stream).writerows(rows); record.write_text(stream.getvalue())
            wheel = {'site_root':str(installed),'distribution':'sfora','version':'0.3.0rc4','record':fact(record),'direct_url':None}
            bundle = root/'bundle'; bundle.mkdir(); manifest = bundle/'bundle.json'; manifest.write_text('{}')
            endpoint = {'bundle':fact(manifest)}
            environment = {'packages':{'torch':{'root':str(site/'torch')}},'files':{}}
            trainer = SimpleNamespace(admit_bundle=lambda *args: ({'environment':environment},{}))
            # Match production guard_helpers: only module-level functions, including decorated wrappers.
            values = dict(vars(original))
            functions = [(f,f.__code__,f.__defaults__,copy.deepcopy(f.__kwdefaults__))
                for f in values.values() if isinstance(f,FunctionType)]
            literals = {k:copy.deepcopy(v) for k,v in values.items() if k != '__builtins__' and
                isinstance(v,(dict,list,tuple,set,frozenset))}
            snapshot = (original,Path(original.__file__),original.__spec__,values,functions,literals)
            context = {'evaluator_reference':original,'launch':{'evaluator_reference':{'root':str(source_path.parent),
                'code':{'evaluate_siglip2_identity_diversity.py':source.fact['sha256']}}},'guards':dict(guards),
                'required_guards':dict(guards),'trainer':trainer,'helper_snapshots':[snapshot],
                'training_context':{'legacy':{'selected':{'source_cpu':{'origins':{'files':{},'native_files':[]}}}}}}
            # Discovery is authenticated before entry; the original boundary grants no installed directory scan.
            list(metadata.distributions(path=[str(installed)]))
            self.assertIsNotNone(PathFinder.find_spec('sfora',[str(installed)]))
            try:
                with patch.object(sys,'path',[str(installed),str(site),str(stdlib),str(stdlib/'lib-dynload')]):
                    yield SimpleNamespace(root=root,source=source,original=original,context=context,endpoint=endpoint,
                        wheel=wheel,site=site,installed=installed,dist=dist,metadata=metadata,rows=rows,record=record)
            finally: sys.modules.pop(original.__name__,None)

    def test_real_metadata_original_red_and_installed_green(self):
        with self.identity_world() as w:
            with w.original.bundle_reads_only(w.context,w.endpoint):
                with self.assertRaisesRegex(ValueError,'bundle-only loader attempted external dependency.*top_level.txt'):
                    w.metadata.packages_distributions()
        with self.identity_world() as w:
            self.assertTrue(hasattr(gate,'_InstalledIdentityBoundary'),'installed identity seam is missing')
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            required = dict(w.context['required_guards'])
            with boundary.scope():
                self.assertEqual(w.metadata.packages_distributions()['sfora'],['sfora'])
            with boundary.scope():
                self.assertEqual(w.metadata.packages_distributions()['sfora'],['sfora'])
            boundary.check()
            self.assertEqual(w.context['required_guards'],required)
            self.assertEqual(w.context['guards'][str(w.record)],w.wheel['record']['sha256'])
            self.assertEqual(w.context['guards'][str(w.dist/'METADATA')],file_sha(w.dist/'METADATA'))
            w.source.check()

    def test_grouped_derivative_retains_original_without_alias(self):
        with self.identity_world() as w:
            original = w.original.grouped_md_origin
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            self.assertTrue('_probe_grouped_md_origin' in boundary.space)
            self.assertIsNone(boundary.space['_probe_grouped_md_origin'](w.context))
            self.assertIs(w.original.grouped_md_origin,original)
            w.source.check()

    def test_exact_production_and_old_test_byte_ast_inverse(self):
        raw = DRIVER.read_bytes()
        original = installed_identity_inverse(raw)
        installed_identity_test_inverse(Path(__file__).read_bytes())
        original_pins = set(re.findall(r'[0-9a-f]{64}',original.decode()))
        self.assertEqual(set(re.findall(r'[0-9a-f]{64}',raw.decode())),original_pins | {
            '95cb8823236408537e04108fd63727fd3a323671f04eb33a6349b23a51ce638d',
            'cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1',
            'f9e19c3c4805ee72b3a9a04bbce65a8645b7501147584322b5c3663de22b1745',
            '7bbd4303cb0724330e5074b859ec4d4a3d5b6f1d40d4182d16c5302ce54e4060'})
        tree = ast.parse(raw)
        execs = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func) == 'exec']
        self.assertEqual(len(execs),3)
        added_execs = [n for n in ast.walk(function(tree,'__init__','_InstalledIdentityBoundary'))
            if isinstance(n,ast.Call) and ast.unparse(n.func) == 'exec']
        self.assertEqual(len(added_execs),2)
        self.assertEqual({ast.unparse(n.args[1]) for n in added_execs},{'self.space','self.grouped_space'})
        self.assertEqual(len([n for n in ast.walk(function(tree,'derive_evaluator_loader'))
            if isinstance(n,ast.Call) and ast.unparse(n.func) == 'exec']),1)
        for changed in (raw.replace(b'identity_boundary.check()',b'identity_boundary.check()\n            pass'),
                raw.replace(b'            reads_only = identity_boundary.scope\n',b''),
                raw.replace(b'            reads_only = identity_boundary.scope\n',b'            reads_only = identity_boundary.scope\n'*2),
                raw.replace(b'def clear_frames(',b'def altered_clear_frames('),raw+b'\nextra = 1\n',
                raw.replace(b'    def read_identity(self):',b'    def extra_identity(self):')):
            with self.assertRaises(ValueError): installed_identity_inverse(changed)
        tests = Path(__file__).read_bytes()
        for changed in (tests+b'\nextra = 1\n',tests.replace(b'def diagnostic_inverse(raw):',b'def altered_inverse(raw):')):
            with self.assertRaises(ValueError): installed_identity_test_inverse(changed)

    def test_active_original_function_tamper_rejected_before_restoration(self):
        with self.identity_world() as w:
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            wrapper = w.original.bundle_reads_only
            body = wrapper.__wrapped__
            cell = wrapper.__closure__[0]
            def foreign_body(*args,**kwargs): yield None
            changes = ((body,'__code__',foreign_body.__code__),
                (body,'__defaults__',('foreign',)),
                (body,'__kwdefaults__',{'foreign':True}),
                (wrapper,'__wrapped__',w.original.require),
                (cell,'cell_contents',w.original.require),
                (w.original.__spec__,'cached','foreign'))
            refusals = []
            def observe(label, action):
                try: action()
                except ValueError: refusals.append((label,True))
                else: refusals.append((label,False))
            with boundary.scope():
                for target, name, value in changes:
                    before = getattr(target,name)
                    setattr(target,name,value)
                    try:
                        observe(name,boundary.check)
                        observe('read '+name,(w.dist/'METADATA').read_bytes)
                    finally: setattr(target,name,before)
                with patch.dict(vars(body),{'foreign':True}):
                    observe('attribute',boundary.check)
                    observe('read attribute',(w.dist/'METADATA').read_bytes)
                boundary.check()
                self.assertEqual((w.dist/'METADATA').read_bytes(),b'Name: sfora\nVersion: 0.3.0rc4\n')
            boundary.check()
            for label, refused in refusals:
                with self.subTest(label): self.assertTrue(refused,'live source tamper accepted')

    def test_identity_bytes_absence_guards_and_site_refusals(self):
        with self.identity_world() as w:
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            top, metadata = w.dist/'top_level.txt', w.dist/'METADATA'
            raw_record, raw_metadata = w.record.read_bytes(), metadata.read_bytes()
            for label, path, value in (('new RECORD row',w.record,raw_record+b'x.py,,\n'),
                    ('wrong metadata bytes',metadata,raw_metadata+b'bad'),
                    ('present top',top,b'sfora\n')):
                with self.subTest(label):
                    path.write_bytes(value)
                    try:
                        with self.assertRaises(ValueError): boundary.check()
                    finally:
                        if path == top: path.unlink()
                        else: path.write_bytes(raw_record if path == w.record else raw_metadata)
            top.symlink_to(w.root/'missing')
            try:
                with self.assertRaises(ValueError): boundary.check()
            finally: top.unlink()
            for path in (w.record,metadata):
                old = w.context['guards'][str(path)]
                w.context['guards'][str(path)] = '0'*64
                try:
                    with self.subTest('wrong guard '+path.name), self.assertRaises(ValueError): boundary.check()
                finally: w.context['guards'][str(path)] = old
            wheel = copy.deepcopy(w.wheel); wheel['site_root'] = str(w.site)
            with self.assertRaises(ValueError): gate._InstalledIdentityBoundary(w.context,w.endpoint,wheel)
            boundary.check()

    def test_record_hash_size_metadata_and_canonical_row_refusals(self):
        with self.identity_world() as w:
            for label, change in (
                    ('hash',lambda r: r[0].__setitem__(1,'sha256='+'A'*43)),
                    ('size',lambda r: r[0].__setitem__(2,str(int(r[0][2])+1))),
                    ('noncanonical hash',lambda r: r[0].__setitem__(1,'sha256='+r[0][1][7:-1]+'B')),
                    ('noncanonical size',lambda r: r[0].__setitem__(2,'0'+r[0][2])),
                    ('missing METADATA',lambda r: r.pop(0)),
                    ('duplicate row',lambda r: r.append(r[0].copy())),
                    ('listed top',lambda r: r.append([w.dist.name+'/top_level.txt','',''])),
                    ('RECORD row',lambda r: r[-1].__setitem__(2,'1'))):
                rows = copy.deepcopy(w.rows); change(rows)
                stream = io.StringIO(); csv.writer(stream).writerows(rows); w.record.write_text(stream.getvalue())
                wheel = copy.deepcopy(w.wheel); wheel['record'] = fact(w.record)
                with self.subTest(label), self.assertRaises(ValueError):
                    gate._InstalledIdentityBoundary(w.context,w.endpoint,wheel)
                w.context['guards'].pop(str(w.record),None); w.context['guards'].pop(str(w.dist/'METADATA'),None)
            for field, replacement in (('Name: sfora','Name: foreign'),('Version: 0.3.0rc4','Version: wrong')):
                path = w.dist/'METADATA'; path.write_text(('Name: sfora\nVersion: 0.3.0rc4\n').replace(field,replacement))
                rows = copy.deepcopy(w.rows)
                rows[0][1] = 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
                rows[0][2] = str(path.stat().st_size)
                stream = io.StringIO(); csv.writer(stream).writerows(rows); w.record.write_text(stream.getvalue())
                wheel = copy.deepcopy(w.wheel); wheel['record'] = fact(w.record)
                with self.subTest(field), self.assertRaises(ValueError): gate._InstalledIdentityBoundary(w.context,w.endpoint,wheel)
                w.context['guards'].pop(str(w.record),None); w.context['guards'].pop(str(path),None)

    def test_original_authentication_and_no_stale_cache_adoption(self):
        with self.identity_world() as w:
            module, context = w.original, w.context
            boundary = gate._InstalledIdentityBoundary(context,w.endpoint,w.wheel)
            source_path = Path(w.source.fact['path']); source_raw = source_path.read_bytes()
            source_path.write_bytes(source_raw+b'\n# extra\n')
            try:
                with self.assertRaises(ValueError): gate._InstalledIdentityBoundary(context,w.endpoint,w.wheel)
            finally: source_path.write_bytes(source_raw)
            with patch.object(module,'distribution_identity_files',lambda *a: ({},set(),set())), self.assertRaises(ValueError):
                boundary.check()
            original_defaults = module.check_file.__defaults__
            module.check_file.__defaults__ = (None,)
            try:
                with self.assertRaises(ValueError): gate._InstalledIdentityBoundary(context,w.endpoint,w.wheel)
            finally: module.check_file.__defaults__ = original_defaults
            with patch.object(module,'__file__',str(w.root/'foreign.py')), self.assertRaises(ValueError): boundary.check()
            with patch.object(module.__spec__,'origin',str(w.root/'foreign.py')), self.assertRaises(ValueError): boundary.check()
            with patch.dict(sys.modules,{module.__name__:SimpleNamespace()}), self.assertRaises(ValueError): boundary.check()
            with patch.dict(module.RUNTIME_SOURCES,{'foreign':set()}), self.assertRaises(ValueError): boundary.check()
            original_code = module.distribution_identity_files.__code__
            module.distribution_identity_files.__code__ = (lambda *args: ({},set(),set())).__code__
            try:
                with self.assertRaises(ValueError): boundary.check()
            finally: module.distribution_identity_files.__code__ = original_code
            context['portable_audits'] = {}
            with self.assertRaises(ValueError): gate._InstalledIdentityBoundary(context,w.endpoint,w.wheel)
            context.pop('portable_audits')
            with boundary.scope(): pass
            cache = context['portable_audits']
            with self.assertRaises(ValueError): gate._InstalledIdentityBoundary(context,w.endpoint,w.wheel)
            context['portable_audits'] = dict(cache)
            with self.assertRaises(ValueError): boundary.check()
            context['portable_audits'] = cache
            entry = cache[boundary.identity]; cache[boundary.identity] = tuple(list(entry))
            with self.assertRaises(ValueError): boundary.check()
            cache[boundary.identity] = entry
            runtime = dict(entry[1]); entry[1][w.root/'foreign.py'] = '0'*64
            with self.assertRaises(ValueError): boundary.check()
            entry[1].clear(); entry[1].update(runtime)
            with patch.dict(boundary.space,{'_probe_distribution_identity_files':lambda *a: ({},set(),set())}), self.assertRaises(ValueError):
                boundary.check()
            with patch.object(boundary.boundary,'__defaults__',(None,)), self.assertRaises(ValueError): boundary.check()
            fn = boundary.space['_probe_distribution_identity_files']
            with patch.object(fn,'__defaults__',(None,)), self.assertRaises(ValueError): boundary.check()
            seam_code = fn.__code__
            # The replacement retains one closure cell, so Python permits it and the dispatch guard must reject it.
            def same_closure():
                value = None
                def replacement(*args): return value
                return replacement
            fn.__code__ = same_closure().__code__
            try:
                with self.assertRaises(ValueError): boundary.check()
            finally: fn.__code__ = seam_code
            boundary.check()

    def test_active_reads_writes_extra_paths_and_failed_exit(self):
        with self.identity_world() as w:
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            unrelated = w.installed/'foreign.py'; unrelated.write_bytes(b'foreign')
            native = w.installed/'foreign.so'; native.write_bytes(b'native')
            direct = w.dist/'direct_url.json'; direct.write_bytes(b'{}')
            import importlib.util
            pyc = Path(importlib.util.cache_from_source(str(w.site/'packaging'/'__init__.py')))
            pyc.parent.mkdir(exist_ok=True); pyc.write_bytes(b'unpinned')
            with boundary.scope():
                for path in (unrelated,native,direct,w.source.fact['path'],pyc):
                    with self.subTest(str(path)), self.assertRaises((ValueError,FileNotFoundError)): Path(path).read_bytes()
                with self.assertRaises(ValueError): os.listdir(w.installed)
                with self.assertRaises(ValueError): (w.dist/'METADATA').open('wb')
                with self.assertRaises(FileNotFoundError): (w.dist/'top_level.txt').read_text()
                with self.assertRaises(ValueError):
                    with boundary.scope(): pass
                with patch.dict(w.context['guards'],{str(w.record):'0'*64}), self.assertRaises(ValueError): w.record.read_text()
            with self.assertRaisesRegex(RuntimeError,'original body failure'):
                with boundary.scope(): raise RuntimeError('original body failure')
            self.assertFalse(boundary.active)
            boundary.check()

    def test_fresh_exit_absence_and_hash_failure_preserve_original_error(self):
        for label, filename, data in (('absence','top_level.txt',b'sfora'),('metadata','METADATA',b'changed'),
                ('record','RECORD',b'changed')):
            with self.subTest(label), self.identity_world() as w:
                boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
                fd = None if filename == 'top_level.txt' else os.open(w.dist/filename,os.O_WRONLY)
                with self.assertRaises((ValueError,RuntimeError)) as raised:
                    with boundary.scope():
                        # A preopened fd or directory entry models an external change without bypassing either hook.
                        if fd is None: (w.dist/filename).symlink_to(w.root/'missing')
                        else:
                            os.write(fd,data); os.ftruncate(fd,len(data)); os.close(fd)
                        raise RuntimeError('original body failure')
                errors = gate._observation_errors(raised.exception)
                self.assertTrue(any(n['type'] == 'builtins.RuntimeError' and n['message'] == 'original body failure'
                    for n in errors['nodes']))
                self.assertTrue(raised.exception.__notes__)

    def test_audit_hook_does_not_retain_context_owner(self):
        class Owner: pass
        for failure in (False,True):
            with self.subTest(failure), self.identity_world() as w:
                owner = Owner(); w.context['context_owner'] = owner
                boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'body failure'):
                        with boundary.scope(): raise RuntimeError('body failure')
                else:
                    with boundary.scope(): pass
                reference, context_owner = weakref.ref(boundary), weakref.ref(owner)
            del boundary, owner, w; gc.collect()
            self.assertIsNone(reference())
            self.assertIsNone(context_owner())

class GroupedNativeComposition(unittest.TestCase):
    identity_world = InstalledIdentityBoundary.identity_world

    @contextmanager
    def native_world(self):
        """Real install/collect/maps/Source and extracted original predicates; fixture files only."""
        import importlib.util
        from contextlib import ExitStack
        from types import MappingProxyType
        with self.identity_world() as w, ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules))
            def module(name, raw):
                path = w.root/(name+'.py'); path.write_text(raw)
                spec = importlib.util.spec_from_file_location(name,path)
                value = importlib.util.module_from_spec(spec); sys.modules[name] = value
                exec(compile(raw,str(path),'exec',dont_inherit=True),vars(value))
                return value
            def extracted(filename, name):
                return ast.unparse(function(tree_of(HERE/filename),name))+'\n'
            def write_json(name, value):
                path = w.root/name; path.write_text(json.dumps(value)); return fact(path)
            native = module('_grouped_native',(HERE/'connected_control_native_authority.py').read_text())
            binary = w.root/'candidate.so'; binary.write_bytes(b'fixture candidate')
            runtime = w.root/'runtime.so'; runtime.write_bytes(b'fixture runtime')
            compiler = w.root/'tileiras'; compiler.write_bytes(b'never executed'); compiler.chmod(0o755)
            manifest = write_json('source-manifest.json',{'ffi.rs':'a'*64})
            binaries = write_json('binaries.json',{'candidate.so':file_sha(binary)})
            evidence = {'source-manifest.json':manifest,'binaries.json':binaries}
            build = write_json('build.json',{'schema':'sfora-cutile-threads-v1','claim_eligible':False,
                'decision':'PASS_SEQUENTIAL_THREAD_EXACTNESS_CACHE_GATE','files_sha256':{n:v['sha256'] for n,v in evidence.items()}})
            provenance = write_json('candidate-proof.json',{'schema':'connected-control-native-file-provenance-v1',
                'file':fact(binary),'kind':'archived-cutile-build','origin':'fixture','evidence':[build,binaries,manifest]})
            runtime_proof = write_json('runtime-proof.json',{'schema':'connected-control-native-file-provenance-v1',
                'file':fact(runtime),'kind':'root-frozen-runtime','origin':'fixture','evidence':[manifest]})
            record = {'schema':'connected-control-native-authority-v2','library':fact(binary),
                'runtime_compiler':fact(compiler),'build_receipt':build,'build_evidence':evidence,'source_manifest':manifest,
                'supplemental':[{'file':fact(binary),'provenance':provenance},{'file':fact(runtime),'provenance':runtime_proof}]}
            authority_fact = write_json('authority.json',record)
            # Substitute only immutable FILE descriptors for tiny local fixtures, never callable code.
            stack.enter_context(patch.object(gate._InstalledIdentityBoundary,'NATIVE_AUTHORITY',
                (authority_fact['path'],authority_fact['sha256'])))
            stack.enter_context(patch.object(gate._InstalledIdentityBoundary,'NATIVE_FILES',
                ((str(binary),file_sha(binary)),(str(runtime),file_sha(runtime)))))
            # Synthetic runtime pins only; every native method retains the real source code.
            native.BINARY_SHA, native.ARCHIVE_SHA = file_sha(binary),build['sha256']
            native_source = requests.Source(native,fact(Path(native.__file__)))
            stack.enter_context(patch.dict(os.environ,{'CUTILE_TILEIRAS_PATH':str(compiler)}))
            group = {w.site/n:h for n,h in w.original.GROUPED_MD_NATIVE_SHA256.items()}
            for p in group: p.parent.mkdir(exist_ok=True); p.write_bytes(b'group fixture')
            four = {}
            for n in ('engines_precompiled','engines_runtime_compiled','graph','heuristic'):
                p = w.site/('libcudnn_'+n+'.so.9'); p.write_bytes(n.encode()); four[str(p)] = file_sha(p)
            mapped = [*map(str,group),*four,str(binary),str(runtime)]
            original = {'packages':{'torch':{'root':str(w.site/'torch')}},'files':dict((str(p),h) for p,h in group.items()),
                'modules':{},'native_files':list(map(str,group))}
            supplement = MappingProxyType({'files':MappingProxyType(dict(four)),'modules':MappingProxyType({})})
            collector = module('qualify_siglip2_substrate_cpu',
                'from pathlib import Path\nimport hashlib\n'+extracted('qualify_connected_probe_serving.py','require')+
                'def imported_origins(extract, packages):\n'
                '    return extract(packages)\n')
            nearest_node = function(tree_of(HERE/'train_siglip2_nearest_ranking.py'),'native_source_api')
            nearest_raw = extracted('qualify_connected_probe_serving.py','require')+'\nNATIVE_MEMBERS = '+repr(set(Path(p).name for p in four))+'\n'
            nearest_raw += 'def native_source_api(context, _owned={}):\n    return _owned[id(context)][2]\n'
            for name in ('audit_origins','exit_rehash'):
                node = next(n for n in nearest_node.body if isinstance(n,ast.FunctionDef) and n.name == name)
                nearest_raw += '\n'.join('    '+line for line in ast.unparse(node).splitlines())+'\n'
            nearest_raw += 'def original_authenticate():\n    pass\n'
            nearest = module('_grouped_nearest',nearest_raw)
            old = module('_grouped_old','from pathlib import Path\n'+extracted('qualify_connected_probe_serving.py','require')+
                extracted('train_siglip2_quadratic_readout.py','audit_origins')+
                extracted('train_siglip2_quadratic_readout.py','exit_rehash')+
                'def bound_file(guards,path,sha):\n    require(guards.setdefault(path,sha) == sha,"fixture guard mismatch")\n')
            fitter = module('_grouped_fitter','def exit_rehash(context):\n    context["old"].exit_rehash(context["legacy"])\n')
            evaluator = module('evaluate_siglip2_connected_mlp',
                '_SOURCE_BUILTINS = tuple(vars(__import__("builtins")).items())\n'
                'def exit_rehash(context,guard):\n    t=context["training_context"]\n    return t["nearest"].native_source_api(t)\n')
            evaluator_source = native.load_evaluator_source(fact(Path(evaluator.__file__)),requests)
            evaluator = evaluator_source.module
            private_ns = {'_nearest_supplement':supplement}
            exec('def audit_origins(*args,**kwargs): pass',private_ns)
            def wrapped(audit_origins):
                def audit(*args,**kwargs): return audit_origins(*args,**kwargs)
                return audit
            original_api = SimpleNamespace(audit_origins=wrapped(private_ns['audit_origins']))
            counters = SimpleNamespace(auth=0,libraries=0,collect=0)
            def collect(packages):
                counters.collect += 1
                files = {}
                for name in mapped:
                    counters.libraries += 1
                    files[name] = group.get(Path(name),file_sha(Path(name)))
                return {'packages':packages,'files':files,'modules':{},'native_files':list(mapped)}
            legacy = {'selected':{'packages':original['packages'],'source_cpu':{'origins':original}},
                'warm_record':{'origins':{'files':{},'modules':{},'native_files':[]}},'source_driver':collector,
                'extract':collect,'prior':{'guards':{}},'guards':{}}
            t = w.context['training_context']; t.clear(); t.update(legacy=legacy,nearest=nearest,old=old,fitter=fitter,
                fit_context={},guards={},native_source_owned={'api':original_api,'authenticate':nearest.original_authenticate})
            class OpaqueOwner:
                def __deepcopy__(self,memo): raise AssertionError('live owner graph copied')
            t['opaque_owner'] = OpaqueOwner()
            nearest.native_source_api.__defaults__[0][id(t)] = (t,nearest.original_authenticate,original_api)
            for mod in (native,requests,observer,nearest,old,fitter,evaluator):
                f = fact(Path(mod.__file__)); w.context['guards'][f['path']] = t['guards'][f['path']] = f['sha256']
            w.context['required_guards'].update({**original['files'],**four,str(Path(collector.__file__)):
                'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38'})
            md_record = w.site/'charset_normalizer-3.4.7.dist-info'/'RECORD'; md_record.parent.mkdir(exist_ok=True)
            md_record.write_text('charset_normalizer/md.cpython-313-aarch64-linux-gnu.so,sha256=EYzysjLHjG1gl9fj8mFGIRrs0HPeSgwZw5AWcSxih7A,201304\n')
            w.context['required_guards'][str(md_record)] = file_sha(md_record)
            alias = w.site/'charset_normalizer'/'md.cpython-313-aarch64-linux-gnu.so'
            md = SimpleNamespace(__file__=str(alias),__spec__=SimpleNamespace(origin=str(alias)))
            stack.enter_context(patch.dict(sys.modules,{'charset_normalizer.md':md}))
            actual_open = Path.open; maps_override = [None]
            def maps():
                return '\n'.join('1000-2000 r-xp 0 %x:%x %d %s' %
                    (os.major(Path(p).stat().st_dev),os.minor(Path(p).stat().st_dev),Path(p).stat().st_ino,p) for p in mapped)
            def open_file(path,*args,**kwargs):
                if str(path) == '/proc/self/maps': return io.StringIO(maps_override[0] if maps_override[0] is not None else maps())
                if str(path) == authority_fact['path']: counters.auth += 1
                if '.so' in path.name: counters.libraries += 1
                return actual_open(path,*args,**kwargs)
            stack.enter_context(patch.object(Path,'open',open_file))
            owner = native.CombinedAuthority(t,authority_fact,observer,requests)
            api = owner.install(evaluator_source,w.context)
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            boundary.bind_native(owner,api,native_source)
            # Only the unavailable remote source/group bytes have fixture stand-ins.
            bound = boundary.grouped_space['bound_file']
            def bound_group(guards,path,sha):
                if Path(path) in group:
                    counters.libraries += 1
                    gate.require(sha == group[Path(path)],'fixture group digest differs')
                    return Path(path)
                if Path(path) == Path(collector.__file__):
                    gate.require(sha == 'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38','fixture source digest differs')
                    return Path(path)
                return bound(guards,path,sha)
            boundary.grouped_space['bound_file'] = bound_group
            boundary.grouped_values['bound_file'] = bound_group
            counters.auth = counters.libraries = counters.collect = 0
            yield SimpleNamespace(w=w,boundary=boundary,owner=owner,api=api,source=native_source,group=group,
                mapped=mapped,maps=maps,maps_override=maps_override,alias=alias,counters=counters,
                binary=binary,runtime=runtime,record=md_record,provenance=Path(provenance['path']),four=four)

    def test_exact_s_and_historical_only_authenticate_without_collecting(self):
        with self.native_world() as g:
            callback = g.boundary.space['_probe_grouped_md_origin']
            g.api.audit_origins(g.owner.context['legacy'])
            inventory,origins = g.owner.inventory,g.owner.context['legacy']['origins']
            g.counters.auth = g.counters.collect = 0
            self.assertEqual(callback(g.w.context),str(g.alias))
            self.assertEqual(g.counters.collect,0)
            self.assertGreater(g.counters.auth,0)
            self.assertEqual(set(g.owner.inventory['native_files']),set(g.mapped))
            self.assertTrue(g.owner.admitted)
            self.assertEqual(callback(g.w.context),str(g.alias))
            self.assertEqual(g.counters.collect,0)
            self.assertIs(g.owner.inventory,inventory)
            self.assertIs(g.owner.context['legacy']['origins'],origins)
        with self.native_world() as g:
            g.mapped.remove(str(g.binary)); g.mapped.remove(str(g.runtime))
            self.assertEqual(g.boundary.space['_probe_grouped_md_origin'](g.w.context),str(g.alias))
            self.assertEqual(g.counters.collect,0)
            self.assertFalse(g.owner.admitted)

    def test_map_mutants_reject_before_authentication_or_library_reads(self):
        with self.native_world() as g:
            original = list(g.mapped)
            extra = g.w.site/'unknown.so'; extra.write_bytes(b'unknown')
            g.w.context['guards'][str(extra)] = file_sha(extra)
            foreign = g.w.site/'other'/'libcudnn_graph.so.9'; foreign.parent.mkdir(); foreign.write_bytes(b'foreign')
            g.w.context['required_guards'][str(foreign)] = file_sha(foreign)
            cases = ('group','alias','unknown','guard basename','deleted','noncanonical','inode','device','missing S',
                'admitted S','malformed','truncated','replaced S')
            for case in cases:
                with self.subTest(case):
                    g.mapped[:] = original; g.maps_override[0] = None; g.owner.admitted = False
                    if case == 'group': g.mapped.remove(str(next(iter(g.group))))
                    elif case == 'alias': g.alias.write_bytes(b'forbidden'); g.mapped.append(str(g.alias))
                    elif case == 'unknown': g.mapped.append(str(extra))
                    elif case == 'guard basename': g.mapped.append(str(foreign))
                    elif case == 'missing S': g.mapped.remove(str(g.runtime))
                    elif case == 'admitted S':
                        g.owner.admitted = True; g.mapped.remove(str(g.binary)); g.mapped.remove(str(g.runtime))
                    elif case == 'replaced S':
                        g.runtime.rename(g.runtime.with_suffix('.saved')); g.runtime.write_bytes(b'fixture runtime')
                    elif case == 'malformed': g.maps_override[0] = g.maps()+'\nnot-a-map /foreign.so'
                    elif case == 'truncated': g.maps_override[0] = g.maps()+'\n1000-2000 r-xp 0 0:0 /foreign.so'
                    else:
                        lines = g.maps().splitlines(); parts = lines[-1].split(maxsplit=5)
                        if case == 'deleted': parts[5] += ' (deleted)'
                        elif case == 'noncanonical': parts[5] = str(g.runtime.parent)+'/./'+g.runtime.name
                        elif case == 'inode': parts[4] = str(int(parts[4])+1)
                        elif case == 'device': parts[3] = '0:0'
                        lines[-1] = ' '.join(parts); g.maps_override[0] = '\n'.join(lines)
                    g.counters.auth = g.counters.libraries = g.counters.collect = 0
                    try:
                        with self.assertRaises((ValueError,FileNotFoundError)):
                            g.boundary.space['_probe_grouped_md_origin'](g.w.context)
                        self.assertEqual((g.counters.auth,g.counters.libraries,g.counters.collect),(0,0,0))
                    finally:
                        if g.alias.exists() or g.alias.is_symlink(): g.alias.unlink()
                        if case == 'replaced S':
                            g.runtime.unlink(); g.runtime.with_suffix('.saved').rename(g.runtime)

    def test_unmapped_md_presence_does_not_grant_bytes_or_native_authority(self):
        with self.native_world() as g:
            g.alias.write_bytes(b'physical wrapper remains untrusted')
            self.assertEqual(g.boundary.space['_probe_grouped_md_origin'](g.w.context),str(g.alias))
            self.assertNotIn(str(g.alias),g.owner.files)
            self.assertNotIn(str(g.alias),g.w.context['guards'])
            self.assertEqual(g.counters.collect,0)

    def test_owned_binding_mutants_reject_before_library_reads(self):
        from types import FunctionType
        with self.native_world() as g:
            auth = g.api.authenticate
            cells = dict(zip(auth.__code__.co_freevars,auth.__closure__,strict=True))
            owned = g.owner.context['control_native_owned']
            checker_source = g.boundary.native[7][0][0]
            counterfeit = FunctionType(auth.__code__,dict(auth.__globals__),auth.__name__,auth.__defaults__,auth.__closure__)
            cases = [
                ('owner',owned,'owner',SimpleNamespace()),('api',owned,'api',SimpleNamespace()),
                ('authentication',owned,'authenticate',lambda:None),
                ('export',vars(g.api),'audit_origins',lambda *a:None),
                ('globals',vars(g.api),'authenticate',counterfeit),
                ('source path',checker_source.fact,'path',str(g.w.root/'foreign.py')),
                ('files',g.owner.files,str(g.runtime),'0'*64),
                ('supplement',vars(g.owner),'supplement',dict(g.owner.supplement)),
                ('registry',g.owner.context['nearest'].native_source_api.__defaults__[0],id(g.owner.context),
                    (g.owner.context,g.owner.context['nearest'].original_authenticate,SimpleNamespace())),
                ('literal',g.source.module.__dict__,'BINARY_SHA','0'*64),
                ('source globals',g.source.module.__dict__,'Path',lambda *a:None),
                ('namespace',g.api.audit_origins.__globals__,'private_audit',lambda *a:None),
                ('legacy native set',g.owner.context['legacy']['selected']['source_cpu']['origins'],
                    'native_files',[*g.mapped,str(g.w.root/'unknown.so')])]
            for label,values,key,replacement in cases:
                with self.subTest(label):
                    before = values[key]; values[key] = replacement
                    g.counters.auth = g.counters.libraries = 0
                    try:
                        with self.assertRaises(ValueError): g.boundary.space['_probe_grouped_md_origin'](g.w.context)
                        self.assertEqual((g.counters.auth,g.counters.libraries),(0,0))
                    finally: values[key] = before
            for target,key,value in ((auth,'__defaults__',(None,)),(auth,'__kwdefaults__',{'bad':1}),
                    (auth,'__name__','forged'),(cells['self'],'cell_contents',SimpleNamespace()),
                    (g.owner.mappings.__func__,'__code__',(lambda self:{}).__code__)):
                before = getattr(target,key); setattr(target,key,value)
                g.counters.auth = g.counters.libraries = 0
                try:
                    with self.assertRaises(ValueError): g.boundary.space['_probe_grouped_md_origin'](g.w.context)
                    self.assertEqual((g.counters.auth,g.counters.libraries),(0,0))
                finally: setattr(target,key,before)
            auth.foreign = True
            try:
                with self.assertRaises(ValueError): g.boundary.check_native()
            finally: del auth.foreign
            g.boundary.check_native()

    def test_initial_owner_api_and_source_forgery_is_not_captured_as_authority(self):
        with self.native_world() as g:
            original = g.boundary.native
            g.boundary.native = None
            for values,key,replacement in ((g.owner.fact,'sha256','0'*64),
                    (g.owner.record['library'],'path','/foreign/candidate.so'),
                    (g.owner.files,'/foreign/additional.so','0'*64)):
                present,before = key in values,values.get(key); values[key] = replacement
                try:
                    with self.assertRaisesRegex(ValueError,'exact frozen grouped native authority'):
                        g.boundary.bind_native(g.owner,g.api,g.source)
                finally:
                    if present: values[key] = before
                    else: del values[key]
            for owner,api,source in ((SimpleNamespace(),g.api,g.source),
                    (g.owner,SimpleNamespace(**vars(g.api)),g.source),
                    (g.owner,g.api,SimpleNamespace(**vars(g.source)))):
                g.boundary.native = None
                with self.assertRaises(ValueError): g.boundary.bind_native(owner,api,source)
            g.boundary.native = None
            code = g.api.authenticate.__code__
            g.api.authenticate.__code__ = code.replace(co_name='forged')
            try:
                with self.assertRaisesRegex(ValueError,'authenticator source'):
                    g.boundary.bind_native(g.owner,g.api,g.source)
            finally: g.api.authenticate.__code__ = code
            # Removing a callable from the supplied Source's cached list cannot admit new code.
            fn = g.owner.mappings.__func__; code = fn.__code__
            table = g.source.functions
            g.source.functions = [row for row in table if row[0] is not fn]
            fn.__code__ = (lambda self:{}).__code__
            try:
                with self.assertRaisesRegex(ValueError,'live source code'):
                    g.boundary.bind_native(g.owner,g.api,g.source)
            finally: fn.__code__ = code; g.source.functions = table
            g.boundary.native = original
            g.boundary.check_native()

    def test_fresh_provenance_digest_source_and_record_failures(self):
        with self.native_world() as g:
            for path in (g.provenance,g.binary,g.record,Path(g.source.fact['path'])):
                original = path.read_bytes(); path.write_bytes(original+b'changed')
                try:
                    with self.subTest(path.name), self.assertRaises(ValueError):
                        g.boundary.space['_probe_grouped_md_origin'](g.w.context)
                finally: path.write_bytes(original)
            # An unchanged digest cannot authorize a different exact grouped RECORD row.
            original = g.record.read_bytes(); g.record.write_bytes(original.replace(b'201304',b'201305'))
            try:
                g.w.context['required_guards'][str(g.record)] = file_sha(g.record)
                g.w.context['guards'][str(g.record)] = file_sha(g.record)
                with self.assertRaisesRegex(ValueError,'exact RECORD'):
                    g.boundary.space['_probe_grouped_md_origin'](g.w.context)
            finally: g.record.write_bytes(original)

    def test_original_exact_four_remains_required_and_audit_called_once(self):
        with self.native_world() as g:
            calls = []
            audit_code = g.api.audit_origins.__code__
            auth_code = g.api.authenticate.__code__
            previous = sys.getprofile()
            def profile(frame,event,arg):
                if event == 'call' and frame.f_code in (audit_code,auth_code): calls.append(frame.f_code)
            sys.setprofile(profile)
            try: g.boundary.space['_probe_grouped_md_origin'](g.w.context)
            finally: sys.setprofile(previous)
            self.assertEqual(calls.count(audit_code),0)
            self.assertEqual(calls.count(auth_code),1)
            g.api.audit_origins(g.owner.context['legacy'],require_exact=True)
            g.mapped.remove(next(iter(g.four)))
            # Grouped admission does not silently strengthen the old optional exact-four flag.
            g.boundary.space['_probe_grouped_md_origin'](g.w.context)
            with self.assertRaisesRegex(ValueError,'exact four'):
                g.api.audit_origins(g.owner.context['legacy'],require_exact=True)

    def test_private_callbacks_release_owner_and_ast_inverses(self):
        with self.native_world() as g:
            callback = g.boundary.grouped_space['_probe_grouped_native_paths']
            reference = weakref.ref(g.boundary)
            native_owner = weakref.ref(g.owner)
            opaque_owner = weakref.ref(g.owner.context['opaque_owner'])
            g.boundary.space['_probe_grouped_md_origin'](g.w.context)
        del g; gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(native_owner())
        self.assertIsNone(opaque_owner())
        with self.assertRaisesRegex(ValueError,'owner released'): callback(None,None,None,None,None)
        with self.identity_world() as w:
            original = tree_of(HERE/'evaluate_siglip2_identity_diversity.py')
            boundary = gate._InstalledIdentityBoundary(w.context,w.endpoint,w.wheel)
            grouped = copy.deepcopy(function(original,'grouped_md_origin'))
            addition = ast.parse('known |= _probe_grouped_native_paths(context, mapped, group, alias, known)').body[0]
            pos = next(i for i,n in enumerate(grouped.body) if 'grouped md mapped native witness differs' in ast.unparse(n))
            grouped.body.insert(pos,addition)
            code = compile(ast.fix_missing_locations(ast.Module(body=[grouped],type_ignores=[])),w.original.__file__,'exec',dont_inherit=True)
            self.assertEqual(next(c for c in code.co_consts if isinstance(c,type(code))),
                boundary.space['_probe_grouped_md_origin'].__code__)
            del grouped.body[pos]
            self.assertEqual(ast.dump(grouped),ast.dump(function(original,'grouped_md_origin')))
            derived = copy.deepcopy(function(original,'bundle_reads_only'))
            for n in ast.walk(derived):
                if isinstance(n,ast.Name) and n.id in ('grouped_md_origin','distribution_identity_files'): n.id = '_probe_'+n.id
            code = compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),w.original.__file__,'exec',dont_inherit=True)
            self.assertEqual(next(c for c in code.co_consts if isinstance(c,type(code))),boundary.boundary.__wrapped__.__code__)
            for n in ast.walk(derived):
                if isinstance(n,ast.Name) and n.id in ('_probe_grouped_md_origin','_probe_distribution_identity_files'): n.id = n.id[7:]
            self.assertEqual(ast.dump(derived),ast.dump(function(original,'bundle_reads_only')))

    def test_only_owned_derivative_and_install_binding_change_original_bytes(self):
        frozen = ROOT/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-installed-control-serving-v4-freeze'
        raw, original = DRIVER.read_bytes(),(frozen/DRIVER.name).read_bytes()
        self.assertEqual(SHA(original),'11019603d8cb88c339ab034c713cd8bce5a03ed429a5c4e960900aefd69467e3')
        begin,end = b'# BEGIN installed identity boundary\n',b'# END installed identity boundary\n'
        restored = raw[:raw.index(begin)]+original[original.index(begin):original.index(end)]+raw[raw.index(end):]
        binding = b'        identity_boundary.bind_native(runtime_authority,api,native_source)\n'
        self.assertEqual(restored.count(binding),1)
        self.assertEqual(restored.replace(binding,b''),original)
        self.assertEqual(SHA((HERE/'evaluate_siglip2_identity_diversity.py').read_bytes()),
            '95cb8823236408537e04108fd63727fd3a323671f04eb33a6349b23a51ce638d')
        authority_raw = (frozen.parent/'connected-control-serving-v4-freeze/native-native-authority.json').read_bytes()
        self.assertEqual(SHA(authority_raw),'cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1')
        authority = json.loads(authority_raw)
        self.assertEqual(gate._InstalledIdentityBoundary.NATIVE_AUTHORITY,
            ('/home/riomus/runs/sfora-connected-control-serving-native-authority-v5/native-authority.json',SHA(authority_raw)))
        self.assertEqual(dict(gate._InstalledIdentityBoundary.NATIVE_FILES),
            {v['file']['path']:v['file']['sha256'] for v in authority['supplemental']})

# END installed identity tests


if __name__ == '__main__':
    limit = 1024**3
    soft,hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft > limit:
        resource.setrlimit(resource.RLIMIT_AS,(limit,hard if hard != resource.RLIM_INFINITY and hard < limit else limit))
    signal.alarm(120)
    unittest.main(verbosity=1)
