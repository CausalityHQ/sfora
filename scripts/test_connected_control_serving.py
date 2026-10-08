#!/usr/bin/env python3
"""Bounded stdlib falsifier:120s/AS1GiB/fixtures16MiB; native work UNRUN."""
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors','ctypes','cupy','triton'}:
            raise AssertionError('native/thirdparty import denied: ' + fullname)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fact(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return fact(path)


def bypass_code(function):
    names = function.__code__.co_freevars
    raw = 'def factory():\n'+''.join(f'    {n}=None\n' for n in names)+\
        '    def fake():\n        return ('+','.join(names)+',) and None\n    return fake\n'
    namespace = {}; exec(raw,namespace)
    return namespace['factory']().__code__


def evaluator_fixture(root, request, f):
    """Full real evaluator exit AST; unrelated source/bundle boundaries are doubles."""
    original = ast.parse((HERE/'evaluate_siglip2_connected_mlp.py').read_bytes())
    node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
    raw = '''from pathlib import Path
FILES = TRAIN_FILES = set()
EVALUATOR_PINS = GENUINE_PINS = {}
NEAREST_EVALUATOR = {'code':{}}
def require(value, message):
    if not value: raise ValueError(message)
def merge_guards(target, values):
    for p,h in values.items(): require(target.setdefault(p,h) == h, 'guard conflict')
def closure(root, digest, names, guards):
    require(digest == 'a'*64 and not names, 'closure mutant')
    return {}
def guard_helpers(context):
    require(context['helpers_intact'], 'helper mutant')
''' + ast.unparse(node) + '\n'
    path = root/'fixture-evaluator.py'
    path.write_text(raw)
    source = request.Source.load(fact(path))
    empty = {'root':str(root),'execution_sha256':'a'*64,'code':{}}
    class Trainer:
        def batch_bound_files(self, guards, items):
            for p,h in items: request.read_file({'path':p,'sha256':h})
        def admit_bundle(self, path, digest):
            request.read_file({'path':str(path/'bundle.json'),'sha256':digest})
            return {}, {}
    class Training:
        def require_no_training(self, context):
            source.module.require(not context.get('training_state'), 'training mutant')
        def helper_guard(self, context):
            source.module.require(not context.get('helper_mutant'), 'training helper mutant')
    f.context['trainer'] = Training()
    manifest = write_json(root/'bundle.json', {})
    context = {'training_context':f.context,'trainer':Trainer(),'guards':{str(path):fact(path)['sha256']},'root':root,
        'args':SimpleNamespace(phase='export',execution_sha256='a'*64),'code':{},'helpers_intact':True,
        'launch':{**{n:copy.deepcopy(empty) for n in ('training','evaluator_reference','nearest_evaluator',
            'genuine_evaluator','reference')},'endpoints':[{'bundle':manifest}]}}
    return source, context, node


def runtime_fixture(root):
    binary = root/'cutile.so'; binary.write_bytes(b'fresh fixture binary')
    source = write_json(root/'source-manifest.json', {'ffi.rs':'a'*64})
    binaries = write_json(root/'binaries.json', {'candidate.so':fact(binary)['sha256']})
    evidence = {'source-manifest.json':source,'binaries.json':binaries}
    receipt = write_json(root/'build-receipt.json', {'schema':'sfora-cutile-threads-v1',
        'claim_eligible':False,'decision':'PASS_SEQUENTIAL_THREAD_EXACTNESS_CACHE_GATE',
        'files_sha256':{n:v['sha256'] for n,v in evidence.items()}})
    provenance = write_json(root/'provenance.json', {'schema':'connected-control-native-file-provenance-v1',
        'file':fact(binary),'kind':'archived-cutile-build','origin':'fixture archived candidate.so',
        'evidence':[receipt,binaries,source]})
    authority = {'schema':'connected-control-native-authority-v1','library':fact(binary),
        'build_receipt':receipt,'build_evidence':evidence,'source_manifest':source,
        'supplemental':[{'file':fact(binary),'provenance':provenance}]}
    return binary, authority, write_json(root/'runtime.json',authority)


def observation_fixture(root, observer):
    bundle = root/'bundle'; bundle.mkdir()
    sources = {'observer':fact(Path(observer.__file__)),
        'test':fact(HERE/'test_observe_connected_serving.py'),
        'bridge':fact(HERE.parent/'src/sfora/connected_compact_serving.py')}
    for role,name in [('trainer','train_siglip2_connected_mlp.py'),('serializer','train_siglip2_substrate_adaptation.py')]:
        path = bundle/name; path.write_bytes((HERE/name).read_bytes()); sources[role] = fact(path)
    manifest = write_json(bundle/'bundle.json',{})
    path = root/'fixture'; path.write_bytes(b'FILE metadata only')
    images = []
    for i in range(32):
        image = root/f'image{i}'; image.write_bytes(bytes([i])); images.append(fact(image))
    value = {'schema':'connected-control-serving-attribution-authority-v1','sources':sources,
        'bundle':{'directory':str(bundle),'manifest':manifest},'gallery':{'file':fact(path),'count':10},
        'native':fact(path),'train_images':images,'control_export_receipt':fact(path),'cache_conditions':'frozen diagnostic fixture',
        'resource_policy':{'body_seconds':120,'host_bytes':8*1024**3,'swap_bytes':0,
            'cuda_allocated_bytes_exclusive':10_000_000_000,'whole_process_seconds':1500,'exit_reserve_seconds':30},
        'both_locks_held':True,'qualification_eligible':False,'state_reuse_eligible':False}
    return value


class ControlTests(unittest.TestCase):
    def test_separate_source_contract(self):
        for name in ('qualify_connected_control_serving.py','connected_control_native_authority.py'):
            self.assertTrue((HERE/name).is_file(), 'missing separate control source: ' + name)

    def test_control_observation_role_and_resource_mutants(self):
        driver = load('_control_role_driver',HERE/'qualify_connected_control_serving.py')
        observer = load('_control_role_observer',HERE/'observe_connected_serving.py')
        raw = Path(observer.__file__).read_bytes(); original_prepare = observer.prepare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = observation_fixture(root,observer)
            def prepare(record):
                return driver.prepare_observation(write_json(root/'observation.json',record),observer)
            self.assertEqual(prepare(value),value)
            with self.assertRaisesRegex(ValueError,'source-only attribution'):
                observer.prepare(root/'observation.json',fact(root/'observation.json')['sha256'])
            for mutant,text in [
                ({**value,'qualified_terminal':value['control_export_receipt']},'control observation'),
                ({**value,'train_images':value['train_images'][:-1]+value['train_images'][:1]},'distinct'),
                ({**value,'resource_policy':{**value['resource_policy'],'whole_process_seconds':1501}},'cap'),
                ({**value,'resource_policy':{**value['resource_policy'],'body_seconds':121}},'cap'),
                ({**value,'resource_policy':{**value['resource_policy'],'exit_reserve_seconds':0}},'reserve'),
                ({**value,'both_locks_held':False},'lock'),({**value,'qualification_eligible':True},'control observation')]:
                with self.subTest(text=text),self.assertRaisesRegex(ValueError,text): prepare(mutant)
        self.assertIs(observer.prepare,original_prepare)
        self.assertEqual(Path(observer.__file__).read_bytes(),raw)

    def test_admit_control_uses_unchanged_unit_reader_and_fullcpu(self):
        driver = load('_control_admit_driver',HERE/'qualify_connected_control_serving.py')
        unit = {'receipt':{'path':'/receipt','sha256':driver.CONTROL_RECEIPT_SHA},'invocation_id':driver.CONTROL_INVOCATION}
        endpoint = {'arm':'control','seed':179061,'bundle':{'path':'/bundle','sha256':'a'*64}}
        context = {'launch':{'stage':'full','panel':'selection','phase':'export','arm':'control','seed':179061,
            'selected_cpu':{'receipt':{'sha256':driver.CPU_RECEIPT_SHA}},'endpoints':[endpoint]}}
        authority = {'control':dict(driver.CONTROL),'control_export':unit}
        observation = {'control_export_receipt':unit['receipt'],'bundle':{'manifest':endpoint['bundle']}}
        calls = []
        def accept(*args,**kw):
            calls.append((args,kw)); return {'original':'unchanged UNIT result'}
        result = driver.admit_control(SimpleNamespace(accept_unit=accept),context,authority,observation)
        self.assertIs(result[0],endpoint)
        self.assertEqual(calls, [((context,unit,'export','control',179061),{'stage':'full','panel':'selection'})])
        for mutant in [{'control':{'arm':'candidate','seed':179061}},
            {'control_export':{**unit,'invocation_id':'0'*32}}]:
            with self.assertRaisesRegex(ValueError,'control061'):
                driver.admit_control(SimpleNamespace(accept_unit=accept),context,authority|mutant,observation)
        with patch.dict(context['launch'],{'selected_cpu':{'receipt':{'sha256':'0'*64}}}), \
            self.assertRaisesRegex(ValueError,'fullCPU'):
            driver.admit_control(SimpleNamespace(accept_unit=accept),context,authority,observation)

    def test_missing_actual_pins_and_candidate_kill_never_import_native(self):
        driver = load('_control_missing_driver',HERE/'qualify_connected_control_serving.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); authority = {k:None for k in driver.KEYS}
            authority.update(schema=driver.SCHEMA,control=dict(driver.CONTROL))
            file = write_json(root/'authority.json',authority)
            with self.assertRaisesRegex(ValueError,'source pins'):
                driver.main(['--authority',file['path'],'--authority-sha256',file['sha256'],'--output',str(root/'out')])
            self.assertFalse((root/'out').exists())
            kill = write_json(root/'kill.json',{'decision':'KILL','pass':True,'stage':'full','panel':'selection'})
            with self.assertRaisesRegex(ValueError,'missing/KILL'):
                driver.requests.read_go({'receipt':kill})
        self.assertFalse(any(n.split('.')[0] in {'torch','numpy','PIL','sfora'} for n in sys.modules))

    def test_reused_twenty_two_call_body_oracles_ties_and_genuine_release(self):
        driver = load('_control_body_driver',HERE/'qualify_connected_control_serving.py')
        helper = load('_control_body_fixture',HERE/'test_connected_serving_requests.py')
        with tempfile.TemporaryDirectory() as directory,helper.public_fixture(Path(directory)) as f:
            report = driver.requests.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
            self.assertEqual(f.events.searches,[1]*10+[32]*10+[1,32])
            self.assertEqual(len(f.events.owners),2)
            self.assertEqual(sum(r['instrumented'] for r in report['calls']),4)
            self.assertEqual(len(f.events.closes),2)
            ties = driver.requests.native_ties(f.packed,f.gallery,f.native,f.observer)
            self.assertTrue(ties['ascending_ordinal_score_bits_exact'])
            self.assertTrue(report['same_group_original_byte_native_parity'])
            self.assertIsNone(sys.getprofile())

    def test_terminal_new_cli_and_actual_original_unit_reader(self):
        driver = load('_control_terminal_driver',HERE/'qualify_connected_control_serving.py')
        helpers = load('_control_terminal_fixture',HERE/'test_siglip2_nearest_ranking.py')
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules):
            root = Path(directory); f = helpers.NativeAdmissionFixture(root)
            prior = {'python':str(Path(sys.executable).resolve()),'python_sha256':fact(Path(sys.executable).resolve())['sha256'],
                'python_version':sys.version}
            f.legacy['selected']['source_cpu']['invocation'] = prior
            unit = {'unit':'control-terminal-fixture','invocation_id':'9'*32,'service_seconds':2,
                'native_peak_rss_kib':1001,'both_locks_held':True}
            before,after,final = (f.cgroup(unit['unit'],i) for i in (1,2,3))
            final['invocation_id'] = unit['invocation_id']
            log = [f"Running as unit: {unit['unit']}.service; invocation ID: {unit['invocation_id']}",
                '\tExit status: 0','Finished with result: success','Main processes terminated with: code=exited/status=0',
                '\tSwaps: 0','Memory swap peak: 0B','Service runtime: 2s','\tMaximum resident set size (kbytes): 1001',
                'FINAL_CGROUP '+json.dumps(final)]
            unit['log'] = f.write('terminal.log', ('\n'.join(log)+'\n').encode())
            pinned = f.write('terminal-input',b'fresh')
            authority_fact = f.write_json('terminal-authority.json',{})
            authority = {'sources':{'control_driver':fact(Path(driver.__file__))},'control_export':{'receipt':pinned}}
            policy = {'body_seconds':120,'host_bytes':8*1024**3,'swap_bytes':0,'cuda_allocated_bytes_exclusive':10_000_000_000,
                'whole_process_seconds':1500,'exit_reserve_seconds':30}
            output = root/'terminal-output'; output.mkdir()
            record = {'schema':'connected-control-serving-diagnostic-v1','status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
                'authority':authority_fact,'sources':authority['sources'],'control':dict(driver.CONTROL),
                'control_export':authority['control_export'],'output':str(output),
                'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
                **{k:False for k in ('quality_read','quality_eligible','qualification_eligible','state_reuse_eligible',
                    'optimization_eligible','product_go')},'full_uncached_exit_pass':True,'same_group_original_byte_native_parity':True,
                'calls':[{'batch':b,'kind':k} for b in (1,32) for k in ['warm_oracle','warm']+['timed']*8]+[
                    {'batch':1,'kind':'observation'},{'batch':32,'kind':'observation'}],
                'body_seconds':1,'ties':{'ascending_ordinal_score_bits_exact':True},'resource_policy':policy,
                'invocation':{**prior,'argv':driver.cli(authority_fact,str(output),driver.__file__),'optimize':0,
                    'invocation_id':unit['invocation_id'],'cuda_visible_devices':'0','cublas_workspace_config':':4096:8'},
                'resources':{'wall_seconds':1,'process_peak_rss_kib':1000,'peak_cuda_allocated_bytes':1,
                    'cgroup_before':before,'cgroup_after':after},'whole_process_seconds':1,
                'input_guards':{pinned['path']:pinned['sha256']}}
            driver.validate_receipt(record,authority,authority_fact)
            for replacement,text in [({'invocation':{**record['invocation'],'argv':['old-evaluator']}},'CLI'),
                ({'qualification_eligible':True},'discarded'),({'calls':record['calls'][:-1]},'22'),
                ({'full_uncached_exit_pass':False},'discarded')]:
                with self.assertRaisesRegex(ValueError,text): driver.validate_receipt(record|replacement,authority,authority_fact)
            unit['receipt'] = write_json(output/'receipt.json',record)
            context = {'training_context':f.context,'guards':{},'common_guards':dict(record['input_guards']),
                'terminal_reader':f.fitter.original_terminal_reader(f.context['fit_context']),
                'helper':SimpleNamespace(zero_events=f.old.zero_events)}
            # Execute the actual terminal tail with the real source-authenticated fresh UNIT reader.
            node = next(n for n in ast.parse(Path(driver.__file__).read_bytes()).body if
                isinstance(n,ast.FunctionDef) and n.name == 'accept_unit')
            start = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id == 'resources' for t in n.targets))
            tail = ast.parse('def tail(context,unit,record): pass').body[0]; tail.body = node.body[start:]
            namespace = dict(vars(driver))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[])),str(driver.__file__),'exec'),namespace)
            self.assertEqual(namespace['tail'](context,unit,record),record)
            f.legacy['invocations'].remove(unit['invocation_id'])
            for key,value,text in [('both_locks_held',False,'locks'),('service_seconds',1501,'duration'),
                ('invocation_id','8'*32,'invocation')]:
                with self.assertRaisesRegex(ValueError,text): namespace['tail'](context,unit|{key:value},record)
            path = Path(unit['log']['path']); raw = path.read_bytes()
            path.write_bytes(raw.replace(b'Exit status: 0',b'Exit status: 1'))
            bad_unit = {**unit,'log':fact(path)}
            with self.assertRaisesRegex(ValueError,'normal-exit|conflicting file'): namespace['tail'](context,bad_unit,record)
            path.write_bytes(raw)
            input_path = Path(pinned['path']); saved = input_path.stat(); input_path.write_bytes(b'stale')
            os.utime(input_path,ns=(saved.st_atime_ns,saved.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'SHA256'): namespace['tail'](context,unit,record)

    def test_publication_cleanup_failure_requires_failed_terminal(self):
        driver = load('_control_cleanup_driver',HERE/'qualify_connected_control_serving.py')
        publisher = load('_control_real_publisher',HERE/'export_siglip2_substrate_adaptation.py')
        tree = ast.parse(Path(driver.__file__).read_bytes())
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run')
        block = next(n for n in run.body if isinstance(n,ast.Try))
        tail = ast.parse('def tail(context,api,evaluator,exit_guard,guard,record,authority,authority_fact,output,policy,owned,locks): pass').body[0]
        tail.body = ast.parse('failures = []').body+block.finalbody+[ast.Return(value=ast.Name(id='record',ctx=ast.Load()))]
        namespace = dict(vars(driver)); namespace['validate_receipt'] = lambda *a:None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[])),str(driver.__file__),'exec'),namespace)
        for corrupt in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); output = root/'out'; events = []
                module = SimpleNamespace(__name__='_control_cleanup_owned')
                sys.modules[module.__name__] = module
                def publish(path,record):
                    publisher.publish(path,record); events.append('published')
                    if corrupt: sys.modules[module.__name__] = SimpleNamespace()
                context = {'guards':{},'helper':SimpleNamespace(publish=publish)}
                resources = {'wall_seconds':1,'process_peak_rss_kib':1,'peak_cuda_allocated_bytes':0}
                api = SimpleNamespace(evaluator_exit=lambda *a:events.append('full-exit'),
                    evidence=lambda:{'complete_fresh_evidence':True})
                locks = SimpleNamespace(check=lambda:events.append('locks'))
                arguments = (context,api,None,None,lambda **kw:dict(resources),{'qualification_eligible':False},
                    None,None,output,{'whole_process_seconds':1500,'exit_reserve_seconds':30},[SimpleNamespace(module=module)],locks)
                try:
                    if corrupt:
                        with self.assertRaisesRegex(ValueError,'owned source registry'): namespace['tail'](*arguments)
                        self.assertTrue((output/'receipt.json').exists())
                    else:
                        result = namespace['tail'](*arguments)
                        self.assertFalse(result['qualification_eligible'])
                        self.assertTrue(result['full_uncached_exit_pass'])
                        self.assertNotIn(module.__name__,sys.modules)
                    self.assertLess(events.index('full-exit'),events.index('published'))
                finally: sys.modules.pop(module.__name__,None)

    def test_combined_real_exit_and_mutant_matrix(self):
        self.assertTrue((HERE/'connected_control_native_authority.py').is_file(), 'combined authority missing')
        request = load('_control_fixture_requests', HERE/'qualify_connected_serving_requests.py')
        observer = load('_control_fixture_observer', HERE/'observe_connected_serving.py')
        native = load('_control_fixture_native', HERE/'connected_control_native_authority.py')
        helpers = load('_control_fixture_nearest_tests', HERE/'test_siglip2_nearest_ranking.py')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            root = Path(directory)
            f = helpers.NativeAdmissionFixture(root)
            # The historical fixture omitted packages; the genuine production collector includes it.
            source_path = Path(f.source.__file__)
            source_path.write_text(source_path.read_text().replace('    return origins\n',
                '    origins["packages"] = packages\n    return origins\n'))
            exec(compile(source_path.read_bytes(),str(source_path),'exec'),vars(f.source))
            f.context['guards'][str(source_path)] = fact(source_path)['sha256']
            f.original_state = [(m,dict(vars(m))) for m in (f.old,f.fitter,f.original,f.source,f.extract)]
            original = f.admit()
            original.audit_origins(f.legacy,require_exact=True)
            f.context['nearest'] = helpers.driver
            sys.modules[helpers.driver.__name__] = helpers.driver
            f.context['guards'][helpers.driver.__file__] = fact(Path(helpers.driver.__file__))['sha256']
            binary, runtime, runtime_fact = runtime_fixture(root)
            source, context, evaluator_node = evaluator_fixture(root, request, f)
            context['guards'][native.__file__] = fact(Path(native.__file__))['sha256']
            context['guards'][request.__file__] = fact(Path(request.__file__))['sha256']
            context['guards'][observer.__file__] = fact(Path(observer.__file__))['sha256']
            combined_files = {**f.files,str(binary):fact(binary)['sha256']}
            def maps(files=None):
                lines = []
                for p in combined_files if files is None else files:
                    value = Path(p).stat()
                    lines.append(f'1000-2000 r-xp 00000000 {os.major(value.st_dev):x}:{os.minor(value.st_dev):x} {value.st_ino} {p}')
                return '\n'.join(lines)
            actual_read_text = Path.read_text
            map_state = [maps()]
            def read_text(path, *args, **kwargs):
                return map_state[0] if str(path) == '/proc/self/maps' else actual_read_text(path,*args,**kwargs)
            with patch.object(native,'ARCHIVE_SHA',runtime['build_receipt']['sha256']), \
                 patch.object(native,'BINARY_SHA',fact(binary)['sha256']), patch.object(Path,'read_text',read_text):
                for mutant,text in [
                    ({**runtime,'supplemental':[]},'provenance'),
                    ({**runtime,'supplemental':runtime['supplemental']*2},'conflicting'),
                    ({**runtime,'supplemental':[{'file':fact(binary),'provenance':{'path':str(root/'missing'),'sha256':'a'*64}}]},'regular'),
                    ({**runtime,'build_evidence':{}},'complete archived')]:
                    mutant_fact = write_json(root/'mutant-runtime.json',mutant)
                    with self.subTest(text=text),self.assertRaisesRegex(ValueError,text):
                        native.CombinedAuthority(f.context,mutant_fact,observer,request)
                historical_file = next(iter(f.files))
                historical_fact = fact(Path(historical_file))
                conflict_proof = write_json(root/'conflict-provenance.json',{
                    'schema':'connected-control-native-file-provenance-v1','file':historical_fact,
                    'kind':'root-frozen-runtime','origin':'conflicting historical member','evidence':[runtime['build_receipt']]})
                conflict = {**runtime,'supplemental':runtime['supplemental']+[{'file':historical_fact,'provenance':conflict_proof}]}
                with self.assertRaisesRegex(ValueError,'conflicting H/S'):
                    native.CombinedAuthority(f.context,write_json(root/'conflicting-runtime.json',conflict),observer,request)
                f.set_origins(combined_files)
                with self.assertRaisesRegex(ValueError,'unknown or changed'):
                    original.audit_origins(f.legacy,require_exact=True)
                authority = native.CombinedAuthority(f.context,runtime_fact,observer,request)
                api = authority.install(source,context)
                saved_owner = f.context['native_source_owned']
                api.audit_origins(f.legacy,require_exact=True)
                with self.assertRaisesRegex(ValueError,'original CPU'):
                    api.audit_origins(f.legacy,initial=True)
                self.assertEqual(f.legacy['origins']['files'],f.files)
                self.assertEqual(api.evidence()['inventory']['files'],combined_files)
                def exit():
                    f.context['fit_context']['phase_seconds'].clear()
                    with redirect_stdout(io.StringIO()): api.evaluator_exit(context,None)
                exit()
                self.assertIs(f.context['native_source_owned'],saved_owner)
                self.assertEqual(ast.dump(api.asts['evaluator_original'],include_attributes=False),
                                 ast.dump(evaluator_node,include_attributes=False))
                for name,(before,after,substitutions) in api.asts['inverses'].items():
                    self.assertEqual(before,after,name)
                    self.assertEqual(substitutions, {'audit':2,'quadratic':0,'fitter':1,'evaluator':1}[name])
                for files, mapped, modules, text in [
                    ({**combined_files,f.bulk[0]['path']:f.bulk[0]['sha256']},list(combined_files),{},'unknown'),
                    (combined_files,list(f.files),{},'mapping'),
                    (combined_files,list(combined_files),{'torch.foreign':str(binary)},'module'),
                    (f.files,list(f.files),{},'supplemental')]:
                    f.set_origins(files,mapped,modules)
                    with self.assertRaisesRegex(ValueError,text): exit()
                f.set_origins(combined_files)
                reduced = {p:h for p,h in combined_files.items() if p != historical_file}
                f.set_origins(reduced)
                map_state[0] = maps(reduced)
                with self.assertRaisesRegex(ValueError,'exact four'): exit()
                f.set_origins(combined_files); map_state[0] = maps()
                for suffix,text in [(' (deleted)','mapping'),('', 'inode')]:
                    saved = map_state[0]
                    map_state[0] = saved.replace(str(binary),str(binary)+suffix) if suffix else \
                        saved.replace(str(binary.stat().st_ino)+' '+str(binary),'1 '+str(binary))
                    with self.assertRaisesRegex(ValueError,text): exit()
                    map_state[0] = saved
                for p in (binary,Path(runtime['build_receipt']['path']),Path(runtime['supplemental'][0]['provenance']['path']),
                          Path(runtime['build_evidence']['binaries.json']['path']),Path(f.source.__file__)):
                    raw,saved = p.read_bytes(),p.stat()
                    try:
                        p.write_bytes(b'x'*len(raw)); os.utime(p,ns=(saved.st_atime_ns,saved.st_mtime_ns))
                        with self.assertRaisesRegex(ValueError,'SHA256'): exit()
                    finally: p.write_bytes(raw)
                # A valid ordinary guard never grants native origin membership.
                context['guards'][f.bulk[0]['path']] = f.bulk[0]['sha256']
                f.set_origins({**combined_files,f.bulk[0]['path']:f.bulk[0]['sha256']})
                with self.assertRaisesRegex(ValueError,'unknown'): exit()
                f.set_origins(combined_files)
                for key,text in [('training_state','training mutant'),('helper_mutant','training helper')]:
                    with patch.dict(f.context,{key:True}),self.assertRaisesRegex(ValueError,text): exit()
                with patch.dict(context,{'helpers_intact':False}),self.assertRaisesRegex(ValueError,'helper mutant'): exit()
                prior = f.legacy['selected']['genuine']['prior']
                prior['images'] = ['bad FIT']
                with self.assertRaisesRegex(ValueError,'FIT image resolution'): exit()
                prior['images'] = []
                genuine = f.legacy['selected']['genuine']
                genuine['selected']['original_rows'] = [0]
                with self.assertRaisesRegex(ValueError,'TRAIN mapping'): exit()
                genuine['selected']['original_rows'] = []
                with patch.object(source.module,'guard_helpers',lambda *a:None), \
                        self.assertRaisesRegex(ValueError,'source changed'): exit()
                with patch.object(f.source,'imported_origins',lambda *a:{}), \
                        self.assertRaisesRegex(ValueError,'binding changed|dependency changed'): exit()
                with patch.dict(f.context['control_native_owned'],{'authenticate':lambda:None}), \
                        self.assertRaisesRegex(ValueError,'ownership'): exit()
                original_auth_code = api.authenticate.__code__
                try:
                    api.authenticate.__code__ = bypass_code(api.authenticate)
                    with self.assertRaisesRegex(ValueError,'authentication code'): exit()
                finally: api.authenticate.__code__ = original_auth_code
                original_auth = f.context['native_source_owned']['authenticate']
                original_auth_code = original_auth.__code__
                try:
                    original_auth.__code__ = bypass_code(original_auth)
                    with self.assertRaisesRegex(ValueError,'original native authentication'): exit()
                finally: original_auth.__code__ = original_auth_code
                with patch.object(authority,'historical',{'files':{},'modules':{}}), \
                        self.assertRaisesRegex(ValueError,'binding changed'): exit()
                with patch.object(api.evaluator_exit,'__defaults__',(None,)), \
                        self.assertRaisesRegex(ValueError,'private function'): exit()
                private_evaluator = next(c.cell_contents for c in api.evaluator_exit.__closure__ if
                    isinstance(c.cell_contents,request.FunctionType) and c.cell_contents.__name__ == 'exit_rehash')
                with patch.dict(private_evaluator.__globals__,{'_control_native_api':lambda *a:api}), \
                        self.assertRaisesRegex(ValueError,'namespace binding'): exit()
                # A mutation arriving at the final bundle boundary is caught by the final audit.
                real_admit = context['trainer'].admit_bundle
                raw = binary.read_bytes()
                def mutate(path,digest):
                    result = real_admit(path,digest)
                    binary.write_bytes(b'x'*len(raw))
                    return result
                with patch.object(context['trainer'],'admit_bundle',mutate),self.assertRaisesRegex(ValueError,'SHA256'): exit()
                binary.write_bytes(raw)
                exit()
                f.unchanged_originals(self)
                self.assertLess(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),16*1024**2)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (1024**3,1024**3))
    signal.alarm(120)
    sys.meta_path.insert(0, NoNative())
    unittest.main()
