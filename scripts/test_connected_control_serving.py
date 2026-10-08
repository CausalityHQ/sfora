#!/usr/bin/env python3
"""Bounded stdlib falsifier:120s/AS1GiB/fixtures16MiB; native work UNRUN."""
import ast
import copy
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

COMPILER_VALIDATOR = '''def validate_runtime_compiler(record, observer):
    require(type(record) is dict and record.keys() ==
        {'schema','library','build_receipt','build_evidence','source_manifest','supplemental','runtime_compiler'} and
        record['schema'] == 'connected-control-native-authority-v2', 'exact combined native authority required')
    file = record['runtime_compiler']
    observer.file_bytes(file)
    path = Path(file['path'])
    require(stat.S_ISREG(path.stat().st_mode) and os.access(path,os.X_OK), 'regular executable runtime compiler FILE required')
    require(os.environ.get('CUTILE_TILEIRAS_PATH') == file['path'], 'exact CUTILE_TILEIRAS_PATH binding required')


'''
ORIGIN_DIAGNOSTIC = '''        changed = [{'path':p,'actual_sha256':h,'expected_sha256':union.get(p)}
            for p,h in origins['files'].items() if union.get(p) != h]
        if changed:
            print(json.dumps({'schema':'connected-control-native-origin-rejection-v1',
                'total_count':len(changed),'origins':changed[:32]},sort_keys=True),file=sys.stderr,flush=True)
'''
PRODUCTION_DELTAS = {
    'qualify_connected_control_serving.py': ('d76376418deef511de97b7c516bb0d8f6eca0c1cd55e49a3898ce72da763517c', [
        ('', "        native_source.module.validate_runtime_compiler(runtime,observer)\n")]),
    'connected_control_native_authority.py': ('2a9a1b4f8cb7303e4d8f58dbcb0dad37699548385de374b2c306afcc30bb9985', [
        ('connected-control-native-authority-v1 FILE with\nexact keys schema/library/build_receipt/build_evidence/source_manifest/supplemental.',
         'connected-control-native-authority-v2 FILE with\nexact keys schema/library/build_receipt/build_evidence/source_manifest/supplemental/runtime_compiler.\n'
         'runtime_compiler is an executable FILE bound to CUTILE_TILEIRAS_PATH, never a native grant.'),
        ('import os\n', 'import json\nimport os\n'),
        ('', COMPILER_VALIDATOR),
        ("        require(type(record) is dict and record.keys() ==\n"
         "            {'schema','library','build_receipt','build_evidence','source_manifest','supplemental'} and\n"
         "            record['schema'] == 'connected-control-native-authority-v1', 'exact combined native authority required')\n",
         '        validate_runtime_compiler(record,self.observer)\n'),
        ("facts = [self.fact,record['library'],", "facts = [self.fact,record['runtime_compiler'],record['library'],"),
        ('', ORIGIN_DIAGNOSTIC)])}


def production_inverse(raw, digest, changes):
    for before,after in reversed(changes):
        if raw.count(after) != 1: raise ValueError('exact production delta differs')
        raw = raw.replace(after,before,1)
    if hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() != digest:
        raise ValueError('whole production AST inverse differs')


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
    compiler = root/'tileiras'; compiler.write_bytes(b'compiler FILE fixture; never executed'); compiler.chmod(0o755)
    source = write_json(root/'source-manifest.json', {'ffi.rs':'a'*64})
    binaries = write_json(root/'binaries.json', {'candidate.so':fact(binary)['sha256']})
    evidence = {'source-manifest.json':source,'binaries.json':binaries}
    receipt = write_json(root/'build-receipt.json', {'schema':'sfora-cutile-threads-v1',
        'claim_eligible':False,'decision':'PASS_SEQUENTIAL_THREAD_EXACTNESS_CACHE_GATE',
        'files_sha256':{n:v['sha256'] for n,v in evidence.items()}})
    provenance = write_json(root/'provenance.json', {'schema':'connected-control-native-file-provenance-v1',
        'file':fact(binary),'kind':'archived-cutile-build','origin':'fixture archived candidate.so',
        'evidence':[receipt,binaries,source]})
    authority = {'schema':'connected-control-native-authority-v2','library':fact(binary),'runtime_compiler':fact(compiler),
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


def measurement_fixture(root, driver):
    helper = load('_control_measurement_fixture',HERE/'test_connected_serving_requests.py')
    with helper.public_fixture(root) as f:
        diagnostic = driver.requests.request_body(f.factory,f.observer,f.reader,lambda:None,f.paths,f.pins,lambda:None)
        ties = driver.requests.native_ties(f.packed,f.gallery,f.native,f.observer)
    authority = {'sources':{'control_driver':fact(Path(driver.__file__))},'control_export':{'receipt':fact(root/'native.so')}}
    authority_fact = write_json(root/'authority.json',authority)
    record = {'schema':'connected-control-serving-diagnostic-v1','status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
        'authority':authority_fact,'sources':authority['sources'],'control':dict(driver.CONTROL),
        'control_export':authority['control_export'],'output':str(root/'out'),
        **{k:False for k in ('quality_read','quality_eligible','qualification_eligible','state_reuse_eligible',
            'optimization_eligible','product_go')},'full_uncached_exit_pass':True,
        'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,'ties':ties,**diagnostic,
        'invocation':{'argv':driver.cli(authority_fact,str(root/'out'),driver.__file__),'optimize':0,
            'cuda_visible_devices':'0','cublas_workspace_config':':4096:8'}}
    return record,authority,authority_fact


def terminal_fixture(root, f, name, invocation, record):
    unit = {'unit':name,'invocation_id':invocation,'service_seconds':2,'native_peak_rss_kib':1001,'both_locks_held':True}
    before,after,final = (f.cgroup(name,i) for i in (1,2,3))
    final['invocation_id'] = invocation
    log = [f'Running as unit: {name}.service; invocation ID: {invocation}', '\tExit status: 0',
        'Finished with result: success','Main processes terminated with: code=exited/status=0','\tSwaps: 0',
        'Memory swap peak: 0B','Service runtime: 2s','\tMaximum resident set size (kbytes): 1001','FINAL_CGROUP '+json.dumps(final)]
    path = root/(name+'.log'); path.write_text('\n'.join(log)+'\n'); unit['log'] = fact(path)
    record.update(cgroup_before=before,cgroup_after=after,wall_seconds=1,process_peak_rss_kib=1000)
    record['invocation']['invocation_id'] = invocation
    return unit


@contextmanager
def genuine_exit_fixture(root):
    """Original source functions throughout; only files, maps and package data are fixtures."""
    request = load('_genuine_requests',HERE/'qualify_connected_serving_requests.py')
    observer = load('_genuine_observer',HERE/'observe_connected_serving.py')
    native = load('_genuine_native',HERE/'connected_control_native_authority.py')
    helpers = load('_genuine_nearest_tests',HERE/'test_siglip2_nearest_ranking.py')
    f = helpers.NativeAdmissionFixture(root)
    virtual = {}
    def closure(name, names, pins=None):
        directory = root/name; directory.mkdir()
        for filename in names:
            candidates = [HERE/filename]
            if pins and (not candidates[0].is_file() or fact(candidates[0])['sha256'] != pins[filename]):
                candidates = list((HERE.parent/'docs/evidence').rglob(filename))
            path = next(p for p in candidates if p.is_file() and (not pins or fact(p)['sha256'] == pins[filename]))
            (directory/filename).write_bytes(path.read_bytes())
        code = {n:fact(directory/n)['sha256'] for n in names}
        execution = write_json(directory/'execution.json',code)
        return {'root':str(directory),'execution_sha256':execution['sha256'],'code':code}
    def module(name, path):
        actual = virtual.get(str(path),path)
        spec = importlib.util.spec_from_file_location(name,path)
        value = importlib.util.module_from_spec(spec); sys.modules[name] = value
        exec(compile(actual.read_bytes(),str(path),'exec',dont_inherit=True),vars(value))
        return value
    source = module('_genuine_collector',HERE/'qualify_siglip2_substrate_cpu.py')
    src = closure('collector',source.FILES)
    extract = module('extract_siglip2_vision_source',Path(src['root'])/'extract_siglip2_vision_source.py')
    f.source = source; f.extract = extract
    f.legacy.update(source_driver=source,extract=extract)
    exporter = module('_genuine_exporter',HERE/'export_siglip2_genuine_views.py')
    reference = module('_genuine_fit_reference',HERE/'export_siglip2_substrate_fit.py')
    ref = closure('fit-reference',reference.FILES)
    export = closure('genuine-export',exporter.FILES)
    images = root/'Img'; images.mkdir(); (root/'images').rename(images/'img')
    targets,panels,offset,base = [],{},0,0
    for name,count,classes,query in [('train',6355,1008,0),('selection',3449,498,1734),('validation',3479,498,1749)]:
        targets.extend(base+i%classes for i in range(count))
        panels[name] = {'original_rows':list(range(offset,offset+count)),'original_class_ids':list(range(base,base+classes))}
        if query: panels[name].update(query=list(range(query)),gallery=list(range(query,count)))
        offset += count; base += classes
    fit = {'schema':'native256-frozen-fit-manifest-v1','fit_images':13283,'fit_identities':2004,'held_images_read':0,
        'quality_read':False,'source_features_reused':False,'teacher_state_reused':False,'dataset_root':str(root),
        'class_names':[str(i) for i in range(2004)],'targets':targets,
        'rows':[{'relative_path':f'Img/img/{i}','train_row':i,'product':str(target),'image_sha256':hashlib.sha256(b'').hexdigest()}
            for i,target in enumerate(targets)]}
    partition = {'schema':'siglip2-identity-mix-partition-v1','global_class_names':fit['class_names'],
        'original_cache':{'sha256':exporter.OLD_CACHE_SHA},'original_fit':{'sha256':exporter.MANIFEST_SHA},
        'panels':panels,'partition_seeds':[179071,179072]}
    prior = f.legacy['prior']
    prior.update(source_driver=source,extract=extract,fit=fit,root=Path(src['root']),code=src['code'],
        args=SimpleNamespace(execution_sha256=src['execution_sha256']),own_root=Path(ref['root']),own_code=ref['code'],
        export_args=SimpleNamespace(execution_sha256=ref['execution_sha256']),images=source.fit_rows(extract,fit),
        all_images=[root/row['relative_path'] for row in fit['rows']])
    genuine = f.legacy['selected']['genuine']
    genuine.update(reference=reference,root=Path(export['root']),code=export['code'],
        args=SimpleNamespace(execution_sha256=export['execution_sha256']),
        launch={'partition':write_json(root/'partition.json',partition),'image_rows':fact(HERE/'extract_siglip2_vision_source.py')},
        selected=exporter.selected_manifest(partition,fit))
    # The original ImageRows AST lives in the initializer's authenticated reference.
    candidates = [p for p in HERE.glob('*.py') if b'class ImageRows' in p.read_bytes()]
    for path in candidates:
        try: exporter.image_rows_node(str(path))
        except ValueError: continue
        image_source = path; break
    else: raise AssertionError('genuine ImageRows source unavailable')
    genuine['launch']['image_rows'] = fact(image_source)
    genuine['selected']['resolved_paths'] = [str(prior['all_images'][i]) for i in genuine['selected']['original_rows']]
    f.legacy['selected']['exporter'] = exporter
    training = module('_genuine_identity_trainer',HERE/'train_siglip2_identity_diversity.py')
    nearest_path = Path(training.NEAREST['root'])/'train_siglip2_nearest_ranking.py'
    virtual[str(nearest_path)] = HERE/nearest_path.name
    nearest = module('_compact_nearest',nearest_path)
    f.context.update(nearest=nearest,trainer=training)
    fit_context = f.context['fit_context']; fit_root = Path(training.READOUT['path']).parent
    for name in f.fitter.FILES | {'execution.json'}: virtual[str(fit_root/name)] = fit_context['root']/name
    fit_context['root'] = fit_root
    fit_context.pop('readout_authentication'); fit_context.pop('readout'); sys.modules.pop('_prototype_signed_readout')
    evaluator_source = native.load_evaluator_source(fact(HERE/'evaluate_siglip2_connected_mlp.py'),request)
    evaluator = evaluator_source.module
    own = closure('evaluator',evaluator.FILES)
    descriptors = {
        'training':closure('training',evaluator.TRAIN_FILES),
        'evaluator_reference':closure('identity-evaluation',evaluator.EVALUATOR_PINS,evaluator.EVALUATOR_PINS),
        'nearest_evaluator':closure('nearest-evaluation',evaluator.NEAREST_EVALUATOR['code'],evaluator.NEAREST_EVALUATOR['code']),
        'genuine_evaluator':closure('genuine-evaluation',evaluator.GENUINE_PINS,evaluator.GENUINE_PINS),
        'reference':closure('reference',evaluator.REFERENCE['code'],evaluator.REFERENCE['code'])}
    original = closure('original-evaluator',evaluator.FILES,evaluator.ORIGINAL_EXPORT_OWNER['code'])
    # Keep the frozen remote CODE descriptor; its bytes are supplied at the filesystem boundary.
    for name in evaluator.FILES: virtual[str(Path(evaluator.ORIGINAL_EXPORT_OWNER['root'])/name)] = Path(original['root'])/name
    archived = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    execution = next(p for p in archived.rglob('execution.json') if fact(p)['sha256'] == evaluator.ORIGINAL_EXPORT_OWNER['execution_sha256'])
    virtual[str(Path(evaluator.ORIGINAL_EXPORT_OWNER['root'])/'execution.json')] = execution
    bundle = root/'bundle'; bundle.mkdir()
    trainer = module('_genuine_connected_trainer',HERE/'train_siglip2_connected_mlp.py')
    code = trainer.FILES | trainer.SERVING_FILES | {'joint_relational_compaction.py'}
    for name in code:
        path = HERE/name if (HERE/name).is_file() else HERE.parent/'src/sfora'/name
        (bundle/name).write_bytes(path.read_bytes())
    for name in ('vision.pt','endpoint.pt','processor.json'): (bundle/name).write_bytes(b'bounded data fixture')
    environment_file = fact(bundle/'processor.json')
    manifest = write_json(bundle/'bundle.json',{'schema':trainer.BUNDLE_SCHEMA,'code':{n:fact(bundle/n)['sha256'] for n in code},
        'files':{n:fact(bundle/n)['sha256'] for n in ('vision.pt','endpoint.pt','processor.json')},
        'endpoint_state_sha256':'a'*64,'base_vision_sha256':'b'*64,'vision_sha256':'c'*64,'encoder_identity':{},
        'scope':{'arm':'control','manifest_sha256':trainer.SCOPE_SHA256,'arm_sha256':trainer.CONTROL_SHA256},
        'environment':{'packages':{n:{'root':str(bundle)} for n in trainer.NATIVE-{'sfora'}},
            'files':{environment_file['path']:environment_file['sha256']},'native_files':{},'vision_constructor':environment_file['path']}})
    context = {'training_context':f.context,'trainer':trainer,'guards':{},'root':Path(own['root']),'code':own['code'],
        'args':SimpleNamespace(phase='export',execution_sha256=own['execution_sha256']),
        'launch':{**descriptors,'endpoints':[{'bundle':manifest}]},'original_evaluator':module('_genuine_original',Path(original['root'])/'evaluate_siglip2_connected_mlp.py')}
    for role,key,filename in [('evaluator_reference','evaluator_reference','evaluate_siglip2_identity_diversity.py'),
        ('nearest_evaluator','nearest_evaluator','evaluate_siglip2_nearest_ranking.py'),
        ('math','genuine_evaluator','evaluate_siglip2_genuine_views.py'),('reference','reference','evaluate_siglip2_prototype_residual.py')]:
        context[role] = module('_genuine_'+role,Path(descriptors[key]['root'])/filename)
    context['helper'] = module('_genuine_export_helper',HERE/'export_siglip2_substrate_adaptation.py')
    context['baseline'] = module('_genuine_baseline',HERE/'evaluate_siglip2_quadratic_readout.py')
    for value in (source,extract,nearest,training,request,observer,native,evaluator,*[context[k] for k in
        ('trainer','evaluator_reference','nearest_evaluator','math','reference','helper','baseline','original_evaluator')]):
        path = Path(value.__file__); digest = fact(virtual.get(str(path),path))['sha256']
        f.context['guards'][str(path)] = context['guards'][str(path)] = digest
    binary,runtime,runtime_fact = runtime_fixture(root)
    map_paths = [*f.files,str(binary)]
    def maps():
        lines = []
        for p in map_paths:
            value = Path(p.removesuffix(' (deleted)')).stat()
            lines.append(f'1000-2000 r-xp 0 {os.major(value.st_dev):x}:{os.minor(value.st_dev):x} {value.st_ino} {p}')
        return '\n'.join(lines)
    actual_open,actual_stat,actual_is_dir,actual_is_file = Path.open,Path.stat,Path.is_dir,Path.is_file
    map_text = [None]
    def open_file(path,*args,**kwargs):
        if str(path) == '/proc/self/maps': return io.StringIO(maps() if map_text[0] is None else map_text[0])
        return actual_open(virtual.get(str(path),path),*args,**kwargs)
    def stat_file(path,*args,**kwargs): return actual_stat(virtual.get(str(path),path),*args,**kwargs)
    virtual_dirs = {str(Path(p).parent) for p in virtual}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ,{'CUTILE_TILEIRAS_PATH':runtime['runtime_compiler']['path']}))
        stack.enter_context(patch.object(Path,'open',open_file))
        stack.enter_context(patch.object(Path,'stat',stat_file))
        stack.enter_context(patch.object(Path,'is_file',lambda p:actual_is_file(virtual.get(str(p),p))))
        stack.enter_context(patch.object(Path,'is_dir',lambda p:str(p) in virtual_dirs or actual_is_dir(p)))
        stack.enter_context(patch.object(native,'ARCHIVE_SHA',runtime['build_receipt']['sha256']))
        stack.enter_context(patch.object(native,'BINARY_SHA',fact(binary)['sha256']))
        packages = {}
        for name in source.PACKAGES:
            directory = root/'packages'/name; directory.mkdir(parents=True)
            path = directory/'__init__.py'; path.write_bytes(b'# package origin data only\n')
            value = ModuleType(name); value.__file__ = str(path); sys.modules[name] = value
            packages[name] = {'root':str(directory)}
        f.legacy['selected']['packages'] = packages
        map_text[0] = ''
        f.legacy['selected']['source_cpu']['origins'] = source.imported_origins(extract,packages)
        map_text[0] = None
        with patch.object(nearest,'NATIVE_PROOF_PINS',f.pins): original_api = nearest.native_source_api(f.context)
        authority = native.CombinedAuthority(f.context,runtime_fact,observer,request)
        api = authority.install(evaluator_source,context)
        guard = evaluator.admission_exit_guard(evaluator.guard_helpers)
        yield SimpleNamespace(f=f,request=request,observer=observer,native=native,source=evaluator_source,evaluator=evaluator,
            context=context,api=api,authority=authority,original_api=original_api,binary=binary,runtime=runtime,runtime_fact=runtime_fact,
            map_paths=map_paths,map_text=map_text,maps=maps,guard=guard,own=own,partition=partition,virtual=virtual)


class ControlTests(unittest.TestCase):
    def test_exact_production_delta_and_whole_ast_inverse(self):
        for name,(digest,changes) in PRODUCTION_DELTAS.items():
            raw = (HERE/name).read_text()
            with self.subTest(source=name): production_inverse(raw,digest,changes)
            for before,after in changes:
                with self.subTest(delta=after),self.assertRaisesRegex(ValueError,'delta'):
                    production_inverse(raw.replace(after,after[:-1]+'?',1),digest,changes)
            with self.subTest(retained=name),self.assertRaisesRegex(ValueError,'AST'):
                production_inverse(raw.replace('raise ValueError(message)','raise RuntimeError(message)',1),digest,changes)

    def test_compiler_rejects_invalid_binding_before_evaluator_authority(self):
        driver = load('_control_compiler_driver',HERE/'qualify_connected_control_serving.py')
        native = load('_control_compiler_native',HERE/'connected_control_native_authority.py')
        observer = load('_control_compiler_observer',HERE/'observe_connected_serving.py')
        run = next(n for n in ast.parse(Path(driver.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name == 'run')
        body = next(n.body for n in run.body if isinstance(n,ast.Try))
        start = next(i+1 for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'runtime')
        end = next(i+1 for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.value) == 'evaluator.authority(eargs)')
        code = compile(ast.Module(body=copy.deepcopy(body[start:end]),type_ignores=[]),driver.__file__,'exec')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _,runtime,_ = runtime_fixture(root)
            compiler = Path(runtime['runtime_compiler']['path']); raw = compiler.read_bytes()
            link = root/'compiler-link'; link.symlink_to(compiler)
            directory_file = root/'compiler-directory'; directory_file.mkdir()
            bad = root/'bad-compiler'; bad.write_bytes(b'wrong compiler'); bad.chmod(0o755)
            entered = []
            evaluator = SimpleNamespace(FILES=set(),check_code=lambda *a:None,closure=lambda *a:{},
                authority=lambda args:(entered.append(args) or {},None))
            descriptor = {'root':str(root),'execution_sha256':'a'*64,'code':{'evaluate_siglip2_connected_mlp.py':'b'*64}}
            evaluator.closure = lambda *a:descriptor['code']
            def admit(record,env):
                namespace = {**vars(driver),'runtime':record,'observer':observer,'observation':{'native':runtime['library']},
                    'native_source':SimpleNamespace(module=native),'owned':[],'output':root/'out',
                    'authority':{'evaluator':descriptor,'evaluation_authority':{'path':str(root/'evaluation.json'),'sha256':'a'*64}}}
                with patch.dict(os.environ,env,clear=True),patch.object(native,'load_evaluator_source',
                        lambda *a:SimpleNamespace(module=evaluator)):
                    exec(code,namespace)
            cases = [
                ('missing env',runtime,{}),('wrong env',runtime,{'CUTILE_TILEIRAS_PATH':str(bad)}),
                ('old schema',{**runtime,'schema':'connected-control-native-authority-v1'},None),
                ('missing FILE',{k:v for k,v in runtime.items() if k != 'runtime_compiler'},None),
                ('wrong hash',{**runtime,'runtime_compiler':{**runtime['runtime_compiler'],'sha256':'0'*64}},None),
                ('missing path',{**runtime,'runtime_compiler':{**runtime['runtime_compiler'],'path':str(root/'missing')}},None),
                ('relative path',{**runtime,'runtime_compiler':{**runtime['runtime_compiler'],'path':'tileiras'}},None),
                ('symlink',{**runtime,'runtime_compiler':{**runtime['runtime_compiler'],'path':str(link)}},None),
                ('directory',{**runtime,'runtime_compiler':{**runtime['runtime_compiler'],'path':str(directory_file)}},None)]
            for label,record,env in cases:
                entered.clear()
                env = env if env is not None else {'CUTILE_TILEIRAS_PATH':record.get('runtime_compiler',runtime['runtime_compiler'])['path']}
                with self.subTest(mutant=label),self.assertRaises(ValueError): admit(record,env)
                self.assertEqual(entered,[],label)
            for label in ('mode','changed bytes'):
                entered.clear()
                try:
                    if label == 'mode': compiler.chmod(0o644)
                    else: compiler.write_bytes(b'x'*len(raw))
                    with self.subTest(mutant=label),self.assertRaises(ValueError):
                        admit(runtime,{'CUTILE_TILEIRAS_PATH':str(compiler)})
                    self.assertEqual(entered,[],label)
                finally: compiler.write_bytes(raw); compiler.chmod(0o755)
            admit(runtime,{'CUTILE_TILEIRAS_PATH':str(compiler)})
            self.assertEqual(len(entered),1)

    def test_origin_diagnostic_is_bounded_and_rejects_real_unknown_or_changed_hashes(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules),genuine_exit_fixture(Path(directory)) as g:
            packages = g.f.legacy['selected']['packages']
            def rejected():
                stderr = io.StringIO()
                with redirect_stderr(stderr),self.assertRaisesRegex(ValueError,'unknown or changed combined native origin'):
                    g.authority.collect(g.f.extract,packages)
                self.assertTrue(stderr.getvalue(),'origin rejection omitted metadata')
                return json.loads(stderr.getvalue())
            unknown = []
            for i in range(35):
                path = Path(directory)/f'unknown{i}.so'; path.write_bytes(bytes([i])); unknown.append(path)
            g.map_paths.extend(str(p) for p in unknown)
            report = rejected()
            self.assertEqual(report['schema'],'connected-control-native-origin-rejection-v1')
            self.assertEqual(report['total_count'],35)
            self.assertEqual(len(report['origins']),32)
            for row in report['origins']:
                self.assertEqual(row,{'path':str(Path(row['path']).resolve()),
                    'actual_sha256':fact(Path(row['path']))['sha256'],'expected_sha256':None})
            del g.map_paths[-35:]
            historical = Path(g.map_paths[0]); raw = historical.read_bytes(); expected = fact(historical)['sha256']
            try:
                historical.write_bytes(b'changed historical bytes')
                self.assertEqual(rejected()['origins'],[{'path':str(historical),
                    'actual_sha256':fact(historical)['sha256'],'expected_sha256':expected}])
            finally: historical.write_bytes(raw)
            # Change S after the initial fresh FILE check; the genuine collector sees its new bytes.
            raw = g.binary.read_bytes(); expected = fact(g.binary)['sha256']; actual_open = Path.open
            def change_at_maps(path,*args,**kwargs):
                if str(path) == '/proc/self/maps': g.binary.write_bytes(b'changed supplemental bytes')
                return actual_open(path,*args,**kwargs)
            try:
                with patch.object(Path,'open',change_at_maps): report = rejected()
                self.assertEqual(report['origins'],[{'path':str(g.binary),
                    'actual_sha256':fact(g.binary)['sha256'],'expected_sha256':expected}])
            finally: g.binary.write_bytes(raw)
            self.assertNotIn(g.runtime['runtime_compiler']['path'],g.authority.files)
            g.authority.collect(g.f.extract,packages)

    def test_compiler_binding_and_current_bytes_survive_full_original_exit(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules),genuine_exit_fixture(Path(directory)) as g:
            file = g.runtime['runtime_compiler']; compiler = Path(file['path']); raw = compiler.read_bytes()
            self.assertIn(file,g.authority.provenance_facts())
            self.assertNotIn(str(compiler),g.authority.files)
            g.evaluator.merge_guards(g.context['guards'],{v['path']:v['sha256'] for v in g.authority.provenance_facts()})
            self.assertEqual(g.context['guards'][str(compiler)],file['sha256'])
            def exit():
                g.f.context['fit_context']['phase_seconds'].clear()
                with redirect_stdout(io.StringIO()): g.api.evaluator_exit(g.context,g.guard)
            exit()
            with patch.dict(os.environ,{'CUTILE_TILEIRAS_PATH':str(compiler)+'.wrong'}), \
                    self.assertRaisesRegex(ValueError,'CUTILE_TILEIRAS_PATH'): exit()
            try:
                compiler.chmod(0o644)
                with self.assertRaisesRegex(ValueError,'executable'): exit()
            finally: compiler.chmod(0o755)
            try:
                saved = compiler.stat(); compiler.write_bytes(b'x'*len(raw))
                os.utime(compiler,ns=(saved.st_atime_ns,saved.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'): exit()
            finally: compiler.write_bytes(raw)
            exit()

    def test_genuine_collector_and_real_proc_maps_agree(self):
        source = load('_control_real_maps_collector',HERE/'qualify_siglip2_substrate_cpu.py')
        extract = load('_control_real_maps_extract',HERE/'extract_siglip2_vision_source.py')
        native = load('_control_real_maps_native',HERE/'connected_control_native_authority.py')
        authority = native.CombinedAuthority.__new__(native.CombinedAuthority)
        mappings = authority.mappings()
        origins = source.imported_origins(extract,{})
        self.assertTrue(mappings)
        self.assertEqual(set(origins['native_files']),set(mappings))
        self.assertEqual(origins['files'],{p:fact(Path(p))['sha256'] for p in mappings})

    def test_complete_parent_acceptance_with_real_evaluator_and_terminal_reader(self):
        driver = load('_control_complete_parent',HERE/'qualify_connected_control_serving.py')
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules),genuine_exit_fixture(Path(directory)) as g:
            root = Path(directory); context = g.context; e = g.evaluator; f = g.f
            g.api.audit_origins(f.legacy,require_exact=True)
            combined = g.api.evidence()
            # Parent admission runs before native imports; receipt evidence describes the finished child.
            g.map_paths.clear()
            archived = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
            exported = json.loads((archived/'connected-mlp-evaluation-full-export-control-179061-v2/receipt.json').read_bytes())
            manifest = context['launch']['endpoints'][0]['bundle']
            bundle = Path(manifest['path']).parent
            bundle.rename(root/'control-179061-bundle')
            manifest = fact(root/'control-179061-bundle'/'bundle.json')
            launch = exported['launch']; launch['execution_sha256'] = g.own['execution_sha256']
            endpoint = next(v for v in launch['endpoints'] if (v['arm'],v['seed']) == ('control',179061))
            endpoint['bundle'] = manifest
            endpoint['terminal']['receipt']['path'] = str(root/'endpoint-receipt.json')
            endpoint['checkpoint']['path'] = str(root/'control-179061-terminal.pt')
            exported.update(source_code=g.own['code'],execution_sha256=g.own['execution_sha256'],source=f.context['source'],
                binding=e.binding({'launch':launch}),origins=combined['historical_projection'])
            export_dir = root/'export'; export_dir.mkdir(); exported['output'] = str(export_dir)
            for name in exported['files']:
                (export_dir/name).write_bytes(b'bounded export data'); exported['files'][name] = fact(export_dir/name)['sha256']
            evaluation_fact = write_json(root/'evaluation.json',launch)
            exported.update(authority=evaluation_fact,authority_sha256=evaluation_fact['sha256'])
            eargs = SimpleNamespace(execution_sha256=g.own['execution_sha256'],authority=Path(evaluation_fact['path']),
                authority_sha256=evaluation_fact['sha256'],phase='export',arm='control',seed=179061,output=export_dir)
            exported['invocation']['argv'] = e.cli(eargs)
            exported['invocation']['argv'][0] = str(Path(g.own['root'])/'evaluate_siglip2_connected_mlp.py')
            export_unit = terminal_fixture(root,f,'export-terminal',driver.CONTROL_INVOCATION,exported)
            common = {str(Path(g.own['root'])/n):h for n,h in g.own['code'].items()}
            common[str(Path(g.own['root'])/'execution.json')] = g.own['execution_sha256']
            exported['input_guards'] = {**common,**exported['origins']['files']}
            export_unit['receipt'] = write_json(export_dir/'receipt.json',exported)
            mapping = {'query':[],'gallery':[],'original_rows':[None]*3449}
            for row in exported['images']:
                for image in row['rows']:
                    mapping[row['role']].append(image['panel_ordinal'])
                    mapping['original_rows'][image['panel_ordinal']] = image['original_row']
            context.update(args=eargs,launch=launch,costs=exported['cost'],preparation_costs=exported['preparation_costs'],
                cpu={'payload_facts':{'control-179061':exported['payload_facts']}},common_guards=common,accepted_units=[],
                terminal_reader=f.fitter.original_terminal_reader(f.context['fit_context']),
                score_context={'partition':{'panels':{'selection':mapping}}})
            f.legacy['selected']['source_cpu'].update(invocation=exported['invocation'],numerical_flags=exported['numerical_flags'])
            obs_root = root/'observation'; obs_root.mkdir()
            observation = observation_fixture(obs_root,g.observer)
            observation['bundle'] = {'directory':str(Path(manifest['path']).parent),'manifest':manifest}
            for role,name in [('trainer','train_siglip2_connected_mlp.py'),('serializer','train_siglip2_substrate_adaptation.py')]:
                observation['sources'][role] = fact(Path(manifest['path']).parent/name)
            observation.update(native=fact(g.binary),control_export_receipt=export_unit['receipt'])
            observation_fact = write_json(root/'observation.json',observation)
            sources = {role:fact(path) for role,path in {
                'control_driver':Path(driver.__file__),'control_native':Path(g.native.__file__),'control_test':Path(__file__),
                'request_driver':HERE/'qualify_connected_serving_requests.py','request_test':HERE/'test_connected_serving_requests.py',
                'observer':Path(g.observer.__file__),'observer_test':HERE/'test_observe_connected_serving.py',
                'bridge':HERE.parent/'src/sfora/connected_compact_serving.py','native_wrapper':HERE.parent/'src/sfora/cutile_int8.py',
                'packing':HERE.parent/'src/sfora/joint_relational_compaction.py'}.items()}
            authority = {'schema':driver.SCHEMA,'sources':sources,'evaluator':g.own,'evaluation_authority':evaluation_fact,
                'control_export':export_unit,'control':dict(driver.CONTROL),'observation':observation_fact,
                'native_runtime':g.runtime_fact,'locks':[]}
            authority_fact = write_json(root/'control-authority.json',authority)
            measurement_root = root/'measurement'; measurement_root.mkdir()
            record,_,_ = measurement_fixture(measurement_root,driver)
            record.update(authority=authority_fact,sources=sources,control_export=export_unit,observation=observation_fact,
                native_runtime=g.runtime_fact,combined_native=combined,output=str(root/'diagnostic'),
                resource_policy=observation['resource_policy'],whole_process_seconds=1)
            record['invocation'].update({k:exported['invocation'][k] for k in ('python','python_sha256','python_version')})
            record['invocation']['argv'] = driver.cli(authority_fact,record['output'],driver.__file__)
            unit = terminal_fixture(root,f,'diagnostic-terminal','9'*32,record)
            record['resources'] = {k:record.pop(k) for k in ('wall_seconds','process_peak_rss_kib','cgroup_before','cgroup_after')}
            record['resources']['peak_cuda_allocated_bytes'] = 1
            required = [authority_fact,observation_fact,g.runtime_fact,evaluation_fact,*sources.values(),*g.authority.provenance_facts(),
                manifest,observation['gallery']['file'],export_unit['receipt'],*observation['train_images']]
            record['input_guards'] = {**common,**{v['path']:v['sha256'] for v in required}}
            output = Path(record['output']); output.mkdir()
            baseline_guards = dict(context['guards'])
            actual_load = driver.requests.Source.load
            def load_source(file):
                result = actual_load(file)
                if file == sources['control_native']:
                    result.module.ARCHIVE_SHA = g.runtime['build_receipt']['sha256']
                    result.module.BINARY_SHA = g.runtime['library']['sha256']
                return result
            def accept(value,change=None,authority_file=authority_fact):
                f.legacy['invocations'].clear(); context['accepted_units'].clear()
                context['guards'] = dict(baseline_guards)
                f.legacy['admission'] = f.original.FlatAdmission(); f.legacy['admission'].init = f.init
                current = {**unit,'receipt':write_json(output/'receipt.json',value),**(change or {})}
                return driver.accept_unit(context,current,authority_file)
            with patch.object(driver,'CONTROL_RECEIPT_SHA',export_unit['receipt']['sha256']), \
                    patch.object(f.context['nearest'],'NATIVE_PROOF_PINS',f.pins), \
                    patch.object(driver.requests.Source,'load',load_source):
                self.assertEqual(accept(record),record)
                mutations = []
                bad = copy.deepcopy(record); bad['combined_native']['inventory']['files'][str(root/'unknown.so')] = 'a'*64
                mutations.append(('combined',bad))
                for key in ('supplemental_files','historical_projection','mapped_identities'):
                    bad = copy.deepcopy(record); bad['combined_native'][key] = {}; mutations.append((key,bad))
                for file in required:
                    bad = copy.deepcopy(record); del bad['input_guards'][file['path']]; mutations.append(('missing guard '+file['path'],bad))
                bad = copy.deepcopy(record); bad['resource_policy']['whole_process_seconds'] = 1501; mutations.append(('cap',bad))
                bad = copy.deepcopy(record); bad['resources']['peak_cuda_allocated_bytes'] = 10_000_000_000; mutations.append(('CUDA cap',bad))
                for label,bad in mutations:
                    with self.subTest(mutant=label),self.assertRaises((ValueError,KeyError)):
                        accept(bad)
                for change in ({'both_locks_held':False},{'service_seconds':1501},{'invocation_id':'8'*32}):
                    with self.subTest(mutant=change),self.assertRaises(ValueError): accept(record,change)
                bad_observation = {**observation,'native':observation['gallery']['file']}
                bad_observation_fact = write_json(root/'bad-observation.json',bad_observation)
                bad_authority = {**authority,'observation':bad_observation_fact}
                bad_authority_fact = write_json(root/'bad-authority.json',bad_authority)
                bad = copy.deepcopy(record); bad.update(authority=bad_authority_fact,observation=bad_observation_fact)
                bad['invocation']['argv'] = driver.cli(bad_authority_fact,record['output'],driver.__file__)
                with self.assertRaisesRegex(ValueError,'terminal native FILE differs'):
                    accept(bad,authority_file=bad_authority_fact)
                with patch.object(context['args'],'execution_sha256','0'*64),self.assertRaisesRegex(ValueError,'original evaluator authority'):
                    accept(record)
                self.assertEqual(accept(record),record)
                self.assertIn(driver.CONTROL_INVOCATION,f.legacy['invocations'])
                self.assertIn(unit['invocation_id'],f.legacy['invocations'])
                self.assertEqual(context['accepted_units'],[export_unit])
                with self.assertRaisesRegex(ValueError,'reused terminal invocation'):
                    driver.accept_unit(context,{**unit,'receipt':fact(output/'receipt.json')},authority_fact)

    def test_startup_frozen_inputs_clock_and_lazy_four_members(self):
        driver = load('_control_startup_driver',HERE/'qualify_connected_control_serving.py')
        run = next(n for n in ast.parse(Path(driver.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name == 'run')
        body = next(n.body for n in run.body if isinstance(n,ast.Try))
        def execute(nodes,values):
            namespace = {**vars(driver),**values}
            exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),driver.__file__,'exec'),namespace)
            return namespace
        start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.value) == 'evaluator.authority(eargs)')
        end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.value).startswith('admit_control('))
        context = {'training_context':{'fit_context':{'unit_started':-1}}}
        execute(body[start:end],{'evaluator':SimpleNamespace(authority=lambda args:(context,None)),'eargs':None})
        with self.subTest(contract='fitter starts with control unit'):
            self.assertEqual(context['training_context']['fit_context']['unit_started'],driver.STARTED)
        frozen = next(n for n in body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'frozen')
        observation = {'bundle':{'manifest':'bundle'},'gallery':{'file':'gallery'},'native':'native',
            'control_export_receipt':'export','train_images':['image']}
        namespace = execute([frozen],{'authority_fact':'authority','authority':{'observation':'observation',
            'native_runtime':'runtime','evaluation_authority':'evaluation'},'sources':{'source':'source'},'observation':observation})
        with self.subTest(contract='explicit bundle and terminal freeze'):
            self.assertTrue({'bundle','export'} <= set(namespace['frozen']))
        start = next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and ast.unparse(n).startswith('api.audit_origins('))
        end = next(i for i in range(start+1,len(body)) if isinstance(body[i],ast.Expr) and ast.unparse(body[i]) == 'guard()')
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules),genuine_exit_fixture(Path(directory)) as g:
            historical = g.map_paths.pop(0)
            execute(body[start:end],{'api':g.api,'context':g.context})
            with self.assertRaisesRegex(ValueError,'exact four'):
                g.api.evaluator_exit(g.context,g.guard)
            g.map_paths.insert(0,historical)
            with redirect_stdout(io.StringIO()): g.api.evaluator_exit(g.context,g.guard)

    def test_genuine_collector_frozen_evaluator_exit_and_mutants(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules):
            root = Path(directory)
            with genuine_exit_fixture(root) as g:
                def exit():
                    g.f.context['fit_context']['phase_seconds'].clear()
                    with redirect_stdout(io.StringIO()): return g.api.evaluator_exit(g.context,g.guard)
                exit()
                self.assertIsNotNone(g.guard)
                self.assertEqual(g.api.asts['inverses']['evaluator'][2],1)
                private = next(c.cell_contents for c in g.api.evaluator_exit.__closure__ if
                    isinstance(c.cell_contents,g.request.FunctionType) and c.cell_contents.__name__ == 'exit_rehash')
                for key,value in vars(g.evaluator).items():
                    if key != 'exit_rehash': self.assertIs(private.__globals__[key],value,key)
                self.assertEqual(g.evaluator.FILES,{'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'})
                historical = g.map_paths.pop(0)
                with self.assertRaisesRegex(ValueError,'exact four'): exit()
                g.map_paths.insert(0,historical)
                unknown = root/'unknown.so'; unknown.write_bytes(b'unknown')
                g.context['guards'][str(unknown)] = fact(unknown)['sha256']
                g.map_paths.append(str(unknown))
                with self.assertRaisesRegex(ValueError,'unknown'): exit()
                g.map_paths.pop()
                g.map_paths.remove(str(g.binary))
                with self.assertRaisesRegex(ValueError,'supplemental'): exit()
                g.map_paths.append(str(g.binary))
                g.map_text[0] = g.maps().replace(str(g.binary),str(g.binary)+' (deleted)')
                with self.assertRaisesRegex(ValueError,'deleted'): exit()
                g.map_text[0] = g.maps().replace(str(g.binary.stat().st_ino)+' '+str(g.binary),'1 '+str(g.binary))
                with self.assertRaisesRegex(ValueError,'inode'): exit()
                g.map_text[0] = None
                raw,saved = g.binary.read_bytes(),g.binary.stat()
                g.binary.write_bytes(b'x'*len(raw)); os.utime(g.binary,ns=(saved.st_atime_ns,saved.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'): exit()
                g.binary.write_bytes(raw)
                backup = g.binary.with_suffix('.saved'); g.binary.rename(backup); g.binary.write_bytes(raw)
                try:
                    with self.assertRaisesRegex(ValueError,'inode'): exit()
                finally: g.binary.unlink(); backup.rename(g.binary)
                foreign = ModuleType('torch.foreign'); foreign.__file__ = sys.modules['torch'].__file__
                with patch.dict(sys.modules,{'torch.foreign':foreign}),self.assertRaisesRegex(ValueError,'module'):
                    exit()
                with patch.dict(g.f.context['control_native_owned'],{'authenticate':lambda:None}),self.assertRaisesRegex(ValueError,'ownership'):
                    exit()
                with patch.object(g.f.source,'imported_origins',lambda *a:{}),self.assertRaisesRegex(ValueError,'binding changed|dependency changed'):
                    exit()
                with patch.dict(g.context['code'],{'foreign.py':'a'*64}),self.assertRaisesRegex(ValueError,'closure'):
                    exit()
                with patch.dict(g.f.context,{'live_training':lambda:object()}),self.assertRaisesRegex(ValueError,'training A'):
                    exit()
                with patch.object(g.context['trainer'],'admit_bundle',lambda *a:({},{})),self.assertRaisesRegex(ValueError,'helper live'):
                    exit()
                exit()
                self.assertLess(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),16*1024**2)

    def test_complete_measurement_receipt_rejects_missing_and_forged_evidence(self):
        driver = load('_control_measurement_driver',HERE/'qualify_connected_control_serving.py')
        with tempfile.TemporaryDirectory() as directory:
            record,authority,authority_fact = measurement_fixture(Path(directory),driver)
            driver.validate_receipt(record,authority,authority_fact)
            # NumPy int64 buffers may expose native long instead of long long on LP64.
            if driver.struct.calcsize('l') == 8:
                long_ids = copy.deepcopy(record)
                for pair in [*[row['native'] for row in long_ids['calls']],
                    *[row['native'] for row in long_ids['original_warmup_oracles'].values()],*long_ids['ties']['native']]:
                    pair[0]['format'] = 'l'
                driver.validate_receipt(long_ids,authority,authority_fact)
            mutants = []
            for key in ('timed','original_warmup_oracles','observations','owners','oracle_semantics','timing_semantics','product_p99'):
                bad = copy.deepcopy(record); del bad[key]; mutants.append((key,bad))
            for key in record['calls'][0]:
                bad = copy.deepcopy(record); del bad['calls'][0][key]; mutants.append(('row '+key,bad))
            for path in [('timed','1'),('original_warmup_oracles','1'),('observations','1'),('owners',0),('ties',),
                ('observations','1','fingerprints',0),('observations','1','tensor_occurrences',0),
                ('observations','1','host_events',0),('calls',0,'native',0),('observations','1','output','raw')]:
                target = record
                for key in path: target = target[key]
                for key in target:
                    bad = copy.deepcopy(record); parent = bad
                    for part in path: parent = parent[part]
                    del parent[key]; mutants.append(('missing '+str((*path,key)),bad))
            changes = [
                (('calls',2,'seconds'),-1), (('calls',2,'instrumented'),True), (('calls',2,'native'),None),
                (('calls',2,'seconds'),float('nan')), (('calls',2,'read_decode_seconds'),float('inf')),
                (('calls',2,'public_call_seconds'),True), (('calls',2,'image_cleanup_seconds'),-1),
                (('calls',2,'seconds'),1), (('calls',0,'instrumented'),False),
                (('timed','1','measured_images_per_second'),0), (('timed','1','seconds'),[1]*8),
                (('original_warmup_oracles','1','output','raw','hex'),'01'*512),
                (('observations','1','fingerprints'),[]), (('observations','1','tensor_occurrences'),[]),
                (('observations','1','callback_seconds'),-1), (('observations','1','complete'),False),
                (('observations','1','fingerprints',0,'occurrences'),99),
                (('observations','1','fingerprints',0,'sha256'),'x'*64),
                (('observations','1','tensor_occurrences',0,'fingerprint'),99),
                (('observations','1','host_events',0,'exclusive_seconds'),-1),
                (('observations','1','exclusive_host_phase_seconds'),{}),
                (('owners',0,'admission_seconds'),-1), (('owners',1,'release_seconds'),float('inf')),
                (('owners',),[record['owners'][0]]), (('ties','native',0,0,'hex'),'00'*80),
                (('calls',2,'native',0,'shape'),[32,10]), (('calls',2,'native',1,'format'),'d')]
            for path,value in changes:
                bad = copy.deepcopy(record); target = bad
                for key in path[:-1]: target = target[key]
                target[path[-1]] = value; mutants.append((str(path),bad))
            for label,bad in mutants:
                with self.subTest(mutant=label),self.assertRaises((ValueError,KeyError,TypeError)):
                    driver.validate_receipt(bad,authority,authority_fact)

    def test_separate_source_contract(self):
        for name in ('qualify_connected_control_serving.py','connected_control_native_authority.py'):
            self.assertTrue((HERE/name).is_file(), 'missing separate control source: ' + name)

    def test_real_evaluator_source_identity_baseline_and_mutants(self):
        request = load('_control_actual_evaluator_requests',HERE/'qualify_connected_serving_requests.py')
        native = load('_control_actual_evaluator_native',HERE/'connected_control_native_authority.py')
        file = fact(HERE/'evaluate_siglip2_connected_mlp.py')
        plain = request.Source.load(file)
        try:
            self.assertEqual([n for n,v in plain.literals.items() if vars(plain.module)[n] != v],['_SOURCE_BUILTINS'])
            with self.assertRaisesRegex(ValueError,'live source changed'): plain.check()
        finally: sys.modules.pop(plain.module.__name__,None)
        self.assertTrue(callable(getattr(native,'load_evaluator_source',None)), 'owned exact evaluator checker missing')
        source = native.load_evaluator_source(file,request)
        module = source.module; baseline = module._SOURCE_BUILTINS
        members = dict(vars(module)); original_request_check = request.Source.check
        try:
            source.check()
            with patch.object(module,'_SOURCE_BUILTINS',tuple(list(baseline))),self.assertRaisesRegex(ValueError,'baseline'):
                source.check()
            import builtins
            with patch.object(builtins,'help',lambda *a:None),self.assertRaisesRegex(ValueError,'builtin'):
                source.check()
            with patch.object(builtins,'all',lambda *a:True),self.assertRaisesRegex(ValueError,'builtin'):
                source.check()
            with patch.dict(vars(builtins),{'__control_fixture_extra__':object()}),self.assertRaisesRegex(ValueError,'builtin'):
                source.check()
            with patch.dict(module.FIRST_SELECTION_OWNER,{'execution_sha256':'0'*64}),self.assertRaisesRegex(ValueError,'live source'):
                source.check()
            initializer = module.source_live_guard.initializer
            code = initializer.__code__
            try:
                initializer.__code__ = bypass_code(initializer)
                with self.assertRaisesRegex(ValueError,'function/closure'): source.check()
            finally: initializer.__code__ = code
            with patch.object(initializer,'__defaults__',(None,)),self.assertRaisesRegex(ValueError,'function/closure'):
                source.check()
            cell = initializer.__closure__[0]; value = cell.cell_contents
            try:
                cell.cell_contents = object()
                with self.assertRaisesRegex(ValueError,'source|function/closure'): source.check()
            finally: cell.cell_contents = value
            with patch.object(module.__spec__,'origin',str(HERE/'not-the-evaluator.py')),self.assertRaisesRegex(ValueError,'live source'):
                source.check()
            source.check()
            self.assertEqual(vars(module).keys(),members.keys())
            self.assertTrue(all(vars(module)[n] is value for n,value in members.items()))
            self.assertIs(request.Source.check,original_request_check)
        finally: sys.modules.pop(module.__name__,None)

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
            measurement_root = root/'measurement'; measurement_root.mkdir()
            measured,_,_ = measurement_fixture(measurement_root,driver)
            record.update({k:measured[k] for k in ('calls','body_seconds','ties','timed','owners','original_warmup_oracles',
                'observations','oracle_semantics','timing_semantics','product_p99')})
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
                 patch.object(native,'BINARY_SHA',fact(binary)['sha256']), patch.object(Path,'read_text',read_text), \
                 patch.dict(os.environ,{'CUTILE_TILEIRAS_PATH':runtime['runtime_compiler']['path']}):
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
                    ({**combined_files,runtime['runtime_compiler']['path']:runtime['runtime_compiler']['sha256']},
                        list(combined_files),{},'unknown'),
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
                for p in (binary,Path(runtime['runtime_compiler']['path']),Path(runtime['build_receipt']['path']),Path(runtime['supplemental'][0]['provenance']['path']),
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
                with patch.object(source,'check',lambda:None), \
                        self.assertRaisesRegex(ValueError,'evaluator checker binding'): exit()
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
                for target in (binary,Path(runtime['runtime_compiler']['path'])):
                    raw = target.read_bytes()
                    def mutate(path,digest):
                        result = real_admit(path,digest)
                        target.write_bytes(b'x'*len(raw))
                        return result
                    try:
                        with patch.object(context['trainer'],'admit_bundle',mutate),self.assertRaisesRegex(ValueError,'SHA256'): exit()
                    finally: target.write_bytes(raw)
                exit()
                f.unchanged_originals(self)
                self.assertLess(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),16*1024**2)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (1024**3,1024**3))
    signal.alarm(120)
    sys.meta_path.insert(0, NoNative())
    unittest.main()
