#!/usr/bin/env python3
"""Stdlib source/admission falsifiers. No native qualification is claimed.

Run: python3 -B scripts/test_siglip2_connected_mlp.py --source-only
The parent runs this once under timeout120 and address-space1GiB after repairs.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from functools import lru_cache
import gc
import weakref
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'train_siglip2_connected_mlp.py'


def rejects(call, text):
    try:
        call()
    except (ValueError, TypeError, KeyError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError('accepted mutant: ' + text)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_contract(d):
    tree = ast.parse(DRIVER.read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name.split('.')[0] for n in top_imports for a in n.names] + [
        n.module.split('.')[0] for n in top_imports if isinstance(n, ast.ImportFrom)]
    assert not set(names) & (d.NATIVE | {'qualify_actual_objective_encoder_gradients',
                                        'train_siglip2_identity_diversity'})
    assert d.FILES == {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}
    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/actual-objective-gradient-v2-freeze'
    admitted = json.loads((evidence/'authority.json').read_text())
    assert d.WITNESS_FILES == admitted['files'].keys() and len(d.WITNESS_FILES) == 7
    assert 'test_connected_encoder_gradients.py' not in d.WITNESS_FILES
    for name,digest in admitted['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    assert len(d.MLP) == 4 and sum(__import__('math').prod(s) for s in d.MLP_SHAPES) == 9921872
    assert d.parameter_roles('control') == (['A', 'C'], [[128,160],[128,1152]], 167936)
    assert d.parameter_roles('candidate') == (['A','C',*d.MLP], [[128,160],[128,1152],*d.MLP_SHAPES], 10089808)
    rejects(lambda: d.parameter_roles('CONTROL2016'), 'arm')
    assert d.policy('cpu')['seconds'] == d.policy('mechanics')['seconds'] == 300
    assert d.policy('train')['seconds'] == 600
    assert all(d.policy(p)['host_bytes'] == 8*1024**3 and d.policy(p)['swap_bytes'] == 0
               for p in ('cpu','mechanics','train'))
    assert d.ADAM['lr'] == 1e-4 and d.RECIPE['encoder_lr'] == 1e-5
    assert d.ADAM['weight_decay'] == .05 and d.ADAM['betas'] == (.9,.999)
    assert all(d.ADAM[k] is False for k in ('amsgrad','maximize','foreach','capturable','differentiable','fused'))
    original = ast.parse((HERE/'train_siglip2_identity_diversity.py').read_text())
    static = next(n.value for n in original.body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id == 'STATIC_KEYS' for t in n.targets))
    assert d.STATIC_KEYS == ast.literal_eval(static)
    assert d.PAYLOAD_KEYS == {'schema','identity','source','A','C',*d.STATIC_KEYS,'optimizer','scaler',
                             'counter','cpu_rng','cuda_rng','numerical_flags','base_vision','encoder','vision_sha256'}


def initializer_selection(d):
    records = [{'arm': a, 'seed': s, 'identity': {'device': dev, 'scope': {'arm': a}},
                'checkpoint': {'path': '/fixture/'+a+str(s), 'sha256': 'a'*64},
                'terminal_state_sha256': 'b'*64}
               for a in ('control','candidate') for s in d.SEEDS for dev in ('cpu','cuda')]
    record = {'qualifications': records}
    for seed in d.SEEDS:
        chosen = d.select_initializer(record, seed)
        assert chosen['arm'] == chosen['identity']['scope']['arm'] == 'control'
        assert chosen['identity']['device'] == 'cpu' and chosen['seed'] == seed
        rejects(lambda: d.select_initializer({'qualifications': records+[chosen]}, seed), 'unique')
        rejects(lambda: d.select_initializer({'qualifications': [r for r in records if r is not chosen]}, seed), 'unique')


def authority_falsifiers(d):
    file = {'path':'/fixture/authority.json', 'sha256':'a'*64}
    unit = {'receipt':file, 'log':file, 'unit':'fixture', 'invocation_id':'a'*32,
            'service_seconds':1., 'native_peak_rss_kib':1, 'both_locks_held':True}
    args = SimpleNamespace(execution_sha256='b'*64, phase='cpu', arm='control', seed=d.SEEDS[0])
    launch = {'schema':d.AUTHORITY_SCHEMA, 'execution_sha256':args.execution_sha256,
              'phase':'cpu', 'arm':'control', 'seed':args.seed, 'recipe':copy.deepcopy(d.RECIPE),
              'resource_policy':d.policy('cpu'), 'both_locks_held':True,
              'original_cpu':{'authority':file, 'terminal':unit},
              'actual_gradient':{'authority':file, 'terminal':unit},
              'witness':{'root':'/fixture/witness', 'files':{n:'a'*64 for n in d.WITNESS_FILES}},
              'selected_cpu':None, 'selected_mechanics':None, 'fresh_control':None}
    d.check_launch(launch, args)
    for key, value in [('schema','wrong'), ('seed',True), ('arm','candidate'),
                       ('resource_policy', {**d.policy('cpu'),'seconds':500}),
                       ('both_locks_held',False), ('selected_cpu',unit)]:
        rejects(lambda k=key,v=value: d.check_launch({**launch,k:v},args), 'launch')
    for key, value in [('classes', {'control':1008,'candidate':2016}), ('microbatch',32),
                       ('clip',2.), ('updates',17), ('encoder_lr',1e-4)]:
        changed = copy.deepcopy(launch)
        changed['recipe'][key] = value
        rejects(lambda: d.check_launch(changed,args), 'launch')
    for value in [{**file,'extra':1}, {**file,'path':'relative'}, {**file,'sha256':'future'}]:
        rejects(lambda: d.file_fact(value), 'FILE')
    for value in [{**unit,'both_locks_held':False}, {**unit,'invocation_id':'bad'},
                  {**unit,'service_seconds':float('nan')}]:
        rejects(lambda: d.check_unit(value), 'UNIT')
    rejects(lambda: d.strict_json('{"x":1,"x":2}'), 'duplicate')
    rejects(lambda: d.strict_json('{"x":NaN}'), 'nonfinite')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'bytes'
        path.write_bytes(b'original')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        d.bound_file({}, path, digest)
        path.write_bytes(b'mutant')
        rejects(lambda: d.bound_file({},path,digest), 'bytes')


def lifetime_restore_math(d):
    tree = ast.parse(DRIVER.read_text())
    text = lambda name: ast.unparse(function(tree,name))
    restore = text('restore')
    assert restore.index('require_no_training') < restore.index('torch.load')
    assert "'cpu' if k == 'step'" in restore and 'pages.copy' in restore
    assert restore.index('construct_encoder') < restore.index('bind_optimizer')
    assert restore.index('bind_optimizer') < restore.index('set_rng_state')
    assert 'mapping_absent' in restore and 'del disk' in restore
    assert 'frozen=' not in restore and 'frozen_cache' not in DRIVER.read_text()
    assert 'canonical_initial_witness' in text('load_initializer')
    assert 'trainer.integrity' in text('load_initializer')
    assert 'trainer.check_payload' in text('check_payload')
    assert 'copy.deepcopy' in text('load_initializer') and "disk['identity']" in text('load_initializer')
    assert '.eval()' in text('construct_encoder')
    assert text('load_inference').index('admit_bundle') < text('load_inference').index('import torch')
    assert 'trainer.exit_rehash' in text('run')
    assert 'loss_terms' in text('update') and 'pixels_for' in text('update')
    assert 'connected.raw_features' in text('update')
    update = text('update')
    assert update.index('unscale_') < update.index('clip_grad_norm_') < update.index('scaler.step')
    assert 'enabled=False' in update and 'requires_grad' in update
    assert 'ranking' in update and 'autograd.grad' in update
    assert 'max_memory_allocated' in text('resource_check')
    assert 'reset_peak' not in DRIVER.read_text()
    assert '444' in text('encoder_facts') and 'nonpersistent' in text('encoder_facts')
    assert 'vision_sha256' in text('inference_outputs') and 'fingerprint' in text('inference_outputs')
    assert 'construct_encoder' in text('load_inference') and "disk['encoder']" in text('load_inference')
    assert 'apply_overlay' in text('construct_encoder') and 'load_vision' in text('construct_encoder')
    assert 'deny_training_dependencies' in text('qualify_bundle')
    assert 'train_siglip2_identity_diversity' not in text('load_inference')
    assert 'qualify_actual_objective' not in text('load_inference')
    assert 'range(1, 9)' in text('arm_run') and 'range(9, 18)' in text('arm_run')
    assert 'diagnostic' in text('arm_run') and 'payload' in text('arm_run')
    old = ast.parse((HERE/'train_siglip2_identity_diversity.py').read_text())
    # Native inference math is the original value-equivalent formula, including
    # the detached residual (serving has no gradients). Training uses the witness.
    assert ast.dump(function(tree,'fullfeature_raw_features'),include_attributes=False) == \
           ast.dump(function(old,'fullfeature_raw_features'),include_attributes=False)
    witness = ast.parse((HERE/'qualify_actual_objective_encoder_gradients.py').read_text())
    assert ast.dump(function(tree,'_processor_cache'),include_attributes=False) == \
           ast.dump(function(witness,'_processor_cache'),include_attributes=False)


class Tensor:
    """Metadata-only stand-in: no numeric/native training is simulated."""
    def __init__(self, shape=(), value=0., dtype='float32', device='cpu'):
        self.shape,self.value,self.dtype = tuple(shape),value,dtype
        self.device = SimpleNamespace(type=device)
        self.requires_grad,self.grad_fn,self.grad,self.is_leaf = False,None,None,True
    def detach(self): return Tensor(self.shape,self.value,self.dtype,self.device.type)
    def to(self, device='cpu', *, copy=False, **kwargs):
        assert copy, 'same-device copies must own storage'
        return Tensor(self.shape,self.value,self.dtype,device)
    def copy_(self, value): self.value = value.value; return self
    def requires_grad_(self, value): self.requires_grad = value; return self
    def __float__(self): return float(self.value)


@contextmanager
def tensor_seam():
    old = {name:sys.modules.get(name) for name in ('torch','torch.nn','torch.nn.functional')}
    torch = ModuleType('torch')
    torch.Tensor,torch.float32 = Tensor,'float32'
    torch.isfinite = lambda v: SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:v.value != float('inf')))
    @contextmanager
    def no_grad(): yield
    torch.no_grad = no_grad
    nn = ModuleType('torch.nn')
    nn.LayerNorm = type('LayerNorm',(),{})
    torch.nn = nn
    nn.functional = ModuleType('torch.nn.functional')
    sys.modules.update({'torch':torch,'torch.nn':nn,'torch.nn.functional':nn.functional})
    try:
        yield torch
    finally:
        for name,module in old.items():
            if module is None: sys.modules.pop(name,None)
            else: sys.modules[name] = module


def overlay_optimizer_seams(d):
    with tensor_seam():
        params = {n:Tensor(shape) for n,shape in zip(d.MLP,d.MLP_SHAPES,strict=True)}
        params.update({f'frozen.{i}':Tensor((1,)) for i in range(444)})
        model = SimpleNamespace(named_parameters=lambda:params.items(),state_dict=lambda:params)
        overlay = {n:Tensor(shape,i+1.) for i,(n,shape) in enumerate(zip(d.MLP,d.MLP_SHAPES,strict=True))}
        identities = {n:id(p) for n,p in params.items()}
        d.apply_overlay(model,overlay)
        assert {n:id(p) for n,p in params.items()} == identities
        assert [params[n].value for n in d.MLP] == [1.,2.,3.,4.]
        assert all(p.value == 0 for n,p in params.items() if n not in d.MLP)
        rejects(lambda:d.apply_overlay(model,{**overlay,'head.mlp.fc2.bias':Tensor((1152,))}), 'names')
        rejects(lambda:d.apply_overlay(model,{n:v for n,v in overlay.items() if n != d.MLP[0]}), 'names')
        for bad in (Tensor((1152,)),Tensor(d.MLP_SHAPES[0],dtype='float16'),Tensor(d.MLP_SHAPES[0],float('inf'))):
            rejects(lambda:d.apply_overlay(model,{**overlay,d.MLP[0]:bad}), 'shape/dtype/role')
        overlay[d.MLP[0]].requires_grad = True
        rejects(lambda:d.apply_overlay(model,overlay), 'shape/dtype/role')
        overlay[d.MLP[0]].requires_grad = False
        for arm in d.ARMS:
            names,shapes,_ = d.parameter_roles(arm)
            groups = [{**d.ADAM,'params':[0,1]}]
            if arm == 'candidate': groups.append({**d.ADAM,'lr':1e-5,'params':[2,3,4,5]})
            identity = {'arm':arm,'parameter_names':names,'parameter_shapes':shapes,
                        'optimizer_defaults':copy.deepcopy(d.ADAM),
                        'optimizer_groups':[{k:v for k,v in g.items() if k != 'params'} for g in groups]}
            saved = {'optimizer':{'state':{i:{'step':Tensor((),8.),'exp_avg':Tensor(s,1.),'exp_avg_sq':Tensor(s,2.)}
                                            for i,s in enumerate(shapes)},'param_groups':groups}}
            d.check_optimizer(saved,identity,8)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['params'].reverse()
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'groups')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['maximize'] = 0
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'groups')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'].pop(len(names)-1)
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'ownership')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][0]['step'].device.type = 'cuda'
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'CPU step')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][len(names)-1]['exp_avg'].shape = (1,)
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'moment')
            changed_identity = copy.deepcopy(identity)
            changed_identity['optimizer_groups'][0]['lr'] = 1e-5
            rejects(lambda:d.check_optimizer(saved,changed_identity,8), 'defaults/rates')
        source = ast.parse((HERE/'train_siglip2_substrate_adaptation.py').read_text())
        pages = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name == 'CheckpointPages')
        method = next(n for n in pages.body if isinstance(n,ast.FunctionDef) and n.name == 'copy')
        ns = {}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<original pages copy>','exec'),ns)
        consumed = []
        owner = SimpleNamespace(consume=lambda v:consumed.append(v))
        owner.copy = lambda value,device='cpu':ns['copy'](owner,value,device)
        step,moment = Tensor((),8.),Tensor((3,),1.)
        copied = d.owned_copy({'step':step,'moments':[moment]},owner,'cpu')
        assert copied['step'] is not step and copied['moments'][0] is not moment and consumed == [step,moment]
        copied['step'].value = 9.
        assert step.value == 8.


def owned_loader_admission(d):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code = {}
        for name in d.FILES | d.SERVING_FILES | {'joint_relational_compaction.py'}:
            raw = DRIVER.read_bytes() if name == DRIVER.name else b'# owned source\n'
            (root/name).write_bytes(raw)
            code[name] = hashlib.sha256(raw).hexdigest()
        files = {}
        for name in ('vision.pt','endpoint.pt','processor.json'):
            (root/name).write_bytes(b'owned artifact')
            files[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        vendor = root/'vendor.py'
        vendor.write_bytes(b'installed source')
        env = {'packages':{name:{'root':str(root/'packages'/name)} for name in d.NATIVE-{'sfora'}},
               'files':{str(vendor):hashlib.sha256(vendor.read_bytes()).hexdigest()},
               'native_files':{str(vendor):hashlib.sha256(vendor.read_bytes()).hexdigest()},'vision_constructor':str(vendor)}
        manifest = {'schema':d.BUNDLE_SCHEMA,'code':code,'files':files,'endpoint_state_sha256':'a'*64,
                    'environment':env,'encoder_identity':{},'base_vision_sha256':'b'*64,'vision_sha256':'c'*64,
                    'scope':{'arm':'control','manifest_sha256':d.SCOPE_SHA256,'arm_sha256':d.CONTROL_SHA256}}
        def write(value):
            p = root/'bundle.json'
            p.write_text(json.dumps(value))
            return hashlib.sha256(p.read_bytes()).hexdigest()
        digest = write(manifest)
        actual,_ = d.admit_bundle(root,digest)
        assert actual == manifest
        (root/'endpoint.pt').write_bytes(b'updated overlay substitution')
        rejects(lambda:d.admit_bundle(root,digest), 'bytes')
        # Actual public loader must reject before importing native packages.
        before = set(sys.modules)
        rejects(lambda:d.load_inference(root,digest,'cpu'), 'bytes')
        assert not {n.split('.')[0] for n in set(sys.modules)-before} & d.NATIVE
        (root/'endpoint.pt').write_bytes(b'owned artifact')
        bad = copy.deepcopy(manifest)
        bad['code']['train_siglip2_identity_diversity.py'] = 'a'*64
        rejects(lambda:d.admit_bundle(root,write(bad)), 'schema/closure')
        bad = copy.deepcopy(manifest)
        bad['environment']['files']['/historical/teacher.npy'] = 'a'*64
        rejects(lambda:d.admit_bundle(root,write(bad)), 'TRAIN data')
        import os
        os.link(root/'vision.pt',root/'alias')
        rejects(lambda:d.admit_bundle(root,write(manifest)), 'single-link')


def restore_seam(d):
    """Execute the actual restore ordering/CPU-step ownership seam with fake I/O.

    Payload/native validators are exercised separately. This check addresses
    the original failure mode: Adam preserves a CPU step alias by reference.
    """
    originals = {n:getattr(d,n) for n in ('bound_file','check_payload','fingerprint','construct_encoder',
        'bind_optimizer','encoder_facts','integrity','payload','mapping_absent')}
    events,consumed = [],[]
    with tensor_seam() as torch, tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'saved.pt'
        path.write_bytes(b'seam')
        disk = {'optimizer':{'param_groups':[],'state':{0:{'step':Tensor((),8.),
                   'exp_avg':Tensor((3,),2.),'exp_avg_sq':Tensor((3,),3.)}}},'scaler':{},'config':{},'buffers':{},
                'provenance':{'encoder':{'source_proof':{'runtime':{}}}},'encoder':{},
                'vision_sha256':'updated','cpu_rng':Tensor((3,),4.,'uint8'),'cuda_rng':[]}
        weak_step = weakref.ref(disk['optimizer']['state'][0]['step'])
        torch.load = lambda *a,**kw:disk
        torch.random = SimpleNamespace(set_rng_state=lambda value:events.append(('rng',value)))
        torch.cuda = SimpleNamespace(set_rng_state_all=lambda values:None)
        class Pages:
            def __init__(self, stream): events.append('pages')
            def consume(self, value): consumed.append(value)
            def copy(self, value, device='cpu'):
                result = value.to(device,copy=True)
                self.consume(value)
                return result
        class Optimizer:
            def load_state_dict(self, value): self.state = value['state']; events.append('optimizer_loaded')
        class Model:
            def named_parameters(self): return iter([(d.MLP[0],Tensor((1,)))])
            def to(self,device): events.append('transfer'); return self
            def eval(self): return self
        state = {}
        def model_free(context, arm, seed, device, initial):
            events.append('static_copied')
            return state
        trainer = SimpleNamespace(require_no_training=lambda context:events.append('release_first'),fresh=model_free,
                                  own_A=lambda *a,**kw:events.append('owners'))
        context = {'trainer':trainer,'guards':{},'legacy':{'original':SimpleNamespace(CheckpointPages=Pages),
            'source_driver':None,'prior':{'entry':{'input':{'preprocessor':{'path':'processor.json'}}}},'selected':{'packages':{}}}}
        identity = {'seed':179061,'device':'cpu','arm':'candidate','base_vision':{},'encoder_identity':{'runtime':{}}}
        def construct(*a,**kw): events.append('construct'); return Model(),Model(),None,{},{}
        def bind(state):
            events.append('bind')
            state['optimizer_object'] = Optimizer()
            state['scaler_object'] = SimpleNamespace(load_state_dict=lambda value:events.append('scaler'))
        d.bound_file = lambda guards,path,digest:Path(path)
        d.check_payload = lambda *a:events.append('validate')
        d.fingerprint = lambda *a,**kw:'digest'
        d.construct_encoder,d.bind_optimizer = construct,bind
        d.encoder_facts = lambda *a,**kw:{'vision_sha256':'updated'}
        d.integrity = lambda *a:events.append('integrity')
        d.payload = lambda *a:{}
        d.mapping_absent = lambda path:events.append('mapping_absent')
        try:
            restored = d.restore(context,path,'file','digest',identity,8)
            steps = restored['optimizer_object'].state
            assert steps[0]['step'] is not disk['optimizer']['state'][0]['step']
            steps[0]['step'].value = 9.
            assert disk['optimizer']['state'][0]['step'].value == 8.
            assert events.index('release_first') < events.index('validate') < events.index('construct')
            assert events.index('transfer') < events.index('bind') < events.index('optimizer_loaded')
            rng = next(v for v in events if isinstance(v,tuple) and v[0] == 'rng')
            assert rng[1] is not disk['cpu_rng']
            assert events.index('optimizer_loaded') < events.index(rng) < events.index('mapping_absent') < events.index('integrity')
            del disk
            consumed.clear()
            gc.collect()
            assert weak_step() is None, 'CPU step must not pin the serialized owner'
        finally:
            for name,value in originals.items(): setattr(d,name,value)


def current_encoder_seam(d):
    """Execute real 448/frozen444/runtime/role and public pre-forward falsifiers."""
    def fp(value):
        if isinstance(value,Tensor): return repr((value.shape,value.dtype,value.value))
        if isinstance(value,dict): return repr([(k,fp(v)) for k,v in sorted(value.items())])
        return repr(value)
    class Module:
        def __init__(self):
            self.training = False
            self._forward_hooks,self._forward_pre_hooks,self._backward_hooks = {},{},{}
            self._non_persistent_buffers_set = set()
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Module()
            self.embeddings._non_persistent_buffers_set = {'position_ids'}
            self.config = SimpleNamespace(_attn_implementation='sdpa',layer_norm_eps=1e-6)
            self.config.to_dict = lambda:{'attention':self.config._attn_implementation}
            self.params = {n:Tensor(shape,1.) for n,shape in zip(d.MLP,d.MLP_SHAPES,strict=True)}
            self.params.update({f'frozen.{i}':Tensor((1,)) for i in range(444)})
            self.position = Tensor((1,256),0.,'int64')
        def named_parameters(self): return self.params.items()
        def state_dict(self): return self.params
        def named_modules(self): return [('',self),('embeddings',self.embeddings)]
        def named_buffers(self): return [('embeddings.position_ids',self.position)]
    class Processor:
        backend = 'torchvision'
        def to_json_string(self): return '{}'
        def __call__(self,**kw): raise AssertionError('mutant reached serving math')
    class Head(Module):
        def __init__(self): super().__init__(); self.training = True; self.weight = Tensor((128,1152),1.)
        def parameters(self): return [self.weight]
        def modules(self): return [self]
        def state_dict(self): return {'weight':self.weight}
    original_cache = d._processor_cache
    with tensor_seam(),tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original,source = SimpleNamespace(fingerprint=fp),SimpleNamespace(module_origin=lambda cls,packages:{'class':cls.__name__},
                                                                            numerical_flags=lambda:{})
        guards = {}
        for i,module in enumerate((original,source)):
            path = root/f'helper{i}.py'
            path.write_text('# admitted helper')
            module.__file__ = str(path)
            guards[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        model,processor = Model(),Processor()
        identity = {'inventory':{n:list(p.shape) for n,p in model.named_parameters()},
                    'nonpersistent':{'embeddings':['position_ids']},'buffers_sha256':fp(dict(model.named_buffers())),
                    'frozen_sha256':fp({n:p for n,p in model.named_parameters() if n not in d.MLP}),
                    'runtime':d.model_structure(model,source,{})}
        state = {'model':model,'encoder_identity':identity,'device':'cpu','arm':'candidate',
                 'processor_object':processor,'processor':{'config':{},'backend':'torchvision','origin':{'class':'Processor'}},
                 'processor_cache':'cache','guards':guards}
        d._processor_cache = lambda *a,**kw:'cache'
        try:
            facts = d.encoder_facts(state,original,source,{},serving=True)
            endpoint = {**state,'modules':{'train_siglip2_substrate_adaptation.py':original,'qualify_siglip2_substrate_cpu.py':source},
                'head_object':Head(),'A':Tensor((128,160),1.),'C':Tensor((128,1152),1.),'means':{'concat':Tensor((160,),0.)},
                'mu_train':Tensor((1152,),0.),'mu_train_provenance':{},'scope':{},'common_statistics':{},'flags':{},
                'vision_sha256':facts['vision_sha256'],
                'manifest':{'environment':{'packages':{}},'encoder_identity':copy.deepcopy(identity),'vision_sha256':facts['vision_sha256']}}
            endpoint['A'].requires_grad = endpoint['C'].requires_grad = True
            endpoint['readout_sha256'] = fp(d.inference_readout_tree(endpoint))
            for name in d.MLP:
                model.params[name].value = 0. # Exact original-four member substituted.
                rejects(lambda:d.inference_outputs(endpoint,[object()]), 'current .data')
                model.params[name].value = 1.
            frozen = model.params['frozen.0']
            frozen.value = 2.
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'frozen444')
            frozen.value = 0.
            model.params[d.MLP[0]].requires_grad = True
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'role/gradient')
            model.params[d.MLP[0]].requires_grad = False
            model.embeddings._non_persistent_buffers_set.clear()
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'nonpersistent')
            model.embeddings._non_persistent_buffers_set.add('position_ids')
            model.position.value = 1.
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'buffers')
            model.position.value = 0.
            model.config._attn_implementation = 'eager'
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'SDPA')
            model.config._attn_implementation = 'sdpa'
            endpoint['head_object'].weight.requires_grad = True
            rejects(lambda:d.inference_outputs(endpoint,[object()]), 'frozen head roles')
            endpoint['head_object'].weight.requires_grad = False
            endpoint['head_object'].weight.value = 2.
            rejects(lambda:d.inference_outputs(endpoint,[object()]), 'current .data')
        finally:
            d._processor_cache = original_cache


def processor_release_seam(d):
    """Exercise NEW actual release paths with the admitted backend's real LRU code."""
    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/actual-objective-processor-cache-source'
    raw = (evidence/'original-image_processing_backends.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == d.BACKEND_SHA
    method = '_fuse_mean_std_and_rescale_factor'
    names = ('transformers.image_processing_backends','transformers.models.siglip.image_processing_siglip')
    previous = {name:sys.modules.get(name) for name in names}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'backend.py'
        path.write_bytes(raw)
        backend = ModuleType(names[0])
        backend.__file__ = str(path)
        backend.torch = SimpleNamespace(tensor=lambda x,device:x,float32='float32')
        code = compile(raw,str(path),'exec',dont_inherit=True)
        cls_code = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == 'TorchvisionBackend')
        function_code = next(c for c in cls_code.co_consts if isinstance(c,CodeType) and c.co_name == method)
        fn = FunctionType(function_code,vars(backend))
        fn.__defaults__ = (None,)*6
        wrapper = lru_cache(maxsize=10)(fn)
        backend.TorchvisionBackend = type('TorchvisionBackend',(),{'__module__':names[0],method:wrapper})
        siglip = ModuleType(names[1])
        siglip.SiglipImageProcessor = type('SiglipImageProcessor',(backend.TorchvisionBackend,),{'__module__':names[1]})
        sys.modules.update(dict(zip(names,(backend,siglip))))
        guards = {str(path):d.BACKEND_SHA}
        class Owner:
            def __init__(self): self.values = []
            def parameters(self): return iter(self.values)
            def buffers(self): return iter(())
        try:
            processor = siglip.SiglipImageProcessor()
            cache = d._processor_cache(processor,guards,empty=True)
            # No preprocessing is simulated: the real memoized method with all
            # optional values None returns (None,None,None) and retains self.
            assert getattr(processor,method)() == (None,None,None)
            ref = weakref.ref(processor)
            endpoint = {'processor_object':processor,'processor_cache':cache,'guards':guards,'modules':{},
                        'model':Owner(),'head_object':Owner(),'A':Owner(),'C':Owner(),'mu_train':Owner()}
            del processor
            gc.collect()
            assert ref() is not None and cache.cache_info().currsize == 1
            d.release_inference(endpoint)
            assert ref() is None and endpoint == {} and cache.cache_info().currsize == 0
            processor = siglip.SiglipImageProcessor()
            getattr(processor,method)()
            refs = []
            state = {k:Owner() for k in d.STATIC_KEYS+('A','C')}
            state.update(processor_object=processor,processor_cache=wrapper,model=Owner(),head_object=Owner(),
                         optimizer_object=SimpleNamespace(state_dict=lambda:{}),scaler_object=Owner())
            # SimpleNamespace cannot be weak-referenced; use the actual owned
            # optimizer seam shape with a weak-referenceable object instead.
            class Optimizer(Owner):
                def state_dict(self): return {}
            state['optimizer_object'] = Optimizer()
            trainer = SimpleNamespace(tensor_weakrefs=lambda context,value:[],release=lambda context,state:state.clear())
            ref = weakref.ref(processor)
            del processor
            d.release({'trainer':trainer,'guards':guards},state)
            assert ref() is None and state == {} and wrapper.cache_info().currsize == 0
        finally:
            wrapper.cache_clear()
            for name,module in previous.items():
                if module is None: sys.modules.pop(name,None)
                else: sys.modules[name] = module


def cost_terminal_falsifiers(d):
    """Actual timing/receipt predicates, with native work replaced by a clock."""
    events = []
    def connected(): pass
    def entry(*args):
        events.append('integrity')
        raise ValueError('entry reached')
    with tensor_seam(), patch.multiple(d,
            time=SimpleNamespace(perf_counter=lambda:events.append('tick') or 0.),
            bound_file=lambda *a:events.append('helper'), integrity=entry):
        context = {'trainer':None,'connected':SimpleNamespace(raw_features=connected,__file__='/helper'),
                   'connected_function':(connected,connected.__code__),'guards':{'/helper':'sha'}}
        rejects(lambda:d.update(context,{'device':'cpu','counter':0},{},1), 'entry reached')
    assert events == ['tick','helper','integrity'], events

    def timed_arm(phase, arm, core):
        clock = [0.]
        identity = {'arm':arm,'seed':d.SEEDS[0]}
        def update(context, state, identity, step):
            seconds = core/(128 if phase == 'train' else 34 if phase == 'mechanics' else 1)
            clock[0] += seconds
            return {'step':step,'core_seconds':seconds,'seconds':seconds}
        def qualify(*args):
            clock[0] += 100.  # Common construction/reload/bundle work cannot dilute core.
            return {}
        mechanics = {'steps':[{'step':s} for s in range(1,18)]}
        context = {'connected_args':SimpleNamespace(phase=phase,output=Path('/fixture')),
                   'connected_terminals':{f'mechanics:{d.SEEDS[0]}:{arm}':mechanics}}
        with patch.multiple(d, time=SimpleNamespace(perf_counter=lambda:clock[0]),
                fresh=lambda *a:{'identity':identity}, update=update,
                tamper_witness=lambda *a:None, image_witness=lambda *a:{},
                save=lambda *a:('file','digest'), release=lambda c,s:s.clear(),
                restore=lambda *a:{'identity':identity}, payload=lambda *a:{},
                fingerprint=lambda *a:'digest', inference_members=lambda *a:{},
                export_bundle=lambda *a:{'sha256':'bundle'}, qualify_bundle=qualify):
            result = d.arm_run(context,arm,d.SEEDS[0],'cpu' if phase == 'cpu' else 'cuda',
                               discarded_update=phase == 'cpu')
        assert result['total_training_core_seconds'] == core
        assert result['arm_work_seconds'] == core+100.
        assert len(result['steps']) == (128 if phase == 'train' else 17 if phase == 'mechanics' else 1)
        assert len(result['replay_steps']) == (17 if phase == 'mechanics' else 0)
        return result
    control = timed_arm('train','control',100.)
    candidate = timed_arm('train','candidate',180.)
    timed_arm('mechanics','candidate',34.)
    cpu = timed_arm('cpu','candidate',1.)
    trainer = SimpleNamespace(tensor_weakrefs=lambda *a:[],release=lambda c,s:s.clear())
    with patch.multiple(d, arm_run=lambda *a,**kw:cpu,
            select_initializer=lambda *a:{'checkpoint':{},'canonical_initial':{}},
            load_initializer=lambda *a:({k:None for k in d.STATIC_KEYS+('A','C')},{}), resource_check=lambda *a:None):
        assert d.cpu_run({'trainer':trainer,'original_cpu_record':{}})['total_training_core_seconds'] == 1.
    assert candidate['arm_work_seconds']/control['arm_work_seconds'] == 1.4
    tree = ast.parse(DRIVER.read_text())
    gate = next(n for n in function(tree,'run').body if isinstance(n,ast.If) and
                'fresh_control_record' in ast.unparse(n))
    code = compile(ast.Module(body=[gate],type_ignores=[]),'<actual in-process cost gate>','exec')
    context = {'fresh_control_record':{**control,'wall_seconds':200.},
               'connected_launch':{'fresh_control':{'service_seconds':260.}}}
    namespace = {**vars(d),'args':SimpleNamespace(phase='train',arm='candidate'),
                 'context':context,'result':candidate,'wall':280.}
    rejects(lambda:exec(code,namespace), 'core/whole cost')
    namespace.update(result={'total_training_core_seconds':140.},wall=300.)
    exec(code,namespace)  # Exact <=1.50 boundary, using matching wall scopes.
    namespace['wall'] = 310.  # Would pass against the longer control service time.
    rejects(lambda:exec(code,namespace), 'core/whole cost')

    file = {'path':'/fixture/authority.json','sha256':'a'*64}
    unit = {'receipt':file,'log':file,'unit':'fixture','invocation_id':'a'*32,
            'service_seconds':260.,'native_peak_rss_kib':1,'both_locks_held':True}
    initializer = {'checkpoint':file,'canonical_initial':{}}
    context = {'connected_args':SimpleNamespace(execution_sha256='b'*64),
               'connected_code':{},'source':{},'original_cpu_record':{'numerical_flags':{}}}
    for phase in ('cpu','mechanics','train'):
        launch = {'schema':d.AUTHORITY_SCHEMA,'execution_sha256':'b'*64,'phase':phase,
                  'arm':'control','seed':d.SEEDS[0],'recipe':copy.deepcopy(d.RECIPE),
                  'resource_policy':d.policy(phase),'both_locks_held':True,
                  'original_cpu':{'authority':file,'terminal':unit},
                  'actual_gradient':{'authority':file,'terminal':unit},
                  'witness':{'root':'/fixture/witness','files':{n:'a'*64 for n in d.WITNESS_FILES}},
                  'selected_cpu':None if phase == 'cpu' else unit,
                  'selected_mechanics':{a:unit for a in d.ARMS} if phase == 'train' else None,
                  'fresh_control':None}
        context['connected_launch'] = launch
        active = 'candidate' if phase == 'cpu' else 'control'
        count = 1 if phase == 'cpu' else 17 if phase == 'mechanics' else 128
        identity = {'arm':active,'seed':d.SEEDS[0],'device':'cpu' if phase == 'cpu' else 'cuda',
                    'method':d.method(launch),'scope':{'arm':'control','manifest_sha256':d.SCOPE_SHA256,
                    'arm_sha256':d.CONTROL_SHA256},'parameter_names':d.parameter_roles(active)[0],
                    'parameter_shapes':d.parameter_roles(active)[1]}
        rows = [{'step':s,'core_seconds':1.,'seconds':1.} for s in range(1,count+1)]
        replay = copy.deepcopy(rows) if phase == 'mechanics' else []
        total = float(len(rows)+len(replay))
        result = {'identity':identity,'training_updates':count,'steps':rows,'replay_steps':replay,
                  'independent_17_vs_8_plus_9':phase == 'mechanics','fresh_first17_replay':True,
                  'total_training_core_seconds':total,'arm_work_seconds':total+100.}
        record = {'schema':d.SCHEMA,'phase':phase,'arm':'control','seed':d.SEEDS[0],
                  'launch':launch,'execution_sha256':'b'*64,'code':{},'source':{},
                  'resource_policy':d.policy(phase),'quality_read':False,'state_reuse_eligible':False,
                  'cost_qualified':False,'model_fit_qualified':phase == 'train',
                  'total_training_core_seconds':total,'wall_seconds':total+110.,'process_peak_rss_kib':1,
                  'peak_cuda_allocated_bytes':0,'numerical_flags':{},
                  'invocation':{'optimize':0,'cuda_visible_devices':'' if phase == 'cpu' else '0',
                                'cublas_workspace_config':None if phase == 'cpu' else ':4096:8'},
                  'result':result,'qualifications':[result],
                  'accepted_initializers':[{'seed':s,**initializer} for s in d.SEEDS]}
        for key in ('pass','exit_rehash_pass','sequential_model_ownership','strict_reload_exact',
                    'native_training_inference_exact','inference_artifact_independent',
                    'bundle_original_dependencies_denied','updated_source_mutants_rejected',
                    'both_locks_held_in_parent_authority','cost_requires_parent_normal_exit_units',
                    'terminal_exit_and_both_locks_require_parent_receipt'):
            record[key] = True
        with patch.multiple(d, check_steps=lambda *a:None, select_initializer=lambda *a:initializer):
            d.check_terminal(context,record,phase,'control',d.SEEDS[0])
            for key,value in [('peak_cuda_allocated_bytes',10_000_000_000),
                              ('peak_cuda_allocated_bytes',-1),('peak_cuda_allocated_bytes',False),
                              ('peak_cuda_allocated_bytes',0.),('peak_cuda_allocated_bytes',None),
                              ('model_fit_qualified',phase != 'train'),('model_fit_qualified',int(phase == 'train')),
                              ('wall_seconds',float('inf')),('total_training_core_seconds',True),
                              ('process_peak_rss_kib',True),('pass',1),
                              ('cost_requires_parent_normal_exit_units',1)]:
                rejects(lambda k=key,v=value:d.check_terminal(context,{**record,k:v},phase,'control',d.SEEDS[0]), 'terminal')
            for key,value in [('cuda_visible_devices','0' if phase == 'cpu' else ''),('optimize',False)]:
                wrong = copy.deepcopy(record)
                wrong['invocation'][key] = value
                rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            for key,value in [('steps',rows[:-1]),('total_training_core_seconds',True),
                              ('arm_work_seconds',float('inf')),('training_updates',float(count))]:
                wrong = copy.deepcopy(record)
                (wrong['qualifications'][0] if phase == 'cpu' else wrong['result'])[key] = value
                rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            wrong = copy.deepcopy(record)
            (wrong['qualifications'][0] if phase == 'cpu' else wrong['result'])['steps'][0]['core_seconds'] = True
            rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            if phase != 'cpu':
                record['peak_cuda_allocated_bytes'] = 9_999_999_999
                d.check_terminal(context,record,phase,'control',d.SEEDS[0])
                record['invocation']['cublas_workspace_config'] = None
                rejects(lambda:d.check_terminal(context,record,phase,'control',d.SEEDS[0]), 'terminal')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-only', action='store_true', required=True)
    p.parse_args()
    assert DRIVER.exists(), 'new trainer missing'
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location('_connected_source_test', DRIVER)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    assert not {n.split('.')[0] for n in set(sys.modules)-before} & d.NATIVE
    source_contract(d)
    initializer_selection(d)
    authority_falsifiers(d)
    lifetime_restore_math(d)
    overlay_optimizer_seams(d)
    owned_loader_admission(d)
    restore_seam(d)
    current_encoder_seam(d)
    processor_release_seam(d)
    cost_terminal_falsifiers(d)
    print('PASS source-only connected MLP contracts/falsifiers; native UNRUN')


if __name__ == '__main__':
    main()
