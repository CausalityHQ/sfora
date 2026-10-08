"""Authenticated production inference closure; historical bundle code is evidence only.

Native byte parity and installed-wheel qualification belong to the parent gate.
"""

import copy
import gc
import hashlib
import inspect
import json
import math
import os
import re
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import CodeType, FunctionType



INFERENCE_SCHEMA = 'siglip2-connected-mlp-inference-v1'


BUNDLE_SCHEMA = 'siglip2-connected-mlp-bundle-v1'


FILES = {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}


SERVING_FILES = {'qualify_siglip2_substrate_cpu.py', 'extract_siglip2_vision_source.py',
                 'train_siglip2_cached_readout.py', 'train_siglip2_substrate_adaptation.py',
                 'prototype_residual_readout.py', 'quadratic_readout.py'}


NATIVE = {'torch','numpy','PIL','transformers','safetensors','torchvision','sfora'}


MLP = tuple('encoder.layers.26.mlp.'+layer+'.'+field for layer in ('fc1','fc2') for field in ('weight','bias'))


MLP_SHAPES = [[4304,1152],[4304],[1152,4304],[1152]]


BACKEND_SHA = '250394884a9f90845cf87b6fc0cf3341337b193428556c6ff61ed2b04bedb692'


SCOPE_SHA256 = '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726'


CONTROL_SHA256 = '1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280'


INFERENCE_KEYS = {'schema','source','arm','config','buffers','processor','head','A','C','means',
                  'mu_train','mu_train_provenance','scope','common_statistics','numerical_flags',
                  'base_vision','encoder','encoder_identity','vision_sha256','fixed_sha256'}


ARMS = ('control','candidate')
_binding = None


def _bind_runtime(historical_code, runtime_guards):
    """Called once by the authenticated bridge with its literal authority record."""
    global _binding
    require(_binding is None and type(historical_code) is tuple and
            type(runtime_guards) is tuple and len(runtime_guards) == 3,
            'authenticated installed runtime binding required')
    require(tuple(sorted(historical_code)) == historical_code and
            {name for name,sha in historical_code} == FILES | SERVING_FILES | {'joint_relational_compaction.py'} and
            all(type(name) is str and type(sha) is str and re.fullmatch('[0-9a-f]{64}',sha)
                for name,sha in historical_code), 'exact historical inference authority required')
    require(tuple(Path(path).name for path,sha in runtime_guards) ==
            ('connected_inference.py','_connected_inference_authority.py','packed_int8.py') and
            runtime_guards[0][0] == __file__, 'installed inference authority paths differ')
    for path,sha in runtime_guards:
        bound_file({},path,sha)
    _binding = historical_code,runtime_guards


def _check_runtime():
    require(_binding is not None, 'authenticated installed runtime binding required')
    for path,sha in _binding[1]:
        bound_file({},path,sha)


def _head_method_code(name):
    _check_runtime()
    raw = Path(__file__).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == _binding[1][0][1], 'installed head source changed before use')
    code = compile(raw,__file__,'exec',dont_inherit=True)
    factory = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == 'head_from')
    residual = next(c for c in factory.co_consts if isinstance(c,CodeType) and c.co_name == 'Residual')
    return next(c for c in residual.co_consts if isinstance(c,CodeType) and c.co_name == name)


def _pack(unit):
    _check_runtime()
    from sfora.packed_int8 import pack_int8_unit_embeddings
    return pack_int8_unit_embeddings(unit)



def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_roles(arm):
    require(type(arm) is str and arm in ARMS, 'fixed arm required')
    names, shapes = ['A','C'], [[128,160],[128,1152]]
    if arm == 'candidate':
        names += list(MLP)
        shapes += copy.deepcopy(MLP_SHAPES)
    return names, shapes, sum(math.prod(shape) for shape in shapes)


def mapping_absent(path):
    stat = Path(path).stat()
    for line in Path('/proc/self/maps').read_text().splitlines():
        _,_,_,device,inode,*_ = line.split(maxsplit=5)
        major,minor = (int(v,16) for v in device.split(':'))
        require((major,minor,int(inode)) != (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino),
                'checkpoint mapping survived independent copy/release')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def file_fact(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and Path(value['path']).is_absolute() and
            isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'exact FILE required')


def bound_file(guards, path, expected):
    file_fact({'path': str(path), 'sha256': expected})
    path = Path(path)
    require(path.resolve() == path and path.is_file(), 'canonical regular FILE required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        buffer = bytearray(1024**2)
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
            os.posix_fadvise(stream.fileno(), stream.tell() - count, count, os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'current FILE bytes differ: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    return path


def batch_bound_files(guards, items):
    """Fresh per occurrence; publish on the owner only after complete success."""
    items = list(items)
    for path, expected in items:
        file_fact({'path': str(path), 'sha256': expected})
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bound_file, {}, path, expected) for path, expected in items]
        paths = [future.result() for future in futures]
    staged = dict(guards)
    for path, (_, expected) in zip(paths, items):
        require(staged.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    guards.update(staged)
    return paths


def read_json(fact, guards):
    file_fact(fact)
    path = bound_file(guards, fact['path'], fact['sha256'])
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == fact['sha256'], 'JSON size/current bytes differ')
    return strict_json(raw)


def fullfeature_raw_features(features, head, A, means, C, mu_train, arm):
    """Public FP32 readout over the actual normalized input, with one concat call."""
    import torch
    from torch.nn import functional as F
    parameter_roles(arm)
    _check_tensor(C, (128, 1152), features.device)
    _check_tensor(mu_train, (1152,), features.device, frozen=True)
    require(torch.isfinite(C).all().item() and torch.isfinite(mu_train).all().item(),
            'finite residual/mean required')
    with torch.autocast(features.device.type, enabled=False):
        raw = raw_features(features, head, A, means, 'concat')
        raw = raw + F.linear(features.detach().float() - mu_train, C)
        require(torch.isfinite(raw).all().item(), 'finite fullfeature raw required')
    return raw


def _processor_cache(processor, guards, *, empty=False):
    """Authenticate the one original self-keyed cache; never adopt prior entries."""
    backend = sys.modules.get('transformers.image_processing_backends')
    siglip = sys.modules.get('transformers.models.siglip.image_processing_siglip')
    require(backend is not None and siglip is not None and
            type(processor) is getattr(siglip, 'SiglipImageProcessor', None) and
            type(processor).__bases__ == (getattr(backend, 'TorchvisionBackend', None),),
            'processor cache owner differs')
    path = Path(backend.__file__)
    require(path.is_absolute() and path.resolve() == path and
            guards.get(str(path)) == BACKEND_SHA, 'processor cache source guard differs')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BACKEND_SHA, 'processor cache source differs')
    name = '_fuse_mean_std_and_rescale_factor'
    wrapper = vars(backend.TorchvisionBackend).get(name)
    require(type(wrapper) is type(lru_cache(maxsize=10)(lambda: None)) and
            inspect.getattr_static(processor, name) is wrapper and
            wrapper.cache_parameters() == {'maxsize': 10, 'typed': False} and
            all(getattr(wrapper, key) == getattr(type(wrapper), key).__get__(wrapper)
                for key in ('cache_info', 'cache_clear')) and wrapper.cache_info().maxsize == 10,
            'processor cache wrapper differs')
    function = getattr(wrapper, '__wrapped__', None)
    # Read-only GC edges bind __wrapped__ to the LRU's actual callable, not a decoy.
    require(type(function) is FunctionType and
            [obj for obj in gc.get_referents(wrapper) if type(obj) is FunctionType] == [function],
            'processor cache wrapper callable differs')
    # Compile only: no re-execution of the backend or replacement of its globals/math.
    module_code = compile(raw, str(path), 'exec', dont_inherit=True)
    class_code = next(c for c in module_code.co_consts if isinstance(c, CodeType) and c.co_name == 'TorchvisionBackend')
    original = next(c for c in class_code.co_consts if isinstance(c, CodeType) and c.co_name == name)
    require(function.__code__ == original and function.__globals__ is vars(backend) and
            function.__defaults__ == (None,) * 6 and function.__kwdefaults__ is None and
            function.__closure__ is None, 'processor cache live code differs')
    require(not empty or wrapper.cache_info().currsize == 0, 'processor cache must initially be empty')
    return wrapper


def apply_overlay(model, encoder):
    """Copy absolute parameters into the exact existing objects; never rebind."""
    import torch
    require(isinstance(encoder,dict) and encoder.keys() == set(MLP), 'exact four overlay names required')
    params = dict(model.named_parameters())
    require(len(params) == 448 and params.keys() == model.state_dict().keys() and set(MLP) <= params.keys(),
            'complete original 448 parameter inventory required')
    for name,shape in zip(MLP,MLP_SHAPES,strict=True):
        value,parameter = encoder[name],params[name]
        require(isinstance(value,torch.Tensor) and list(value.shape) == shape == list(parameter.shape) and
                value.dtype == parameter.dtype == torch.float32 and not value.requires_grad and value.grad_fn is None and
                torch.isfinite(value).all().item(), 'absolute finite FP32 overlay shape/dtype/role differs: '+name)
    with torch.no_grad():
        for name in MLP:
            params[name].copy_(encoder[name])


def model_structure(model, packages):
    """Original runtime module facts, separate from updated bytes and roles."""
    import torch
    modules = []
    for name,module in model.named_modules():
        require(not module.training and not module._forward_hooks and not module._forward_pre_hooks and
                not module._backward_hooks and not getattr(module,'gradient_checkpointing',False), 'encoder mode/hooks differ')
        row = {'name':name,**module_origin(type(module),packages),'training':module.training}
        row['attributes'] = {key:value for key,value in vars(module).items() if not key.startswith('_') and
            (value is None or isinstance(value,(bool,int,float,str)) or isinstance(value,(tuple,list)) and
             all(isinstance(item,(bool,int,float,str)) for item in value))}
        if hasattr(module,'config'):
            row['attn_implementation'] = module.config._attn_implementation
        if isinstance(module,torch.nn.LayerNorm):
            require(module.eps == model.config.layer_norm_eps, 'LayerNorm epsilon differs')
        modules.append(row)
    require(model.config._attn_implementation == 'sdpa', 'original SDPA implementation required')
    return json.loads(json.dumps({'config':model.config.to_dict(),'attn_implementation':model.config._attn_implementation,
                                  'modules':modules},allow_nan=False))


def construct_encoder(construct_context, config, buffers, processor_config, base,
                      overlay=None):
    """One genuine CPU factory/strict base load, then exact overlay and transfer."""
    import torch
    from transformers import AutoImageProcessor
    guards = construct_context['guards']
    fact = base['checkpoint']
    path = bound_file(guards,fact['path'],fact['sha256'])
    model = construct(config,construct_context).eval()
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        with path.open('rb') as stream:
            pages = CheckpointPages(stream)
            require(disk.keys() == {'vision','buffers','config','runtime','cpu_rng'} and disk['config'] == config and
                    fingerprint(disk['buffers']) == fingerprint(buffers) and
                    len(disk['vision']) == 448, 'complete authenticated original base differs')
            digest = fingerprint(disk['vision'],consumed=pages.consume)
            require(base.get('sha256',digest) == digest, 'original typed base vision identity differs')
            load_vision(model,disk['vision'],pages)
            require(dict(model.named_buffers()).keys() == buffers.keys() == {'embeddings.position_ids'},
                    'complete nonpersistent buffer inventory differs')
            with torch.no_grad():
                for name,value in model.named_buffers():
                    require(value.shape == disk['buffers'][name].shape and value.dtype == disk['buffers'][name].dtype,
                            'buffer shape/dtype differs')
                    value.copy_(disk['buffers'][name])
                    pages.consume(disk['buffers'][name])
            del value
            require(fingerprint(model.state_dict()) == digest, 'genuine strict copied original448 differs')
        processor = AutoImageProcessor.from_pretrained(processor_config,local_files_only=True,backend='torchvision')
        cache = _processor_cache(processor,guards,empty=True)
        require('position_ids' in model.embeddings._non_persistent_buffers_set and
                torch.equal(dict(model.named_buffers())['embeddings.position_ids'],torch.arange(256).expand(1,-1)),
                'original nonpersistent position buffer differs')
        if overlay is not None:
            apply_overlay(model,overlay)
        structure = model_structure(model,construct_context['packages'])
        base = {'checkpoint':copy.deepcopy(fact),'sha256':digest}
        del pages
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    return model,processor,cache,base,structure


def encoder_facts(state, packages, *, serving=False):
    import torch
    model = state['model']
    params = dict(model.named_parameters())
    ident = state['encoder_identity']
    require(len(params) == 448 and params.keys() == model.state_dict().keys() == ident['inventory'].keys() and
            sum(n not in MLP for n in params) == 444, 'exact four/frozen444 encoder inventory differs')
    for name,p in params.items():
        require(list(p.shape) == ident['inventory'][name] and p.dtype == torch.float32 and p.device.type == state['device'] and
                p.grad is None and p.is_leaf and p.grad_fn is None and
                p.requires_grad is (not serving and state['arm'] == 'candidate' and name in MLP),
                'encoder parameter shape/dtype/device/role/gradient differs: '+name)
    nonpersistent = {n:sorted(m._non_persistent_buffers_set) for n,m in model.named_modules() if m._non_persistent_buffers_set}
    require(nonpersistent == ident['nonpersistent'] == {'embeddings':['position_ids']}, 'nonpersistent registration differs')
    buffers = dict(model.named_buffers())
    require(buffers.keys() == {'embeddings.position_ids'} and
            fingerprint(buffers) == ident['buffers_sha256'], 'complete current buffers differ')
    require(model_structure(model,packages) == ident['runtime'], 'current config/runtime/module origin differs')
    processor = state['processor_object']
    require(json.loads(processor.to_json_string()) == state['processor']['config'] and
            processor.backend == state['processor']['backend'] and
            module_origin(type(processor),packages) == state['processor']['origin'], 'current processor origin/config differs')
    require(_processor_cache(processor,state['guards']) is state['processor_cache'], 'authenticated processor cache changed')
    frozen = fingerprint({n:p for n,p in params.items() if n not in MLP})
    require(frozen == ident['frozen_sha256'], 'current frozen444 bytes differ')
    return {'vision_sha256':fingerprint(model.state_dict()),
            'encoder':{n:fingerprint(params[n]) for n in MLP}}


def admit_bundle(directory, digest):
    _check_runtime()
    directory,guards = Path(directory),{}
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), 'canonical owned bundle required')
    manifest = read_json({'path':str(directory/'bundle.json'),'sha256':digest},guards)
    require(manifest.keys() == {'schema','code','files','endpoint_state_sha256','environment','encoder_identity',
                               'base_vision_sha256','vision_sha256','scope'} and manifest['schema'] == BUNDLE_SCHEMA and
            manifest['code'].keys() == FILES | SERVING_FILES | {'joint_relational_compaction.py'} and
            manifest['files'].keys() == {'vision.pt','endpoint.pt','processor.json'} and
            manifest['scope'] == {'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256} and
            all(re.fullmatch('[0-9a-f]{64}',manifest[k]) for k in ('endpoint_state_sha256','base_vision_sha256','vision_sha256')),
            'exact updated owned bundle schema/closure differs')
    for name,digest in {**manifest['code'],**manifest['files']}.items():
        path = bound_file(guards,directory/name,digest)
        require(path.stat().st_nlink == 1 and not path.is_symlink(), 'regular single-link owned bundle required')
    require(tuple(sorted(manifest['code'].items())) == _binding[0],
            'unsupported historical inference closure')
    env = manifest['environment']
    require(env.keys() == {'packages','files','native_files','vision_constructor'} and env['vision_constructor'] in env['files'] and
            env['packages'].keys() == NATIVE-{'sfora'} and set(env['native_files']) <= set(env['files']),
            'admitted serving environment differs')
    roots = [Path(v['root']) for v in env['packages'].values()]
    require(all(any(Path(p).is_relative_to(r) for r in roots) or env['native_files'].get(p) == h for p,h in env['files'].items()),
            'serving environment contains TRAIN data')
    batch_bound_files(guards,env['files'].items())
    for path,sha in _binding[1]:
        bound_file(guards,path,sha)
    return manifest,guards


def inference_readout_tree(endpoint):
    return {'arm':endpoint['arm'],'scope':endpoint['scope'],'common_statistics':endpoint['common_statistics'],
            'A':endpoint['A'].detach(),'C':endpoint['C'].detach(),'means':endpoint['means'],'mu_train':endpoint['mu_train'],
            'mu_train_provenance':endpoint['mu_train_provenance'],'head':dict(endpoint['head_object'].state_dict()),
            'A_trainable':endpoint['A'].requires_grad,'C_trainable':endpoint['C'].requires_grad}


def owned_copy(value, pages, device='cpu'):
    import torch
    if isinstance(value,torch.Tensor):
        return pages.copy(value,device)
    if isinstance(value,dict):
        return {k:owned_copy(v,pages,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(owned_copy(v,pages,device) for v in value)
    return copy.deepcopy(value)


def load_inference(directory, bundle_sha256, device):
    """Installed public loader uses authenticated historical evidence without executing it."""
    directory = Path(directory)
    manifest,guards = admit_bundle(directory,bundle_sha256)
    require(device in ('cpu','cuda'), 'fixed serving device required')
    modules = {'runtime':sys.modules[__name__]}
    import torch
    path = directory/'endpoint.pt'
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        require(disk.keys() == INFERENCE_KEYS and disk['schema'] == INFERENCE_SCHEMA and disk['arm'] in ARMS and
                fingerprint(disk) == manifest['endpoint_state_sha256'] and
                fingerprint({k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and
                numerical_flags() == disk['numerical_flags'] and
                disk['vision_sha256'] == manifest['vision_sha256'] and disk['encoder_identity'] == manifest['encoder_identity'] and
                disk['base_vision']['sha256'] == manifest['base_vision_sha256'] and
                disk['scope']['arm'] == 'control' and disk['scope']['payload']['scope_sha256'] == CONTROL_SHA256 and
                len(disk['scope']['payload']['class_names']) == 1008,
                'complete original-scope/updated inference identity differs')
        env = manifest['environment']
        construct_context = {'packages':env['packages'],'guards':guards,
            'sources':{'native_environment':{'vision_constructor':{'path':env['vision_constructor']}}}}
        owned_base = {'checkpoint':{'path':str(directory/'vision.pt'),'sha256':manifest['files']['vision.pt']},
                      'sha256':disk['base_vision']['sha256']}
        model,processor,cache,_,structure = construct_encoder(construct_context,disk['config'],disk['buffers'],
            directory/'processor.json',owned_base,disk['encoder'])
        # The overlay is copied by apply_overlay within the shared strict CPU
        # constructor before device transfer; the full updated identity is checked.
        require(fingerprint(model.state_dict()) == disk['vision_sha256'] and
                structure == disk['encoder_identity']['runtime'], 'updated full448 portable reload differs')
        model.requires_grad_(False).eval().to(device)
        with path.open('rb') as stream:
            pages = CheckpointPages(stream)
            copied = {k:owned_copy(disk[k],pages,device) for k in ('head','A','C','means','mu_train','common_statistics')}
            head = head_from('control',tensors=copied.pop('head')).to(device).requires_grad_(False).train()
            endpoint = {**copied,'A':torch.nn.Parameter(copied['A'],requires_grad=True),
                'C':torch.nn.Parameter(copied['C'],requires_grad=True),'head_object':head,'model':model,
                'processor_object':processor,'processor_cache':cache,'processor':copy.deepcopy(disk['processor']),
                'arm':disk['arm'],'scope':copy.deepcopy(disk['scope']),
                'mu_train_provenance':copy.deepcopy(disk['mu_train_provenance']),
                'encoder_identity':copy.deepcopy(disk['encoder_identity']),'vision_sha256':disk['vision_sha256'],
                'flags':copy.deepcopy(disk['numerical_flags']),'device':device,'modules':modules,'guards':guards,'manifest':manifest,'directory':directory}
            endpoint['readout_sha256'] = fingerprint(inference_readout_tree(endpoint))
            del copied,pages,model,processor,head
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    require(encoder_facts(endpoint,manifest['environment']['packages'],serving=True)['vision_sha256'] ==
            endpoint['vision_sha256'], 'public updated encoder identity differs')
    return endpoint


def inference_outputs(endpoint, images):
    import torch
    from torch.nn import functional as F
    modules,device = endpoint['modules'],endpoint['device']
    _check_runtime()
    for module in modules.values():
        bound_file({},module.__file__,endpoint['guards'][module.__file__])
    for filename in SERVING_FILES | {'joint_relational_compaction.py'}:
        path = endpoint['directory']/filename
        bound_file({},path,endpoint['guards'][str(path)])
    require(0 < len(images) <= 32 and numerical_flags() == endpoint['flags'], 'serving batch/numerics differ')
    require(endpoint['encoder_identity'] == endpoint['manifest']['encoder_identity'] and
            endpoint['vision_sha256'] == endpoint['manifest']['vision_sha256'] and
            all(p.grad is None and not p.requires_grad and p.dtype == torch.float32 and p.device.type == device
                for p in endpoint['head_object'].parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in endpoint['head_object'].modules()), 'serving authenticated encoder/frozen head roles/hooks differ')
    require(encoder_facts(endpoint,endpoint['manifest']['environment']['packages'],serving=True)['vision_sha256'] ==
            endpoint['vision_sha256'] and fingerprint(inference_readout_tree(endpoint)) == endpoint['readout_sha256'],
            'current .data updated encoder/readout/role substitution rejected')
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else []
    pixels = endpoint['processor_object'](images=images,return_tensors='pt')['pixel_values']
    require(pixels.shape == (len(images),3,256,256) and pixels.dtype == torch.float32 and torch.isfinite(pixels).all().item(),
            'owned processor pixels differ')
    with torch.no_grad():
        with torch.autocast(device,dtype=torch.float16,enabled=device == 'cuda'):
            pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device,enabled=False):
            features = F.normalize(pooled.float(),dim=1)
            raw = fullfeature_raw_features(features,endpoint['head_object'],endpoint['A'],endpoint['means'],endpoint['C'],
                endpoint['mu_train'],endpoint['arm'])
            require((raw.norm(dim=1) > 0).all().item(), 'nonzero portable raw required')
            unit = F.normalize(raw,dim=1)
            packed = _pack(unit.cpu())
    require(torch.equal(cpu_rng,torch.random.get_rng_state()) and
            all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if device == 'cuda' else [],strict=True)),
            'serving complete RNG changed')
    return {'raw':raw.cpu(),'unit':unit.cpu(),'codes':packed.codes.cpu(),'inverse_norms':packed.inverse_norms.cpu(),'wire':packed.to_bytes()}


def release_inference(endpoint):
    modules = tuple(endpoint['modules'].values())
    refs = [weakref.ref(endpoint[n]) for n in ('model','processor_object','head_object','A','C','mu_train')]
    refs += [weakref.ref(p) for p in (*endpoint['model'].parameters(),*endpoint['model'].buffers(),
                                    *endpoint['head_object'].parameters(),*endpoint['head_object'].buffers())]
    cache = endpoint['processor_cache']
    require(_processor_cache(endpoint['processor_object'],endpoint['guards']) is cache, 'serving processor teardown authority differs')
    cache.cache_clear()
    require(cache.cache_info().currsize == 0, 'serving processor teardown failed')
    endpoint.clear()
    for module in modules:
        require(sys.modules.pop(module.__name__,None) is module, 'owned serving registry changed')
    gc.collect()
    require(all(ref() is None for ref in refs), 'serving model/processor/readout lifetime survived release')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        sys.modules['torch'].cuda.empty_cache()


def fingerprint(value, frozen=None, consumed=None):
    """Typed, length-framed complete tree hash; tensor device is not identity."""
    import torch
    digest = hashlib.sha256()
    def frame(raw):
        raw = raw.encode() if isinstance(raw, str) else raw
        digest.update(str(len(raw)).encode() + b':' + raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            frame('Tensor')
            key = (item.data_ptr(), item._version, str(item.dtype), tuple(item.shape))
            fact = frozen.get(key) if frozen is not None else None
            if fact is None:
                raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                fact = (str(item.dtype), tuple(item.shape), hashlib.sha256(memoryview(raw)).hexdigest())
            if consumed is not None:
                consumed(item)
            visit(fact)
        elif isinstance(item, dict):
            frame('dict'); frame(str(len(item)))
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            frame(type(item).__name__); frame(str(len(item)))
            for child in item:
                visit(child)
        else:
            frame(type(item).__name__); frame(repr(item))
    visit(value)
    return digest.hexdigest()


class CheckpointPages:
    """Drop only consumed complete pages of this read-only-use Torch private mmap.

    fadvise alone cannot drop mapped pages. Validate the actual VMA against the
    open file before madvise; later aliases can refault unchanged archive bytes.
    Advice is not a resource-fit guarantee. Never use on mutated mapped tensors.
    """
    def __init__(self, stream):
        import ctypes
        self.fd = stream.fileno()
        stat = os.fstat(self.fd)
        self.size, self.page = stat.st_size, os.sysconf('SC_PAGESIZE')
        matches = []
        for line in Path('/proc/self/maps').read_text().splitlines():
            span, mode, offset, device, inode, *_ = line.split(maxsplit=5)
            major, minor = (int(part, 16) for part in device.split(':'))
            if (major, minor, int(inode)) == (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                matches.append((span, mode, int(offset, 16)))
        require(len(matches) == 1, 'checkpoint mapping identity differs')
        span, mode, offset = matches[0]
        self.start, end = (int(part, 16) for part in span.split('-'))
        require(mode == 'rw-p' and offset == 0 and self.start % self.page == 0 and
                end - self.start == (self.size + self.page - 1) // self.page * self.page,
                'checkpoint mapping identity differs')
        self.madvise = ctypes.CDLL(None, use_errno=True).madvise
        self.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        self.madvise.restype = ctypes.c_int

    def release(self, address, count):
        import ctypes
        import mmap
        if count == 0:
            return
        require(count > 0 and self.start <= address <= address + count <= self.start + self.size,
                'checkpoint mapping range differs')
        start = (address + self.page - 1) // self.page * self.page
        end = (address + count) // self.page * self.page
        if end > start:
            if self.madvise(start, end - start, mmap.MADV_DONTNEED) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            os.posix_fadvise(self.fd, start - self.start, end - start, os.POSIX_FADV_DONTNEED)

    def consume(self, value):
        require(value.device.type == 'cpu', 'checkpoint CPU tensor required')
        # A strided view need not consume its storage's gaps; leave those pages
        # alone. Vision weights/moments are contiguous; full hashing is unchanged.
        if value.is_contiguous():
            self.release(value.data_ptr(), value.numel() * value.element_size())

    def copy(self, value, device='cpu'):
        result = value.to(device, copy=True)  # Blocking; also owns same-device CPU steps.
        self.consume(value)
        return result


def load_vision(model, vision, pages):
    """Keep the genuine strict recursive loader; release each copied module's bytes."""
    handles = []
    try:
        for prefix, module in model.named_modules():
            prefix = prefix + '.' if prefix else ''
            keys = tuple(prefix + name for name, _ in
                         list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
                         if prefix + name in vision)
            def consumed(module, incompatible, keys=keys):
                for key in keys:
                    pages.consume(vision[key])
            handles.append(module.register_load_state_dict_post_hook(consumed))
        model.load_state_dict(vision, strict=True)
    finally:
        for handle in handles:
            handle.remove()


def numerical_flags():
    import torch
    return {'default_dtype': str(torch.get_default_dtype()), 'default_device': str(torch.get_default_device()),
            'grad_enabled': torch.is_grad_enabled(), 'inference_mode': torch.is_inference_mode_enabled(),
            'autocast_cpu': torch.is_autocast_enabled('cpu'), 'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'autocast_cpu_dtype': str(torch.get_autocast_dtype('cpu')),
            'autocast_cuda_dtype': str(torch.get_autocast_dtype('cuda')),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'sdpa_flash': torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_math': torch.backends.cuda.math_sdp_enabled(),
            'sdpa_mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_cudnn': torch.backends.cuda.cudnn_sdp_enabled()}


def loaded_module_origin(name, module, packages):
    message = 'loaded native module origin differs: ' + name
    if name in ('torch.ops', 'torch.classes'):
        attr = name.split('.')[1]
        backing_name, marker = 'torch._' + attr, '_' + attr + '.py'
        class_name = {'ops': '_Ops', 'classes': '_Classes'}[attr]
        cls = type(module)
        require(inspect.ismodule(module) and module.__name__ == name and
                cls.__module__ == backing_name and cls.__name__ == cls.__qualname__ == class_name and
                cls is getattr(sys.modules.get(backing_name), class_name, None) and
                module is getattr(sys.modules.get(backing_name), attr, None) and
                module is getattr(sys.modules.get('torch'), attr, None) and
                getattr(module, '__file__', None) == marker, message)
        path = Path(module_origin(cls, packages)['file'])
        require(path == Path(packages['torch']['root']) / marker, message)
    elif getattr(module, '__file__', None):
        path = Path(module.__file__).resolve()
    else:
        return None
    require(path.is_relative_to(Path(packages[name.split('.')[0]]['root'])) and path.is_file(), message)
    return path


def loaded_origins(packages):
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in packages:
            loaded_module_origin(name, module, packages)


def module_origin(cls, packages):
    path = Path(inspect.getfile(cls)).resolve()
    package = cls.__module__.split('.')[0]
    require(package in packages and path.is_relative_to(Path(packages[package]['root'])) and path.is_file(),
            'class module origin differs: ' + cls.__module__)
    return {'class': cls.__module__ + '.' + cls.__qualname__, 'file': str(path)}


def construct(config, context):
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    loaded_origins(context['packages'])
    require(module_origin(SiglipVisionModel, context['packages'])['file'] ==
            context['sources']['native_environment']['vision_constructor']['path'], 'loaded constructor origin differs')
    for cls in (SiglipVisionModel, SiglipVisionConfig):
        path = module_origin(cls, context['packages'])['file']
        require(path in context['guards'],
                'loaded vision/config source hash differs')
        bound_file({},path,context['guards'][path])
    require(torch.get_default_dtype() == torch.float32 and str(torch.get_default_device()) == 'cpu',
            'FP32 CPU constructor defaults required')
    rng, flags = torch.random.get_rng_state().clone(), numerical_flags()
    resolved = SiglipVisionConfig.from_dict(config)
    resolved._attn_implementation = 'sdpa'
    with torch.random.fork_rng(devices=[]):
        model = SiglipVisionModel(resolved).float()
    require(torch.equal(rng, torch.random.get_rng_state()) and numerical_flags() == flags,
            'constructor changed CPU RNG/numerical flags')
    require(hasattr(model, 'embeddings') and hasattr(model, 'encoder') and hasattr(model, 'head') and
            not hasattr(model, 'vision_model'), 'installed direct vision layout required')
    return model


def head_from(arm, tensors=None, values=None, features=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    require(arm in ARMS, 'fixed head arm required')
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.primary = nn.Linear(1152, 128)
            self.down = nn.Linear(1152, 32, bias=False)
            self.up = nn.Linear(32, 128, bias=False)
            self.register_buffer('center', torch.zeros(1152))
            self.register_buffer('preactivation_std', torch.ones(()))
        def residual(self, unit):
            z = self.down(unit - self.center)
            return self.up(.5 * z if arm == 'control' else F.gelu(z, approximate='none'))
        def forward(self, source):
            unit = F.normalize(source.float(), dim=1)
            return self.primary(unit) + self.residual(unit)
    with torch.random.fork_rng(devices=[]):
        head = Residual().float()
    if tensors is not None:
        head.load_state_dict(tensors, strict=True)
    else:
        require(values is not None and features is not None and features.shape == (13283, 1152),
                'complete FIT initializer required')
        with torch.no_grad():
            head.primary.load_state_dict({'weight': values['head.weight'], 'bias': values['head.bias']}, strict=True)
            unit = F.normalize(features, dim=1)
            head.center.copy_(unit.mean(0))
            nn.init.kaiming_uniform_(head.down.weight, a=5**.5,
                                    generator=torch.Generator().manual_seed(179034))
            std = head.down(unit - head.center).std(unbiased=False)
            require(torch.isfinite(std).item() and float(std) > 0, 'FIT preactivation std undefined')
            head.preactivation_std.copy_(std)
            head.down.weight.div_(std)
            head.up.weight.zero_()
    require(sum(p.numel() for p in head.parameters()) == 188544, '188544 head scalars required')
    return head


def feature_width(arm):
    if type(arm) is not str or arm not in ('linear', 'concat'):
        raise ValueError('fixed signed readout arm required')
    return 32 if arm == 'linear' else 160


def basis(Z, H0, arm):
    import torch

    feature_width(arm)
    if tuple(Z.shape) != (H0.shape[0], 32) or tuple(H0.shape) != (Z.shape[0], 128):
        raise ValueError('signed basis widths/order differ')
    return Z if arm == 'linear' else torch.cat((Z, H0), dim=1)


def check_weight(A, device, arm):
    import torch

    _check_tensor(A, (128, feature_width(arm)), device)
    require(isinstance(A, torch.nn.Parameter) and A.requires_grad and
                       A.is_leaf and A.grad_fn is None, 'trainable leaf Parameter A required')


def check_means(means, device):
    require(isinstance(means, dict) and means.keys() == {'linear', 'concat'},
                       'both fixed signed means required')
    selected = str(means['linear'].device)
    require(selected in ('cpu', str(device)), 'means must be CPU or selected device')
    for arm, value in means.items():
        _check_tensor(value, (feature_width(arm),), selected, frozen=True)


def raw_features(features, base, A, means, arm):
    import torch
    from torch.nn import functional as F

    device = features.device
    require(device.type in ('cpu', 'cuda'), 'CPU or CUDA device required')
    _check_features(features, device)
    source = _check_base(base, device)
    check_weight(A, device, arm)
    check_means(means, device)
    with torch.autocast(device.type, enabled=False):
        with torch.no_grad():
            _finite(torch, [features, A, *source, *means.values()])
            x = features.detach().float()
            h0 = base(x).detach()
            z = base.down(F.normalize(x, dim=1) - base.center)
            phi = basis(z, h0, arm) - means[arm].detach().to(device=device)
            _check_tensor(h0, (features.shape[0], 128), device, frozen=True)
            _finite(torch, [h0, z, phi])
        raw = h0 + F.linear(phi.detach(), A)
        _finite(torch, [raw])
    return raw


def _check_tensor(value, shape, device, frozen=False, dtype="torch.float32"):
    require(tuple(value.shape) == shape and str(value.dtype) == dtype and
             str(value.device) == str(device) and str(value.layout) == "torch.strided",
             "tensor shape/dtype/device/layout differs")
    if frozen:
        require(not value.requires_grad and value.grad_fn is None, "frozen detached tensor required")


def _check_base(base, device):
    require(type(base).__qualname__ == "head_from.<locals>.Residual" and
             not {"forward", "residual"}.intersection(base.__dict__), "original control factory required")
    for name in ("forward", "residual"):
        method = getattr(type(base), name)
        require(method.__code__.co_qualname == "head_from.<locals>.Residual." + name and
                 method.__code__ == _head_method_code(name) and method.__globals__ is globals() and
                 method.__defaults__ is None and method.__kwdefaults__ is None and
                 Path(method.__code__.co_filename) == Path(__file__),
                 "original source method required")
    residual = type(base).residual
    captured = dict(zip(residual.__code__.co_freevars, residual.__closure__ or (), strict=True))
    require("arm" in captured and captured["arm"].cell_contents == "control", "control source factory required")
    shapes = {"primary.weight": (128, 1152), "primary.bias": (128,),
              "down.weight": (32, 1152), "up.weight": (128, 32)}
    params, buffers = dict(base.named_parameters()), dict(base.named_buffers())
    require(params.keys() == shapes.keys() and buffers.keys() == {"center", "preactivation_std"},
             "complete source parameters/buffers required")
    for name, shape in shapes.items():
        _check_tensor(params[name], shape, device, frozen=True)
    _check_tensor(buffers["center"], (1152,), device, frozen=True)
    _check_tensor(buffers["preactivation_std"], (), device, frozen=True)
    return [*params.values(), *buffers.values()]


def _check_features(features, device, train=False):
    shape = tuple(features.shape)
    require((shape == (6355, 1152) if train else
              len(shape) == 2 and shape[0] > 0 and shape[1] == 1152), "feature matrix shape differs")
    dtype = str(features.dtype)
    require(dtype == "torch.float32" if train else dtype in
             ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"), "floating feature dtype required")
    _check_tensor(features, shape, device, dtype=dtype)


def _finite(torch, values):
    require(all(torch.isfinite(value).all().item() for value in values), "nonfinite readout tensor")
