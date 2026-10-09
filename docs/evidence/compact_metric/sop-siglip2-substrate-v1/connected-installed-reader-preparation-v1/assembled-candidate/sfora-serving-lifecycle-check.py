"""Real release body plus candidate lifecycle; fake native/cache/admission seams."""
import ast, gc, hashlib, json, sys, tempfile, types, weakref
from pathlib import Path
from unittest.mock import patch
root=Path('/home/rb/worktrees/sfora-positive-causality')
source=(root/'src/sfora/connected_inference.py').read_text();tree=ast.parse(source)
original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='release_inference')
legacy=ast.get_source_segment(source,original)
new=legacy.replace('def release_inference(endpoint):\n','def release_inference(endpoint):\n    if "_serving" in endpoint:\n        return _serving_release(endpoint)\n',1)
assert ast.dump(ast.Module(body=ast.parse(new).body[0].body[1:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=original.body,type_ignores=[]),include_attributes=False)
class Leaf: pass
class Cache:
    def __init__(self,owner): self.owner=owner
    def cache_clear(self): self.owner=None
    def cache_info(self): return types.SimpleNamespace(currsize=0)
class Processor:
    def __init__(self): self.cache=Cache(self)
class Model:
    def __init__(self): self.parameters_= [Leaf()];self.buffers_=[Leaf()]
    def parameters(self): return self.parameters_
    def buffers(self): return self.buffers_
def require(ok,message):
    if not ok: raise ValueError(message)
def make_endpoint(module,context):
    processor=Processor()
    return {'model':Model(),'processor_object':processor,'processor_cache':processor.cache,'head_object':Model(),'A':Leaf(),'C':Leaf(),'mu_train':Leaf(),'modules':{'runtime':module},'guards':{},'_serving':context}
rows=[]
for failure in (None,'artifact','environment','anchors','release','pre_registry','post_registry','foreign_owner','multiple'):
    calls=[];sentinel=ValueError('original release sentinel');art_error=ValueError('artifact sentinel');env_error=ValueError('environment sentinel');anchor_error=ValueError('anchor sentinel')
    module=types.ModuleType('_serving_lifecycle_test');sys.modules[module.__name__]=module
    helper_module=types.ModuleType('_serving_owned_helper');sys.modules[helper_module.__name__]=helper_module
    def artifact(*args,**kwargs):
        calls.append('artifact')
        if failure in ('artifact','multiple'): raise art_error
    expected={'site_packages':'/synthetic-only'};encoded=json.dumps(expected).encode()
    def environment(*args,**kwargs):
        calls.append('environment')
        if failure in ('environment','multiple'): raise env_error
        return expected
    def anchors(*args):
        calls.append('anchors')
        if failure in ('anchors','multiple'): raise anchor_error
        if failure=='post_registry':sys.modules[helper_module.__name__]=types.ModuleType(helper_module.__name__)
    def helper_binding(binding): calls.append('helper_binding')
    def cache(processor,guards):
        calls.append('release')
        if failure=='release': raise sentinel
        return processor.cache
    namespace=vars(module);namespace.update(require=require,sys=sys,weakref=weakref,gc=gc,hashlib=hashlib,json=json,_check_runtime=lambda:calls.append('runtime'),_serving_helper_binding=helper_binding,batch_bound_files=anchors,_serving_json=json.loads,_processor_cache=cache)
    exec(compile(Path(__file__).with_name('sfora-serving-lifecycle.py').read_text(),'lifecycle-candidate','exec',dont_inherit=True),namespace)
    exec(compile(Path(__file__).with_name('sfora-serving-failed-load.py').read_text(),'failed-load-candidate','exec',dont_inherit=True),namespace)
    exec(compile(new,'real-release-with-dispatch','exec',dont_inherit=True),namespace)
    helpers=(None,None,types.SimpleNamespace(admit_serving_artifact=artifact),None,types.SimpleNamespace(verify_installed_environment=environment))
    context={'helpers':helpers,'helper_guards':(),'checker':None,'directory':'/synthetic-only','serving_sha256':'s','fragment_sha256':{'origin.json':'a','origin-owners.json':'b'},'anchors':{},'installed_raw':encoded,'installed_sha256':hashlib.sha256(encoded).hexdigest(),'origin_raw':b'{}','owners_raw':b'{}','owned_registry':((module.__name__,module),(helper_module.__name__,helper_module))}
    endpoint=make_endpoint(module,context)
    outside=endpoint['model'] if failure=='foreign_owner' else None
    if failure=='pre_registry':sys.modules[helper_module.__name__]=types.ModuleType(helper_module.__name__)
    torch=types.ModuleType('torch');torch.Tensor=Leaf;torch.cuda=types.SimpleNamespace(is_initialized=lambda:False)
    error=None
    assert 'torch' not in sys.modules
    sys.modules['torch']=torch
    try:
        try: namespace['release_inference'](endpoint)
        except BaseException as caught: error=caught
    finally:
        if sys.modules.get('torch') is torch: del sys.modules['torch']
    assert endpoint=={},failure
    assert calls.count('release')==1 and all(label in calls for label in ('runtime','artifact','environment','anchors')),calls
    if failure is None: assert error is None,error
    else: assert error is not None,failure
    if failure=='release':assert error is sentinel
    if failure in ('artifact','multiple'):assert error is art_error
    if failure=='multiple':assert len(error.__notes__)==2,error.__notes__
    if failure=='foreign_owner':assert outside is not None and 'lifetime survived' in str(error)
    if failure in ('pre_registry','post_registry'):
        assert sys.modules[helper_module.__name__] is not helper_module
    assert sys.modules[module.__name__] is module
    rows.append(failure or 'normal')
    outside=None
    del sys.modules[module.__name__]
    del sys.modules[helper_module.__name__]
assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
print('PASS9 release lifecycle cases: genuine legacy body preserved; all exit categories attempted; primary identity retained; endpoint cleared; foreign owner/registry remain rejected and untouched; native UNRUN')
