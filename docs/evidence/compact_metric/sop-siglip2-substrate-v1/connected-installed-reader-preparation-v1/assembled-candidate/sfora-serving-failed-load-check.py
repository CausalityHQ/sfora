"""Tiny real mmap, actual mapping predicate, owned runtime frame cleanup only."""
import ast, gc, mmap, os, sys, tempfile, types, weakref
from pathlib import Path
root=Path('/home/rb/worktrees/sfora-positive-causality')
raw=(root/'src/sfora/connected_inference.py').read_text()
original_mapping=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='mapping_absent')
class Leaf:
    def __init__(self,mapping=None):self.mapping=mapping
class Cache:
    def __init__(self,owner):self.owner=owner
    def cache_clear(self):self.owner=None
    def cache_info(self):return types.SimpleNamespace(currsize=0)
class Processor:
    def __init__(self):self.cache=Cache(self)
class Model:pass
def require(ok,message):
    if not ok:raise ValueError(message)
rows=[]
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory)
    for name in ('vision.pt','endpoint.pt'):(folder/name).write_bytes(b'opaque-test-bytes'*256)
    for foreign in (False,True):
        module=types.ModuleType('_failed_serving_test');ns=vars(module)
        holder=[];calls=[];primary=RuntimeError('original payload rejection')
        context={'directory':str(folder),'guards':{}}
        ns.update(Path=Path,os=os,gc=gc,sys=sys,weakref=weakref,require=require,mmap=mmap,Model=Model,Processor=Processor,Leaf=Leaf,primary=primary,hold=holder.append if foreign else lambda value:None,_processor_cache=lambda processor,guards:processor.cache,_serving_full_exit=lambda context:calls.append('full_exit'))
        exec(compile(ast.Module(body=[original_mapping],type_ignores=[]),'genuine-mapping-predicate','exec'),ns)
        exec(compile(Path(__file__).with_name('sfora-serving-failed-load.py').read_text(),'failed-load-candidate','exec',dont_inherit=True),ns)
        exec('''def construct_encoder(): pass

def _serving_load_payload(context):
    model, processor, head = Model(), Processor(), Model()
    with (Path(context['directory'])/'endpoint.pt').open('rb') as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    disk = {'tensor':Leaf(mapping)}
    hold(disk['tensor'])
    raise primary
''',ns)
        torch=types.ModuleType('torch');torch.Tensor=Leaf
        assert 'torch' not in sys.modules;sys.modules['torch']=torch
        caught=None
        try:
            try:ns['_serving_load_payload'](context)
            except BaseException as error:
                try:ns['_serving_failed_load'](context,error)
                except BaseException as final:caught=final
        finally:
            if sys.modules.get('torch') is torch:del sys.modules['torch']
        assert caught is primary
        assert calls==['full_exit']
        if foreign:
            assert holder and holder[0].mapping[:1]==b'o'
            assert any('mapping survived' in note for note in primary.__notes__)
            holder.clear();gc.collect()
        else:
            assert not getattr(primary,'__notes__',[]),primary.__notes__
        ns['mapping_absent'](folder/'endpoint.pt');ns['mapping_absent'](folder/'vision.pt')
        trace = primary.__traceback__
        while trace is not None:
            if trace.tb_frame.f_globals is ns and trace.tb_frame.f_code is ns['_serving_load_payload'].__code__:
                assert 'disk' not in trace.tb_frame.f_locals
            trace = trace.tb_next
        rows.append(foreign)
assert rows==[False,True]
assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
print('PASS real mmap failed-load cleanup: original rejection object retained; owned finished aliases released; external retained mapping remains alive and rejected; full exit attempted; native UNRUN')
