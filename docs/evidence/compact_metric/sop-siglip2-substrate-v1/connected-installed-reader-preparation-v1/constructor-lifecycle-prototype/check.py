"""Actual constructor AST with tiny fake native dependencies; no real ML imports."""
import ast, copy, hashlib, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
root=next(parent for parent in Path(__file__).resolve().parents if (parent/'src/sfora/connected_inference.py').is_file())
original=(root/'src/sfora/connected_inference.py').read_text()
old=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='construct_encoder')
raw=ast.get_source_segment(original,old) if '--original' in sys.argv else Path(__file__).with_name('candidate.py').read_text()
new=ast.parse(raw).body[0]
old_try=next(n for n in old.body if isinstance(n,ast.Try));new_try=next(n for n in new.body if isinstance(n,ast.Try))
assert ast.dump(ast.Module(body=old_try.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=new_try.body,type_ignores=[]),include_attributes=False)
legacy=new_try.finalbody[1:] if '--original' in sys.argv else new_try.finalbody[1].body
assert ast.dump(ast.Module(body=legacy,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=old_try.finalbody[1:],type_ignores=[]),include_attributes=False)
assert ast.dump(old.body[-1],include_attributes=False)==ast.dump(new.body[-1],include_attributes=False)
class Buffer:
    shape=(1,);dtype='fake-int'
    def copy_(self,value): return self
    def expand(self,*args): return self
class Model:
    embeddings=types.SimpleNamespace(_non_persistent_buffers_set={'position_ids'})
    def eval(self): return self
    def named_buffers(self): return [('embeddings.position_ids',Buffer())]
    def state_dict(self): return dict.fromkeys(range(448))
class Pages:
    def __init__(self,stream): pass
    def consume(self,value): pass
class NoGrad:
    def __enter__(self): pass
    def __exit__(self,*args): pass
class AutoProcessor:
    @staticmethod
    def from_pretrained(*args,**kwargs): return object()
torch=types.ModuleType('torch');torch.no_grad=NoGrad;torch.arange=lambda n:Buffer();torch.equal=lambda a,b:True
transformers=types.ModuleType('transformers');transformers.AutoImageProcessor=AutoProcessor
config={};buffer={'embeddings.position_ids':Buffer()}
def load(*args,**kwargs): return {'vision':dict.fromkeys(range(448)),'buffers':buffer,'config':config,'runtime':{},'cpu_rng':None}
torch.load=load
cases=[]
with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'torch':torch,'transformers':transformers}):
    path=Path(directory)/'opaque.pt';path.write_bytes(b'opaque')
    for serving in (False,True):
        for body_error in (False,True):
            for gc_error,map_error in ((False,False),(False,True),(True,False),(True,True)):
                primary=RuntimeError('actual body sentinel');collect_error=OSError('collection sentinel');mapping_error=ValueError('mapping sentinel');attempts=[]
                def require(ok,message):
                    if not ok: raise ValueError(message)
                def load_vision(*args):
                    if body_error: raise primary
                def collect():
                    attempts.append('collection')
                    if gc_error: raise collect_error
                def mapping_absent(value):
                    attempts.append('mapping')
                    if map_error: raise mapping_error
                ns={'require':require,'bound_file':lambda guards,path,sha:Path(path),'construct':lambda *args:Model(),'CheckpointPages':Pages,'fingerprint':lambda *args,**kwargs:'h','load_vision':load_vision,'gc':types.SimpleNamespace(collect=collect),'mapping_absent':mapping_absent,'_processor_cache':lambda *args,**kwargs:object(),'model_structure':lambda *args:{},'copy':copy}
                exec(compile(raw,'actual-constructor-candidate','exec',dont_inherit=True),ns)
                context={'guards':{},'packages':{}}
                if serving: context['_serving']={}
                failure=None
                try: ns['construct_encoder'](context,config,buffer,path,{'checkpoint':{'path':str(path),'sha256':'h'},'sha256':'h'})
                except BaseException as error: failure=error
                expected = primary if body_error else None
                if serving:
                    assert attempts==['collection','mapping']
                    if expected is None: expected=collect_error if gc_error else mapping_error if map_error else None
                else:
                    assert attempts==(['collection'] if gc_error else ['collection','mapping'])
                    if gc_error: expected=collect_error
                    elif map_error: expected=mapping_error
                assert failure is expected,(serving,body_error,gc_error,map_error,failure,expected)
                if serving and body_error:
                    notes=getattr(primary,'__notes__',[])
                    assert len(notes)==int(gc_error)+int(map_error)
                cases.append((serving,body_error,gc_error,map_error))
assert len(cases)==16
assert not any(name.split('.')[0] in {'torch','numpy','PIL','transformers'} for name in sys.modules)
print('PASS16 constructor cases: original numerical try/return AST and legacy cleanup AST exact; artifact body error identity retained, both cleanup checks attempted; cleanup-only failures reject; native UNRUN')
