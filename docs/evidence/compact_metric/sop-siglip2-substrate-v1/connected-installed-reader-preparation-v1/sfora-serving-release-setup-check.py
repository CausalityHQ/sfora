import ast,gc,sys,weakref
from pathlib import Path
from types import SimpleNamespace
source=Path('src/sfora/connected_inference.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_serving_release')
class Leaf:pass
class BadModel:
 def parameters(self):raise RuntimeError('member iterator failed')
 def buffers(self):return ()
class Model:
 def parameters(self):return ()
 def buffers(self):return ()
for missing in ('A','model','owned_registry','iterator'):
 calls=[];context={'owned_registry':()};endpoint={'_serving':context,'modules':{},'model':Model(),'processor_object':Leaf(),'head_object':Model(),'A':Leaf(),'C':Leaf(),'mu_train':Leaf(),'means':{},'common_statistics':{}}
 if missing=='owned_registry':del context[missing]
 elif missing=='iterator':endpoint['model']=BadModel()
 else:endpoint[missing]=object()
 def original_release(value):calls.append('original_release');value.clear()
 ns={'gc':gc,'sys':sys,'weakref':weakref,'_serving_tensor_refs':lambda *a:None,'_serving_helper_binding':lambda *a:calls.append('binding'),'release_inference':original_release,'_serving_full_exit':lambda *a:calls.append('full_exit'),'require':lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg))}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-release>','exec'),ns)
 try:ns['_serving_release'](endpoint)
 except BaseException:pass
 else:raise AssertionError('bad release member accepted')
 assert 'full_exit' in calls and 'original_release' in calls and not endpoint,(missing,calls)
print('PASS release setup failures still attempt original cleanup and full exit')
