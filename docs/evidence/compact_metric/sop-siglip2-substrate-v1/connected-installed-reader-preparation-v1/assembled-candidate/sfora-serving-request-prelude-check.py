import ast
import sys
import types
from pathlib import Path
from unittest.mock import patch
original=next(n for n in ast.parse(Path('src/sfora/connected_inference.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='inference_outputs')
candidate=ast.parse(Path(__file__).with_name('sfora-serving-inference-outputs.py').read_text()).body[0]
first=next(i for i,s in enumerate(candidate.body) if isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='require')
old_first=next(i for i,s in enumerate(original.body) if isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='require')
assert ast.dump(ast.Module(body=candidate.body[first:],type_ignores=[]))==ast.dump(ast.Module(body=original.body[old_first:],type_ignores=[]))
assert ast.dump(ast.Module(body=candidate.body[5].orelse,type_ignores=[]))==ast.dump(ast.Module(body=original.body[4:old_first],type_ignores=[]))
candidate.body=candidate.body[:first]
candidate.body.append(ast.Return(value=ast.Constant(value=True)))
events=[]
def forbidden(*args):raise AssertionError('full admission during request')
ns={'_check_runtime':lambda:events.append('source'),'_serving_helper_binding':lambda v:events.append('helpers'),'_serving_live_identity':lambda e:events.append('identity'),'bound_file':lambda *v:events.append(('read',str(v[1]))),'SERVING_FILES':{'old.py'},'_serving_prepare':forbidden,'_serving_full_exit':forbidden}
exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),'<request-prelude>','exec'),ns)
torch=types.ModuleType('torch');nn=types.ModuleType('torch.nn');nn.functional=types.SimpleNamespace()
endpoint={'modules':{},'device':'cuda','_serving':{'helpers':{},'helper_guards':{},'checker':object()}}
with patch.dict(sys.modules,{'torch':torch,'torch.nn':nn}):
 for _ in range(2):assert ns['inference_outputs'](endpoint,[]) is True
 assert events==['source','helpers','source','identity']*2
 events.clear()
 legacy={'modules':{'old':types.SimpleNamespace(__file__='/old.py')},'device':'cuda','directory':Path('/legacy'),'guards':{'/old.py':'sha','/legacy/old.py':'sha','/legacy/joint_relational_compaction.py':'sha'}}
 assert ns['inference_outputs'](legacy,[]) is True
 assert events[0]=='source' and set(events[1:])=={('read','/old.py'),('read','/legacy/old.py'),('read','/legacy/joint_relational_compaction.py')}
print('PASS: original numerical suffix and legacy reads; two serving requests avoid full admission')
