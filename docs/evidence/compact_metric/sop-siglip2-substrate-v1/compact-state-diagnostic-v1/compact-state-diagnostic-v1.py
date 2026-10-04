import ast,hashlib,json,runpy,sys,copy
from pathlib import Path
ROOT=Path('/home/riomus/runs/sfora-so400-compact-ranking-evaluation-source-v20')
SOURCE=ROOT/'evaluate_siglip2_compact_ranking.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='9acc2332dc4c2471299f92a634bab65243a9b9addbd3823af3471716dc7eb7d0'
HELPER=ROOT/'endpoint-state-mismatch-report-v1.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='3e125a3972d4026186964d99253adea391bbb9a2f6754718802c1f9ca98196c3'
ns={};exec(HELPER.read_text(),ns)
report=ns['endpoint_state_mismatch_report']
tree=ast.parse(SOURCE.read_bytes());native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_export')
first=next(n for n in ast.walk(native) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and any(isinstance(x,ast.Constant) and x.value=='independent complete frozen vision/updated A differs' for x in ast.walk(n)))
stop=next(n.lineno for n in ast.walk(native) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='native_train_witness' for x in ast.walk(n)))
seen=False;context=None;serialized_config=None;run_namespace=None;before=None
class DiagnosticStop(ValueError):pass
def trace(frame,event,arg):
 global seen,context,serialized_config,run_namespace,before
 if frame.f_code.co_filename!=str(SOURCE) or frame.f_code.co_name not in ('run','authenticate_payloads','native_export'):return None
 if frame.f_code.co_name=='run' and event=='line' and 'context' in frame.f_locals:
  context=frame.f_locals['context'];run_namespace=frame.f_globals;before=frame.f_locals.get('before')
 if frame.f_code.co_name=='authenticate_payloads' and event=='line':
  disk=frame.f_locals.get('disk')
  if isinstance(disk,dict) and 'config' in disk:serialized_config=copy.deepcopy(disk['config'])
 if frame.f_code.co_name=='native_export' and event=='line':
  if frame.f_lineno==first.lineno and not seen:
   seen=True;loc=frame.f_locals;d=report(loc['facts'],loc['model_facts'],loc['state'])
   actual=loc['state']['model'].config.to_dict()
   if 'config' in d['fields']:
    d['config_differences']={str(k):{'expected_type':type(serialized_config.get(k)).__name__,'actual_type':type(actual.get(k)).__name__,'expected_repr':repr(serialized_config.get(k)),'actual_repr':repr(actual.get(k))} for k in set(serialized_config)|set(actual) if type(serialized_config.get(k))!=type(actual.get(k)) or repr(serialized_config.get(k))!=repr(actual.get(k))}
   print(json.dumps(d,sort_keys=True),flush=True)
  if frame.f_lineno==stop:raise DiagnosticStop('diagnostic stops before first image forward')
 return trace
sys.argv=[str(SOURCE),*sys.argv[1:]]
sys.settrace(trace)
try:
 runpy.run_path(str(SOURCE),run_name='__main__')
finally:
 sys.settrace(None)
 print(json.dumps({'event':'COMPACT_DIAGNOSTIC_STOP','observed':seen,'qualification_eligible':False,'state_reuse_eligible':False,'images':0,'quality_read':False}),flush=True)
 if context is not None:
  run_namespace['exit_rehash'](context)
  if before is not None:
   resource=run_namespace['resources'](context,before)
   print(json.dumps({'event':'COMPACT_DIAGNOSTIC_EXIT','exit_rehash_pass':True,'resources':resource}),flush=True)
