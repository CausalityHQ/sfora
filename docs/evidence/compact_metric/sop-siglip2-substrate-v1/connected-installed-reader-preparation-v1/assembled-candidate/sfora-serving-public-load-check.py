import ast
from pathlib import Path
raw=Path(__file__).with_name('sfora-serving-failed-load.py').read_text()
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='load_serving_inference')
for mode in ('normal','payload','postguard','release'):
 events=[];prepared={};endpoint={'owned':True};primary=ValueError(mode)
 def prepare(*a,**kw):events.append('prepare');return prepared
 def payload(context):
  assert context is prepared;events.append('payload')
  if mode=='payload':raise primary
  return endpoint
 def binding(value):
  events.append('postguard')
  if mode in ('postguard','release'):raise primary
 def release(value):
  assert value is endpoint;events.append('release');value.clear()
  if mode=='release':raise RuntimeError('cleanup')
 def failed(context,error):
  assert context is prepared and error is primary;events.append('failed_exit');raise error
 ns={'_serving_prepare':prepare,'_serving_load_payload':payload,'_serving_helper_binding':binding,'_serving_release':release,'_serving_failed_load':failed}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<public-loader>','exec'),ns)
 try:
  result=ns['load_serving_inference']('/artifact',trusted_serving_sha256='s',trusted_fragment_sha256={},installed_environment='/env',trusted_installed_environment_sha256='e',serving_helpers=())
 except BaseException as error:
  assert mode!='normal' and error is primary
  assert events==(['prepare','payload','failed_exit'] if mode=='payload' else ['prepare','payload','postguard','release','failed_exit'])
  if mode=='release':assert any('cleanup' in note for note in error.__notes__)
 else:assert mode=='normal' and result is endpoint and events==['prepare','payload','postguard']
print('PASS four actual public-wrapper paths: normal, payload rejection, post-load guard, release rejection; primary preserved')
