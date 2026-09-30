import hashlib,json,runpy,sys
from pathlib import Path
source=Path('/home/riomus/runs/sfora-full-valid-anchor-public-source-v3/benchmark_full_valid_anchor_serving.py')
expected='497cd7d946d3723f77960ea1b18b84b46119a741f1284184d64dfa758386d22e'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
output=Path('/home/riomus/runs/sfora-full2000-benchmark-admission-observer-v1.json')
assert not output.exists()
def observe(frame,event,arg):
 if event=='line' and frame.f_code.co_filename==str(source) and frame.f_lineno==66:
  sys.settrace(None)
  actual=frame.f_locals['before'];saved=frame.f_locals['admitted']['models']['public_f16']
  differences=[]
  def compare(a,b,path):
   if a==b:return
   if isinstance(a,dict) and isinstance(b,dict):
    for key in a.keys()|b.keys():
     if key not in a or key not in b:differences.append({'path':path,'key_repr':repr(key),'key_type':type(key).__name__,'actual_has':key in a,'saved_has':key in b})
     else:compare(a[key],b[key],path+'/'+str(key))
   else:differences.append({'path':path,'actual':a,'saved':b,'actual_type':type(a).__name__,'saved_type':type(b).__name__})
  compare(actual,saved,'')
  with output.open('x') as stream:json.dump({'schema':'full2000-benchmark-admission-diagnostic-v1','source_sha256':expected,'actual':actual,'saved':saved,'differences':differences,'canonical_json_equal':json.loads(json.dumps(actual))==saved,'qualification':False,'timing_measured':False},stream,indent=2,sort_keys=True)
 return observe
sys.argv=[str(source),*sys.argv[1:]]
sys.settrace(observe)
runpy.run_path(str(source),run_name='__main__')
