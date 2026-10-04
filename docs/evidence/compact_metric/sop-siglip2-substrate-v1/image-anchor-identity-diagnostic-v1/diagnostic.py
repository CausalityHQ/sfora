"""Engineering-only metadata observation; not qualification/state/quality authority."""
import hashlib,importlib.util,json
from pathlib import Path
P=Path('/home/riomus/runs/sfora-so400-smooth-ap-train-candidate-179061-v1/resume.pt')
R=P.parent/'receipt.json'
S=Path('/home/riomus/runs/sfora-so400-smooth-ap-train-source-v1/train_siglip2_compact_ranking.py')
pins={P:'4f79c3690b874385a286d09a70c4e0e9320eb3a691b207e9e52d143bacfe90b3',R:'49a0d62d96864d6eec833aafe6c9676ae2c9be512ec624eae56997fd656d12c3',S:'74a3cbcc3c1a82f21a36793723d57901783ef1f126617ffdea75bc6e402ba679'}
def bound():
 for p,h in pins.items():
  with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==h
bound()
r=json.loads(R.read_text());spec=importlib.util.spec_from_file_location('diagnostic_original',S);t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
import torch
d=torch.load(P,map_location='cpu',weights_only=True,mmap=True)
a,b=d['identity'],r['identity'];diff=[]
def walk(x,y,path):
 if type(x) is not type(y):diff.append({'path':path,'checkpoint_type':type(x).__name__,'receipt_type':type(y).__name__,'checkpoint_value':repr(x),'receipt_value':repr(y)});return
 if isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:walk(x[k],y[k],path+'.'+str(k))
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y)
  for i,(u,v) in enumerate(zip(x,y,strict=True)):walk(u,v,path+'['+str(i)+']')
 elif x!=y:diff.append({'path':path,'checkpoint_value':repr(x),'receipt_value':repr(y)})
walk(a,b,'identity')
print(json.dumps({'observation_only':True,'qualification_eligible':False,'images':0,'quality_read':False,'native_identity_equals_receipt':a==b,'json_canonical_identity_equals_receipt':json.loads(json.dumps(a,allow_nan=False))==b,'identity_differences':diff,'payload_keys_match_original':d.keys()==t.PAYLOAD_KEYS,'schema_matches_original':d['schema']==t.SCHEMA,'source_matches_receipt':d['source']==r['source'],'numerical_flags_match_receipt':d['numerical_flags']==r['numerical_flags'],'counter':d['counter']},sort_keys=True))
assert not torch.cuda.is_initialized()
del d,a,b
bound()
