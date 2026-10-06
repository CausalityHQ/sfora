"""Engineering-only synthetic CPU/CUDA connected-readout math gate; no model/data/quality."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
started=time.perf_counter()
a=Path(sys.argv[1]); expected=sys.argv[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
require(not sys.flags.optimize and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','fixed unoptimized GPU invocation')
require(sha(a)==expected,'authority differs')
f=json.loads(a.read_text());root=a.parent
for name,digest in f['files'].items():require(sha(root/name)==digest,'current source differs '+name)
require(sha(sys.executable)==f['python_sha256'],'interpreter differs')
mods={n:runpy.run_path(str(root/n)) for n in ('quadratic_readout.py','prototype_residual_readout.py','connected_residual_readout.py','train_siglip2_cached_readout.py')}
from types import SimpleNamespace
primitive=SimpleNamespace(**mods['quadratic_readout.py']);readout=SimpleNamespace(**mods['prototype_residual_readout.py']);connected=mods['connected_residual_readout.py']['raw_features']
import torch
from torch.nn import functional as F
torch.manual_seed(179061);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
shapes={'primary.weight':(128,1152),'primary.bias':(128,),'down.weight':(32,1152),'up.weight':(128,32),'center':(1152,),'preactivation_std':()}
tensors={n:torch.randn(s)*.03 for n,s in shapes.items()};tensors['preactivation_std'].fill_(1)
rows=[]
for device in ('cpu','cuda'):
 head=mods['train_siglip2_cached_readout.py']['head_from']('control',tensors=tensors).to(device)
 head.requires_grad_(False);frozen=[p.detach().clone() for p in head.state_dict().values()]
 for zero in (True,False):
  x=torch.randn(16,1152,device=device,requires_grad=True)
  A=torch.nn.Parameter(torch.randn(128,160,device=device)*.01)
  C=torch.nn.Parameter(torch.zeros(128,1152,device=device) if zero else torch.randn(128,1152,device=device)*.01)
  means={'linear':torch.randn(32,device=device)*.02,'concat':torch.randn(160,device=device)*.02};mu=torch.randn(1152,device=device)*.02
  original=readout.raw_features(x,head,A,means,'concat',primitive)+F.linear(x.detach()-mu,C)
  actual=connected(x,head,A,means,C,mu,primitive,readout)
  require(torch.equal(original,actual),'forward source parity differs')
  probe=torch.randn_like(actual);grads=torch.autograd.grad((actual*probe).sum(),(x,A,C))
  y=x.detach().clone().requires_grad_(True);aa=A.detach().clone().requires_grad_(True);cc=C.detach().clone().requires_grad_(True)
  h=head(y);z=head.down(F.normalize(y,dim=1)-head.center);phi=torch.cat((z,h),dim=1)-means['concat']
  oracle=(h+F.linear(phi,aa))+F.linear(y-mu,cc);g2=torch.autograd.grad((oracle*probe).sum(),(y,aa,cc))
  require(torch.equal(actual,oracle) and all(torch.equal(g,h) for g,h in zip(grads,g2)),'independent expression/gradient parity differs')
  require(all(torch.isfinite(g).all().item() and torch.count_nonzero(g).item()>0 for g in grads),'finite nonzero feature/A/C gradients required')
  try:
   with torch.no_grad():connected(x,head,A,means,C,mu,primitive,readout)
  except ValueError:pass
  else:raise ValueError('no-grad disconnect accepted')
  require(all(torch.equal(p,q) for p,q in zip(head.state_dict().values(),frozen)) and all(p.grad is None for p in head.parameters()),'frozen source changed')
  rows.append({'device':device,'zero_C':zero,'forward_bitwise':True,'gradient_bitwise':True,'gradient_norms':[float(g.norm()) for g in grads]})
  if device=='cuda':torch.cuda.synchronize()
for name,digest in f['files'].items():require(sha(root/name)==digest,'exit source changed')
require(sha(a)==expected,'exit authority changed')
result={'schema':'connected-residual-native-math-v1','pass':True,'engineering_only':True,'model_fit_qualified':False,'state_reuse_eligible':False,'quality_read':False,'rows':rows,'wall_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'invocation_id':os.environ['INVOCATION_ID'],'authority_sha256':expected}
require(result['wall_seconds']<300 and result['peak_cuda_allocated_bytes']<10000000000,'resource cap')
with Path(f['output']).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
