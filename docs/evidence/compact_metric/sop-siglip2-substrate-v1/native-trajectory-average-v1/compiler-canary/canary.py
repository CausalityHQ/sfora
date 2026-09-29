import hashlib
from pathlib import Path
import numpy as np
import torch
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings

compiler=Path('/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras')
library=Path('/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so')
assert hashlib.sha256(compiler.read_bytes()).hexdigest()=='df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae'
assert hashlib.sha256(library.read_bytes()).hexdigest()=='39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c'
root=Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1')
values={}
for role,count in (('query',32),('gallery',6245)):
 values[role]=PackedInt8Embeddings(torch.from_numpy(np.load(root/(role+'.codes.npy'),allow_pickle=False)[:count].copy()),torch.from_numpy(np.load(root/(role+'.inverse.npy'),allow_pickle=False)[:count].copy()))
q,g=values['query'],values['gallery']
torch.backends.cuda.matmul.allow_tf32=False
scores=(q.codes.float().cuda()@g.codes.float().cuda().T)*q.inverse_norms.float().cuda()[:,None]*g.inverse_norms.float().cuda()[None,:]
order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:10]
with CutilePackedInt8Gallery.open_packed(library,g) as native:
 actual=native.search_packed(q)
assert np.array_equal(actual[0],order.cpu().numpy())
assert np.array_equal(actual[1].view(np.uint32),scores.gather(1,order).cpu().numpy().view(np.uint32))
print('PASS authenticated compiler PATH / native B32x6245 top10 ordinals and float32 score bits; no model/quality read')
