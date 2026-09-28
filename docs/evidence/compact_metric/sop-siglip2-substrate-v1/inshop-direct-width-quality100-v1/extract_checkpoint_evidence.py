import json,sys,hashlib
from pathlib import Path
import torch
root=Path(sys.argv[1])
c=torch.load(root/'checkpoint.pt',weights_only=True,map_location='cpu',mmap=True)
e=c['training_evidence']
out={k:c[k] for k in ('training_width','seed','arm','updates','vision_lr','freeze_first_blocks','half_fit_products','tail_blocks_dropped')}
out['training_evidence']=e
out['checkpoint_sha256']=hashlib.file_digest((root/'checkpoint.pt').open('rb'),'sha256').hexdigest()
out['fixture_sha256']=hashlib.file_digest((root/'native_fp16_fit_fixture.pt').open('rb'),'sha256').hexdigest()
out['held_values_sha256']=hashlib.file_digest((root/'held_values.npy').open('rb'),'sha256').hexdigest()
print(json.dumps(out))
