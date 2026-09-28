import hashlib,json,os,sys,time
from pathlib import Path
import torch
root=Path('/home/riomus/runs/sfora-inshop-fit-recovery-v1')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=['/home/riomus/runs/sfora-inshop-fit-positive-tail-v1','/home/riomus/runs','/home/riomus/sfora-siglip2-deployed-batch-v1/src']
import probe as pinned
from export_inshop_siglip2_train_features import MODEL_HASHES
from sfora.unicom_inshop import parse_inshop_partition
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA,digest_rows,split
from train_sop_siglip2_compact import sha256,export_all
started=time.monotonic()
run=Path('/home/riomus/runs/sfora-inshop-valid-anchor-confirm-freeze_emb-179026-v1')
tail=Path('/home/riomus/runs/sfora-inshop-fit-positive-tail-v1/receipt.json')
dataset=Path('/home/riomus/datasets/In-shop Clothes Retrieval Benchmark')
snapshot=Path('/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c')
assert sha256(Path(pinned.__file__))=='1f655cebb7155c1b980bb5222ac3894b582f916f4c9c85a2cc203f14d1978b69'
assert sha256(tail)=='953be9e6eab846f22e3e187f043eebfaf3161af0bbe4a752f6c68987ffb7fb49'
assert sha256(run/'receipt.json')==pinned.RECEIPT_SHA
assert sha256(Path(sys.modules[export_all.__module__].__file__))==pinned.TRAIN_HELPER_SHA
assert sha256(dataset/'Eval/list_eval_partition.txt')==PARTITION_SHA
for name,digest in MODEL_HASHES.items(): assert sha256(snapshot/name)==digest
old=json.loads(tail.read_text()); receipt=json.loads((run/'receipt.json').read_text())
assert sha256(run/'checkpoint.pt')==old['checkpoint_sha256']==receipt['checkpoint_sha256']
train=tuple(r for r in parse_inshop_partition(dataset) if r.split=='train')
fit,held=split(tuple(r.label for r in train))
assert len(fit)==13283 and len(held)==12599
assert digest_rows(fit)==old['fit_rows_sha256']==receipt['fit_rows_sha256']
assert all(train[i].image_path.is_file() for i in fit)
labels=sorted({train[i].label for i in fit}); assert len(labels)==2004
checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
assert checkpoint['classifier'].shape==(2004,128)
assert checkpoint['head']['weight'].shape==(128,1024)
assert torch.isfinite(checkpoint['classifier']).all()
assert checkpoint['seed']==179026 and checkpoint['arm']=='freeze_emb' and checkpoint['updates']==1000
assert not any(isinstance(v,torch.Tensor) and tuple(v.shape)==(13283,128) for v in checkpoint.values())
result={'claim_eligible':False,'decision':'PASS_CPU_AUTHORITY_ONLY','fit_rows':13283,'fit_products':2004,'expected_feature_sha256':old['features_sha256'],'checkpoint_sha256':old['checkpoint_sha256'],'fit_rows_sha256':old['fit_rows_sha256'],'pinned_export_helper_sha256':pinned.TRAIN_HELPER_SHA,'classifier_shape':list(checkpoint['classifier'].shape),'wall_seconds':time.monotonic()-started,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'source_sha256':sha256(Path(__file__))}
with (root/'cpu-preflight.json').open('x') as output: json.dump(result,output,indent=2);output.write('\n')
print(json.dumps(result),flush=True)
