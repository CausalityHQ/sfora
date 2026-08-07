#!/usr/bin/env python3
"""Export trained BN-Inception pre-head avg+max features for geometry matching."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sfora.data import load_image_retrieval_bundle
from sfora.image_end_to_end import ImageEndToEndConfig, _TorchImageDataset, _default_transform_factory, _torchvision_model_factory
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--num-workers',type=int,default=8); a=p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    report=json.loads(a.report.read_text()); cfg=ImageEndToEndConfig.model_validate(report['config']); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=_torchvision_model_factory(cfg); model.load_state_dict({k:v for k,v in ck['state_dict'].items() if k not in {'metric_proxies','metric_proxy_labels'}},strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); tf=_default_transform_factory(cfg,False)
    b=load_image_retrieval_bundle(dataset_name='inshop',dataset_root=a.dataset_root,seed=cfg.seed)
    def enc(ex):
        dl=DataLoader(_TorchImageDataset(ex,tf),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=torch.cuda.is_available()); vs=[]; ys=[]
        with torch.no_grad():
            for im,y in dl:
                x=model.model.features(im.to(device)); v=(x.mean((2,3))+x.amax((2,3))).flatten(1); vs.append(v.cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(vs),np.concatenate(ys)
    out={}
    for n,ex in [('train',b.train),('query',b.query),('gallery',b.gallery)]: out[n],out[n+'_labels']=enc(ex)
    a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,**out); print({k:list(v.shape) for k,v in out.items()})
if __name__=='__main__': main()
