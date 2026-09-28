"""Bounded saved-checkpoint audit; no training, official reads or gate rewrite."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import SiglipConfig, SiglipVisionModel

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from train_inshop_siglip2_unseen_gallery import width_geometry

CHECKPOINT = {128:"fe56d0941618a0fad5d3395dd48d1988edc2d9a496a0329b57bde4160576d2a1",
              256:"ebccc08788a15b6b9536310dc2516ee0f8aa275ae6465b19ba6ecd17927bd811"}
FIXTURE = {128:"955c9d45dae6db42ab9f14fcc96cdd9354d3d462a55c7c2c00eb5d30b013917f",
           256:"7587b75aacc1620dc98d3ed4d3e7dca6183294dcaacf6770e4f1449754b83835"}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def frozen(name):
    return name.startswith('embeddings.') or (name.startswith('encoder.layers.') and int(name.split('.')[2])<12)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path)
    parser.add_argument('--model-snapshot',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        assert frozen('embeddings.position_embedding.weight') and frozen('encoder.layers.11.layer_norm1.weight')
        assert not frozen('encoder.layers.12.layer_norm1.weight') and not frozen('head.layer_norm.weight')
        return
    if any(x is None for x in (args.runs_root,args.model_snapshot,args.output)) or args.output.exists():
        raise ValueError('audit needs fresh output and pinned artifacts')
    started=time.monotonic()
    result={'schema':'sfora-inshop-direct-width-saved-audit-v1','claim_eligible':False,
            'original_gate_still_failed':True,'arms':{},'source_sha256':sha(__file__)}
    try:
        assert sha(args.runs_root/'receipt.json')=='4a0c2dd476fd151d26ceedf4fa823dc3753c64ccea9a2f1e70f5a3dc08c00c2b'
        assert sha(args.model_snapshot/'model.safetensors')=='fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a'
        assert sha(args.model_snapshot/'config.json')=='172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104'
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        config=SiglipConfig.from_pretrained(args.model_snapshot,local_files_only=True).vision_config
        pixels_hash=None
        for width in (128,256):
            directory=args.runs_root/str(width)
            assert sha(directory/'checkpoint.pt')==CHECKPOINT[width] and sha(directory/'native_fp16_fit_fixture.pt')==FIXTURE[width]
            saved=torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu',mmap=True)
            fixture=torch.load(directory/'native_fp16_fit_fixture.pt',weights_only=True,map_location='cpu')
            assert saved['training_width']==width and saved['updates']==100 and saved['seed']==179024
            assert saved['arm']=='freeze_emb' and saved['freeze_first_blocks']==12
            current=hashlib.sha256(fixture['pixels'].numpy().tobytes()).hexdigest()
            assert pixels_hash is None or current==pixels_hash
            pixels_hash=current
            assert fixture['pixels'].shape==(64,3,256,256) and fixture['pixels'].dtype==torch.float16
            changed=[]; frozen_count=0
            with safe_open(args.model_snapshot/'model.safetensors',framework='pt',device='cpu') as base:
                for name,value in saved['vision'].items():
                    assert value.dtype==torch.float32 and torch.isfinite(value).all()
                    original=base.get_tensor('vision_model.'+name).float()
                    if frozen(name):
                        assert torch.equal(value,original),name
                        frozen_count+=1
                    elif not torch.equal(value,original):changed.append(name)
            assert frozen_count>0 and changed
            assert all(torch.isfinite(v).all() for v in saved['head'].values()) and torch.isfinite(saved['classifier']).all()
            model=SiglipVisionModel(config)
            model.load_state_dict(saved['vision'],strict=True)
            model=model.half().cuda().eval()
            head=nn.Linear(1024,width)
            head.load_state_dict(saved['head'],strict=True)
            head=head.cuda().eval()
            features=[]
            for pixels in fixture['pixels'].split(32):
                with torch.autocast('cuda',enabled=False):
                    pooled=model(pixel_values=pixels.cuda()).pooler_output
                features.append(F.normalize(compact_head_features(pooled,head,output_dim=width),dim=1).cpu())
            values=torch.cat(features)
            packed=pack_int8_unit_embeddings(values)
            assert torch.equal(packed.codes,fixture['codes']) and torch.equal(packed.inverse_norms,fixture['inverse_norms'])
            result['arms'][str(width)]={'checkpoint_sha256':CHECKPOINT[width],'fixture_sha256':FIXTURE[width],
                'frozen_tensor_count':frozen_count,'changed_trainable_tensor_names':changed,
                'finite_saved_parameters':True,'strict_native_fp16_reload_exact_codes_and_norms':True,
                'fit64_geometry':width_geometry(values),'pixels_sha256':current}
            del model,head,saved,fixture,features,values,packed
            torch.cuda.empty_cache()
        a,b=result['arms']['128'],result['arms']['256']
        result['geometry_guard']=all(b['fit64_geometry'][k]>=.5*a['fit64_geometry'][k] for k in ('variance','effective_rank'))
        assert result['geometry_guard']
        result['decision']='GO_SAVED_CHECKPOINT_DESIGN_REVIEW'
    except Exception as error:
        result.update(decision='KILL_SAVED_CHECKPOINT_AUDIT',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['seconds']=time.monotonic()-started
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
