#!/usr/bin/env python3
"""Bounded source-qualified native12 continuation;17 mechanics state is discarded."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pe_large_coverage as coverage
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

pair,trained,teacher=coverage.pair,coverage.trained,coverage.teacher
CPU_CODE_SHA='7755af1d64e2f9c42fd5d60eae96cdb67923a7f28fae219315926b035881d955'
CPU=Path('/home/riomus/runs/sfora-large-coverage-cpu-v2')
CPU_SHA='fbb88035f2bdebaa243cdf4e3e2011fc91ec15a0d6722422b448b2ff4b88c61b'


def startup(root,execution_sha):
    path=root/'large-coverage-training-execution.json'
    assert pair.sha(path)==execution_sha
    code=json.loads(path.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items()),'coverage training code differs'
    control,_,source,prior,cpu_code=coverage.startup(root,CPU_CODE_SHA)
    assert all(code[n]==h for n,h in cpu_code.items()) and len(code)==len(cpu_code)+1
    assert pair.sha(CPU/'receipt.json')==CPU_SHA
    proof=json.loads((CPU/'receipt.json').read_text())
    assert proof['pass'] and proof['code']==cpu_code and proof['optimizer_updates']==0
    assert proof['source_whole_sha256']==source['teacher_whole_sha256']
    return control,source,prior,proof,code


def fp16(model,x):
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        return model(pixel_values=x).pooler_output.float()


def packed_equal(a,b):
    pa,pb=pack_int8_unit_embeddings(a.cpu()),pack_int8_unit_embeddings(b.cpu())
    assert np.array_equal(pa.codes,pb.codes) and np.array_equal(pa.inverse_norms,pb.inverse_norms)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=('half','full'),required=True)
    parser.add_argument('--updates',type=int,choices=(17,100),required=True)
    parser.add_argument('--mechanics',type=Path)
    parser.add_argument('--mechanics-sha256')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control,source,prior,proof,code=startup(root,args.execution_sha256)
    assert torch.cuda.is_available() and not args.output.exists()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    flags=teacher.qualified.numerical_flags()
    assert flags==prior['numerical_flags']
    mechanics=None
    if args.updates==100:
        assert args.mechanics and args.mechanics_sha256 and pair.sha(args.mechanics/'receipt.json')==args.mechanics_sha256
        mechanics=json.loads((args.mechanics/'receipt.json').read_text())
        assert mechanics['advance'] and mechanics['updates']==17 and mechanics['arm']=='full'
        assert mechanics['execution_sha256']==args.execution_sha256 and mechanics['cpu_authority_sha256']==CPU_SHA
        assert mechanics['training_state_discarded'] and not mechanics['quality_read']
    else:
        assert args.arm=='full' and not args.mechanics and not args.mechanics_sha256
    args.output.mkdir(exist_ok=False)
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    arm=proof['arms'][args.arm]
    init=CPU/(args.arm+'.npz')
    assert pair.sha(init)==arm['initializers_sha256']
    with np.load(init,allow_pickle=False) as values:
        bank=torch.from_numpy(values['bank'].copy()).cuda()
        classifier=nn.Parameter(torch.from_numpy(values['classifier'].copy()).cuda())
        target=torch.from_numpy(values['target'].copy()).cuda()
    assert pair.smoke.digest({'bank':bank})==arm['bank_sha256'] and pair.smoke.digest({'classifier':classifier})==arm['classifier_sha256']
    assert pair.smoke.digest({'target':target})==arm['target_sha256'] and not bank.requires_grad
    positives=pair.smoke.member_bank_positive_ordinals(target.cpu().numpy(),allow_singletons=True).cuda()
    model,head,processor,inventory=coverage.load_native(control,source)
    assert json.loads(json.dumps(inventory))==proof['inventory'] and coverage.frozen_digest(model,inventory)==proof['frozen_prefix_sha256']
    model.cuda().train()
    head.cuda().train()
    params=coverage.parameters(model,head,classifier)
    members=[p for _,p in params]
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-4},{'params':[classifier],'lr':1e-4}],weight_decay=.05)
    assert len({id(p) for p in members})==len(members) and not optimizer.state
    _,scaler=pair.smoke.training_precision('fp16',device='cuda')
    assert scaler.get_scale()==128
    frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    runtime=trained.base.runtime_identity(model)
    def groups():
        return {**{str(i):dict(model.encoder.layers[i].named_parameters()) for i in range(12,24)},'post_layernorm':dict(model.post_layernorm.named_parameters()),'pool':dict(model.head.named_parameters()),'compact_head':dict(head.named_parameters()),'classifier':{'classifier':classifier}}
    initial_groups={n:pair.smoke.digest(v) for n,v in groups().items()}
    images,_=pair.augmented_images(control.dataset_root,arm['rows'],tuple(arm['batches'][0][:2]),1)
    calibration=pair.pixels(processor,images,'large').cuda()
    with torch.no_grad():
        a=model(pixel_values=calibration).pooler_output.float()
        b=fp16(model,calibration)
        initial_cos={'pooled':F.cosine_similarity(a,b).tolist(),'compact':F.cosine_similarity(pair.smoke.compact_head_features(a,head),pair.smoke.compact_head_features(b,head)).tolist()}
        assert all(min(v)>=.999 for v in initial_cos.values())
    del images,a,b
    cpu_rng,cuda_rng=torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
    rows=[]
    seen=set()
    training_started=time.perf_counter()
    for step,batch in enumerate(arm['batches'][:args.updates],1):
        torch.cuda.synchronize()
        tick=time.perf_counter()
        images,rgb=pair.augmented_images(control.dataset_root,arm['rows'],tuple(batch),step)
        pixels=pair.pixels(processor,images,'large')
        pixel_sha=pair.smoke.digest({'pixels':pixels})
        if step==1:
            assert rgb==arm['first_rgb_sha256'] and pixel_sha==arm['first_pixels_sha256']
        optimizer.zero_grad(set_to_none=True)
        bank_version=bank._version
        raws=[]
        ce_sum=rank_sum=0.
        for offset in range(0,64,16):
            index=torch.tensor(batch[offset:offset+16],device='cuda')
            pooled=fp16(model,pixels[offset:offset+16].cuda())
            ce,rank,raw=coverage.terms(pooled,head,classifier,bank,target,positives,index,arm['rank_active'][step-1])
            loss=(ce+8*rank)/4
            assert torch.isfinite(pooled).all() and torch.isfinite(raw).all() and torch.isfinite(loss)
            scaler.scale(loss).backward()
            raws.append(raw.detach())
            ce_sum+=float(ce.detach())/4
            rank_sum+=float(rank.detach())/4
            del index,pooled,ce,rank,raw,loss
        assert bank._version==bank_version
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in members)
        assert all(p.grad is None and p._version==frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad)
        gradients={str(i):sum(float(p.grad.double().norm()) for p in model.encoder.layers[i].parameters()) for i in range(12,24)}
        assert all(v>0 and np.isfinite(v) for v in gradients.values())
        norm=torch.nn.utils.clip_grad_norm_(members,1,error_if_nonfinite=True)
        scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale()>=scale,'optimizer update skipped'
        assert len(optimizer.state)==len(members)
        assert all(int(s['step'])==step for s in optimizer.state.values())
        assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
        assert all(torch.isfinite(p).all() for p in members)
        refresh,positions=pair.smoke.member_bank_refresh_rows(tuple(batch))
        detached=torch.cat(raws)
        bank[torch.tensor(refresh,device='cuda')]=pair.smoke.member_bank_refresh_values(detached,detached,torch.tensor(positions,device='cuda'),live_head=False)
        assert bank._version==bank_version+1 and torch.isfinite(bank).all()
        del detached,raws,images,pixels
        seen.update(int(v) for v in target[torch.tensor(batch,device='cuda')].tolist())
        assert torch.cuda.max_memory_allocated()<10_000_000_000
        torch.cuda.synchronize()
        row={'step':step,'seconds':time.perf_counter()-tick,'ce':ce_sum,'rank':rank_sum,'loss':ce_sum+8*rank_sum,'scale':scaler.get_scale(),'preclip_norm':float(norm),'gradient_norms':gradients,'rgb_sha256':rgb,'pixels_sha256':pixel_sha,'gradient_input_identities':len(seen)}
        rows.append(row)
        print(json.dumps(row),flush=True)
        if mechanics and args.arm=='full' and step<=17:
            old=mechanics['steps'][step-1]
            assert all(row[k]==old[k] for k in ('ce','rank','loss','scale','preclip_norm','gradient_norms','rgb_sha256','pixels_sha256','gradient_input_identities')),'fresh100 first17 mechanics differs'
    training_wall=time.perf_counter()-training_started
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    assert trained.base.runtime_identity(model)==runtime and coverage.frozen_digest(model,inventory)==proof['frozen_prefix_sha256']
    final_groups={n:pair.smoke.digest(v) for n,v in groups().items()}
    assert all(final_groups[n]!=h for n,h in initial_groups.items())
    if args.updates==100:
        assert len(seen)==len(arm['classes'])
    training={'arm':args.arm,'updates':args.updates,'steps':rows,'training_wall_seconds':training_wall,'images_per_second':args.updates*64/training_wall,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'initial_group_sha256':initial_groups,'final_group_sha256':final_groups,'quality_read':False}
    pair.smoke.save(args.output/'training.json',training)
    del optimizer,members,params
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    classifier.grad=None
    model.eval()
    head.eval()
    with TemporaryDirectory(dir=root,prefix='discard-coverage-mechanics-') as tmp,torch.random.fork_rng(devices=[0]):
        checkpoint=(args.output/'native.pt') if args.updates==100 else Path(tmp)/'native.pt'
        saved={'vision':model.state_dict(),'head':head.state_dict(),'classifier':classifier.detach(),'bank':bank,'classes':arm['classes']}
        assert len(saved['vision'])==400
        torch.save(saved,checkpoint)
        del saved
        disk=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        loaded=type(model)(copy.deepcopy(model.config)).float().eval()
        loaded.load_state_dict(disk['vision'],strict=True)
        assert pair.smoke.digest(trained.base.whole_state(loaded))==pair.smoke.digest(trained.base.whole_state(model))
        assert coverage.frozen_digest(loaded,inventory)==proof['frozen_prefix_sha256']
        loaded.cuda()
        loaded_head=nn.Linear(1024,128).eval().cuda()
        loaded_head.load_state_dict(disk['head'],strict=True)
        with torch.no_grad():
            a,b=fp16(model,calibration),fp16(loaded,calibration)
            assert torch.equal(a,b)
            va=F.normalize(pair.smoke.compact_head_features(a,head),dim=1)
            vb=F.normalize(pair.smoke.compact_head_features(b,loaded_head),dim=1)
            assert torch.equal(va,vb)
            packed_equal(va,vb)
            f32=loaded(pixel_values=calibration).pooler_output.float()
            terminal_cos={'pooled':F.cosine_similarity(b,f32).tolist(),'compact':F.cosine_similarity(vb,F.normalize(pair.smoke.compact_head_features(f32,loaded_head),dim=1)).tolist()}
            assert all(min(v)>=.999 for v in terminal_cos.values())
        updated_whole=pair.smoke.digest(trained.base.whole_state(model))
        updated_head=pair.smoke.digest(head.state_dict())
        assert torch.equal(disk['classifier'],classifier.cpu()) and torch.equal(disk['bank'],bank.cpu())
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    assert all(pair.sha(root/n)==h for n,h in code.items()) and pair.sha(teacher.TEACHER)==teacher.TEACHER_SHA and pair.sha(init)==arm['initializers_sha256']
    assert json.loads(json.dumps(trained.native.environment(model,processor)))==source['environment'] and teacher.qualified.numerical_flags()==flags
    assert coverage.frozen_digest(model,inventory)==proof['frozen_prefix_sha256'] and torch.cuda.max_memory_allocated()<10_000_000_000
    receipt={**training,'advance':True,'execution_sha256':args.execution_sha256,'cpu_authority_sha256':CPU_SHA,'source_checkpoint_sha256':teacher.TEACHER_SHA,'updated_whole_sha256':updated_whole,'updated_head_sha256':updated_head,'training_sha256':pair.sha(args.output/'training.json'),'training_state_discarded':args.updates==17,'strict400_reload_whole_head_packed_exact':True,'initial_cosines':initial_cos,'terminal_cosines':terminal_cos,'frozen_source_code_environment_rng_preserved':True,'total_seconds':time.perf_counter()-started,'claim_eligible':False,'new_model_quality_measured':False}
    if args.updates==100:
        receipt['checkpoint_sha256']=pair.sha(args.output/'native.pt')
    receipt['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    pair.smoke.save(args.output/'receipt.json',receipt)
    print('PASS native coverage mechanics/continuation; no model-quality claim',flush=True)


if __name__=='__main__':
    main()
