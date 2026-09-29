"""Fixed native optimization budget and complete process-boundary state."""
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch
from torch import nn
import qualify_pe_large_coverage_checkpoint as previous

coverage=previous.coverage
pair=coverage.pair
TOTAL_UPDATES=2000
PREVIOUS_SHA='15eba89fd3eb2fa1e86566ccf5166b9335c87387eb72774f9829fead229828f0'
FULL=Path('/home/riomus/runs/sfora-large-coverage-full-official-b32-v1')
FULL_AUDIT_SHA='2092703d0801c28613e8f6b485f242836fc1cdfc1653b6ca373bcf9a564b24a6'
HALF=Path('/home/riomus/runs/sfora-large-coverage-half-official-b32-v1')
HALF_AUDIT_SHA='b1e38ad84bb0c29078d207d9751fdc2d835cb38742caa7215d6320f3504815df'


def startup(root,execution_sha):
    path=root/'large-optimization-execution.json'
    assert pair.sha(path)==execution_sha
    code=json.loads(path.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items()),'optimization execution code differs'
    assert pair.sha(root/'large-coverage-checkpoint-execution.json')==PREVIOUS_SHA
    old=json.loads((root/'large-coverage-checkpoint-execution.json').read_text())
    assert all(code[n]==h for n,h in old.items())
    control,source,prior,proof,_=previous.training.startup(root,previous.TRAIN_CODE_SHA)
    assert pair.sha(FULL/'cpu-audit.json')==FULL_AUDIT_SHA and pair.sha(HALF/'cpu-audit.json')==HALF_AUDIT_SHA
    audit=json.loads((FULL/'cpu-audit.json').read_text())
    assert audit['pass'] and pair.sha(FULL/'receipt.json')==audit['receipt_sha256']
    effect=audit['matched_pool_comparison']
    assert not effect['survivor'] and all(effect[k]['product_lower95']>0 for k in ('per_query_r1','per_query_ap'))
    assert pair.sha(HALF/'receipt.json')==effect['control_receipt_sha256'] and effect['control_audit_sha256']==HALF_AUDIT_SHA
    previous.selected.helpers(root,code)
    return control,source,prior,proof,code


def schedule(target):
    target=np.asarray(target,dtype=np.int64)
    names=sorted(set(target.tolist()))
    assert names==list(range(len(names))) and len(names)>=64
    rng=np.random.default_rng(pair.SEED)
    order=rng.permutation(names)
    members={c:np.flatnonzero(target==c) for c in names}
    batches=np.asarray([[int(rng.choice(members[int(c)])) for c in order[(step*64+np.arange(64))%len(order)]] for step in range(TOTAL_UPDATES)],dtype=np.int64)
    assert batches.shape==(TOTAL_UPDATES,64) and all(len(set(target[b]))==64 for b in batches)
    assert np.array_equal(batches[:100],coverage.schedule(target))
    assert set(target[batches.ravel()].tolist())==set(names)
    return batches


def initializers(proof,arm):
    path=previous.training.CPU/(arm+'.npz')
    assert pair.sha(path)==proof['arms'][arm]['initializers_sha256']
    with np.load(path,allow_pickle=False) as values:
        result={k:torch.from_numpy(values[k].copy()) for k in ('bank','classifier','target')}
    for k in result:
        assert pair.smoke.digest({k:result[k]})==proof['arms'][arm][k+'_sha256']
    return result


def fresh(control,source,proof,arm,device):
    model,head,processor,inventory=coverage.load_native(control,source)
    assert json.loads(json.dumps(inventory))==proof['inventory']
    assert coverage.frozen_digest(model,inventory)==proof['frozen_prefix_sha256']
    model.to(device).train()
    head.to(device).train()
    values=initializers(proof,arm)
    classifier=nn.Parameter(values['classifier'].to(device))
    bank,target=values['bank'].to(device),values['target'].to(device)
    params=coverage.parameters(model,head,classifier)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5},{'params':head.parameters(),'lr':1e-4},{'params':[classifier],'lr':1e-4}],weight_decay=.05)
    scaler=pair.smoke.training_precision('fp16',device='cuda')[1] if str(device).startswith('cuda') else None
    return {'model':model,'head':head,'processor':processor,'inventory':inventory,'classifier':classifier,'bank':bank,'target':target,'positive':pair.smoke.member_bank_positive_ordinals(target.cpu().numpy(),allow_singletons=True).to(device),'params':params,'optimizer':optimizer,'scaler':scaler,'counter':0}


def fingerprint(value):
    digest=hashlib.sha256()
    def frame(value):
        encoded=value.encode()
        digest.update(str(len(encoded)).encode()+b':'+encoded)
    def visit(v):
        frame(type(v).__name__)
        if isinstance(v,torch.Tensor):
            frame(pair.smoke.digest({'tensor':v}))
        elif isinstance(v,Mapping):
            frame(str(len(v)))
            for k in sorted(v,key=repr):
                visit(k)
                visit(v[k])
        elif isinstance(v,(list,tuple)):
            frame(str(len(v)))
            for item in v:
                visit(item)
        else:
            frame(repr(v))
    visit(value)
    return digest.hexdigest()


def identity(state,proof,arm,execution_sha,schedule_sha):
    return {'schema':'native-optimization-resume-v1','arm':arm,'total_updates':TOTAL_UPDATES,'seed':pair.SEED,'execution_sha256':execution_sha,'source_checkpoint_sha256':coverage.teacher.TEACHER_SHA,'frozen_prefix_sha256':proof['frozen_prefix_sha256'],'initializers_sha256':proof['arms'][arm]['initializers_sha256'],'schedule_sha256':schedule_sha,'parameter_names':[n for n,_ in state['params']],'model_roles':[(n,p.requires_grad) for n,p in state['model'].named_parameters()],'runtime':coverage.trained.base.runtime_identity(state['model']),'buffers_sha256':fingerprint(dict(state['model'].named_buffers())),'optimizer_groups':[{k:v for k,v in group.items() if k!='params'} for group in state['optimizer'].param_groups],'precision':'cuda_fp16' if state['scaler'] else 'cpu_float32'}


def payload(state,base):
    return {'identity':{**base,'global_step':state['counter']},'vision':state['model'].state_dict(),'buffers':dict(state['model'].named_buffers()),'head':state['head'].state_dict(),'classifier':state['classifier'].detach(),'bank':state['bank'],'optimizer':state['optimizer'].state_dict(),'scaler':state['scaler'].state_dict() if state['scaler'] else None,'cpu_rng':torch.random.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if state['scaler'] else []}


def save(state,base,path):
    assert not path.exists() and not path.with_suffix('.part').exists()
    state['optimizer'].zero_grad(set_to_none=True)
    assert coverage.frozen_digest(state['model'],state['inventory'])==base['frozen_prefix_sha256']
    assert fingerprint(dict(state['model'].named_buffers()))==base['buffers_sha256']
    assert [n for n,_ in state['params']]==base['parameter_names']
    assert [id(p) for g in state['optimizer'].param_groups for p in g['params']]==[id(p) for _,p in state['params']]
    assert [{k:v for k,v in g.items() if k!='params'} for g in state['optimizer'].param_groups]==base['optimizer_groups']
    tmp=path.with_suffix('.part')
    torch.save(payload(state,base),tmp)
    os.replace(tmp,path)
    return pair.sha(path)


def restore(state,base,path,sha,expected_step):
    assert pair.sha(path)==sha,'resume checkpoint digest differs'
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert set(saved)==set(payload(state,base)),'complete optimizer/bank/RNG resume state missing'
    assert saved['identity']=={**base,'global_step':expected_step},'resume identity/counter differs'
    assert len(saved['vision'])==400 and 0<expected_step<=TOTAL_UPDATES
    assert [n for n,_ in state['params']]==base['parameter_names']
    assert [{k:v for k,v in g.items() if k!='params'} for g in state['optimizer'].param_groups]==base['optimizer_groups']
    assert all(int(s['step'])==expected_step for s in saved['optimizer']['state'].values())
    assert len(saved['optimizer']['state'])==len(state['params'])
    assert saved['optimizer']['param_groups']==state['optimizer'].state_dict()['param_groups'],'saved optimizer ordering/hyperparameters differ'
    state['model'].load_state_dict(saved['vision'],strict=True)
    state['head'].load_state_dict(saved['head'],strict=True)
    with torch.no_grad():
        state['classifier'].copy_(saved['classifier'])
        state['bank'].copy_(saved['bank'])
        buffers=dict(state['model'].named_buffers())
        assert buffers.keys()==saved['buffers'].keys()
        for n,v in buffers.items():
            v.copy_(saved['buffers'][n])
    state['optimizer'].load_state_dict(saved['optimizer'])
    if state['scaler']:
        state['scaler'].load_state_dict(saved['scaler'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        assert saved['scaler'] is None and not saved['cuda_rng']
    torch.random.set_rng_state(saved['cpu_rng'])
    state['counter']=expected_step
    assert coverage.frozen_digest(state['model'],state['inventory'])==base['frozen_prefix_sha256']
    assert fingerprint(dict(state['model'].named_buffers()))==base['buffers_sha256']
    assert fingerprint(payload(state,base))==fingerprint(saved),'restored native optimizer state differs'


def step(state,pixels,batch,rank_active,micro=16):
    model,head=state['model'],state['head']
    optimizer,scaler=state['optimizer'],state['scaler']
    members=[p for _,p in state['params']]
    device=next(model.parameters()).device
    assert model.training and head.training and len(batch)%micro==0 and len(pixels)==len(batch)
    versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    bank_version=state['bank']._version
    ce_sum=rank_sum=0.
    raw_rows=[]
    for offset in range(0,len(batch),micro):
        index=torch.tensor(batch[offset:offset+micro],device=device)
        x=pixels[offset:offset+micro].to(device)
        pooled=previous.training.fp16(model,x) if scaler else model(pixel_values=x).pooler_output
        ce,rank,raw=coverage.terms(pooled,head,state['classifier'],state['bank'],state['target'],state['positive'],index,rank_active)
        factor=micro/len(batch)
        loss=(ce+8*rank)*factor
        assert torch.isfinite(pooled).all() and torch.isfinite(raw).all() and torch.isfinite(loss)
        (scaler.scale(loss) if scaler else loss).backward()
        ce_sum+=float(ce.detach())*factor
        rank_sum+=float(rank.detach())*factor
        raw_rows.append(raw.detach())
        del pooled,ce,rank,raw,loss,x,index
    assert state['bank']._version==bank_version
    if scaler:
        scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in members)
    assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters() if not p.requires_grad)
    gradients={str(i):sum(float(p.grad.double().norm()) for p in model.encoder.layers[i].parameters()) for i in range(12,24)}
    assert all(v>0 and np.isfinite(v) for v in gradients.values())
    norm=torch.nn.utils.clip_grad_norm_(members,1,error_if_nonfinite=True)
    if scaler:
        scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale()>=scale,'optimizer update skipped'
    else:
        optimizer.step()
    state['counter']+=1
    assert len(optimizer.state)==len(members) and all(int(s['step'])==state['counter'] for s in optimizer.state.values())
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    rows,positions=pair.smoke.member_bank_refresh_rows(tuple(batch))
    raw=torch.cat(raw_rows)
    state['bank'][torch.tensor(rows,device=device)]=pair.smoke.member_bank_refresh_values(raw,raw,torch.tensor(positions,device=device),live_head=False)
    assert state['bank']._version==bank_version+1 and torch.isfinite(state['bank']).all() and all(torch.isfinite(p).all() for p in members)
    return {'step':state['counter'],'ce':ce_sum,'rank':rank_sum,'loss':ce_sum+8*rank_sum,'scale':scaler.get_scale() if scaler else None,'preclip_norm':float(norm),'gradient_norms':gradients}
