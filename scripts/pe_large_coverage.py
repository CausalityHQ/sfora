"""Fixed F5 source and actual full-identity continuation inputs/objective."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import qualify_pe_large_inshop_official as selected

pair, trained, teacher = selected.pair, selected.trained, selected.teacher
OFFICIAL_CODE_SHA = 'c7c17085982d364a8d7d261102f1c047e40a067db8a69204596947fe4e4cf158'
OFFICIAL = Path('/home/riomus/runs/sfora-large-official-b32-v1')
OFFICIAL_SHA = '174282ce87459a71b23594a6a679c3c4c4cd55046fb0dbf925ef7542aea26f24'
OFFICIAL_AUDIT_SHA = '35071ed2baa8bbfe11d3ebb3b0387c22b1233a4c70341a2bf01e74bb356e8238'
FIT = Path('/home/riomus/runs/sfora-large-teacher-fit-targets-v1')
FIT_SHA = '277f9b774470355f76c16675a9c7808abe602c83612d83f4d90e4d37cb5507fe'


def startup(root, execution_sha):
    path = root / 'large-coverage-execution.json'
    assert pair.sha(path) == execution_sha
    code = json.loads(path.read_text())
    assert all(pair.sha(root/n) == h for n,h in code.items()), 'coverage execution code differs'
    control,frozen,cpu,prior,cast,old = selected.startup(root,OFFICIAL_CODE_SHA)
    assert all(code[n] == h for n,h in old.items())
    assert pair.sha(OFFICIAL/'receipt.json') == OFFICIAL_SHA and pair.sha(OFFICIAL/'cpu-audit.json') == OFFICIAL_AUDIT_SHA
    audit = json.loads((OFFICIAL/'cpu-audit.json').read_text())
    assert audit['pass'] and not audit['advance_minimum_dated_reference_screen'] and audit['receipt_sha256']==OFFICIAL_SHA
    return control,frozen,cpu,prior,code


def schedule(target, seed=pair.SEED):
    target=np.asarray(target,dtype=np.int64)
    names=sorted(set(target.tolist()))
    assert names==list(range(len(names))) and len(names)>=64
    rng=np.random.default_rng(seed)
    order=rng.permutation(names)
    members={c:np.flatnonzero(target==c) for c in names}
    batches=[]
    for step in range(100):
        classes=order[(step*64+np.arange(64))%len(order)]
        batches.append([int(rng.choice(members[int(c)])) for c in classes])
    batches=np.asarray(batches,dtype=np.int64)
    assert batches.shape==(100,64) and all(len(set(target[b].tolist()))==64 for b in batches)
    assert set(target[batches.ravel()].tolist())==set(names)
    return batches


def inputs(control,frozen):
    from sfora.unicom_inshop import parse_inshop_partition
    assert pair.sha(FIT/'receipt.json') == FIT_SHA
    fit=json.loads((FIT/'receipt.json').read_text())
    assert fit['fit_manifest']==frozen['fit_manifest'] and pair.sha(FIT/'teacher.fit.npy')==fit['fit_sha256']
    held_path=selected.serving.HELD
    assert pair.sha(held_path/'large.held.npy')==json.loads((held_path/'receipt.json').read_text())['held_sha256']
    rows=frozen['fit_manifest']+frozen['held_manifest']
    values=np.concatenate([np.load(FIT/'teacher.fit.npy',allow_pickle=False),np.load(held_path/'large.held.npy',allow_pickle=False)])
    assert values.dtype==np.float32 and values.shape==(25882,128) and np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5,rtol=0)
    lookup={r['relative_path']:i for i,r in enumerate(rows)}
    assert len(lookup)==25882
    records=[r for r in parse_inshop_partition(control.dataset_root) if r.split=='train']
    order=[lookup[str(r.image_path.relative_to(control.dataset_root))] for r in records]
    assert len(order)==25882 and all(rows[i]['product']==r.label for i,r in zip(order,records,strict=True))
    saved=torch.load(teacher.TEACHER,map_location='cpu',weights_only=True,mmap=True)
    old_names=sorted(set(r['product'] for r in frozen['fit_manifest']))
    assert [old_names[i] for i in frozen['target']]==[r['product'] for r in frozen['fit_manifest']]
    assert saved['classifier'].shape==(2004,128) and saved['classifier'].dtype==torch.float32
    old_proxy=saved['classifier'].detach().clone()
    assert torch.isfinite(old_proxy).all() and (old_proxy.norm(dim=1)>0).all()
    result={}
    for arm,indices in (('half',list(range(len(frozen['fit_manifest'])))),('full',order)):
        manifest=[rows[i] for i in indices]
        names=sorted(set(r['product'] for r in manifest))
        classes={name:i for i,name in enumerate(names)}
        target=np.asarray([classes[r['product']] for r in manifest],dtype=np.int64)
        bank=torch.from_numpy(values[indices].copy())
        classifier=torch.empty((len(names),128),dtype=torch.float32)
        for i,name in enumerate(names):
            if name in old_names:
                classifier[i]=old_proxy[old_names.index(name)]
            else:
                classifier[i]=F.normalize(bank[torch.from_numpy(target==i)].mean(dim=0),dim=0)
        assert torch.equal(classifier[torch.tensor([classes[n] for n in old_names])],old_proxy)
        assert torch.isfinite(classifier).all() and (classifier.norm(dim=1)>0).all()
        batches=schedule(target)
        counts=np.bincount(target)
        rank=[bool((counts[target[b]]>1).all()) for b in batches]
        result[arm]={'rows':manifest,'classes':names,'target':torch.from_numpy(target),'bank':bank,'classifier':classifier,'batches':batches,'rank_active':rank}
    assert len(result['half']['classes'])==2004 and len(result['full']['classes'])==3997
    return result


def load_native(control,cpu):
    model,processor=pair.smoke.load_arm(control,'large')
    saved=torch.load(teacher.TEACHER,map_location='cpu',weights_only=True,mmap=True)
    assert len(saved['vision'])==400
    model.load_state_dict(saved['vision'],strict=True)
    head=nn.Linear(1024,128)
    head.load_state_dict(saved['head'],strict=True)
    assert pair.smoke.digest(trained.base.whole_state(model))==cpu['teacher_whole_sha256']
    assert pair.smoke.digest(head.state_dict())==cpu['teacher_head_sha256']
    assert json.loads(json.dumps(trained.native.environment(model,processor)))==cpu['environment']
    inventory=pair.smoke.freeze_prefix(model,'large')
    model.train()
    head.train()
    roots=('embeddings.',)+tuple('encoder.layers.'+str(i)+'.' for i in range(12))
    assert all(p.requires_grad==(not n.startswith(roots)) for n,p in model.named_parameters())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    return model,head,processor,inventory


def terms(source,head,classifier,bank,target,positive,index,rank_active):
    raw=pair.smoke.compact_head_features(source,head)
    ce=pair.smoke.sharded_mask_arcface_loss(raw,classifier,target[index],torch.arange(128,device=source.device).unsqueeze(0),margin=.3,scale=64)
    rank=pair.smoke.member_bank_rank_loss(raw,bank,head,positive[index],index,live_head=False) if rank_active else ce.new_zeros(())
    return ce,rank,raw


def parameters(model,head,classifier):
    return [(n,p) for n,p in model.named_parameters() if p.requires_grad]+[('compact_head.'+n,p) for n,p in head.named_parameters()]+[('classifier',classifier)]


def frozen_digest(model,inventory):
    return pair.smoke.digest(pair.smoke.frozen_state(model,'large',inventory))
