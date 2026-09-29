#!/usr/bin/env python3
"""Actual native CPU source/full-identity inputs and accumulation qualification."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import pe_large_coverage as coverage

pair,trained,teacher=coverage.pair,coverage.trained,coverage.teacher


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    root=Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control,frozen,cpu,_,code=coverage.startup(root,args.execution_sha256)
    target=np.repeat(np.arange(75),2)
    batches=coverage.schedule(target)
    assert batches.shape==(100,64) and len(set(target[batches.ravel()]))==75
    assert all(len(set(target[b]))==64 for b in batches) and np.array_equal(batches,coverage.schedule(target))
    pools=coverage.inputs(control,frozen)
    model,head,processor,inventory=coverage.load_native(control,cpu)
    original=pair.smoke.digest(trained.base.whole_state(model))
    prefix=coverage.frozen_digest(model,inventory)
    full=pools['full']
    classifier=nn.Parameter(full['classifier'].clone())
    bank=full['bank']
    bank_sha=pair.smoke.digest({'bank':bank})
    positives=pair.smoke.member_bank_positive_ordinals(full['target'].numpy(),allow_singletons=True)
    lookup={r['relative_path']:i for i,r in enumerate(full['rows'])}
    indices=torch.tensor([lookup[r['relative_path']] for r in frozen['fit_manifest'][:2]])
    assert (positives[indices]>=0).any(dim=1).all()
    cpu_rng=torch.random.get_rng_state().clone()
    images,_=pair.augmented_images(control.dataset_root,frozen['fit_manifest'],(0,1),None)
    pixels=pair.pixels(processor,images,'large')
    assert pair.smoke.digest({'pixels':pixels})==cpu['first_two_fit_pixels_sha256']
    params=coverage.parameters(model,head,classifier)
    source=model(pixel_values=pixels).pooler_output
    ce,rank,_=coverage.terms(source,head,classifier,bank,full['target'],positives,indices,True)
    loss=ce+8*rank
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in params)
    reference={n:p.grad.detach().clone() for n,p in params}
    reference_loss=float(loss.detach())
    del source,ce,rank,loss
    for _,p in params:
        p.grad=None
    partial=[]
    for i in range(2):
        source=model(pixel_values=pixels[i:i+1]).pooler_output
        ce,rank,_=coverage.terms(source,head,classifier,bank,full['target'],positives,indices[i:i+1],True)
        loss=(ce+8*rank)/2
        partial.append(float(loss.detach()))
        loss.backward()
    assert abs(sum(partial)-reference_loss)/max(abs(reference_loss),1e-8)<=2e-4
    differences={}
    null_gradients={}
    reference_squared=difference_squared=0.
    for n,p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        reference_norm=float(reference[n].double().norm())
        difference_norm=float((p.grad-reference[n]).double().norm())
        reference_squared+=reference_norm**2
        difference_squared+=difference_norm**2
        difference=difference_norm/max(reference_norm,1e-8)
        if n.endswith('.self_attn.k_proj.bias'):
            # A shared key bias cancels in softmax; dividing roundoff by zero is invalid.
            bound=torch.finfo(torch.float32).eps*p.numel()**.5
            assert max(reference_norm,float(p.grad.double().norm()),difference_norm)<=bound
            null_gradients[n]={'reference_norm':reference_norm,'difference_norm':difference_norm,'absolute_l2_roundoff_bound':bound}
        else:
            assert difference<=2e-4,'native CPU accumulation gradient differs: '+n
        differences[n]=difference
    global_difference=(difference_squared/reference_squared)**.5
    assert global_difference<=2e-4 and len(null_gradients)==12
    gradients={str(i):sum(float(p.grad.double().norm()) for p in model.encoder.layers[i].parameters() if p.grad is not None) for i in range(12,24)}
    assert all(v>0 for v in gradients.values())
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    assert coverage.frozen_digest(model,inventory)==prefix and pair.smoke.digest(trained.base.whole_state(model))==original
    assert pair.smoke.digest(head.state_dict())==cpu['teacher_head_sha256'] and pair.smoke.digest({'bank':bank})==bank_sha
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    assert pair.smoke.member_bank_refresh_rows((7,0,7,8))==((0,7,8),(1,2,3))
    for _,p in params:
        p.grad=None
    del reference,params,classifier,positives,source,ce,rank,loss
    parameter=next(model.embeddings.parameters())
    before=parameter.detach().clone()
    version=parameter._version
    with torch.no_grad():
        parameter.data.flatten()[0].add_(.01)
        assert parameter._version==version and coverage.frozen_digest(model,inventory)!=prefix
        parameter.copy_(before)
    assert coverage.frozen_digest(model,inventory)==prefix and pair.smoke.digest(trained.base.whole_state(model))==original
    arms={}
    for arm,pool in pools.items():
        batch=tuple(map(int,pool['batches'][0]))
        images,rgb=pair.augmented_images(control.dataset_root,pool['rows'],batch,1)
        x=pair.pixels(processor,images,'large')
        target=pool['target'].numpy()
        exposure=np.bincount(target[pool['batches'].ravel()],minlength=len(pool['classes']))
        assert exposure.min()>=1 and len(set(target.tolist()))==len(pool['classes'])
        positives=pair.smoke.member_bank_positive_ordinals(target,allow_singletons=True)
        valid=positives>=0
        assert not ((positives==torch.arange(len(target))[:,None]) & valid).any()
        assert all(torch.equal(pool['target'][pos[pos>=0]],torch.full((int((pos>=0).sum()),),int(pool['target'][i]))) for i,pos in enumerate(positives))
        arms[arm]={'rows':pool['rows'],'classes':pool['classes'],'batches':pool['batches'].tolist(),'rank_active':pool['rank_active'],'first_rgb_sha256':rgb,'first_pixels_sha256':pair.smoke.digest({'pixels':x}),'target_sha256':pair.smoke.digest({'target':pool['target']}),'bank_sha256':pair.smoke.digest({'bank':pool['bank']}),'classifier_sha256':pair.smoke.digest({'classifier':pool['classifier']}),'actual_gradient_input_identities':len(pool['classes']),'minimum_class_exposure':int(exposure.min()),'maximum_class_exposure':int(exposure.max()),'original_trained_proxy_rows_exact':True,'positive_self_class_mapping_exact':True}
    original_sha=pair.sha
    with patch.object(pair,'sha',lambda p:'altered' if Path(p)==Path(__file__) else original_sha(p)):
        try:
            coverage.startup(root,args.execution_sha256)
        except AssertionError as error:
            assert str(error)=='coverage execution code differs'
        else:
            raise AssertionError('changed coverage CPU driver accepted')
    assert all(pair.sha(root/n)==h for n,h in code.items()) and pair.sha(teacher.TEACHER)==teacher.TEACHER_SHA
    assert json.loads(json.dumps(trained.native.environment(model,processor)))==cpu['environment']
    args.output.mkdir(exist_ok=False)
    for arm,pool in pools.items():
        path=args.output/(arm+'.npz')
        np.savez(path,bank=pool['bank'].numpy(),classifier=pool['classifier'].numpy(),target=pool['target'].numpy())
        arms[arm]['initializers_sha256']=pair.sha(path)
    pair.smoke.save(args.output/'receipt.json',{'pass':True,'code':code,'execution_sha256':args.execution_sha256,'source_checkpoint_sha256':teacher.TEACHER_SHA,'source_whole_sha256':original,'source_head_sha256':cpu['teacher_head_sha256'],'frozen_prefix_sha256':prefix,'inventory':inventory,'arms':arms,'actual_native_CPU_B2_vs_2B1_loss_gradient_relative_tolerance':2e-4,'actual_native_CPU_gradient_relative_differences':differences,'actual_native_CPU_global_gradient_relative_error':global_difference,'actual_native_CPU_null_key_bias_roundoff':null_gradients,'actual_native_CPU_reference_loss':reference_loss,'actual_native_CPU_accumulated_loss':sum(partial),'actual_native_twelve_block_gradient_norms':gradients,'source_head_bank_rng_content_unchanged':True,'changed_driver_rejected':True,'duplicate_refresh_last_occurrence_exact':True,'frozen_data_mutation_detected_and_restored':True,'optimizer_updates':0,'official_quality_read':False,'new_model_quality_measured':False,'claim_eligible':False})
    print('PASS actual native12-block CPU source/grad-aggregation and actual all-ID/proxy/bank/pixel authority; no optimizer/model-quality update')


if __name__=='__main__':
    main()
