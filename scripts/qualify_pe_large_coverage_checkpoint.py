#!/usr/bin/env python3
"""Actual updated-source CPU authority then fixed-checkpoint official B32 confirmation."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import train_pe_large_coverage as training
from sfora.joint_relational_compaction import PackedInt8Embeddings,pack_int8_unit_embeddings

coverage=training.coverage
selected=coverage.selected
pair,trained,teacher=coverage.pair,coverage.trained,coverage.teacher
TRAIN_CODE_SHA='b4b8ae6341192d78934e9aa51d50aab04e1107fd2e523d9cb21b257e8466295f'
PROTOCOL=Path('/home/riomus/runs/sfora-large-inshop-official-qual-v2/official-cpu-proof.json')
PROTOCOL_SHA='be7b27c68b5e2895063d4f24b2b9d4e396a77ed79b2426596893c293df2c20f2'


def startup(root,execution_sha,arm,training_sha):
    path=root/'large-coverage-checkpoint-execution.json'
    assert pair.sha(path)==execution_sha
    code=json.loads(path.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items()),'coverage checkpoint code differs'
    control,source,prior,proof,old=training.startup(root,TRAIN_CODE_SHA)
    assert len(code)==len(old)+1 and all(code[n]==h for n,h in old.items())
    run=Path('/home/riomus/runs/sfora-large-coverage-'+arm+'-100-v1')
    assert pair.sha(run/'receipt.json')==training_sha
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['advance'] and receipt['updates']==100 and receipt['arm']==arm and not receipt['quality_read']
    assert receipt['execution_sha256']==TRAIN_CODE_SHA and receipt['cpu_authority_sha256']==training.CPU_SHA
    assert receipt['strict400_reload_whole_head_packed_exact'] and receipt['frozen_source_code_environment_rng_preserved']
    assert pair.sha(run/'native.pt')==receipt['checkpoint_sha256'] and pair.sha(run/'training.json')==receipt['training_sha256']
    assert receipt['steps'][-1]['gradient_input_identities']==len(proof['arms'][arm]['classes'])
    assert pair.sha(PROTOCOL)==PROTOCOL_SHA
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['pass'] and protocol['train_query_gallery_ids_disjoint'] and protocol['prior_official_benchmark_exposure']
    assert pair.sha(control.dataset_root/'Eval/list_eval_partition.txt')==protocol['partition_sha256']==selected.PARTITION_SHA
    official=selected.helpers(root,code)
    return control,source,prior,proof,run,receipt,protocol,official,code


def updated(control,source,proof,run,receipt):
    model,head,processor,inventory=coverage.load_native(control,source)
    disk=torch.load(run/'native.pt',map_location='cpu',weights_only=True,mmap=True)
    assert len(disk['vision'])==400
    model.load_state_dict(disk['vision'],strict=True)
    head.load_state_dict(disk['head'],strict=True)
    assert pair.smoke.digest(trained.base.whole_state(model))==receipt['updated_whole_sha256']
    assert pair.smoke.digest(head.state_dict())==receipt['updated_head_sha256']
    assert coverage.frozen_digest(model,inventory)==proof['frozen_prefix_sha256']
    model.requires_grad_(False).eval()
    head.requires_grad_(False).eval()
    return model,head,processor


def same_runtime(native,public):
    a,b=(trained.base.runtime_identity(m) for m in (native,public))
    # from_pretrained records F32 load metadata; public from_config leaves it unset.
    assert a['config'].pop('dtype')=='float32' and b['config'].pop('dtype') is None
    assert a==b


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--arm',choices=('half','full'),required=True)
    parser.add_argument('--training-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qualify-cpu',action='store_true')
    parser.add_argument('--cpu-proof',type=Path)
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--audit-cpu',action='store_true')
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--control',type=Path)
    parser.add_argument('--control-sha256')
    parser.add_argument('--control-audit-sha256')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control,source,prior,proof,run,receipt,protocol,official,code=startup(root,args.execution_sha256,args.arm,args.training_sha256)
    if args.qualify_cpu:
        assert not torch.cuda.is_available() and not args.output.exists()
        model,head,processor=updated(control,source,proof,run,receipt)
        arm=proof['arms'][args.arm]
        images,_=pair.augmented_images(control.dataset_root,arm['rows'],tuple(arm['batches'][0][:2]),1)
        pixels=pair.pixels(processor,images,'large')
        from transformers import AutoConfig
        loaded=type(model)(AutoConfig.from_pretrained(control.large_snapshot,local_files_only=True).vision_config).float().requires_grad_(False).eval()
        same_runtime(model,loaded)
        disk=torch.load(run/'native.pt',map_location='cpu',weights_only=True,mmap=True)
        loaded.load_state_dict(disk['vision'],strict=True)
        loaded_head=nn.Linear(1024,128).requires_grad_(False).eval()
        loaded_head.load_state_dict(disk['head'],strict=True)
        with torch.inference_mode():
            a,b=model(pixel_values=pixels).pooler_output,loaded(pixel_values=pixels).pooler_output
            assert torch.equal(a,b)
            va,vb=F.normalize(pair.smoke.compact_head_features(a,head),dim=1),F.normalize(pair.smoke.compact_head_features(b,loaded_head),dim=1)
            assert torch.equal(va,vb)
            training.packed_equal(va,vb)
        assert pair.smoke.digest(trained.base.whole_state(loaded))==receipt['updated_whole_sha256']
        loaded.half()
        cast_sha=pair.smoke.digest(trained.base.whole_state(loaded))
        assert cast_sha!=receipt['updated_whole_sha256'] and all(p.dtype==torch.float16 for p in loaded.parameters())
        original=pair.sha
        with patch.object(pair,'sha',lambda p:'altered' if Path(p)==Path(__file__) else original(p)):
            try:
                startup(root,args.execution_sha256,args.arm,args.training_sha256)
            except AssertionError as error:
                assert str(error)=='coverage checkpoint code differs'
            else:
                raise AssertionError('altered checkpoint driver accepted')
        assert all(pair.sha(root/n)==h for n,h in code.items()) and pair.sha(run/'native.pt')==receipt['checkpoint_sha256']
        pair.smoke.save(args.output,{'pass':True,'arm':args.arm,'code':code,'execution_sha256':args.execution_sha256,'training_receipt_sha256':args.training_sha256,'checkpoint_sha256':receipt['checkpoint_sha256'],'updated_f32_whole_sha256':receipt['updated_whole_sha256'],'updated_f16_whole_sha256':cast_sha,'updated_head_sha256':receipt['updated_head_sha256'],'environment':source['environment'],'strict_updated_native_CPU_B2_whole_head_packed_exact':True,'frozen_prefix_matches_original':True,'protocol_sha256':PROTOCOL_SHA,'changed_driver_rejected':True,'quality_read':False,'optimizer_updates':0})
        print('PASS actual updated native CPU source/strict reload/F16 cast; no official quality')
        return
    assert args.cpu_proof and args.cpu_sha256
    assert pair.sha(args.cpu_proof)==args.cpu_sha256
    qualified=json.loads(args.cpu_proof.read_text())
    assert qualified['pass'] and qualified['code']==code and qualified['checkpoint_sha256']==receipt['checkpoint_sha256'] and qualified['arm']==args.arm
    if args.audit_cpu:
        assert not torch.cuda.is_available() and args.receipt_sha256 and not (args.output/'cpu-audit.json').exists()
        assert pair.sha(args.output/'receipt.json')==args.receipt_sha256
        measured=json.loads((args.output/'receipt.json').read_text())
        assert measured['code']==code and measured['cpu_authority_sha256']==args.cpu_sha256 and measured['checkpoint_sha256']==receipt['checkpoint_sha256']
        assert measured['native_all_query_top10_ordinal_score_bits_exact'] and measured['source_state_rng_environment_code_library_preserved']
        q,g=(selected.fp16.load_packed(args.output,measured,role) for role in ('query','gallery'))
        qlabels,glabels=(tuple(r['product'] for r in protocol['protocol'][role]) for role in ('query','gallery'))
        quality,intervals,advance=selected.metrics(official,q,g,qlabels,glabels,torch.device('cpu'))
        assert all(np.max(np.abs(np.asarray(v)-np.asarray(measured['quality'][k])))<1e-6 for k,v in quality.items())
        assert all(abs(v-measured['quality_intervals'][k][n])<1e-6 for k,row in intervals.items() for n,v in row.items())
        assert advance==measured['advance_minimum_dated_reference_screen']
        comparison={}
        if args.control:
            assert args.arm=='full' and pair.sha(args.control/'receipt.json')==args.control_sha256 and pair.sha(args.control/'cpu-audit.json')==args.control_audit_sha256
            baseline=json.loads((args.control/'receipt.json').read_text())
            audit=json.loads((args.control/'cpu-audit.json').read_text())
            assert audit['pass'] and audit['receipt_sha256']==args.control_sha256 and baseline['arm']=='half' and baseline['code']==code
            for metric in ('per_query_r1','per_query_ap'):
                delta=np.asarray(quality[metric])-np.asarray(baseline['quality'][metric])
                comparison[metric]={'mean_difference':float(delta.mean())}
                for kind,groups in (('product',np.asarray(qlabels)),('query',np.arange(len(delta)))):
                    comparison[metric][kind+'_lower95']=pair.bootstrap_lower(delta,groups)
                    comparison[metric][kind+'_upper95']=-pair.bootstrap_lower(-delta,groups)
            comparison['survivor']=advance and all(comparison[k]['product_lower95']>0 for k in ('per_query_r1','per_query_ap'))
            comparison['control_receipt_sha256']=args.control_sha256
            comparison['control_audit_sha256']=args.control_audit_sha256
        assert all(pair.sha(root/n)==h for n,h in code.items())
        pair.smoke.save(args.output/'cpu-audit.json',{'pass':True,'receipt_sha256':args.receipt_sha256,'quality':quality,'quality_intervals':intervals,'advance_minimum_dated_reference_screen':advance,'matched_pool_comparison':comparison,'prior_official_benchmark_exposure':True,'claim_eligible':False})
        print('PASS complete saved official wire CPU quality/CI/decision replay',flush=True)
        return
    assert torch.cuda.is_available() and not args.output.exists()
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    flags=teacher.qualified.numerical_flags()
    assert flags==prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder=Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot,checkpoint=run/'native.pt',expected_checkpoint_sha256=receipt['checkpoint_sha256'],model_file_sha256=pair.smoke.MODEL_HASHES,precision='fp16_native',device=torch.device('cuda'))
    def preserved():
        assert pair.smoke.digest(trained.base.whole_state(encoder.vision))==qualified['updated_f16_whole_sha256'] and pair.smoke.digest(encoder.head.state_dict())==qualified['updated_head_sha256']
        assert json.loads(json.dumps(trained.native.environment(encoder.vision,encoder.processor)))==qualified['environment']
        assert all(p.grad is None for m in (encoder.vision,encoder.head) for p in m.parameters())
        assert not encoder.vision.training and not encoder.head.training
        assert all(p.dtype==torch.float16 for p in encoder.vision.parameters())
        assert all(p.dtype==torch.float32 for p in encoder.head.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
    preserved()
    reference,reference_head,_=updated(control,source,proof,run,receipt)
    reference.half().cuda()
    reference_head.cuda()
    same_runtime(reference,encoder.vision)
    assert pair.smoke.digest(trained.base.whole_state(reference))==qualified['updated_f16_whole_sha256']
    cpu_rng,cuda_rng=torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
    arrays={}
    for role in ('query','gallery'):
        chunks=[]
        for start in range(0,len(protocol['protocol'][role]),32):
            images=[]
            for row in protocol['protocol'][role][start:start+32]:
                path=control.dataset_root/row['relative_path']
                assert pair.sha(path)==row['image_sha256']
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            actual=encoder.encode_images(images)
            if start==0:
                with torch.inference_mode():
                    pixels=pair.pixels(encoder.processor,images,'large').cuda().half()
                    pooled=reference(pixel_values=pixels).pooler_output
                    expected=pack_int8_unit_embeddings(F.normalize(pair.smoke.compact_head_features(pooled,reference_head),dim=1).cpu())
                selected.fp16.same(actual,expected)
            chunks.append(actual)
            assert torch.cuda.max_memory_allocated()<10_000_000_000
            if (start//32+1)%64==0 or start+32>=len(protocol['protocol'][role]):
                print(json.dumps({role+'_public_B32_images':min(start+32,len(protocol['protocol'][role]))}),flush=True)
        arrays[role]=PackedInt8Embeddings(torch.cat([x.codes for x in chunks]),torch.cat([x.inverse_norms for x in chunks]))
    q,g=arrays['query'],arrays['gallery']
    assert pair.sha(selected.serving.LIBRARY)==selected.serving.LIBRARY_SHA
    with selected.serving.CutilePackedInt8Gallery.open_packed(selected.serving.LIBRARY,g) as native:
        for start in range(0,len(q.codes),32):
            block=PackedInt8Embeddings(q.codes[start:start+32].contiguous(),q.inverse_norms[start:start+32].contiguous())
            actual=native.search_packed(block)
            scores=(block.codes.float().cuda()@g.codes.float().cuda().T)*block.inverse_norms.float().cuda()[:,None]*g.inverse_norms.float().cuda()[None,:]
            order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:10]
            selected.pilot.equal(actual,(order.cpu().numpy(),scores.gather(1,order).cpu().numpy()))
    qlabels,glabels=tuple(r['product'] for r in protocol['protocol']['query']),tuple(r['product'] for r in protocol['protocol']['gallery'])
    quality,intervals,advance=selected.metrics(official,q,g,qlabels,glabels,torch.device('cuda'))
    preserved()
    assert pair.smoke.digest(trained.base.whole_state(reference))==qualified['updated_f16_whole_sha256'] and pair.smoke.digest(reference_head.state_dict())==qualified['updated_head_sha256']
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    assert teacher.qualified.numerical_flags()==flags and all(pair.sha(root/n)==h for n,h in code.items())
    assert pair.sha(run/'native.pt')==receipt['checkpoint_sha256'] and pair.sha(PROTOCOL)==PROTOCOL_SHA and pair.sha(selected.serving.LIBRARY)==selected.serving.LIBRARY_SHA
    assert pair.sha(control.dataset_root/'Eval/list_eval_partition.txt')==selected.PARTITION_SHA and torch.cuda.max_memory_allocated()<10_000_000_000
    args.output.mkdir(exist_ok=False)
    facts={}
    for role,values in arrays.items():
        for field,value in (('codes',values.codes),('inverse',values.inverse_norms)):
            path=args.output/(role+'.'+field+'.npy')
            np.save(path,value.numpy(),allow_pickle=False)
            facts[role+'_'+field+'_sha256']=pair.sha(path)
    pair.smoke.save(args.output/'receipt.json',{**facts,'arm':args.arm,'code':code,'execution_sha256':args.execution_sha256,'cpu_authority_sha256':args.cpu_sha256,'training_receipt_sha256':args.training_sha256,'checkpoint_sha256':receipt['checkpoint_sha256'],'quality':quality,'quality_intervals':intervals,'advance_minimum_dated_reference_screen':advance,'query_images':14218,'gallery_images':12612,'native_all_query_top10_ordinal_score_bits_exact':True,'independent_updated_native_original_preprocessor_first32_each_role_packed_exact':True,'source_state_rng_environment_code_library_preserved':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'optimizer_updates':0,'batch':32,'prior_official_benchmark_exposure':True,'claim_eligible':False})
    print('GO minimum official screen' if advance else 'KILL minimum official quality screen',flush=True)


if __name__=='__main__':
    main()
