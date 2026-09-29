#!/usr/bin/env python3
"""Fixed F5/native-F16 official InShop qualification; prior TEST exposure retained."""
import argparse
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import measure_pe_large_public_precision_pair as pilot
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

pair, trained, teacher, fp16, serving = pilot.pair, pilot.trained, pilot.teacher, pilot.fp16, pilot.serving
PILOT_CODE_SHA = 'e1d6c4259a3bc1413a68bd139bbf25c3931ea8270b04449666271750941dadd9'
SELECTED_B1 = Path('/home/riomus/runs/sfora-large-public-fp16-b1-v2')
SELECTED_B1_SHA = '5d4302653b9fa7c2f3fa94ff4e5958f54f8081050fb4bc86d31885faa00ba848'
SELECTED_B1_AUDIT_SHA = '42c75d1f7922f5c0301d38862c648b10e92cad92dc74f515838457e2f975574d'
LATENCY = Path('/home/riomus/runs/sfora-large-public-precision-pair-v1')
LATENCY_SHA = 'fe041e1e0611a05fa54633852047271424ae9968750053c18183937e41f7e4af'
LATENCY_AUDIT_SHA = 'b63acb7a72b3a82b9c8464fe818aa16cb9f51cda88dc0ce96bec80a81d7ee651'
PARTITION_SHA = 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'


def startup(root, execution_sha):
    manifest = root / 'large-official-execution.json'
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n,h in code.items()), 'official execution code differs'
    control, frozen, cpu, prior, old, authority = pilot.startup(root, PILOT_CODE_SHA, SELECTED_B1, SELECTED_B1_SHA, SELECTED_B1_AUDIT_SHA)
    assert all(code[n] == h for n,h in old.items())
    assert pair.sha(LATENCY / 'latency.json') == LATENCY_SHA and pair.sha(LATENCY / 'latency-cpu-audit.json') == LATENCY_AUDIT_SHA
    latency = json.loads((LATENCY / 'latency.json').read_text())
    audit = json.loads((LATENCY / 'latency-cpu-audit.json').read_text())
    assert latency['pilot_go'] and audit['pass'] and audit['pilot_go'] and audit['receipt_sha256'] == LATENCY_SHA
    assert pair.sha(control.dataset_root / 'Eval/list_eval_partition.txt') == PARTITION_SHA
    return control, frozen, cpu, prior, authority, code


def helpers(root, code):
    # Import added legacy helpers only AFTER all original closed authority guards.
    import evaluate_inshop_siglip2_official as official
    assert Path(inspect.getfile(inspect.unwrap(official.score_asymmetric))).resolve() == root / 'evaluate_inshop_siglip2_official.py'
    assert Path(inspect.getfile(official.parse_inshop_partition)).resolve() == root / 'src/sfora/unicom_inshop.py'
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).is_file() and Path(name).resolve().is_relative_to(root):
            relative = str(Path(name).resolve().relative_to(root))
            assert code.get(relative) == pair.sha(Path(name)), 'added imported source unqualified: ' + relative
    return official


def metrics(official, query, gallery, qlabels, glabels, device):
    for packed in (query,gallery):
        assert packed.codes.dtype == torch.int8 and packed.codes.ndim == 2 and packed.codes.shape[1] == 128
        assert packed.inverse_norms.dtype == torch.float16 and packed.inverse_norms.shape == (len(packed.codes),)
        norms = torch.linalg.vector_norm(packed.codes.float(),dim=1)
        assert packed.codes.min() >= -127 and (norms > 0).all() and torch.equal(norms.reciprocal().half(),packed.inverse_norms)
    quality = official.score_asymmetric(query.codes.float().to(device),gallery.codes.float().to(device),tuple(qlabels),tuple(glabels),query_inverse=query.inverse_norms.float().to(device),gallery_inverse=gallery.inverse_norms.float().to(device))
    intervals = {}
    for name in ('per_query_r1','per_query_ap'):
        values = np.asarray(quality[name])
        intervals[name] = {}
        for kind,groups in (('product',np.asarray(qlabels)),('query',np.arange(len(values)))):
            intervals[name][kind+'_lower95'] = pair.bootstrap_lower(values,groups)
            intervals[name][kind+'_upper95'] = -pair.bootstrap_lower(-values,groups)
    return quality, intervals, bool(quality['recall_at_1'] > .967)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qualify-cpu',action='store_true')
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--audit-cpu',action='store_true')
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--b32',type=Path)
    parser.add_argument('--b32-sha256')
    parser.add_argument('--b32-audit-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control,frozen,cpu,prior,authority,code = startup(root,args.execution_sha256)
    official = helpers(root,code)
    records = official.parse_inshop_partition(control.dataset_root)
    train = tuple(r for r in records if r.split == 'train')
    queries = tuple(r for r in records if r.split == 'query')
    gallery = tuple(r for r in records if r.split == 'gallery')
    assert (len(train),len(queries),len(gallery)) == (25882,14218,12612)
    assert set(r.label for r in queries) == set(r.label for r in gallery)
    assert not set(r.label for r in train) & set(r.label for r in queries)
    assert {r.label for r in train} == {r['product'] for r in frozen['fit_manifest']+frozen['held_manifest']}
    qlabels,glabels = tuple(r.label for r in queries),tuple(r.label for r in gallery)
    if args.qualify_cpu:
        assert not torch.cuda.is_available() and not args.output.exists() and not args.audit_cpu and not args.b32
        packed = fp16.load_packed(pilot.B32,json.loads((pilot.B32/'receipt.json').read_text()),'held')
        q = PackedInt8Embeddings(packed.codes[frozen['query']],packed.inverse_norms[frozen['query']])
        g = PackedInt8Embeddings(packed.codes[frozen['gallery']],packed.inverse_norms[frozen['gallery']])
        labels = [r['product'] for r in frozen['held_manifest']]
        quality = official.score_asymmetric(q.codes.float(),g.codes.float(),tuple(labels[i] for i in frozen['query']),tuple(labels[i] for i in frozen['gallery']),query_inverse=q.inverse_norms.float(),gallery_inverse=g.inverse_norms.float())
        old = json.loads((pilot.B32/'receipt.json').read_text())['quality']
        assert all(np.max(np.abs(np.asarray(v)-np.asarray(old[k]))) < 1e-6 for k,v in quality.items())
        rows = {role:[{'relative_path':str(r.image_path.relative_to(control.dataset_root)),'product':r.label,'image_sha256':pair.sha(r.image_path)} for r in group] for role,group in (('query',queries),('gallery',gallery))}
        original = pair.sha
        with patch.object(pair,'sha',lambda p:'altered' if Path(p)==Path(__file__) else original(p)):
            try:
                startup(root,args.execution_sha256)
            except AssertionError as error:
                assert str(error) == 'official execution code differs'
            else:
                raise AssertionError('altered official driver accepted')
        assert all(pair.sha(root/n)==h for n,h in code.items())
        pair.smoke.save(args.output,{'pass':True,'code':code,'execution_sha256':args.execution_sha256,'source_checkpoint_sha256':teacher.TEACHER_SHA,'fp16_whole_sha256':authority['fp16_whole_sha256'],'partition_sha256':PARTITION_SHA,'protocol':rows,'query_images':14218,'gallery_images':12612,'query_gallery_products':len(set(qlabels)),'train_products':len(set(r.label for r in train)),'train_query_gallery_ids_disjoint':True,'actual_wire_scorer_all_TRAIN_per_query_exact':True,'changed_driver_rejected':True,'official_images_decoded':0,'official_quality_read':False,'prior_official_benchmark_exposure':True,'optimizer_updates':0})
        print('PASS actual official standard protocol/image authority and TRAIN packed-scorer CPU replay; no official quality')
        return
    proof = root/'official-cpu-proof.json'
    assert args.cpu_sha256 and pair.sha(proof)==args.cpu_sha256
    qualified = json.loads(proof.read_text())
    assert qualified['pass'] and qualified['code']==code and qualified['fp16_whole_sha256']==authority['fp16_whole_sha256']
    for role,group in (('query',queries),('gallery',gallery)):
        assert [(r['relative_path'],r['product']) for r in qualified['protocol'][role]] == [(str(r.image_path.relative_to(control.dataset_root)),r.label) for r in group]
    if args.audit_cpu:
        assert not torch.cuda.is_available() and args.receipt_sha256
        path = args.output/'receipt.json'
        assert pair.sha(path)==args.receipt_sha256 and not (args.output/'cpu-audit.json').exists()
        receipt = json.loads(path.read_text())
        assert receipt['code']==code and receipt['cpu_authority_sha256']==args.cpu_sha256 and receipt['source_state_rng_environment_code_library_preserved']
        assert receipt['source_checkpoint_sha256']==teacher.TEACHER_SHA and receipt['optimizer_updates']==0 and receipt['prior_official_benchmark_exposure']
        q = fp16.load_packed(args.output,receipt,'query')
        if receipt['batch']==32:
            g = fp16.load_packed(args.output,receipt,'gallery')
            assert receipt['native_all_query_top10_ordinal_score_bits_exact']
        else:
            assert args.b32 and pair.sha(args.b32/'receipt.json')==args.b32_sha256==receipt['b32_receipt_sha256']
            g = fp16.load_packed(args.b32,json.loads((args.b32/'receipt.json').read_text()),'gallery')
            assert receipt['native_all_public_B1_query_r1_exact']
        quality,intervals,advance = metrics(official,q,g,qlabels,glabels,torch.device('cpu'))
        assert all(np.max(np.abs(np.asarray(v)-np.asarray(receipt['quality'][k])))<1e-6 for k,v in quality.items())
        assert all(abs(v-receipt['quality_intervals'][k][n])<1e-6 for k,row in intervals.items() for n,v in row.items())
        assert advance==receipt['advance_minimum_dated_reference_screen'] and all(pair.sha(root/n)==h for n,h in code.items())
        pair.smoke.save(args.output/'cpu-audit.json',{'pass':True,'advance_minimum_dated_reference_screen':advance,'receipt_sha256':args.receipt_sha256,'quality':quality,'quality_intervals':intervals,'claim_eligible':False,'prior_official_benchmark_exposure':True})
        print('PASS complete official saved-wire CPU per-query/interval/screen replay')
        return
    assert torch.cuda.is_available() and not args.output.exists()
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder,Siglip2CompactIndex
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve()==root/'src/sfora/siglip2_compact_serving.py'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    flags=teacher.qualified.numerical_flags()
    assert flags==prior['numerical_flags']
    torch.cuda.reset_peak_memory_stats()
    encoder=Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot,checkpoint=teacher.TEACHER,expected_checkpoint_sha256=teacher.TEACHER_SHA,model_file_sha256=pair.smoke.MODEL_HASHES,precision='fp16_native',device=torch.device('cuda'))
    def preserved():
        assert pair.smoke.digest(trained.base.whole_state(encoder.vision))==authority['fp16_whole_sha256'] and pair.smoke.digest(encoder.head.state_dict())==cpu['teacher_head_sha256']
        assert json.loads(json.dumps(trained.native.environment(encoder.vision,encoder.processor)))==cpu['environment']
        assert all(p.grad is None for m in (encoder.vision,encoder.head) for p in m.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.vision.modules())
    preserved()
    cpu_rng,cuda_rng=torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
    def images(role,start,end):
        result=[]
        for r in qualified['protocol'][role][start:end]:
            path=control.dataset_root/r['relative_path']
            assert pair.sha(path)==r['image_sha256']
            with Image.open(path) as image:
                result.append(image.convert('RGB'))
        return result
    facts={'batch':32 if not args.b32 else 1}
    if not args.b32:
        arrays={}
        for role in ('query','gallery'):
            chunks=[]
            for start in range(0,len(qualified['protocol'][role]),32):
                batch=images(role,start,start+32)
                actual=encoder.encode_images(batch)
                if start==0:
                    with torch.inference_mode():
                        pixels=pair.pixels(encoder.processor,batch,'large').cuda().half()
                        pooled=encoder.vision(pixel_values=pixels).pooler_output
                        reference=pack_int8_unit_embeddings(torch.nn.functional.normalize(pair.smoke.compact_head_features(pooled,encoder.head),dim=1).cpu())
                    fp16.same(actual,reference)
                chunks.append(actual)
                assert torch.cuda.max_memory_allocated()<10_000_000_000
                if (start//32+1)%32==0 or start+32>=len(qualified['protocol'][role]):
                    print(json.dumps({role+'_public_B32_images':min(start+32,len(qualified['protocol'][role]))}),flush=True)
            arrays[role]=PackedInt8Embeddings(torch.cat([v.codes for v in chunks]),torch.cat([v.inverse_norms for v in chunks]))
        q,g=arrays['query'],arrays['gallery']
        with serving.CutilePackedInt8Gallery.open_packed(serving.LIBRARY,g) as native:
            for start in range(0,len(q.codes),32):
                block=PackedInt8Embeddings(q.codes[start:start+32].contiguous(),q.inverse_norms[start:start+32].contiguous())
                actual=native.search_packed(block)
                scores=(block.codes.float().cuda()@g.codes.float().cuda().T)*block.inverse_norms.float().cuda()[:,None]*g.inverse_norms.float().cuda()[None,:]
                order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:10]
                pilot.equal(actual,(order.cpu().numpy(),scores.gather(1,order).cpu().numpy()))
        facts['native_all_query_top10_ordinal_score_bits_exact']=True
    else:
        assert pair.sha(args.b32/'receipt.json')==args.b32_sha256 and pair.sha(args.b32/'cpu-audit.json')==args.b32_audit_sha256
        receipt=json.loads((args.b32/'receipt.json').read_text())
        audit=json.loads((args.b32/'cpu-audit.json').read_text())
        assert receipt['advance_minimum_dated_reference_screen'] and audit['pass'] and audit['advance_minimum_dated_reference_screen'] and receipt['code']==code and audit['receipt_sha256']==args.b32_sha256
        g=fp16.load_packed(args.b32,receipt,'gallery')
        captured,hits=[],[]
        with Siglip2CompactIndex(encoder,serving.CutilePackedInt8Gallery.open_packed(serving.LIBRARY,g)) as index:
            actual_search=index.gallery.search_packed
            def capture(query):
                assert query.codes.shape==(1,128)
                captured.append(query)
                return actual_search(query)
            with patch.object(index.gallery,'search_packed',capture):
                for i in range(14218):
                    ordinals,_=index.search_images(images('query',i,i+1))
                    hits.append(int(qlabels[i]==glabels[int(ordinals[0,0])]))
                    assert torch.cuda.max_memory_allocated()<10_000_000_000
                    if (i+1)%256==0 or i+1==14218:
                        print(json.dumps({'official_public_B1_queries':i+1}),flush=True)
        q=PackedInt8Embeddings(torch.cat([v.codes for v in captured]),torch.cat([v.inverse_norms for v in captured]))
        arrays={'query':q}
        facts.update({'native_all_public_B1_query_r1_exact':True,'b32_receipt_sha256':args.b32_sha256,'b32_audit_sha256':args.b32_audit_sha256})
    quality,intervals,advance=metrics(official,q,g,qlabels,glabels,torch.device('cuda'))
    if args.b32:
        assert hits==quality['per_query_r1']
    preserved()
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    assert teacher.qualified.numerical_flags()==flags and pair.sha(teacher.TEACHER)==teacher.TEACHER_SHA and pair.sha(serving.LIBRARY)==serving.LIBRARY_SHA and pair.sha(control.dataset_root/'Eval/list_eval_partition.txt')==PARTITION_SHA
    assert all(pair.sha(root/n)==h for n,h in code.items()) and torch.cuda.max_memory_allocated()<10_000_000_000
    args.output.mkdir(exist_ok=False)
    for name,values in arrays.items():
        for field,value in (('codes',values.codes),('inverse',values.inverse_norms)):
            path=args.output/(name+'.'+field+'.npy')
            np.save(path,value.numpy(),allow_pickle=False)
            facts[name+'_'+field+'_sha256']=pair.sha(path)
    facts.update({'code':code,'execution_sha256':args.execution_sha256,'source_checkpoint_sha256':teacher.TEACHER_SHA,'cpu_authority_sha256':args.cpu_sha256,'query_images':14218,'gallery_images':12612,'query_gallery_products':len(set(qlabels)),'quality':quality,'quality_intervals':intervals,'advance_minimum_dated_reference_screen':advance,'minimum_dated_reference_r1':.967,'strongest_current_reference_verified':False,'source_state_rng_environment_code_library_preserved':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'numerical_flags':flags,'optimizer_updates':0,'official_read':True,'prior_official_benchmark_exposure':True,'claim_eligible':False,'p99_certified':False})
    pair.smoke.save(args.output/'receipt.json',facts)
    print('GO minimum dated-reference official screen' if advance else 'KILL minimum dated-reference official quality screen')


if __name__=='__main__':
    main()
