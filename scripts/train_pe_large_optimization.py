#!/usr/bin/env python3
"""Discard-only native GPU resume mechanics for the frozen optimization budget."""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pe_large_optimization as driver

pair=driver.pair
CPU=Path('/home/riomus/runs/sfora-large-optimization-v2')
CPU_SHA='c1e3e15a06ff6723a83713543cbdc07d8e9c78d3178ea7bd5c042fd91b23b143'
CPU_CODE_SHA='cb0e89a375c60d7fbebb54d09b10c07d193f3ca24002cf59d71fc8c5764af2b9'
OLD_FULL=Path('/home/riomus/runs/sfora-large-coverage-full-100-v1')
OLD_FULL_SHA='49f64a7e273e37e2df5efcf2249de90b153cee486bb123d61702889d0bb34eb5'


def startup(root,execution_sha):
    path=root/'large-optimization-gpu-execution.json'
    assert pair.sha(path)==execution_sha
    code=json.loads(path.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items())
    assert pair.sha(root/'large-optimization-execution.json')==CPU_CODE_SHA
    cpu_code=json.loads((root/'large-optimization-execution.json').read_text())
    assert len(code)==len(cpu_code)+1 and all(code[n]==h for n,h in cpu_code.items())
    assert pair.sha(root/'large-coverage-checkpoint-execution.json')==driver.PREVIOUS_SHA
    old=json.loads((root/'large-coverage-checkpoint-execution.json').read_text())
    assert all(code[n]==h for n,h in old.items())
    control,source,prior,proof,_=driver.previous.training.startup(root,driver.previous.TRAIN_CODE_SHA)
    assert pair.sha(driver.FULL/'cpu-audit.json')==driver.FULL_AUDIT_SHA
    assert pair.sha(driver.HALF/'cpu-audit.json')==driver.HALF_AUDIT_SHA
    audit=json.loads((driver.FULL/'cpu-audit.json').read_text())
    effect=audit['matched_pool_comparison']
    assert audit['pass'] and not effect['survivor']
    assert pair.sha(driver.FULL/'receipt.json')==audit['receipt_sha256']
    assert pair.sha(driver.HALF/'receipt.json')==effect['control_receipt_sha256']
    assert effect['control_audit_sha256']==driver.HALF_AUDIT_SHA
    assert all(effect[k]['product_lower95']>0 for k in ('per_query_r1','per_query_ap'))
    assert pair.sha(CPU/'optimization-cpu-proof-v3.json')==CPU_SHA
    cpu=json.loads((CPU/'optimization-cpu-proof-v3.json').read_text())
    assert cpu['pass'] and cpu['execution_sha256']==CPU_CODE_SHA and cpu['code']==cpu_code
    assert cpu['actual_native_CPU_resume_update1_then_fresh_objects_update2_exact']
    assert not cpu['quality_read'] and not cpu['GPU_training_qualified']
    driver.previous.selected.helpers(root,code)
    return control,source,prior,proof,code,cpu


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--startup-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    assert not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control,source,prior,proof,code,cpu=startup(root,args.execution_sha256)
    if args.startup_only:
        assert not torch.cuda.is_available()
        pair.smoke.save(args.output,{'pass':True,'execution_sha256':args.execution_sha256,'cpu_authority_sha256':CPU_SHA,'quality_read':False,'GPU_training_qualified':False})
        print('PASS GPU driver frozen authority and actual import closure; no updates')
        return
    assert torch.cuda.is_available()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    flags=driver.coverage.teacher.qualified.numerical_flags()
    assert flags==prior['numerical_flags']
    assert pair.sha(OLD_FULL/'receipt.json')==OLD_FULL_SHA
    old=json.loads((OLD_FULL/'receipt.json').read_text())
    args.output.mkdir(exist_ok=False)
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    state=driver.fresh(control,source,proof,'full','cuda')
    arm=proof['arms']['full']
    batches=driver.schedule(state['target'].cpu().numpy())
    schedule_sha=pair.smoke.digest({'batches':torch.from_numpy(batches)})
    assert schedule_sha==cpu['schedules']['full']['schedule_sha256']
    base=driver.identity(state,proof,'full',args.execution_sha256,schedule_sha)
    images,_=pair.augmented_images(control.dataset_root,arm['rows'],tuple(batches[0][:2]),1)
    calibration=pair.pixels(state['processor'],images,'large').cuda()
    def cosines(s):
        with torch.no_grad():
            a=s['model'](pixel_values=calibration).pooler_output.float()
            b=driver.previous.training.fp16(s['model'],calibration)
            result={'pooled':F.cosine_similarity(a,b).tolist(),'compact':F.cosine_similarity(pair.smoke.compact_head_features(a,s['head']),pair.smoke.compact_head_features(b,s['head'])).tolist()}
            assert all(min(v)>=.999 for v in result.values())
            return result
    initial_cos=cosines(state)
    del images
    cpu_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all()
    counts=np.bincount(state['target'].cpu().numpy())
    target=state['target'].cpu().numpy()
    def update(s,step):
        torch.cuda.synchronize()
        tick=time.perf_counter()
        batch=tuple(batches[step-1])
        images,rgb=pair.augmented_images(control.dataset_root,arm['rows'],batch,step)
        pixels=pair.pixels(s['processor'],images,'large')
        pixel_sha=pair.smoke.digest({'pixels':pixels})
        rank=bool((counts[target[list(batch)]]>1).all())
        assert rank==arm['rank_active'][step-1]
        row=driver.step(s,pixels,batch,rank)
        row.update(rgb_sha256=rgb,pixels_sha256=pixel_sha)
        reference=old['steps'][step-1]
        assert all(row[k]==reference[k] for k in row),'first17 differs from closed native100 recipe'
        assert torch.cuda.max_memory_allocated()<10_000_000_000
        torch.cuda.synchronize()
        row['seconds']=time.perf_counter()-tick
        print(json.dumps(row),flush=True)
        return row
    with TemporaryDirectory(prefix='discard-optimization-mechanics-',dir=root) as tmp:
        path=Path(tmp)/'step8.pt'
        rows=[]
        for step in range(1,18):
            rows.append(update(state,step))
            if step==8:
                sha=driver.save(state,base,path)
        expected=driver.fingerprint(driver.payload(state,base))
        del state
        gc.collect()
        torch.cuda.empty_cache()
        state=driver.fresh(control,source,proof,'full','cuda')
        assert driver.identity(state,proof,'full',args.execution_sha256,schedule_sha)==base
        driver.restore(state,base,path,sha,expected_step=8)
        resumed=[]
        for step in range(9,18):
            row=update(state,step)
            assert {k:v for k,v in row.items() if k!='seconds'}=={k:v for k,v in rows[step-1].items() if k!='seconds'}
            resumed.append(row)
        assert driver.fingerprint(driver.payload(state,base))==expected,'native GPU resume state differs'
        assert torch.equal(cpu_rng,torch.random.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
        driver.save(state,base,Path(tmp)/'step17.pt')
        del state['optimizer']
        state['model'].eval()
        state['head'].eval()
        disk=torch.load(Path(tmp)/'step17.pt',map_location='cpu',weights_only=True,mmap=True)
        with torch.random.fork_rng(devices=[0]):
            loaded=type(state['model'])(copy.deepcopy(state['model'].config)).float().eval()
            loaded.load_state_dict(disk['vision'],strict=True)
            assert driver.coverage.frozen_digest(loaded,state['inventory'])==proof['frozen_prefix_sha256']
            assert pair.smoke.digest(driver.coverage.trained.base.whole_state(loaded))==pair.smoke.digest(driver.coverage.trained.base.whole_state(state['model']))
            loaded.cuda()
            head=nn.Linear(1024,128).eval().cuda()
            head.load_state_dict(disk['head'],strict=True)
            with torch.no_grad():
                a=driver.previous.training.fp16(state['model'],calibration)
                b=driver.previous.training.fp16(loaded,calibration)
                assert torch.equal(a,b)
                va=F.normalize(pair.smoke.compact_head_features(a,state['head']),dim=1)
                vb=F.normalize(pair.smoke.compact_head_features(b,head),dim=1)
                assert torch.equal(va,vb)
                driver.previous.training.packed_equal(va,vb)
            terminal_cos=cosines({'model':loaded,'head':head})
    admission=100*max(float(np.median([r['seconds'] for r in rows[2:]])),sum(r['seconds'] for r in rows)/17)+30
    assert admission<=269,'100-update chunk resource admission fails'
    assert torch.cuda.max_memory_allocated()<10_000_000_000
    assert all(pair.sha(root/n)==h for n,h in code.items())
    assert pair.sha(driver.coverage.teacher.TEACHER)==driver.coverage.teacher.TEACHER_SHA
    assert driver.coverage.teacher.qualified.numerical_flags()==flags
    assert json.loads(json.dumps(driver.coverage.trained.native.environment(state['model'],state['processor'])))==source['environment']
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    pair.smoke.save(args.output/'receipt.json',{'advance':True,'arm':'full','updates':17,'execution_sha256':args.execution_sha256,'cpu_authority_sha256':CPU_SHA,'source_checkpoint_sha256':driver.coverage.teacher.TEACHER_SHA,'schedule_sha256':schedule_sha,'steps':rows,'resumed_steps':resumed,'terminal_state_fingerprint':expected,'native_17_equals_serialized8_plus9_exact':True,'strict400_reload_whole_head_packed_exact':True,'initial_cosines':initial_cos,'terminal_cosines':terminal_cos,'training_state_discarded':True,'quality_read':False,'claim_eligible':False,'chunk100_admission_seconds':admission,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'total_seconds':time.perf_counter()-started})
    print('PASS discarded17 native GPU optimizer/bank/scaler/RNG resume mechanics',flush=True)


if __name__=='__main__':
    main()
