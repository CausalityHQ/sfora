#!/usr/bin/env python3
"""One immutable100-update chunk of the fixed2000 budget, with complete resume."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import pe_large_optimization as driver
import train_pe_large_optimization as mechanics

pair=driver.pair
MECHANICS_CODE_SHA='b050801f6f15434873443f979dbd1c9645df1f0e065c2ad5a99b9a8e963baa50'


def startup(root,execution_sha,mechanics_path,mechanics_sha):
    manifest=root/'large-optimization-chunk-execution.json'
    assert pair.sha(manifest)==execution_sha
    code=json.loads(manifest.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items())
    assert pair.sha(root/'large-optimization-gpu-execution.json')==MECHANICS_CODE_SHA
    prior_code=json.loads((root/'large-optimization-gpu-execution.json').read_text())
    assert len(code)==len(prior_code)+1 and all(code[n]==h for n,h in prior_code.items())
    assert pair.sha(root/'large-optimization-execution.json')==mechanics.CPU_CODE_SHA
    cpu_code=json.loads((root/'large-optimization-execution.json').read_text())
    assert all(code[n]==h for n,h in cpu_code.items())
    assert pair.sha(root/'large-coverage-checkpoint-execution.json')==driver.PREVIOUS_SHA
    assert all(code[n]==h for n,h in json.loads((root/'large-coverage-checkpoint-execution.json').read_text()).items())
    control,source,prior,proof,_=driver.previous.training.startup(root,driver.previous.TRAIN_CODE_SHA)
    assert pair.sha(driver.FULL/'cpu-audit.json')==driver.FULL_AUDIT_SHA and pair.sha(driver.HALF/'cpu-audit.json')==driver.HALF_AUDIT_SHA
    audit=json.loads((driver.FULL/'cpu-audit.json').read_text())
    effect=audit['matched_pool_comparison']
    assert audit['pass'] and not effect['survivor'] and all(effect[k]['product_lower95']>0 for k in ('per_query_r1','per_query_ap'))
    assert pair.sha(driver.FULL/'receipt.json')==audit['receipt_sha256']
    assert pair.sha(driver.HALF/'receipt.json')==effect['control_receipt_sha256'] and effect['control_audit_sha256']==driver.HALF_AUDIT_SHA
    assert pair.sha(mechanics.CPU/'optimization-cpu-proof-v3.json')==mechanics.CPU_SHA
    cpu=json.loads((mechanics.CPU/'optimization-cpu-proof-v3.json').read_text())
    assert cpu['pass'] and cpu['code']==cpu_code and cpu['execution_sha256']==mechanics.CPU_CODE_SHA
    assert pair.sha(mechanics_path/'receipt.json')==mechanics_sha
    qualified=json.loads((mechanics_path/'receipt.json').read_text())
    assert qualified['advance'] and qualified['updates']==17 and qualified['arm']=='full'
    assert qualified['execution_sha256']==MECHANICS_CODE_SHA and qualified['cpu_authority_sha256']==mechanics.CPU_SHA
    assert qualified['source_checkpoint_sha256']==driver.coverage.teacher.TEACHER_SHA
    assert qualified['training_state_discarded'] and not qualified['quality_read']
    assert qualified['native_17_equals_serialized8_plus9_exact'] and qualified['strict400_reload_whole_head_packed_exact']
    assert qualified['chunk100_admission_seconds']<=269 and qualified['peak_cuda_allocated_bytes']<10_000_000_000
    driver.previous.selected.helpers(root,code)
    return control,source,prior,proof,code,cpu,qualified


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--mechanics',type=Path,required=True)
    parser.add_argument('--mechanics-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=('half','full'),required=True)
    parser.add_argument('--chunk-end',type=int,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--previous-sha256')
    parser.add_argument('--startup-only',action='store_true')
    args=parser.parse_args()
    assert 100<=args.chunk_end<=driver.TOTAL_UPDATES and args.chunk_end%100==0
    start=args.chunk_end-100
    assert bool(start)==bool(args.previous)==bool(args.previous_sha256)
    assert not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    root=Path(__file__).resolve().parent
    control,source,prior,proof,code,cpu,qualified=startup(root,args.execution_sha256,args.mechanics,args.mechanics_sha256)
    if args.startup_only:
        assert not torch.cuda.is_available() and start==0
        pair.smoke.save(args.output,{'pass':True,'execution_sha256':args.execution_sha256,'mechanics_sha256':args.mechanics_sha256,'quality_read':False,'optimizer_updates':0})
        print('PASS chunk frozen authority and actual import closure; zero updates')
        return
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    flags=driver.coverage.teacher.qualified.numerical_flags()
    assert flags==prior['numerical_flags']
    args.output.mkdir(exist_ok=False)
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    state=driver.fresh(control,source,proof,args.arm,'cuda')
    arm=proof['arms'][args.arm]
    target=state['target'].cpu().numpy()
    batches=driver.schedule(target)
    schedule_sha=pair.smoke.digest({'batches':torch.from_numpy(batches)})
    assert schedule_sha==cpu['schedules'][args.arm]['schedule_sha256']
    identity=driver.identity(state,proof,args.arm,args.execution_sha256,schedule_sha)
    previous=None
    if start:
        assert pair.sha(args.previous/'receipt.json')==args.previous_sha256
        previous=json.loads((args.previous/'receipt.json').read_text())
        assert previous['advance'] and previous['completed_step']==start and previous['arm']==args.arm
        assert previous['execution_sha256']==args.execution_sha256 and previous['mechanics_sha256']==args.mechanics_sha256
        assert previous['source_checkpoint_sha256']==driver.coverage.teacher.TEACHER_SHA and previous['schedule_sha256']==schedule_sha
        driver.restore(state,identity,args.previous/'resume.pt',previous['checkpoint_sha256'],expected_step=start)
    else:
        assert state['counter']==0 and not state['optimizer'].state
    old_root=Path('/home/riomus/runs/sfora-large-coverage-'+args.arm+'-100-v1')
    old_sha=mechanics.OLD_FULL_SHA if args.arm=='full' else '00a8ced0364f6ace599764256f27062ba81ef822f9d04685f4f2abfd0b22f183'
    assert pair.sha(old_root/'receipt.json')==old_sha
    old=json.loads((old_root/'receipt.json').read_text())
    cpu_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all()
    counts=np.bincount(target)
    seen=set(target[batches[:start].ravel()].tolist())
    rows=[]
    training_started=time.perf_counter()
    for step in range(start+1,args.chunk_end+1):
        torch.cuda.synchronize()
        tick=time.perf_counter()
        batch=tuple(batches[step-1].tolist())
        images,rgb=pair.augmented_images(control.dataset_root,arm['rows'],batch,step)
        pixels=pair.pixels(state['processor'],images,'large')
        pixel_sha=pair.smoke.digest({'pixels':pixels})
        rank=bool((counts[target[list(batch)]]>1).all())
        row=driver.step(state,pixels,batch,rank)
        row.update(rgb_sha256=rgb,pixels_sha256=pixel_sha)
        if step<=100:
            assert rank==arm['rank_active'][step-1]
            assert all(row[k]==old['steps'][step-1][k] for k in row),'fresh first100 differs from matched native recipe'
        seen.update(target[list(batch)].tolist())
        assert torch.cuda.max_memory_allocated()<10_000_000_000
        torch.cuda.synchronize()
        row.update(seconds=time.perf_counter()-tick,gradient_input_identities=len(seen),rank_active=rank)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del images,pixels
    training_wall=time.perf_counter()-training_started
    assert state['counter']==args.chunk_end and len(seen)==len(arm['classes'])
    checkpoint_sha=driver.save(state,identity,args.output/'resume.pt')
    assert driver.coverage.trained.base.runtime_identity(state['model'])==identity['runtime']
    assert json.loads(json.dumps(driver.coverage.trained.native.environment(state['model'],state['processor'])))==source['environment']
    assert driver.coverage.frozen_digest(state['model'],state['inventory'])==proof['frozen_prefix_sha256']
    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True))
    assert all(pair.sha(root/n)==h for n,h in code.items()) and pair.sha(driver.coverage.teacher.TEACHER)==driver.coverage.teacher.TEACHER_SHA
    assert driver.coverage.teacher.qualified.numerical_flags()==flags and torch.cuda.max_memory_allocated()<10_000_000_000
    exposure=np.bincount(target[batches[:args.chunk_end].ravel()],minlength=len(arm['classes']))
    receipt={'advance':True,'arm':args.arm,'completed_step':args.chunk_end,'chunk_start':start,'updates':100,'total_updates':driver.TOTAL_UPDATES,'execution_sha256':args.execution_sha256,'cpu_authority_sha256':mechanics.CPU_SHA,'mechanics_sha256':args.mechanics_sha256,'source_checkpoint_sha256':driver.coverage.teacher.TEACHER_SHA,'schedule_sha256':schedule_sha,'previous_receipt_sha256':args.previous_sha256,'previous_checkpoint_sha256':previous['checkpoint_sha256'] if previous else None,'checkpoint_sha256':checkpoint_sha,'steps':rows,'training_wall_seconds':training_wall,'images_per_second':6400/training_wall,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'actual_gradient_input_identities':len(seen),'minimum_id_exposure':int(exposure.min()),'maximum_id_exposure':int(exposure.max()),'frozen_source_code_environment_rng_preserved':True,'quality_read':False,'claim_eligible':False,'total_seconds':time.perf_counter()-started}
    pair.smoke.save(args.output/'receipt.json',receipt)
    print('PASS native100-update persisted optimization chunk; no quality claim',flush=True)


if __name__=='__main__':
    main()
