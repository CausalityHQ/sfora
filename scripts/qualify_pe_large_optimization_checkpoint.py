#!/usr/bin/env python3
"""Authenticate fixed2000 state; reuse the unchanged source/official qualifier."""
import argparse
import json
import sys
from pathlib import Path
import torch
import pe_large_optimization as driver
import qualify_pe_large_coverage_checkpoint as confirmation

pair=driver.pair
TRAIN_CODE_SHA='a7c27140c1cd53d1aee3b41f6e1c7bfb43173a467cab3855cfc2c92cac41bf7f'
MECHANICS_SHA='41cdd66db34b6e55d7aca72945133b6e25d0cea10cf18aae2a666d4ff4d1be1d'
CPU_SHA='c1e3e15a06ff6723a83713543cbdc07d8e9c78d3178ea7bd5c042fd91b23b143'


def authority(root,execution_sha,arm,training_sha):
    manifest=root/'large-optimization-checkpoint-execution.json'
    assert pair.sha(manifest)==execution_sha
    code=json.loads(manifest.read_text())
    assert all(pair.sha(root/n)==h for n,h in code.items()),'coverage checkpoint code differs'
    assert pair.sha(root/'large-optimization-chunk-execution.json')==TRAIN_CODE_SHA
    prior_code=json.loads((root/'large-optimization-chunk-execution.json').read_text())
    assert len(code)==len(prior_code)+1 and all(code[n]==h for n,h in prior_code.items())
    terminal_path=Path(f'/home/riomus/runs/sfora-large-optimization-{arm}-2000-v1/receipt.json')
    assert terminal_path.is_file(),'fixed2000 terminal training authority unavailable'
    assert pair.sha(terminal_path)==training_sha,'fixed2000 terminal training receipt differs'
    assert pair.sha(Path('/home/riomus/runs/sfora-large-optimization-v2/optimization-cpu-proof-v3.json'))==CPU_SHA
    assert pair.sha(Path('/home/riomus/runs/sfora-large-optimization-mechanics-v2/receipt.json'))==MECHANICS_SHA
    control,source,prior,proof,_=confirmation.training.startup(root,confirmation.TRAIN_CODE_SHA)
    assert pair.sha(driver.FULL/'cpu-audit.json')==driver.FULL_AUDIT_SHA and pair.sha(driver.HALF/'cpu-audit.json')==driver.HALF_AUDIT_SHA
    assert pair.sha(confirmation.PROTOCOL)==confirmation.PROTOCOL_SHA
    protocol=json.loads(confirmation.PROTOCOL.read_text())
    assert protocol['pass'] and protocol['train_query_gallery_ids_disjoint'] and protocol['prior_official_benchmark_exposure']
    assert pair.sha(control.dataset_root/'Eval/list_eval_partition.txt')==protocol['partition_sha256']==confirmation.selected.PARTITION_SHA
    last=None
    last_sha=None
    training_seconds=0.
    for end in range(100,2001,100):
        run=Path(f'/home/riomus/runs/sfora-large-optimization-{arm}-{end}-v1')
        value=json.loads((run/'receipt.json').read_text())
        assert value['advance'] and value['arm']==arm and value['completed_step']==end and value['chunk_start']==end-100
        assert value['updates']==100 and value['total_updates']==2000 and not value['quality_read']
        assert value['execution_sha256']==TRAIN_CODE_SHA and value['mechanics_sha256']==MECHANICS_SHA and value['cpu_authority_sha256']==CPU_SHA
        assert value['source_checkpoint_sha256']==driver.coverage.teacher.TEACHER_SHA and value['frozen_source_code_environment_rng_preserved']
        assert [r['step'] for r in value['steps']]==list(range(end-99,end+1))
        assert value['peak_cuda_allocated_bytes']<10_000_000_000 and value['actual_gradient_input_identities']==len(proof['arms'][arm]['classes'])
        assert value['previous_receipt_sha256']==last_sha
        assert value['previous_checkpoint_sha256']==(last['checkpoint_sha256'] if last else None)
        if last:
            assert value['schedule_sha256']==last['schedule_sha256']
        log=(root/f'{arm}-{end}.log').read_text()
        assert 'Finished with result: success' in log and 'Main processes terminated with: code=exited/status=0' in log and 'Memory swap peak: 0B' in log
        last=value
        last_sha=pair.sha(run/'receipt.json')
        training_seconds+=value['training_wall_seconds']
    assert last_sha==training_sha and pair.sha(run/'resume.pt')==last['checkpoint_sha256']
    last['aggregate_training_wall_seconds']=training_seconds
    official=confirmation.selected.helpers(root,code)
    return control,source,prior,proof,run,last,protocol,official,code


def startup(root,execution_sha,arm,training_sha):
    control,source,prior,proof,training_run,terminal,protocol,official,code=authority(root,execution_sha,arm,training_sha)
    run=Path(f'/home/riomus/runs/sfora-large-optimization-{arm}-weights-v1')
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['pass'] and receipt['completed_step']==2000 and receipt['arm']==arm
    assert receipt['execution_sha256']==execution_sha and receipt['training_receipt_sha256']==training_sha
    assert receipt['source_resume_sha256']==terminal['checkpoint_sha256'] and receipt['source_checkpoint_sha256']==driver.coverage.teacher.TEACHER_SHA
    assert pair.sha(run/'native.pt')==receipt['checkpoint_sha256']
    return control,source,prior,proof,run,receipt,protocol,official,code


def export():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--arm',choices=('half','full'),required=True)
    parser.add_argument('--training-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--export-cpu',action='store_true',required=True)
    args=parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    assert args.output==Path(f'/home/riomus/runs/sfora-large-optimization-{args.arm}-weights-v1')
    torch.set_num_threads(8)
    root=Path(__file__).resolve().parent
    _,_,_,proof,run,terminal,_,_,code=authority(root,args.execution_sha256,args.arm,args.training_sha256)
    disk=torch.load(run/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
    identity=disk['identity']
    assert identity['global_step']==identity['total_updates']==2000 and identity['arm']==args.arm
    assert identity['execution_sha256']==TRAIN_CODE_SHA and identity['schedule_sha256']==terminal['schedule_sha256']
    assert identity['source_checkpoint_sha256']==driver.coverage.teacher.TEACHER_SHA
    assert identity['initializers_sha256']==proof['arms'][args.arm]['initializers_sha256'] and identity['frozen_prefix_sha256']==proof['frozen_prefix_sha256']
    assert len(disk['vision'])==400 and driver.fingerprint(disk['buffers'])==identity['buffers_sha256']
    assert len(disk['optimizer']['state'])==len(identity['parameter_names']) and all(int(s['step'])==2000 for s in disk['optimizer']['state'].values())
    assert [{k:v for k,v in g.items() if k!='params'} for g in disk['optimizer']['param_groups']]==identity['optimizer_groups']
    assert identity['precision']=='cuda_fp16' and disk['scaler'] and len(disk['cuda_rng'])==1
    values={**disk['vision'],**disk['head'],'classifier':disk['classifier'],'bank':disk['bank']}
    assert all(torch.isfinite(v).all() for v in values.values())
    assert all(torch.isfinite(v).all() for s in disk['optimizer']['state'].values() for v in s.values() if isinstance(v,torch.Tensor))
    args.output.mkdir(exist_ok=False)
    torch.save({k:disk[k] for k in ('vision','head','classifier','bank')}|{'classes':proof['arms'][args.arm]['classes']},args.output/'native.pt')
    whole={**disk['vision'],**{'runtime.'+n:v for n,v in disk['buffers'].items()}}
    assert all(pair.sha(root/n)==h for n,h in code.items())
    pair.smoke.save(args.output/'receipt.json',{'pass':True,'arm':args.arm,'completed_step':2000,'execution_sha256':args.execution_sha256,'training_receipt_sha256':args.training_sha256,'source_resume_sha256':terminal['checkpoint_sha256'],'source_checkpoint_sha256':driver.coverage.teacher.TEACHER_SHA,'checkpoint_sha256':pair.sha(args.output/'native.pt'),'updated_whole_sha256':pair.smoke.digest(whole),'updated_head_sha256':pair.smoke.digest(disk['head']),'aggregate_training_wall_seconds':terminal['aggregate_training_wall_seconds'],'aggregate_images_per_second':128000/terminal['aggregate_training_wall_seconds'],'optimizer_updates':0,'quality_read':False,'claim_eligible':False})
    print('PASS fixed2000 exact serving-weight export; source/official qualification still required')


if __name__=='__main__':
    if '--export-cpu' in sys.argv:
        export()
    else:
        # Keep original native/public/scorer/parity/CI implementation unchanged.
        with confirmation.patch.object(confirmation,'startup',startup):
            confirmation.main()
