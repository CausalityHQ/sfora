#!/usr/bin/env python3
"""Actual native CPU optimizer/bank/RNG process-boundary qualification."""
import argparse
import gc
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
assert importlib.util.find_spec('pe_large_optimization') is not None, 'native optimizer/bank/RNG resume capability is missing'
import pe_large_optimization as driver


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(driver.pair.SEED)
    root=Path(__file__).resolve().parent
    control,source,prior,proof,code=driver.startup(root,args.execution_sha256)
    schedules={}
    for arm in ('half','full'):
        target=driver.initializers(proof,arm)['target']
        batches=driver.schedule(target.numpy())
        assert np.array_equal(batches[:100],np.asarray(proof['arms'][arm]['batches']))
        exposure=np.bincount(target.numpy()[batches.ravel()],minlength=len(proof['arms'][arm]['classes']))
        schedules[arm]={'schedule_sha256':driver.pair.smoke.digest({'batches':torch.from_numpy(batches)}),'minimum_id_exposure':int(exposure.min()),'maximum_id_exposure':int(exposure.max()),'first100_schedule_exact':True}
    state=driver.fresh(control,source,proof,'full','cpu')
    identity=driver.identity(state,proof,'full',args.execution_sha256,schedules['full']['schedule_sha256'])
    arm=proof['arms']['full']
    batches=[tuple(arm['batches'][i][:2]) for i in range(2)]
    pixels=[]
    for i,batch in enumerate(batches,1):
        images,_=driver.pair.augmented_images(control.dataset_root,arm['rows'],batch,i)
        pixels.append(driver.pair.pixels(state['processor'],images,'large'))
    with TemporaryDirectory(prefix='discard-native-cpu-resume-',dir=root) as tmp:
        path=Path(tmp)/'step1.pt'
        first=driver.step(state,pixels[0],batches[0],arm['rank_active'][0],micro=1)
        sha=driver.save(state,identity,path)
        second=driver.step(state,pixels[1],batches[1],arm['rank_active'][1],micro=1)
        expected=driver.fingerprint(driver.payload(state,identity))
        del state
        gc.collect()
        state=driver.fresh(control,source,proof,'full','cpu')
        for changed in ({**identity,'parameter_names':list(reversed(identity['parameter_names']))},{**identity,'total_updates':100},{**identity,'source_checkpoint_sha256':'0'*64}):
            try:
                driver.restore(state,changed,path,sha,expected_step=1)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed resume identity accepted')
        try:
            driver.restore(state,identity,path,sha,expected_step=2)
        except AssertionError:
            pass
        else:
            raise AssertionError('wrong global resume counter accepted')
        driver.restore(state,identity,path,sha,expected_step=1)
        actual=driver.step(state,pixels[1],batches[1],arm['rank_active'][1],micro=1)
        assert second==actual and driver.fingerprint(driver.payload(state,identity))==expected,'actual native CPU resume differs'
        assert driver.coverage.frozen_digest(state['model'],state['inventory'])==proof['frozen_prefix_sha256']
        old=Path('/home/riomus/runs/sfora-large-coverage-full-100-v1/native.pt')
        try:
            driver.restore(state,identity,old,'2f5b82cd3ca209303d693bd9f27e4d1845ba8310d15069505de4148001d3da4c',expected_step=100)
        except AssertionError:
            pass
        else:
            raise AssertionError('weight-only100 checkpoint accepted as optimizer resume')
    assert all(driver.pair.sha(root/n)==h for n,h in code.items())
    driver.pair.smoke.save(args.output,{'pass':True,'execution_sha256':args.execution_sha256,'code':code,'source_checkpoint_sha256':driver.coverage.teacher.TEACHER_SHA,'actual_native_CPU_resume_update1_then_fresh_objects_update2_exact':True,'whole_head_classifier_bank_AdamW_CPU_RNG_exact':True,'wrong_counter_parameter_order_budget_source_rejected':True,'old_weight_only100_checkpoint_rejected':True,'temporary_states_discarded':True,'first_update':first,'second_update':second,'terminal_state_fingerprint':expected,'schedules':schedules,'quality_read':False,'GPU_training_qualified':False,'claim_eligible':False})
    print('PASS actual native CPU AdamW/bank/RNG fresh-object resume; no GPU/quality qualification')


if __name__=='__main__':
    main()
